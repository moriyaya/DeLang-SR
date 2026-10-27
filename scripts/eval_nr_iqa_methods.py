#!/usr/bin/env python3
"""Evaluate no-reference IQA metrics for one method or many method folders.

Metrics:
- NIQE: lower is better.
- MUSIQ, MANIQA, CLIP-IQA: higher is better.

The metrics are no-reference metrics, so GT images are not used as metric
inputs. If --gt_dir is provided, this script only uses it to verify image
pairing and records the matched GT path in the per-image table.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_NAMES = ["niqe", "musiq", "maniqa", "clipiqa"]
TABLE_NAMES = {
    "niqe": "NIQE",
    "musiq": "MUSIQ",
    "maniqa": "MANIQA",
    "clipiqa": "CLIP-IQA",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute NIQE/MUSIQ/MANIQA/CLIP-IQA for restored test images."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_dir", type=Path, help="One method output image folder.")
    group.add_argument(
        "--methods_root",
        type=Path,
        help="Root folder whose immediate subfolders are method output folders.",
    )
    parser.add_argument("--gt_dir", type=Path, default=None, help="Optional GT folder for pairing check.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--method", type=str, default=None, help="Method name for --pred_dir mode.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively inside each method folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda, cuda:0, cpu, etc. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRIC_NAMES,
        choices=METRIC_NAMES,
        help="Subset of no-reference metrics to compute.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N image limit for debugging.")
    parser.add_argument("--resize_max_side", type=int, default=None, help="Optional downsize before scoring.")
    parser.add_argument(
        "--strict_gt",
        action="store_true",
        help="Skip images without matched GT when --gt_dir is provided.",
    )
    return parser.parse_args()


def resolve_device(device):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[nr-iqa] CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def list_images(folder, recursive=False):
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_id_from_stem(stem):
    match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else stem


def build_gt_maps(gt_dir):
    if gt_dir is None:
        return {}, {}
    gt_paths = list_images(gt_dir, recursive=False)
    by_name = {p.name: p for p in gt_paths}
    by_key = {}
    for path in gt_paths:
        by_key[path.stem] = path
        by_key[image_id_from_stem(path.stem)] = path
    return by_name, by_key


def match_gt(pred_path, gt_by_name, gt_by_key):
    if not gt_by_name and not gt_by_key:
        return None
    if pred_path.name in gt_by_name:
        return gt_by_name[pred_path.name]
    if pred_path.stem in gt_by_key:
        return gt_by_key[pred_path.stem]
    return gt_by_key.get(image_id_from_stem(pred_path.stem))


def prepare_metric_inputs(path, resize_max_side=None):
    if resize_max_side is None:
        return str(path)

    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        max_side = max(w, h)
        if max_side <= resize_max_side:
            return str(path)
        scale = resize_max_side / float(max_side)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    import torch

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def create_metrics(metric_names, device):
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError(
            "pyiqa is required. Install it in your server environment with: pip install pyiqa"
        ) from exc

    metrics = {}
    for name in metric_names:
        print(f"[nr-iqa] creating metric: {name}")
        metrics[name] = pyiqa.create_metric(name, device=device).to(device)
        metrics[name].eval()
    return metrics


def score_image(path, metrics, device, resize_max_side=None):
    import torch

    scores = {}
    metric_input = prepare_metric_inputs(path, resize_max_side=resize_max_side)
    if isinstance(metric_input, torch.Tensor):
        metric_input = metric_input.to(device)
    with torch.no_grad():
        for name, metric in metrics.items():
            try:
                value = metric(metric_input)
                scores[name] = float(value.item() if hasattr(value, "item") else value)
            except Exception as exc:
                print(f"[nr-iqa] failed {name} on {path.name}: {exc}")
                scores[name] = float("nan")
    return scores


def finite_mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def finite_std(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.std(values, ddof=0)) if values else float("nan")


def format_latex_value(metric, value):
    if not math.isfinite(value):
        return "--"
    if metric == "niqe":
        return f"{value:.3f}"
    return f"{value:.4f}"


def write_method_outputs(method_output_dir, method, rows, summary):
    method_output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame([summary])

    xlsx_path = method_output_dir / f"{method}_nr_iqa_per_image.xlsx"
    csv_path = method_output_dir / f"{method}_nr_iqa_per_image.csv"
    summary_json = method_output_dir / f"{method}_nr_iqa_summary.json"
    latex_path = method_output_dir / f"{method}_nr_iqa_latex_row.txt"

    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_image", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latex_metrics = [
        format_latex_value("niqe", summary.get("NIQE_mean", float("nan"))),
        format_latex_value("musiq", summary.get("MUSIQ_mean", float("nan"))),
        format_latex_value("maniqa", summary.get("MANIQA_mean", float("nan"))),
        format_latex_value("clipiqa", summary.get("CLIP-IQA_mean", float("nan"))),
    ]
    latex_path.write_text(f"{method} & " + " & ".join(latex_metrics) + r" \\" + "\n", encoding="utf-8")
    return xlsx_path


def evaluate_method(
    pred_dir,
    method,
    output_dir,
    metrics,
    device,
    gt_by_name,
    gt_by_key,
    recursive=False,
    limit=None,
    resize_max_side=None,
    strict_gt=False,
):
    pred_paths = list_images(pred_dir, recursive=recursive)
    if limit is not None:
        pred_paths = pred_paths[:limit]
    if not pred_paths:
        raise FileNotFoundError(f"No images found in {pred_dir}")

    print(f"[nr-iqa] evaluating {method}: {len(pred_paths)} images")
    rows = []
    missing_gt = 0
    for idx, pred_path in enumerate(pred_paths, start=1):
        gt_path = match_gt(pred_path, gt_by_name, gt_by_key)
        if gt_by_name and gt_path is None:
            missing_gt += 1
            if strict_gt:
                continue
        scores = score_image(pred_path, metrics, device, resize_max_side=resize_max_side)
        row = {
            "index": idx,
            "file": pred_path.name,
            "pred_path": str(pred_path),
            "gt_file": gt_path.name if gt_path is not None else "",
            "gt_path": str(gt_path) if gt_path is not None else "",
        }
        for name in metrics:
            row[TABLE_NAMES[name]] = scores[name]
        rows.append(row)
        if idx % 25 == 0 or idx == len(pred_paths):
            print(f"[nr-iqa] {method}: {idx}/{len(pred_paths)}")

    summary = {
        "method": method,
        "pred_dir": str(pred_dir),
        "num_images": len(rows),
        "missing_gt": missing_gt,
    }
    for name in metrics:
        col = TABLE_NAMES[name]
        vals = [row[col] for row in rows]
        summary[f"{col}_mean"] = finite_mean(vals)
        summary[f"{col}_std"] = finite_std(vals)

    xlsx_path = write_method_outputs(output_dir / method, method, rows, summary)
    print(f"[nr-iqa] wrote {xlsx_path}")
    return summary


def method_dirs_from_root(methods_root):
    return sorted(p for p in methods_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (args.pred_dir / "nr_iqa_metrics") if args.pred_dir else (args.methods_root / "nr_iqa_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = create_metrics(args.metrics, device)
    gt_by_name, gt_by_key = build_gt_maps(args.gt_dir)

    if args.pred_dir is not None:
        method_items = [(args.method or args.pred_dir.name, args.pred_dir)]
    else:
        method_items = [(p.name, p) for p in method_dirs_from_root(args.methods_root)]

    summaries = []
    for method, pred_dir in method_items:
        summaries.append(
            evaluate_method(
                pred_dir=pred_dir,
                method=method,
                output_dir=output_dir,
                metrics=metrics,
                device=device,
                gt_by_name=gt_by_name,
                gt_by_key=gt_by_key,
                recursive=args.recursive,
                limit=args.limit,
                resize_max_side=args.resize_max_side,
                strict_gt=args.strict_gt,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_xlsx = output_dir / "nr_iqa_all_methods_summary.xlsx"
    summary_csv = output_dir / "nr_iqa_all_methods_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print("\n[nr-iqa] all-method summary:")
    print(summary_df.to_string(index=False))
    print(f"[nr-iqa] wrote {summary_xlsx}")


if __name__ == "__main__":
    main()
