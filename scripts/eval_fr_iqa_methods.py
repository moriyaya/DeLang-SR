#!/usr/bin/env python3
"""Evaluate full-reference RELLISUR metrics for one or many method folders.

Metrics:
- PSNR, SSIM: higher is better.
- LPIPS, DISTS, ExpErr, ColorErr: lower is better.

Outputs:
- one per-image Excel/CSV for each method
- one summary JSON and LaTeX row for each method
- one all-method summary Excel/CSV
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_METRICS = ["PSNR", "SSIM", "LPIPS", "DISTS", "ExpErr", "ColorErr"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/LPIPS/DISTS/ExpErr/ColorErr against GT."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_dir", type=Path, help="One method output image folder.")
    group.add_argument(
        "--methods_root",
        type=Path,
        help="Root folder whose immediate subfolders are method output folders.",
    )
    parser.add_argument("--gt_dir", type=Path, required=True, help="GT/reference image folder.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--method", type=str, default=None, help="Method name for --pred_dir mode.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N image limit for debugging.")
    parser.add_argument("--resize_to_gt", action="store_true", help="Resize prediction to GT size if needed.")
    parser.add_argument(
        "--primary_space",
        choices=["rgb", "y"],
        default="rgb",
        help="Which PSNR/SSIM space is used for the main table columns.",
    )
    parser.add_argument(
        "--skip_deep_metrics",
        action="store_true",
        help="Only compute PSNR/SSIM/ExpErr/ColorErr; skip LPIPS and DISTS.",
    )
    return parser.parse_args()


def list_images(folder, recursive=False):
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_id_from_stem(stem):
    match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else stem


def build_gt_maps(gt_dir):
    gt_paths = list_images(gt_dir, recursive=False)
    by_name = {p.name: p for p in gt_paths}
    by_key = {}
    for path in gt_paths:
        by_key[path.stem] = path
        by_key[image_id_from_stem(path.stem)] = path
    return by_name, by_key


def match_gt(pred_path, gt_by_name, gt_by_key):
    if pred_path.name in gt_by_name:
        return gt_by_name[pred_path.name]
    if pred_path.stem in gt_by_key:
        return gt_by_key[pred_path.stem]
    return gt_by_key.get(image_id_from_stem(pred_path.stem))


def read_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def np01(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_to_y(rgb01):
    # BT.601 luma, common for SR evaluation.
    return 0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]


def psnr_ssim_rgb(pred01, gt01):
    psnr = float(peak_signal_noise_ratio(gt01, pred01, data_range=1.0))
    ssim = float(structural_similarity(gt01, pred01, channel_axis=2, data_range=1.0))
    return psnr, ssim


def psnr_ssim_y(pred01, gt01):
    pred_y = rgb_to_y(pred01)
    gt_y = rgb_to_y(gt01)
    psnr = float(peak_signal_noise_ratio(gt_y, pred_y, data_range=1.0))
    ssim = float(structural_similarity(gt_y, pred_y, data_range=1.0))
    return psnr, ssim


def exposure_error(pred01, gt01):
    return float(abs(rgb_to_y(pred01).mean() - rgb_to_y(gt01).mean()))


def color_error(pred01, gt01):
    return float(np.abs(pred01.mean(axis=(0, 1)) - gt01.mean(axis=(0, 1))).sum())


def resolve_device(device):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[fr-iqa] CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def tensor01(img):
    import torch

    arr = np01(img)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def create_deep_metrics(device, skip=False):
    if skip:
        return {}
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError(
            "pyiqa is required for LPIPS and DISTS. Install with: pip install pyiqa"
        ) from exc

    metrics = {}
    for name in ["lpips", "dists"]:
        print(f"[fr-iqa] creating metric: {name}")
        metrics[name] = pyiqa.create_metric(name, device=device).to(device)
        metrics[name].eval()
    return metrics


def deep_metric_value(metric, pred_img, gt_img, device):
    import torch

    pred = tensor01(pred_img).to(device)
    gt = tensor01(gt_img).to(device)
    with torch.no_grad():
        value = metric(pred, gt)
    return float(value.item() if hasattr(value, "item") else value)


def finite_mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def finite_std(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.std(values, ddof=0)) if values else float("nan")


def format_value(metric, value):
    if not math.isfinite(value):
        return "--"
    if metric == "PSNR":
        return f"{value:.3f}"
    return f"{value:.4f}"


def write_method_outputs(method_output_dir, method, rows, summary):
    method_output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame([summary])

    xlsx_path = method_output_dir / f"{method}_fr_iqa_per_image.xlsx"
    csv_path = method_output_dir / f"{method}_fr_iqa_per_image.csv"
    summary_json = method_output_dir / f"{method}_fr_iqa_summary.json"
    latex_path = method_output_dir / f"{method}_fr_iqa_latex_row.txt"

    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_image", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latex_values = [format_value(m, summary.get(f"{m}_mean", float("nan"))) for m in TABLE_METRICS]
    latex_path.write_text(f"{method} & " + " & ".join(latex_values) + r" \\" + "\n", encoding="utf-8")
    return xlsx_path


def evaluate_method(
    pred_dir,
    method,
    output_dir,
    gt_by_name,
    gt_by_key,
    deep_metrics,
    device,
    recursive=False,
    limit=None,
    resize_to_gt=False,
    primary_space="rgb",
):
    pred_paths = list_images(pred_dir, recursive=recursive)
    if limit is not None:
        pred_paths = pred_paths[:limit]
    if not pred_paths:
        raise FileNotFoundError(f"No images found in {pred_dir}")

    print(f"[fr-iqa] evaluating {method}: {len(pred_paths)} images")
    rows = []
    missing_gt = 0
    shape_mismatch = 0

    for idx, pred_path in enumerate(pred_paths, start=1):
        gt_path = match_gt(pred_path, gt_by_name, gt_by_key)
        if gt_path is None:
            missing_gt += 1
            continue

        pred_img = read_rgb(pred_path)
        gt_img = read_rgb(gt_path)
        if pred_img.size != gt_img.size:
            if resize_to_gt:
                pred_img = pred_img.resize(gt_img.size, Image.Resampling.LANCZOS)
            else:
                shape_mismatch += 1
                continue

        pred01 = np01(pred_img)
        gt01 = np01(gt_img)
        psnr_rgb, ssim_rgb = psnr_ssim_rgb(pred01, gt01)
        psnr_y, ssim_y = psnr_ssim_y(pred01, gt01)
        row = {
            "index": idx,
            "file": pred_path.name,
            "pred_path": str(pred_path),
            "gt_file": gt_path.name,
            "gt_path": str(gt_path),
            "PSNR_RGB": psnr_rgb,
            "SSIM_RGB": ssim_rgb,
            "PSNR_Y": psnr_y,
            "SSIM_Y": ssim_y,
            "PSNR": psnr_y if primary_space == "y" else psnr_rgb,
            "SSIM": ssim_y if primary_space == "y" else ssim_rgb,
            "LPIPS": float("nan"),
            "DISTS": float("nan"),
            "ExpErr": exposure_error(pred01, gt01),
            "ColorErr": color_error(pred01, gt01),
        }

        if "lpips" in deep_metrics:
            row["LPIPS"] = deep_metric_value(deep_metrics["lpips"], pred_img, gt_img, device)
        if "dists" in deep_metrics:
            row["DISTS"] = deep_metric_value(deep_metrics["dists"], pred_img, gt_img, device)

        rows.append(row)
        if idx % 25 == 0 or idx == len(pred_paths):
            print(f"[fr-iqa] {method}: {idx}/{len(pred_paths)}")

    summary = {
        "method": method,
        "pred_dir": str(pred_dir),
        "num_images": len(rows),
        "missing_gt": missing_gt,
        "shape_mismatch": shape_mismatch,
        "primary_space": primary_space,
    }
    for metric in ["PSNR", "SSIM", "LPIPS", "DISTS", "ExpErr", "ColorErr", "PSNR_RGB", "SSIM_RGB", "PSNR_Y", "SSIM_Y"]:
        vals = [row[metric] for row in rows]
        summary[f"{metric}_mean"] = finite_mean(vals)
        summary[f"{metric}_std"] = finite_std(vals)

    xlsx_path = write_method_outputs(output_dir / method, method, rows, summary)
    print(f"[fr-iqa] wrote {xlsx_path}")
    return summary


def method_dirs_from_root(methods_root):
    return sorted(p for p in methods_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def main():
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (args.pred_dir / "fr_iqa_metrics") if args.pred_dir else (args.methods_root / "fr_iqa_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    gt_by_name, gt_by_key = build_gt_maps(args.gt_dir)
    deep_metrics = create_deep_metrics(device=device, skip=args.skip_deep_metrics)

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
                gt_by_name=gt_by_name,
                gt_by_key=gt_by_key,
                deep_metrics=deep_metrics,
                device=device,
                recursive=args.recursive,
                limit=args.limit,
                resize_to_gt=args.resize_to_gt,
                primary_space=args.primary_space,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_xlsx = output_dir / "fr_iqa_all_methods_summary.xlsx"
    summary_csv = output_dir / "fr_iqa_all_methods_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print("\n[fr-iqa] all-method summary:")
    print(summary_df.to_string(index=False))
    print(f"[fr-iqa] wrote {summary_xlsx}")


if __name__ == "__main__":
    main()
