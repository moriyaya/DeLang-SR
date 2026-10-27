#!/usr/bin/env python
"""Build degradation-language metadata from RELLISUR LQ/HQ pairs.

The script converts paired low-light SR supervision into reproducible language
supervision for DeLang.  It never modifies images; it writes JSONL files with
paths, pseudo degradation attributes, prompts, and restoration instructions.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

ATTRIBUTES = {
    "underexposure": ["mild", "moderate", "severe"],
    "shadow_noise": ["low", "medium", "high"],
    "contrast_loss": ["mild", "moderate", "strong"],
    "detail_loss": ["mild", "moderate", "severe"],
    "illumination": ["uniform", "non-uniform"],
    "color_cast": ["neutral", "warm", "cool", "greenish", "magenta"],
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "RELLISUR-Dataset"
    default_output = Path(__file__).resolve().parents[1] / "metadata"

    parser = argparse.ArgumentParser(
        description="Generate DeLang metadata from RELLISUR LQ/HQ image pairs."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--stats-size", type=int, default=256)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "x2_train",
            "x4_train",
            "x2_test_lq_only",
            "x4_test_lq_only",
        ],
        help="Metadata splits to build.",
    )
    return parser.parse_args()


def list_images(path):
    images = sorted(p for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError(f"No images found in {path}")
    return images


def exposure_tag_from_name(name):
    match = re.search(r"-([0-9]+(?:\.[0-9]+)?)\.[^.]+$", name)
    return float(match.group(1)) if match else None


def read_rgb(path, size=None):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0


def luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gradient_strength(y):
    gy = np.abs(np.diff(y, axis=0)).mean()
    gx = np.abs(np.diff(y, axis=1)).mean()
    return float(gx + gy)


def local_illumination_std(y):
    pil = Image.fromarray(np.uint8(np.clip(y * 255.0, 0, 255)))
    local = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=15)), dtype=np.float32) / 255.0
    return float(local.std())


def dark_noise_score(y):
    pil = Image.fromarray(np.uint8(np.clip(y * 255.0, 0, 255)))
    smooth = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    residual = np.abs(y - smooth)
    mask = y < 0.20
    if mask.mean() < 0.02:
        return float(residual.mean())
    return float(residual[mask].mean())


def compute_stats(lq_path, hq_path=None, stats_size=256):
    lq = read_rgb(lq_path, stats_size)
    y_lq = luminance(lq)

    stats = {
        "lq_mean": float(y_lq.mean()),
        "lq_std": float(y_lq.std()),
        "dark_ratio": float((y_lq < 0.15).mean()),
        "noise_score": dark_noise_score(y_lq),
        "illumination_var": local_illumination_std(y_lq),
        "lq_edge": gradient_strength(y_lq),
    }

    if hq_path is not None:
        hq = read_rgb(hq_path, stats_size)
        y_hq = luminance(hq)
        stats.update(
            {
                "hq_mean": float(y_hq.mean()),
                "hq_std": float(y_hq.std()),
                "exposure_gap": float(y_hq.mean() - y_lq.mean()),
                "contrast_gap": float(y_hq.std() - y_lq.std()),
                "detail_gap": float(gradient_strength(y_hq) - gradient_strength(y_lq)),
                "rgb_gap": [float(v) for v in (hq.mean(axis=(0, 1)) - lq.mean(axis=(0, 1)))],
            }
        )
    else:
        # LQ-only test-time estimate.  The values are not oracle labels; they
        # are deployable prompts inferred without the reference image.
        stats.update(
            {
                "hq_mean": None,
                "hq_std": None,
                "exposure_gap": float(max(0.0, 0.38 - y_lq.mean())),
                "contrast_gap": float(max(0.0, 0.18 - y_lq.std())),
                "detail_gap": float(max(0.0, 0.075 - gradient_strength(y_lq))),
                "rgb_gap": [float(v) for v in (lq.mean(axis=(0, 1)) - lq.mean())],
            }
        )

    return stats


def quantiles(rows, key):
    values = np.asarray([r["stats"][key] for r in rows], dtype=np.float32)
    return np.quantile(values, [0.33, 0.66]).tolist()


def level3(value, t1, t2, names):
    if value < t1:
        return names[0]
    if value < t2:
        return names[1]
    return names[2]


def classify_color_cast(rgb_gap):
    r, g, b = rgb_gap
    spread = max(rgb_gap) - min(rgb_gap)
    if spread < 0.035:
        return "neutral"
    if r > g + 0.035 and b > g + 0.035:
        return "magenta"
    if r >= g and r >= b:
        return "warm"
    if b >= r and b >= g:
        return "cool"
    if g >= r and g >= b:
        return "greenish"
    return "magenta"


def add_attributes(rows):
    noise_q = quantiles(rows, "noise_score")
    detail_q = quantiles(rows, "detail_gap")
    illum_q = quantiles(rows, "illumination_var")

    for row in rows:
        s = row["stats"]
        exposure_strength = 0.65 * s["exposure_gap"] + 0.35 * s["dark_ratio"]
        attrs = {
            "underexposure": level3(exposure_strength, 0.25, 0.48, ATTRIBUTES["underexposure"]),
            "shadow_noise": level3(s["noise_score"], noise_q[0], noise_q[1], ATTRIBUTES["shadow_noise"]),
            "contrast_loss": level3(s["contrast_gap"], 0.06, 0.14, ATTRIBUTES["contrast_loss"]),
            "detail_loss": level3(s["detail_gap"], detail_q[0], detail_q[1], ATTRIBUTES["detail_loss"]),
            "illumination": "non-uniform" if s["illumination_var"] >= illum_q[1] else "uniform",
            "color_cast": classify_color_cast(s["rgb_gap"]),
        }
        row["attributes"] = attrs
        row["attribute_ids"] = {
            name: ATTRIBUTES[name].index(value) for name, value in attrs.items()
        }
        row["attribute_vector"] = make_attribute_vector(row)
    return rows


def make_attribute_vector(row):
    s = row["stats"]
    attrs = row["attributes"]
    return [
        float(np.clip(s["exposure_gap"] / 0.55, 0.0, 1.0)),
        float(np.clip(s["dark_ratio"], 0.0, 1.0)),
        float(np.clip(s["noise_score"] / 0.08, 0.0, 1.0)),
        float(np.clip(max(0.0, s["contrast_gap"]) / 0.30, 0.0, 1.0)),
        float(np.clip(max(0.0, s["detail_gap"]) / 0.15, 0.0, 1.0)),
        1.0 if attrs["illumination"] == "non-uniform" else 0.0,
        float(row["scale"]) / 4.0,
    ]


def phrase_join(parts):
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def build_prompt(row):
    a = row["attributes"]
    scale = row["scale"]
    degradations = phrase_join(
        [
            f"{a['underexposure']} underexposure",
            f"{a['shadow_noise']} shadow noise",
            f"{a['contrast_loss']} contrast loss",
            f"{a['detail_loss']} detail degradation",
            f"{a['illumination']} illumination",
            None if a["color_cast"] == "neutral" else f"a {a['color_cast']} color cast",
        ]
    )
    prompt = (
        f"Restore a real low-light image with {degradations}. "
        f"Perform {scale}x super-resolution while preserving scene structure and natural colors."
    )
    instruction = (
        "Enhance dark regions, suppress shadow artifacts, recover fine details, "
        "and avoid over-exposure or hallucinated content."
    )
    if a["color_cast"] != "neutral":
        instruction += f" Correct the {a['color_cast']} color cast without changing object colors."
    return prompt, instruction


def collect_rows(dataset_root, lq_rel, hq_rel, scale, stats_size):
    lq_dir = dataset_root / lq_rel
    hq_dir = dataset_root / hq_rel if hq_rel is not None else None
    rows = []

    for lq_path in list_images(lq_dir):
        hq_path = None
        if hq_dir is not None:
            candidate = hq_dir / lq_path.name
            if not candidate.exists():
                raise FileNotFoundError(f"Missing HQ pair for {lq_path}: {candidate}")
            hq_path = candidate

        stats = compute_stats(lq_path, hq_path, stats_size=stats_size)
        rows.append(
            {
                "id": lq_path.stem,
                "lq": str(lq_path),
                "hq": str(hq_path) if hq_path is not None else None,
                "scale": scale,
                "exposure_tag": exposure_tag_from_name(lq_path.name),
                "stats": stats,
            }
        )

    rows = add_attributes(rows)
    for row in rows:
        prompt, instruction = build_prompt(row)
        row["prompt"] = prompt
        row["instruction"] = instruction
        row["method_note"] = (
            "Attributes are estimated from the LQ image only; no HQ reference is used."
            if row["hq"] is None
            else "Attributes are automatically derived from the real LQ/HQ RELLISUR pair."
        )
    return rows


def write_jsonl(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return output_path


def build_split(name, dataset_root, stats_size):
    split_map = {
        "x2_train": ("Train/LLLR", "Train/NLHR-Duplicates/X2", 2),
        "x4_train": ("Train/LLLR", "Train/NLHR-Duplicates/X4", 4),
        "x2_test_lq_only": ("Test_crop/LLLR", None, 2),
        "x4_test_lq_only": ("Test_crop/LLLR", None, 4),
    }
    if name not in split_map:
        raise ValueError(f"Unknown split {name}. Available: {sorted(split_map)}")
    lq_rel, hq_rel, scale = split_map[name]
    return collect_rows(dataset_root, lq_rel, hq_rel, scale, stats_size)


def summarize(rows):
    summary = {}
    for attr_name, choices in ATTRIBUTES.items():
        counts = {choice: 0 for choice in choices}
        for row in rows:
            counts[row["attributes"][attr_name]] += 1
        summary[attr_name] = counts
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    manifest = {
        "dataset_root": str(args.dataset_root),
        "stats_size": args.stats_size,
        "attribute_schema": ATTRIBUTES,
        "splits": {},
    }

    for split in args.splits:
        rows = build_split(split, args.dataset_root, args.stats_size)
        output_path = args.output_dir / f"rellisur_{split}_lang.jsonl"
        write_jsonl(rows, output_path)
        manifest["splits"][split] = {
            "path": str(output_path),
            "num_samples": len(rows),
            "summary": summarize(rows),
        }
        print(f"Wrote {len(rows):5d} rows -> {output_path}")

    manifest_path = args.output_dir / "rellisur_language_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
