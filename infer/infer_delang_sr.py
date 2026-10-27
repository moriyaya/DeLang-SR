#!/usr/bin/env python3
"""Full DeLang-SR inference with conditions estimated from LQ images only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from accelerate.utils import set_seed
from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import free_memory
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer.delang_sr_infer_model import DeLang_SR_Infer
from scripts.build_rellisur_language_metadata import (
    IMAGE_EXTS, add_attributes, build_prompt, compute_stats,
)


def collect_inputs(source):
    if source.is_dir():
        images = sorted(p.resolve() for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    elif source.suffix.lower() == ".txt":
        images = []
        for line in source.read_text().splitlines():
            if line.strip():
                path = Path(line.strip()).expanduser()
                images.append((source.parent / path).resolve() if not path.is_absolute() else path)
    else:
        images = [source.resolve()]
    if not images or any(not p.is_file() for p in images):
        raise ValueError("Provide an image, a nonempty image folder, or a valid input list.")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Input stems must be unique because outputs are saved as <stem>.png.")
    return images


def lq_conditions(images, scale, metadata_path=None, stats_size=256):
    if metadata_path:
        records = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        by_id = {row["id"]: row for row in records}
        rows = []
        for image in images:
            if image.stem not in by_id:
                raise ValueError(f"Missing LQ-only metadata for {image.name}")
            row = by_id[image.stem]
            if row.get("hq") is not None or int(row["scale"]) != scale:
                raise ValueError("Inference metadata must be LQ-only and match the requested scale.")
            if len(row["attribute_vector"]) != 7:
                raise ValueError("Expected a seven-dimensional degradation descriptor.")
            rows.append(row)
        return rows
    rows = [{"id": p.stem, "lq": str(p), "hq": None, "scale": scale,
             "stats": compute_stats(p, hq_path=None, stats_size=stats_size)} for p in images]
    add_attributes(rows)
    for row in rows:
        row["prompt"], row["instruction"] = build_prompt(row)
    return rows


@torch.inference_mode()
def encode_prompts(sd3_path, prompts, device, dtype, batch_size):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        sd3_path, transformer=None, vae=None, torch_dtype=dtype,
    ).to(device)
    encoded, pooled = [], []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        embeds, _, pools, _ = pipe.encode_prompt(
            prompt=batch, prompt_2=batch, prompt_3=batch,
            do_classifier_free_guidance=False,
        )
        encoded.append(embeds.cpu())
        pooled.append(pools.cpu())
    pipe.to("cpu")
    del pipe
    free_memory()
    return torch.cat(encoded), torch.cat(pooled)


def prepare_image(image, scale, process_size, mode):
    target = (image.width * scale, image.height * scale)
    if mode == "paper":
        # Preserve the preprocessing of the original benchmark inference script.
        if min(image.size) < process_size // scale:
            ratio = (process_size // scale) / min(image.size)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.BICUBIC)
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.BICUBIC)
        image = image.resize((image.width - image.width % 8, image.height - image.height % 8), Image.Resampling.LANCZOS)
        return image, image.size
    image = image.resize(target, Image.Resampling.BICUBIC)
    image = ImageOps.expand(image, border=(0, 0, (-target[0]) % 16, (-target[1]) % 16))
    return image, target


def main(args):
    if args.process_size < 16 or args.process_size % 16:
        raise ValueError("process_size must be a positive multiple of 16.")
    if args.prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be positive.")
    set_seed(args.seed)
    images = collect_inputs(args.input)
    rows = lq_conditions(images, args.upscale, args.metadata_path)
    args.output.mkdir(parents=True, exist_ok=True)
    for image in images:
        if (args.output / (image.stem + ".png")).resolve() == image.resolve():
            raise ValueError("The output folder must not overwrite input images.")
    index_path = args.checkpoint / "model_index.json"
    model_index = json.loads(index_path.read_text()) if index_path.exists() else {}
    if model_index.get("scale", args.upscale) != args.upscale:
        raise ValueError("Use the checkpoint released for the requested scale.")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.weight_dtype]
    prompts = [f"{r['prompt']} {r.get('instruction', '')}".strip() for r in rows]
    unique_prompts = sorted(set(prompts))
    prompt_ids = {p: i for i, p in enumerate(unique_prompts)}
    embeddings, pooled = encode_prompts(args.sd3_path, unique_prompts, args.device, dtype, args.prompt_batch_size)
    net = DeLang_SR_Infer(
        sd3_path=args.sd3_path, lora_path=str(args.checkpoint), device=args.device,
        prompt_embed_dim=embeddings.shape[-1], weight_dtype=dtype,
        mid_timestep=700, sigma_min=0.15, sigma_max=0.95,
        sigma_base_channels=32, num_degradation_tokens=4, token_base_channels=64,
        attribute_vector_dim=7,
    ).eval()
    run = {"scale": args.upscale, "checkpoint": str(args.checkpoint), "seed": args.seed,
           "preprocess": args.preprocess, "process_size": args.process_size,
           "weight_dtype": args.weight_dtype, "gt_used_for_conditions": False, "images": []}
    with torch.inference_mode():
        for path, row, prompt in tqdm(list(zip(images, rows, prompts)), desc="DeLang-SR"):
            with Image.open(path) as im:
                original = im.convert("RGB")
            prepared, output_size = prepare_image(original, args.upscale, args.process_size, args.preprocess)
            tensor = to_tensor(prepared).unsqueeze(0).to(device=args.device, dtype=dtype) * 2 - 1
            pid = prompt_ids[prompt]
            output, seconds = net(
                tensor, embeddings[pid:pid + 1].to(args.device, dtype=dtype),
                pooled[pid:pid + 1].to(args.device, dtype=dtype),
                torch.tensor([row["attribute_vector"]], device=args.device, dtype=torch.float32),
                tile_size=args.process_size // 8, tile_overlap=args.process_size // 16,
            )
            restored = to_pil_image((output[0].float().cpu() * 0.5 + 0.5).clamp(0, 1))
            if args.preprocess == "native":
                restored = restored.crop((0, 0, *output_size))
            restored.save(args.output / f"{path.stem}.png")
            run["images"].append({"file": path.name, "input_hw": [original.height, original.width],
                                  "output_hw": [restored.height, restored.width], "model_time_s": seconds})
    (args.output / "conditions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Saved {len(images)} restored images to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input_image", type=Path, required=True)
    parser.add_argument("--output", "--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", "--lora_path", type=Path, required=True)
    parser.add_argument("--sd3_path", default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--upscale", type=int, choices=[2, 4], required=True)
    parser.add_argument("--metadata_path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process_size", type=int, default=1024)
    parser.add_argument("--weight_dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--prompt_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--preprocess", choices=["paper", "native"], default="paper",
                        help="paper preserves benchmark preprocessing; native returns exactly scale*input dimensions.")
    main(parser.parse_args())
