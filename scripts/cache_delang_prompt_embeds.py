#!/usr/bin/env python
"""Cache unique SD3 prompt embeddings for DeLang metadata."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import free_memory
from tqdm.auto import tqdm


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Cache DeLang SD3 prompt embeddings.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sd3-path", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--prompt-mode",
        choices=["prompt", "instruction", "prompt_instruction"],
        default="prompt_instruction",
    )
    return parser.parse_args()


def dtype_from_name(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def read_prompts(metadata_path, prompt_mode):
    prompts = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompt = record.get("prompt", "")
            instruction = record.get("instruction", "")
            if prompt_mode == "prompt":
                text = prompt
            elif prompt_mode == "instruction":
                text = instruction
            else:
                text = f"{prompt} {instruction}".strip()
            prompts.append(text)
    return prompts


def main():
    args = parse_args()
    weight_dtype = dtype_from_name(args.dtype)
    prompts = read_prompts(args.metadata, args.prompt_mode)
    unique_prompts = sorted(set(prompts))
    prompt_to_id = {prompt: idx for idx, prompt in enumerate(unique_prompts)}
    record_prompt_ids = torch.tensor([prompt_to_id[p] for p in prompts], dtype=torch.long)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.sd3_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    pipe = pipe.to(args.device)

    prompt_embeds = []
    pooled_prompt_embeds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(unique_prompts), args.batch_size), desc="Encoding prompts"):
            batch = unique_prompts[start : start + args.batch_size]
            embeds, _, pooled, _ = pipe.encode_prompt(
                prompt=batch,
                prompt_2=batch,
                prompt_3=batch,
                negative_prompt=[""] * len(batch),
                negative_prompt_2=[""] * len(batch),
                negative_prompt_3=[""] * len(batch),
                do_classifier_free_guidance=False,
            )
            prompt_embeds.append(embeds.cpu())
            pooled_prompt_embeds.append(pooled.cpu())

    pipe = pipe.to("cpu")
    del pipe
    free_memory()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": str(args.metadata),
            "prompt_mode": args.prompt_mode,
            "prompt_texts": unique_prompts,
            "record_prompt_ids": record_prompt_ids,
            "prompt_embeds": torch.cat(prompt_embeds, dim=0),
            "pooled_prompt_embeds": torch.cat(pooled_prompt_embeds, dim=0),
        },
        args.output,
    )
    print(f"Cached {len(unique_prompts)} unique prompts for {len(prompts)} records -> {args.output}")


if __name__ == "__main__":
    main()
