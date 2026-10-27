#!/usr/bin/env python3
"""Package full-model adapters without training logs, datasets, or base weights."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


FILES = [
    "sd3_adapter/adapter_config.json", "sd3_adapter/adapter_model.safetensors",
    "vae_encoder_adapter/adapter_config.json", "vae_encoder_adapter/adapter_model.safetensors",
    "delang_modules.pt", "model_index.json", "README.md", "NOTICE", "STABLE_DIFFUSION_3_LICENSE.txt",
]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def public_metadata(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=[2, 4], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for relative in FILES:
        if not (args.checkpoint / relative).is_file():
            raise FileNotFoundError(args.checkpoint / relative)
    for relative in ("sd3_adapter/adapter_config.json", "vae_encoder_adapter/adapter_config.json"):
        config = json.loads((args.checkpoint / relative).read_text())
        if config.get("base_model_name_or_path") not in (None, "stabilityai/stable-diffusion-3-medium-diffusers"):
            raise ValueError(f"Replace private base-model paths before packaging: {relative}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, "w:gz", compresslevel=1) as archive:
        for relative in FILES:
            archive.add(args.checkpoint / relative, arcname=f"delang_sr_x{args.scale}/{relative}", recursive=False, filter=public_metadata)
    print(json.dumps({
        "scale": args.scale, "checkpoint_step": 20000,
        "base_model": "stabilityai/stable-diffusion-3-medium-diffusers",
        "url": f"https://github.com/moriyaya/DeLang-SR/releases/download/v1.0.0/delang_sr_x{args.scale}.tar.gz",
        "size_bytes": args.output.stat().st_size, "sha256": digest(args.output),
        "files": {relative: digest(args.checkpoint / relative) for relative in FILES},
    }, indent=2))


if __name__ == "__main__":
    main()
