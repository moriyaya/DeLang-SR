#!/usr/bin/env python3
"""Download and verify the official DeLang-SR adapters from GitHub Releases."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(scale, output_dir, manifest):
    entry = manifest[str(scale)]
    destination = output_dir / f"delang_sr_x{scale}"
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite: {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".delang-download-", dir=output_dir) as tmp:
        work = Path(tmp)
        archive = work / "checkpoint.tar.gz"
        print(f"Downloading x{scale} adapters ({entry['size_bytes'] / 1e6:.1f} MB)...", flush=True)
        urllib.request.urlretrieve(entry["url"], archive)
        if archive.stat().st_size != entry["size_bytes"] or sha256(archive) != entry["sha256"]:
            raise ValueError("Download checksum mismatch; the model has not been installed.")
        extracted = work / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                target = (extracted / member.name).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise ValueError("Unsafe checkpoint archive path.")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                else:
                    raise ValueError("Checkpoint archives must not contain links or devices.")
        model_root = extracted / f"delang_sr_x{scale}"
        for relative, expected in entry["files"].items():
            if sha256(model_root / relative) != expected:
                raise ValueError(f"Model file checksum mismatch: {relative}")
        shutil.move(str(model_root), destination)
    print(f"Installed verified adapters: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=[2, 4], nargs="+", default=[2, 4])
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "models.json"
    manifest = json.loads(manifest_path.read_text())
    for scale in dict.fromkeys(args.scale):
        download(scale, args.output_dir, manifest)


if __name__ == "__main__":
    main()
