"""Check verified installation and refusal to overwrite a model."""

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.download_models import download, sha256


class DownloadTests(unittest.TestCase):
    def make_package(self, root):
        archive = root / "model.tar.gz"
        content = b'{"scale": 2}\n'
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("delang_sr_x2/model_index.json")
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
        return {"2": {"url": archive.as_uri(), "size_bytes": archive.stat().st_size,
                      "sha256": sha256(archive),
                      "files": {"model_index.json": hashlib.sha256(content).hexdigest()}}}

    def test_verified_install_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_package(root)
            output = root / "checkpoints"
            download(2, output, manifest)
            self.assertTrue((output / "delang_sr_x2/model_index.json").is_file())
            with self.assertRaises(FileExistsError):
                download(2, output, manifest)

    def test_bad_hash_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_package(root)
            manifest["2"]["sha256"] = "0" * 64
            output = root / "checkpoints"
            with self.assertRaises(ValueError):
                download(2, output, manifest)
            self.assertFalse((output / "delang_sr_x2").exists())


if __name__ == "__main__":
    unittest.main()
