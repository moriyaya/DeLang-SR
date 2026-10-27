import json
import os
import random
import sys

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


ATTRIBUTE_SCHEMA = {
    "underexposure": ["mild", "moderate", "severe"],
    "shadow_noise": ["low", "medium", "high"],
    "contrast_loss": ["mild", "moderate", "strong"],
    "detail_loss": ["mild", "moderate", "severe"],
    "illumination": ["uniform", "non-uniform"],
    "color_cast": ["neutral", "warm", "cool", "greenish", "magenta"],
}

ATTRIBUTE_NAMES = list(ATTRIBUTE_SCHEMA.keys())


def _read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class RELLISURLanguageDataset(torch.utils.data.Dataset):
    """Paired RELLISUR dataset with degradation-language metadata.

    Each record contains paired LQ/HQ paths, a degradation-language prompt,
    a continuous attribute vector, and discrete attribute labels.
    """

    def __init__(
        self,
        metadata_path,
        resolution,
        prompt_mode="prompt_instruction",
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.records = _read_jsonl(metadata_path)
        if not self.records:
            raise ValueError(f"No records found in metadata: {metadata_path}")
        self.resolution = resolution
        self.prompt_mode = prompt_mode
        self.crop_preproc = transforms.Compose(
            [
                transforms.RandomCrop(
                    (resolution, resolution),
                    pad_if_needed=True,
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def __len__(self):
        return len(self.records)

    def _prompt_text(self, record):
        prompt = record.get("prompt", "")
        instruction = record.get("instruction", "")
        if self.prompt_mode == "prompt":
            return prompt
        if self.prompt_mode == "instruction":
            return instruction
        return f"{prompt} {instruction}".strip()

    def _attribute_labels(self, record):
        raw_ids = record.get("attribute_ids", {})
        labels = []
        for name in ATTRIBUTE_NAMES:
            if name in raw_ids:
                labels.append(int(raw_ids[name]))
            else:
                value = record["attributes"][name]
                labels.append(ATTRIBUTE_SCHEMA[name].index(value))
        return torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        record = self.records[idx]
        if record.get("hq") is None:
            raise ValueError(
                "Training DeLang requires paired HQ paths in training metadata."
            )

        with Image.open(record["hq"]) as im:
            gt_img = im.convert("RGB").copy()
        gt_size = gt_img.size[0]

        with Image.open(record["lq"]) as im:
            lq_img = im.convert("RGB").copy()
        lq_img = lq_img.resize((gt_size, gt_size), Image.Resampling.LANCZOS)

        seed = torch.randint(0, 2**32, (1,)).item()
        torch_state = torch.get_rng_state()
        random_state = random.getstate()

        torch.manual_seed(seed)
        random.seed(seed)
        img_gt = self.crop_preproc(gt_img)

        torch.manual_seed(seed)
        random.seed(seed)
        img_lq = self.crop_preproc(lq_img)

        torch.set_rng_state(torch_state)
        random.setstate(random_state)

        img_gt = F.normalize(
            F.to_tensor(img_gt),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        img_lq = F.normalize(
            F.to_tensor(img_lq),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        attr_vec = torch.tensor(record["attribute_vector"], dtype=torch.float32)
        attr_labels = self._attribute_labels(record)

        return {
            "lq": img_lq,
            "hq": img_gt,
            "prompt": self._prompt_text(record),
            "attr_vec": attr_vec,
            "attr_labels": attr_labels,
            "index": torch.tensor(idx, dtype=torch.long),
            "path": record["lq"],
        }
