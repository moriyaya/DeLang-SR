# Training DeLang-SR

## Dependencies and data

Install `requirements-training.txt` after installing PyTorch and torchvision. Obtain the
SD3 Medium Diffusers-format model with its three text encoders. For DINOv3:

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

Download the official ConvNeXt-Large LVD-1689M pretrained weights after following
the DINOv3 access conditions. Place them in `pretrained/` and set
`dinov3_repo_path` and `dinov3_weights_path` in the selected configuration.
The complete upstream repository is used locally and is excluded from this repo.

Obtain RELLISUR from the dataset authors and preserve the following structure:

```text
RELLISUR-Dataset/
  Train/LLLR/
  Train/NLHR-Duplicates/X2/
  Train/NLHR-Duplicates/X4/
  Test_crop/LLLR/
  Test_crop/NLHR-Duplicates/X2/
  Test_crop/NLHR-Duplicates/X4/
```

Build paired training conditions and input-only testing conditions:

```bash
python scripts/build_rellisur_language_metadata.py \
  --dataset-root /path/to/RELLISUR-Dataset --output-dir metadata \
  --splits x2_train x4_train x2_test_lq_only x4_test_lq_only

python scripts/cache_delang_prompt_embeds.py \
  --metadata metadata/rellisur_x2_train_lang.jsonl \
  --sd3-path /path/to/stable-diffusion-3 \
  --output metadata/rellisur_x2_train_prompt_embeds.pt --dtype bf16

python scripts/cache_delang_prompt_embeds.py \
  --metadata metadata/rellisur_x4_train_lang.jsonl \
  --sd3-path /path/to/stable-diffusion-3 \
  --output metadata/rellisur_x4_train_prompt_embeds.pt --dtype bf16
```

Set `sd3_path`, the two DINOv3 paths, and `output_dir` in each YAML. Start training:

```bash
NUM_PROCESSES=1 bash sh/train_x2.sh
NUM_PROCESSES=1 bash sh/train_x4.sh
```

## Configuration of the released checkpoints

| Setting | x2 | x4 |
|---|---:|---:|
| Training crop resolution | 1024 | 864 |
| Steps | 20000 | 20000 |
| Batch size per process | 1 | 1 |
| Gradient accumulation | 1 | 1 |
| Precision | bf16 | bf16 |
| Learning rate | 5e-5 | 5e-5 |
| Optimizer | AdamW | AdamW |
| VAE encoder / SD3 LoRA ranks | 96 / 128 | 96 / 128 |
| Scheduler index | 700 | 700 |
| Strength bounds | [0.15, 0.95] | [0.15, 0.95] |
| Visual tokens | 4 | 4 |

These YAML values were checked against the configuration files saved with the
released training runs. Dataset paths and output paths are portable placeholders.
The training source and tensor shapes remain compatible with the released
adapters. Training a new run is not a promise of bitwise-identical weights.

The objective uses weights 5 (latent refinement), 5 (DINOv3 feature distance),
0.7 (L1), 0.2 (strength prior), 0.5 (low frequency), 0.5 (high frequency), and
0.1 (attribute alignment). Optimizer betas are (0.9, 0.999), weight decay is
1e-4, epsilon is 1e-8, and gradient clipping is 1.0. LoRA alpha equals rank,
initialization is Gaussian, and dropout is 0. See `train/delang_sr.py` for the
explicit target modules.

## Saved artifacts

Each `weight-<step>/` contains `sd3_adapter/`, `vae_encoder_adapter/`, and
`delang_modules.pt`. The attribute alignment head is used only during training.
Model adapters do not contain optimizer, scheduler, or RNG states; resuming the
optimizer trajectory from such a checkpoint is not supported. The training
script fails explicitly if an optimizer resume is requested.
