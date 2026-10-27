#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

INPUT_IMAGE=${INPUT_IMAGE:-"/path/to/RELLISUR-Dataset/Test_crop/LLLR"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/delang_sr_x4"}
SD3_PATH=${SD3_PATH:-"/path/to/stable-diffusion-3"}
LORA_PATH=${LORA_PATH:-"./checkpoints/delang_sr_x4"}

python infer/infer_delang_sr.py \
  --input_image "${INPUT_IMAGE}" \
  --output_dir "${OUTPUT_DIR}" \
  --sd3_path "${SD3_PATH}" \
  --lora_path "${LORA_PATH}" \
  --device "${DEVICE:-cuda:0}" \
  --weight_dtype "${WEIGHT_DTYPE:-bf16}" \
  --upscale 4 \
  --process_size 1024
