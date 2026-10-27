#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2

CONFIG_PATH=${CONFIG_PATH:-"./configs/delang_sr_x4.yaml"}

accelerate launch \
  --num_processes=${NUM_PROCESSES:-1} \
  --num_machines=1 \
  train/delang_sr.py --config "${CONFIG_PATH}"
