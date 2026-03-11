#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_xlsum_smoke.yaml}"

uv run python -m adapter.train \
  --config "$CONFIG_PATH" \
  --no-auto_generate_outputs
