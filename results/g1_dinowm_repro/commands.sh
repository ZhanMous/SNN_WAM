#!/usr/bin/env bash
# G1 DINO-WM Minimal Reproduction Smoke Commands
# This file records the exact commands executed for this smoke run.

set -euo pipefail

# Command 1: Environment checks
conda run -n snnwam-libero python --version
conda run -n snnwam-libero python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi || true

# Command 2: Quality gate
conda run -n snnwam-libero bash scripts/quality_gate.sh

# Command 3: Smoke reproduction (dry_run, 1 step, mock data)
conda run -n snnwam-libero python src/train/train_offline.py \
  --config configs/smoke/g0_patch_latent_smoke.yaml \
  --dry_run --max_steps 1 \
  --output_dir results/g1_dinowm_repro/

# Command 4: Run tests
conda run -n snnwam-libero python -m pytest tests/ -v
