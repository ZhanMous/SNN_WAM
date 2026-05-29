#!/usr/bin/env bash
set -euo pipefail
/home/zhan_shaoji/miniconda3/envs/snnwam-libero/bin/python3 src/train/train_offline.py --config configs/reportable/libero_spatial_wam_gru_dinov2s_no_future.yaml
