#!/usr/bin/env python3
"""Multi-seed DINO-WM baseline rerun with explicit future_actions interface.

Trains DINOwMTransformer on seeds 0, 1, 2, then runs multi-horizon eval
and planning sanity for each. Produces reproducibility artifacts.

Usage:
    python scripts/rerun_dinowm_baseline_multiseed.py
    python scripts/rerun_dinowm_baseline_multiseed.py --seeds 0 1 2
    python scripts/rerun_dinowm_baseline_multiseed.py --dry_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/reportable/dinowm_baseline_rerun.yaml"
TRAIN_SCRIPT = REPO_ROOT / "scripts/train_dinowm_baseline.py"
EVAL_SCRIPT = REPO_ROOT / "src/eval/dinowm_eval_offline.py"
PLANNING_SCRIPT = REPO_ROOT / "src/eval/dwm_g4_planning_sanity.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--skip_train", action="store_true", help="Skip training, only run eval/planning.")
    p.add_argument("--skip_eval", action="store_true", help="Skip eval/planning, only train.")
    return p.parse_args()


def run_cmd(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
    return result.returncode


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"  SEED {seed}")
        print(f"{'#'*60}")

        # --- Training ---
        if not args.skip_train:
            run_dir = f"results/runs/dinowm_transformer_baseline_rerun_seed{seed}"
            cmd = [
                sys.executable, str(TRAIN_SCRIPT),
                "--config", str(CONFIG),
                "--seed", str(seed),
                "--device", args.device,
            ]
            if args.dry_run:
                cmd.extend(["--dry_run", "--max_steps", "1"])
            rc = run_cmd(cmd, f"Training seed={seed}")
            if rc != 0:
                failures.append(f"train_seed{seed}")
                continue

        # --- Multi-horizon eval ---
        if not args.skip_eval:
            run_dir = f"results/runs/dinowm_transformer_baseline_rerun_seed{seed}"
            cmd = [
                sys.executable, str(EVAL_SCRIPT),
                "--run_dir", run_dir,
                "--horizons", "1", "2", "4",
                "--action_mode", "real",
                "--device", args.device,
            ]
            if args.dry_run:
                cmd.extend(["--max_steps", "1"])
            rc = run_cmd(cmd, f"Eval (real actions) seed={seed}")
            if rc != 0:
                failures.append(f"eval_real_seed{seed}")

            # Shuffle ablation
            cmd_shuffle = [
                sys.executable, str(EVAL_SCRIPT),
                "--run_dir", run_dir,
                "--horizons", "1", "2", "4",
                "--action_mode", "shuffle",
                "--shuffle_seeds", "0", "1", "2",
                "--device", args.device,
            ]
            if args.dry_run:
                cmd_shuffle.extend(["--max_steps", "1"])
            rc = run_cmd(cmd_shuffle, f"Eval (shuffle ablation) seed={seed}")
            if rc != 0:
                failures.append(f"eval_shuffle_seed{seed}")

            # Persistence baseline
            cmd_persist = [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{REPO_ROOT}'); "
                f"from scripts.eval_persistence_baseline import eval_persistence; "
                f"import torch, json; from pathlib import Path; "
                f"from torch.utils.data import DataLoader, Subset; "
                f"from src.data.patch_latent_dataset import create_dinowm_transition_dataset; "
                f"cache = Path('{REPO_ROOT}/latents/libero_spatial/dinov2_vits14_patch'); "
                f"ds = create_dinowm_transition_dataset(cache, context_len=3, future_horizon=4, split='val'); "
                f"n=len(ds); ntrain=int(n*0.9); "
                f"loader = DataLoader(Subset(ds, list(range(ntrain, n))), batch_size=32, collate_fn=lambda s: {{k: torch.stack([x[k] for x in s]) if isinstance(s[0][k], torch.Tensor) else [x[k] for x in s] for k in s[0]}}); "
                f"for h in [1,2,4]: "
                f"  m = eval_persistence(loader, eval_horizon=h); "
                f"  print(f'H={{h}}: cos_err={{m[\"patch_cosine_error\"]:.6f}}'); "
                f"  Path('{REPO_ROOT}/{run_dir}/baselines').mkdir(parents=True, exist_ok=True); "
                f"  (Path('{REPO_ROOT}/{run_dir}/baselines/persistence_h'+str(h)+'.json')).write_text(json.dumps(m, indent=2))"
            ]
            if not args.dry_run:
                rc = run_cmd(cmd_persist, f"Persistence baseline seed={seed}")
                if rc != 0:
                    failures.append(f"persistence_seed{seed}")

            # --- Planning sanity ---
            cmd_plan = [
                sys.executable, str(PLANNING_SCRIPT),
                "--run_dir", run_dir,
                "--n_samples", "50",
                "--horizon", "2",
                "--random_baseline_type", "dataset",
                "--device", args.device,
            ]
            if args.dry_run:
                pass  # planning doesn't have --max_steps
            rc = run_cmd(cmd_plan, f"Planning sanity seed={seed}")
            if rc != 0:
                failures.append(f"planning_seed{seed}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    if failures:
        print(f"  FAILURES: {failures}")
        return 1
    print("  All seeds completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
