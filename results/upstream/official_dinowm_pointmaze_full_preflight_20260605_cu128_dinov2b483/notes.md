# Official DINO-WM Upstream Reproduction

This package records commands for the unmodified upstream DINO-WM repo. It is not evidence of reproduction until `summary.json` is accompanied by official training metrics/checkpoints and planning outputs from `plan.py`.

Required success evidence:
- official repo commit
- DATASET_DIR with official task data
- train.py logs/checkpoint under ckpt_base_path/outputs/<model_name>
- plan.py outputs under upstream plan_outputs

## 2026-06-05 Execution Status

The full upstream PointMaze train command was launched on this WSL host with the
audited default train settings (`training.epochs=100`,
`training.batch_size=32`, `env.dataset.n_rollout=null`, `env.num_workers=16`).
WSL restarted before the Python wrapper could record a return code. The last
captured train log reached full dataset loading (`Loaded 2000 rollouts`) and no
epoch metric, checkpoint, or Python traceback was written.

See `wsl_crash_analysis.md` for the system-level crash evidence and decision.
Do not cite this package as a successful upstream train reproduction.
