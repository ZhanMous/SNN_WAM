# SNN-WAM

SNN-WAM 是一个研究仓库，用来探索 **SNN 版 latent world model / world-action model** 能否在语言条件机器人操作中提供可复现的动态建模证据。

## Route Reset: 2026-06-01

旧的 direct action BC 路线已经冻结为诊断证据，不再作为主实验路线。主要原因不是“还差几轮调参”，而是 direct action behavior cloning 没有成为检验 WAM / future latent / SNN dynamics 的稳定底座：WAM-GRU future/no-future 在 matched LIBERO closed-loop diagnostic 中均为 `0/30`，而 expert replay 为 `27/30`。

新的主线从 DINO-WM 复现开始：

```text
image_t
  -> frozen DINOv2 spatial patch features
  -> action-conditioned latent world model
  -> future patch features
  -> action sequence optimization / planning
```

目标不是用 SNN 微调已有 ANN world model，也不是 ANN-to-SNN conversion；目标是在 DINO-WM 式 latent world-model 问题上，研究是否能直接训练一个 SNN world model，后续再比较 ES/EGGROLL-style low-rank optimization 与梯度训练 baseline。

旧路线归档点：

- branch/tag: `legacy_bc_policy_diagnostics_20260601`
- report: `docs/WEEKLY_REPORT_2026-06-01.md`
- reflection: `docs/EXPERIMENT_REFLECTION_REPORT_2026-06-01.md`
- new route plan: `docs/DINOWM_SNN_WORLDMODEL_PLAN.md`

## Current Goal

当前阶段先复现 DINO-WM 的最小机制，再把 temporal predictor 替换为 SNN world model：

- 输入：DINOv2 spatial patch features、action sequence、可选语言/任务条件。
- 输出：future spatial patch features，不以 direct action BC 为主目标。
- 规划：在 latent space 中通过 action sequence optimization 选择动作序列。
- 对比：DINO-WM-style ANN/Transformer predictor、GRU predictor、直接训练的 SNN predictor。
- 证据：future patch feature error、multi-step latent drift、planning objective、closed-loop planning success、spike rate / SynOps proxy。

## Non-goals

- 不直接训练完整 VLA/WAM 大模型。
- 不直接上 OpenVLA 7B。
- 不直接上真实 Unitree。
- 不先写论文故事。
- 不把 direct action BC 结果写成 WAM / SNN dynamics 证据。
- 不声称 EGGROLL 已复现；只允许写成 EGGROLL-style / low-rank ES direct training 研究。
- 不声称 spike rate 等于真实硬件能耗。

## Repository Map

- `docs/project_plan.md`: 从 `SNN_WAM_plan.pdf` 提取的紧凑项目计划摘要。
- `docs/TOP_LEVEL_SCIENTIFIC_PLAN.md`: 新路线总规划、阶段 gate、artifact schema、claim ledger schema、stop rules 和 Claude Code prompt。
- `docs/DINOWM_SNN_WORLDMODEL_PLAN.md`: 2026-06-01 新路线计划，DINO-WM 复现到直接训练 SNN latent world model。
- `docs/EXPERIMENT_REFLECTION_REPORT_2026-06-01.md`: direct action BC 旧路线反思报告。
- `docs/WEEKLY_REPORT_2026-06-01.md`: 旧路线诊断周报。
- `docs/PROJECT_CONTRACT.md`: 项目边界、依赖政策、科学主张和结果产物契约。
- `docs/EXPERIMENT_PROTOCOL.md`: 实验生命周期、结果目录和复评估协议。
- `docs/ENVIRONMENT.md`: LIBERO-first environment setup and verification workflow.
- `docs/DATA_CONTRACT.md`: raw LIBERO trajectory inspection contract and leakage risks.
- `docs/LIBERO_ACTION_SEMANTICS.md`: G2.5 action/observation alignment audit.
- `docs/SPLIT_POLICY.md`: train/val/test and closed-loop initial-state split rules.
- `docs/NORMALIZATION_POLICY.md`: train-only normalization rules.
- `docs/FUTURE_LATENT_CONTRACT.md`: future latent target contract before WAM-style claims.
- `docs/AUDIT_DATASET_LEAKAGE.md`: dataset leakage audit and G2.5 status.
- `docs/LIBERO_DATA_CONTRACT.md`: real LIBERO schema contract from the inspected demonstration.
- `docs/DATA_RISKS.md`: data risk register and resolved/unresolved status.
- `docs/LOCAL_PATHS_TEMPLATE.md`: non-committed local path template for `LIBERO_DATASET_ROOT`.
- `docs/RESULT_ARTIFACTS.md`: 可引用结果产物登记表。
- `docs/CLAIMS_LEDGER.md`: 科学主张登记表，所有 claim 必须指向结果文件。
- `.agents/skills/`: repo-scoped Codex skills for this project; restart Codex from the repository root to load them.
- `configs/`: 后续实验配置占位目录。
- `src/data/trajectory_window.py`: G2 causal trajectory-window dataset v1；当前仍不得放模型代码或训练代码。
- `tests/`: 最小 pytest 框架。
- `scripts/smoke_check.sh`: 本仓库 smoke test 入口。
- `scripts/quality_gate.sh`: environment report plus smoke test entrypoint.
- `scripts/inspect_libero_data.py`: raw demonstration inspection script with mock mode.
- `scripts/inspect_libero_demo.py`: locate the first real demo under `LIBERO_DATASET_ROOT` and update data docs.
- `scripts/check_libero_action_alignment.py`: G2.5 real-demo action/window alignment diagnostic.
- `scripts/bootstrap_libero_check.py`: G1.5 LIBERO bootstrap checker before real G2 dataset work.
- `scripts/download_libero_minimal.sh`: safe wrapper for the official minimal LIBERO suite downloader.
- `src/models/temporal_mlp.py`: action-only MLP temporal baseline; no GRU/SNN/future-latent path.
- `src/train/train_offline.py`: config-driven offline trainer for the first MLP action baseline.
- `results/`: 后续实验结果目录。

## Gate Ladder

| Gate | Name | Deliverable |
| --- | --- | --- |
| G0 | Repo Gate | directory skeleton, `AGENTS.md`, `README.md`, pytest, smoke script |
| G1 | Environment Gate | torch/LIBERO import, version record, minimal demo log |
| G1.5 | LIBERO Bootstrap Gate | official LIBERO repo path, data root, real `.hdf5` demo, inspection report |
| G2 | Dataset Gate | trajectory shape, window alignment, no-future-leakage tests |
| G3 | Model Gate | MLP/GRU/SNN forward shape, parameter count, latency smoke |
| G4 | Training Gate | tiny-batch overfit, decreasing loss, checkpoint save |
| G5 | Metric Gate | synthetic metric tests, action MSE, future cosine, spike rate |
| G6 | Rollout Gate | fixed initial states, episode CSV, failure videos, success rate |
| G7 | Robustness Gate | noise/delay/frame drop curves with fixed seeds |
| G8 | Evidence Gate | `docs/RESULT_ARTIFACTS.md` matches real files |
| G9 | Claim Gate | every paper/report sentence points to evidence |

## Smoke Tests

```bash
bash scripts/smoke_check.sh
bash scripts/quality_gate.sh
python3 -m pytest -q
python src/train/train_offline.py --config configs/libero_spatial_mlp.yaml --dry_run --max_steps 1
```

`train_offline.py --dry_run` 只运行 deterministic mock action data to validate
the MLP training path, metrics logging, and checkpoints. Mock dry-run outputs
are not scientific evidence and must not be cited as LIBERO results.

## Source Plan

原始计划来自仓库根目录的 `SNN_WAM_plan.pdf`。因为初始仓库中没有 `docs/project_plan.md`，本仓库补建了一个摘要版 markdown，作为后续工程任务的轻量入口。
