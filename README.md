# SNN-WAM

SNN-WAM 是一个第一阶段研究仓库，用来把实验计划转化为可复现、可复评估、可审计的工程基础。当前目标不是训练完整具身大模型，而是为后续验证 `SNN temporal/world-action adapter` 建立清晰边界、产物契约和 smoke tests。

## Stage 1 Goal

第一阶段只验证冻结视觉/语言表示之后的 temporal/world-action adapter 思路：

- 输入：当前视觉 latent、语言条件、动作历史和可选机器人状态。
- 输出：future action chunk 与 future visual latent chunk。
- 对比：MLP、GRU、SNN adapter。
- 证据：action error、future latent error、closed-loop success rate、spike rate、鲁棒性曲线等结果文件。

## Non-goals

- 不直接训练完整 VLA/WAM 大模型。
- 不直接上 OpenVLA 7B。
- 不直接上真实 Unitree。
- 不先写论文故事。
- 不在 bootstrap 阶段写模型代码、训练代码、LIBERO 集成代码或硬件控制代码。

## Repository Map

- `docs/project_plan.md`: 从 `SNN_WAM_plan.pdf` 提取的紧凑项目计划摘要。
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
```

当前 smoke tests 只检查研究仓库骨架、文档边界和依赖政策，不运行模型训练或仿真。

## Source Plan

原始计划来自仓库根目录的 `SNN_WAM_plan.pdf`。因为初始仓库中没有 `docs/project_plan.md`，本仓库补建了一个摘要版 markdown，作为后续工程任务的轻量入口。
