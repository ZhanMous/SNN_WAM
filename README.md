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
- `docs/RESULT_ARTIFACTS.md`: 可引用结果产物登记表。
- `docs/CLAIMS_LEDGER.md`: 科学主张登记表，所有 claim 必须指向结果文件。
- `.agents/skills/`: repo-scoped Codex skills for this project; restart Codex from the repository root to load them.
- `configs/`: 后续实验配置占位目录。
- `src/`: 后续实现占位目录；当前不得放模型代码。
- `tests/`: 最小 pytest 框架。
- `scripts/smoke_check.sh`: 本仓库 smoke test 入口。
- `results/`: 后续实验结果目录。

## Smoke Tests

```bash
bash scripts/smoke_check.sh
python3 -m pytest -q
```

当前 smoke tests 只检查研究仓库骨架、文档边界和依赖政策，不运行模型训练或仿真。

## Source Plan

原始计划来自仓库根目录的 `SNN_WAM_plan.pdf`。因为初始仓库中没有 `docs/project_plan.md`，本仓库补建了一个摘要版 markdown，作为后续工程任务的轻量入口。
