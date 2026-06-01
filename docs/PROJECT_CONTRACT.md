# Project Contract

## Scope

本项目第一阶段原本验证 `SNN temporal/world-action adapter`。2026-06-01 后，主路线调整为 DINO-WM-style latent world modeling：在冻结 DINOv2 spatial patch features 上预测 action-conditioned future patch latents，并研究直接训练 SNN world model 的可行性。

当前允许规划和后续实现的最小研究问题：

- 复现最小 DINO-WM-style latent dynamics prediction。
- 给定 DINOv2 spatial patch features 和 action sequence，预测 future patch features。
- 用 action sequence optimization / planning 评估 learned latent world model。
- 公平比较 ANN/Transformer、GRU 和直接训练的 SNN latent world models。
- 研究 EGGROLL-style / low-rank ES 是否能直接训练小型 SNN world model。

## Canonical Scientific Standards

`docs/PROJECT_CONTRACT.md` 和 `docs/EXPERIMENT_PROTOCOL.md` 是本仓库的 canonical 研究标准。`.agents/skills/` 可以提供工作流帮助，但不能替代本契约。

### Gate Ladder

项目按 G0-G9 gate 推进。每个 gate 的交付物必须在进入下一 gate 前落到仓库文档、测试或 `results/` 产物中。

| Gate | 名称 | Codex 必须交付 |
| --- | --- | --- |
| G0 | Repo Gate | 目录、`AGENTS.md`、`README.md`、pytest、smoke 脚本 |
| G1 | Environment Gate | torch/LIBERO import、版本记录、最小 demo 日志 |
| G1.5 | LIBERO Bootstrap Gate | `LIBERO_REPO_ROOT`、官方下载脚本、数据根目录、真实 `.hdf5` demo 和 inspection report |
| G2 | Dataset Gate | trajectory shape、window 对齐、无未来泄漏测试 |
| G3 | Model Gate | MLP/GRU/SNN forward shape、参数量、latency smoke |
| G4 | Training Gate | tiny-batch overfit、loss 下降、checkpoint 保存 |
| G5 | Metric Gate | synthetic metric tests、action MSE、future cosine、spike rate |
| G6 | Rollout Gate | fixed initial states、episode CSV、failure videos、success rate |
| G7 | Robustness Gate | noise/delay/frame drop 曲线，固定 seeds |
| G8 | Evidence Gate | `docs/RESULT_ARTIFACTS.md` 与真实文件一致 |
| G9 | Claim Gate | 论文/汇报中的每句话都能指向证据 |

G0 是 repo bootstrap gate。G1 之后允许引入阶段必要依赖，但必须先更新 Dependency Policy、环境记录和 smoke tests。G2 不能开始，直到 G1.5 已经检查过至少一个真实 LIBERO HDF5 demonstration 文件。

### Baseline Fairness

任何 “SNN 优于 ANN/GRU” 或 “SNN 更鲁棒” 的 claim 必须满足：

- 使用相同数据 split、seed 列表、trajectory window 规则和 evaluation split。
- 使用相同 frozen DINOv2 patch encoder、输入字段、action horizon、future horizon 和 target normalization。
- 使用可比较的训练预算、调参预算、早停规则和评估次数。
- 闭环评估使用相同 task list、initial states、episode seeds、max steps 和 success 判定。
- 报告每个模型的参数量、推理延迟和主要训练超参；如果容量不匹配，claim 必须降级为 observation。
- 至少包含 copy-last-latent、ANN/Transformer world model、GRU world model、random/untrained SNN 等基线，避免把容量或优化预算误写成 SNN 效果。
- 如果继续引用旧 direct action BC 诊断，必须保留 action-only baseline 语义，且只能作为 legacy diagnostic，不得升级为当前 WAM/world-model 证据。

### WAM Evidence Standard

本项目中的 WAM/world-model claim 不等于 direct action BC。要称为 WAM-style 或 DINO-WM-style evidence，必须同时满足：

- 模型显式预测 action-conditioned future spatial patch latents。
- 保留 copy-last-latent、random action rollout 或 non-world-model baseline。
- 结果显示 latent world model 对 multi-step horizon degradation、planning objective 或 closed-loop planning success 至少一项有可审计贡献。
- claim 必须指向 metrics、planning 结果、rollout 结果或 robustness 表，不能只引用训练 loss。

### ES Evidence Standard

ES/EGGROLL-style training 分两类处理：

1. **Post-training ES**: 用于已经训练好的模型，优化 closed-loop success、jerk/smoothness、安全惩罚或 spike 稀疏约束等不可微目标。此时必须比较 surrogate/Adam-only checkpoint。
2. **Direct SNN world-model ES**: 这是 2026-06-01 后用户明确指定的新研究目标。允许从随机初始化的小型 SNN latent world model 出发，用 EGGROLL-style / low-rank ES 直接优化 recorded-trajectory latent prediction fitness。

Direct ES claim 必须满足：

- 目标是小型 SNN latent world model，不是完整 VLA/WAM。
- fitness 明确记录 future patch latent error、multi-step drift、spike-rate proxy 等组成项。
- 固定 perturbation seeds、population size、rank、sigma、generation budget。
- 对比 copy-last-latent、random SNN、ANN/GRU gradient-trained baseline。
- surrogate-gradient SNN 可作为对照 baseline，但不能作为 direct-trained SNN 主结果的来源。
- 只在结果支持时写成 “direct ES improves X under Y setting”，不能写成 full EGGROLL reproduction。

## Non-goals

- 不直接训练完整 VLA/WAM 大模型。
- 不直接上 OpenVLA 7B。
- 不直接上真实 Unitree。
- 不先写论文故事。
- 不把 direct action BC 作为当前主验证路线。
- 不把 spike rate 或 SynOps proxy 写成真实能耗结论。
- 不在没有结果文件时提出科学主张。

## Dependency Policy

G0 bootstrap 阶段只允许：

- Python stdlib
- Bash
- git
- pytest

G1 `snnwam-libero` environment baseline 允许 PyTorch、NumPy、h5py、PyYAML 和官方 LIBERO 安装流程，用于验证 torch/LIBERO import、版本记录和最小 demo 日志。

默认 `snnwam-libero` 环境不允许引入下列后续阶段依赖：

- maniskill
- openvla
- transformers
- spikingjelly
- snntorch
- norse
- unitree

如果后续需要新增依赖，必须先更新本文件，说明阶段、用途、安装环境和可替代方案。

## Result Artifact Contract

每个正式实验必须保存到 `results/runs/<run_id>/`。`run_id` 应包含 suite、模型、seed 和时间信息。

每个结果目录至少包含：

- `config.yaml`: 完整配置和路径。
- `metrics.csv` 或 `metrics.json`: 每个 epoch、split、task、seed 的指标。
- `run.log`: 可审计运行日志。
- `git_commit.txt`: 运行时 commit hash；如果仓库未提交，必须记录 dirty 状态。
- `environment.txt`: Python、OS、关键包版本和硬件摘要。
- `seeds.txt`: 使用的 random seed 列表和 seed 设置位置。
- `command.sh`: 产生该结果的完整命令。
- `split.json`: 数据集、task、trajectory、episode 或 held-out split 描述。
- `notes.md`: 异常、观察和人工检查记录。

闭环评估还必须包含：

- `eval_rollout.csv`: 每个 episode 的 task、seed、success、steps 和 failure mode。
- `failure_videos/` 或 `failure_frames/`: 可复查失败案例。

## Scientific Claims Policy

- 每个 claim 必须登记在 `docs/CLAIMS_LEDGER.md`。
- 每个被引用的结果必须先登记在 `docs/RESULT_ARTIFACTS.md`。
- 每个 claim 必须指向一个或多个 `results/` 下的结果文件。
- 如果结果文件缺失，只能写成 hypothesis 或 observation，不能写成 conclusion。
- claim 必须区分离线 evidence、closed-loop evidence 和 robustness evidence。
- 多 seed 结果应保留每个 seed 的原始文件，汇总表不能替代原始证据。
