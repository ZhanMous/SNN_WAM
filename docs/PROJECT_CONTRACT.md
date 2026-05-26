# Project Contract

## Scope

本项目第一阶段只验证 `SNN temporal/world-action adapter`。它是冻结视觉/语言编码器之后的时序适配模块，用于学习短期动态、future latent prediction 和 action chunk prediction。

第一阶段允许规划和后续实现的最小研究问题：

- 给定当前视觉表示、语言指令、动作历史和可选状态，预测未来动作。
- 显式预测 action-conditioned future visual latent。
- 公平比较 MLP、GRU 和 SNN temporal adapters。
- 用离线指标和闭环仿真指标共同评估。

## Canonical Scientific Standards

`docs/PROJECT_CONTRACT.md` 和 `docs/EXPERIMENT_PROTOCOL.md` 是本仓库的 canonical 研究标准。`.agents/skills/` 可以提供工作流帮助，但不能替代本契约。

### Baseline Fairness

任何 “SNN 优于 MLP/GRU” 或 “SNN 更鲁棒” 的 claim 必须满足：

- 使用相同数据 split、seed 列表、trajectory window 规则和 evaluation split。
- 使用相同 frozen encoders、输入字段、action horizon、future horizon 和 target normalization。
- 使用可比较的训练预算、调参预算、早停规则和评估次数。
- 闭环评估使用相同 task list、initial states、episode seeds、max steps 和 success 判定。
- 报告每个模型的参数量、推理延迟和主要训练超参；如果容量不匹配，claim 必须降级为 observation。
- 至少包含 action-only baseline 与 future-latent WAM variant，避免把容量或辅助 loss 误写成 SNN 效果。

### WAM Evidence Standard

本项目中的 WAM claim 不等于“普通 BC 上加一个头”。要称为 WAM-style evidence，必须同时满足：

- 模型显式预测 action-conditioned future visual latent。
- 保留 action-only baseline，例如 BC-GRU、SNN action-only 或 `lambda_future=0`。
- 结果显示 future latent objective 对 multi-step horizon degradation、closed-loop success 或 robustness 至少一项有可审计贡献。
- claim 必须指向 `metrics.csv`、rollout 结果或 robustness 表，不能只引用训练 loss。

### ES Evidence Standard

ES/EGGROLL-style post-training 不是第一阶段必要条件。只有在满足以下条件时才能写成有必要：

- 已有 surrogate-gradient 或 Adam baseline。
- fitness 包含不可微或系统级目标，例如 closed-loop success、jerk/smoothness、安全惩罚或 spike 稀疏约束。
- 在固定 task、initial states、seed 和评估预算下，ES post-training 优于 surrogate-only 或普通后处理。
- 没有上述证据时，ES 只能写成 future work 或 exploratory result。

## Non-goals

- 不直接训练完整 VLA/WAM 大模型。
- 不直接上 OpenVLA 7B。
- 不直接上真实 Unitree。
- 不先写论文故事。
- 不把 spike rate 或 SynOps proxy 写成真实能耗结论。
- 不在没有结果文件时提出科学主张。

## Dependency Policy

当前 bootstrap 阶段只允许：

- Python stdlib
- Bash
- git
- pytest

当前阶段不允许引入大型依赖，包括但不限于：

- torch / pytorch
- numpy
- libero
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
