# Experiment Protocol

## Stage 1 Protocol

第一阶段验证对象是 `SNN temporal/world-action adapter`，不是完整 VLA/WAM、OpenVLA 7B、真实 Unitree 或论文叙事。

一个实验只有在满足以下条件后才可以称为可复现：

- 有固定配置文件。
- 有固定数据来源和 split 说明。
- 有固定 seed 列表。
- 有固定输出目录。
- 有可重复运行命令。
- 有结果文件和日志。
- 有 claims ledger 条目或明确声明“不产生科学主张”。
- 有 causal window alignment 检查，确认没有 future leakage。

## Required Experiment Lifecycle

1. Define: 写清楚实验问题、模型组、数据 split、seed、指标和预期产物。
2. Run: 用单条命令启动实验，并把 stdout/stderr 写入结果目录。
3. Record: 保存配置、metrics、日志、代码版本、环境信息和人工 notes。
4. Evaluate: 离线评估与闭环评估分开保存，不混写。
5. Audit: 检查每个 claim 是否能指向 `results/` 下的文件。
6. Report: 只汇报已有 evidence；没有结果文件的内容不写成结论。

## Causal Data Rules

任何 trajectory window dataset 或评估切片必须满足：

- 输入只能包含当前或过去的 observation、state、action history 和语言指令。
- 输入不得包含 future observation、future state、future action target、reward、success label 或 episode outcome。
- `target_actions` 和 `target_future_latents` 只能作为监督目标，不能进入模型输入或 normalization fit 的未来部分。
- normalization/statistics 必须只从 train split 估计，不能从 val/test 或 held-out tasks 泄漏。
- 在任何 dataset 结果可引用前，必须有 no-future-leakage 测试覆盖窗口边界、horizon 和 split。

## Fair Comparison Rules

MLP、GRU、SNN 或 WAM variant 的对比必须共享：

- 相同 frozen encoders、data split、window definition、history length、action horizon 和 future horizon。
- 相同 seed list、optimizer budget、early stopping policy、evaluation split 和 metric implementation。
- 相同 closed-loop task list、initial states、episode seeds、max steps 和 success definition。

如果参数量、延迟、训练预算或调参预算不同，必须在 `notes.md` 和结果表中报告，并把对应 claim 降级为 observation。

## Result Directory Layout

```text
results/runs/<run_id>/
  config.yaml
  metrics.csv
  run.log
  git_commit.txt
  environment.txt
  seeds.txt
  command.sh
  split.json
  notes.md
  eval_rollout.csv
  failure_videos/
```

离线实验可以省略 `eval_rollout.csv` 和 `failure_videos/`，但必须明确标注为 offline-only。

## Minimum Metrics

离线 adapter 实验至少记录：

- action MSE 或 action L1
- future latent cosine error
- multi-step latent error by horizon
- inference latency

SNN adapter 实验还必须记录：

- spike rate
- spike penalty weight
- temporal state reset policy

闭环仿真实验至少记录：

- closed-loop success rate
- completion steps
- failure mode count
- per-episode success/failure evidence

## Metric Definitions

- `action_mse`: 对预测 action chunk 和 demonstration action chunk 在相同 horizon、相同 action normalization 下计算 mean squared error。
- `action_l1`: 同上，但计算 mean absolute error。
- `future_latent_cosine_error`: `1 - cosine_similarity(z_hat_future, z_target_future)`，必须按 horizon 分开记录后再汇总。
- `multi_step_latent_error`: 按 future horizon index 记录的 latent error 曲线，用于判断 horizon degradation。
- `spike_rate`: spike tensor 的 mean firing fraction；不能写成真实能耗。
- `inference_latency`: 固定 batch/device/settings 下的 forward latency，必须记录测量环境。
- `closed_loop_success_rate`: 成功 episode 数除以总 episode 数，必须保存 per-episode `eval_rollout.csv`。

## Re-evaluation Checklist

复评估者必须能从结果目录回答：

- 运行了哪个 config？
- 使用了哪个 commit？
- 使用了哪些 seed？
- 输入数据和 split 是什么？
- 指标从哪个文件读取？
- claim 对应哪些结果文件？
- 如果结果失败，失败案例在哪里？
