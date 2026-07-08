# Group Meeting: 2026-06-08 Three-Week Progress

## 0. One-Sentence Summary

近三周围绕“不依赖梯度训练 SNN 世界模型”这个目标，我完成了三条前置工作线：ES/EGGROLL 方法复现与本地适配、nano-egg int8 ES 预训练代码适配、SNN-WAM 从直接 BC 路线切换到 DINO-WM patch-latent 世界模型路线；目前最重要的结论不是 SNN 已经训练成功，而是旧 action-BC 路线证据不足、DINO-WM 路线的工程门控已部分打通，但真实 patch-latent baseline 和 ES 优化仍未通过可报告门槛。

## 1. Current Research Question

核心问题：

```text
能否不用反向传播/梯度训练 SNN，
让 SNN 直接作为 action-conditioned latent world model，
预测未来 DINOv2 spatial patch features，
并最终支持 latent planning?
```

当前定位：

- SNN 不是 policy head，也不是 ANN-to-SNN 转换结果。
- ES/EGGROLL 是直接训练 SNN 世界模型的候选方法，但必须先证明 latent world-model task 本身可学。
- 不再把旧的 `DINO CLS/state/action history -> action regression` 当主路线。

不应在本次组会中声称：

- Direct ES 已能训练 SNN world model。
- SNN 已优于 ANN/GRU/Transformer。
- DINO-WM 已在本仓库完整复现。
- 当前结果支持低功耗或 embodied foundation model 结论。

## 2. Workstream A: ES / EGGROLL Reproduction

目标：

理解 EGGROLL / low-rank ES 如何在大模型上做无梯度优化，并评估它能否迁移到 SNN latent world model。

本地工作：

- 检查并运行 `Replay_EGGROLL/HyperscaleES`。
- 复现 0.1B RWKV + EGGROLL 的 LLM bandit 训练路径。
- 针对本机 JAX/sharding 环境做了工程修复：
  - generate batch 使用真实 `device_put` dummy prompts 编译，避免 `ShapeDtypeStruct` 与 sharding 路径不兼容。
  - fitness 计算显式转到 host `np.array` 后再放回 device。
  - validation 使用非 `shard_map` 路径，避免 `pcast/pvary` 在单卡/非分片路径中出错。
  - RWKV7 scan map 增加 `qkv` 参数路径兼容。

观察到的证据：

| Task | Setting | Observed result | Status |
|---|---|---:|---|
| GSM8K | 0.1B RWKV, EGGROLL, 32 generations | validation score `0.0`; fitness mostly `0.0` | Engineering run only |
| Countdown | 0.1B RWKV, EGGROLL, 32 generations | validation score about `0.01`; fitness mostly flat | Engineering run only |

解释：

- 路径能编译、生成、算 fitness、执行 update。
- 但当前日志不支持“ES 优化有效”这个结论，`lora_updates` 大多数 epoch 为 `0`，fitness 也基本没有上升。
- 这说明把 EGGROLL 迁移到 SNN-WAM 前，需要先做 toy sanity：符号方向、扰动规模、fitness 归一化、参数子空间是否真的产生非零更新。

Evidence:

- External repo, not registered in SNN-WAM result ledger:
  - `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/experiments/paper_repro/gsm8k_0.1B.log`
  - `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/experiments/paper_repro/countdownn_0.1B.log`
  - Modified files in that repo: `llm_experiments/general_do_evolution.py`, `llm_experiments/utils.py`, `src/hyperscalees/models/llm/rwkv7.py`

## 3. Workstream B: nano-egg

目标：

研究一个更小、更透明的 EGGROLL-style int8 pretraining 实现，作为后续 SNN direct-training toy baseline 的参考。

本地工作：

- 检查 `nano-egg` single-file implementation。
- 明确其训练对象：minGRU language model，整数权重/激活，fitness 为 token-level cross-entropy bits-per-byte。
- 针对本地资源做了适配：
  - 将 JAX memory fraction 默认从 `0.95` 降到 `0.55`。
  - 增加 `update_microbatch_size`，让 ES update 可以拆成更小 microbatch。
  - 修正 `num_sequences/group_size`，保证数据 batch 在设备数较小时仍可整除。
  - SLURM/sweep 脚本改为允许外部覆盖 `XLA_PYTHON_CLIENT_MEM_FRACTION`。

实验结果：

| 配置项 | 值 |
|---|---|
| 模型 | RWKV-7 nano-egg，4.86M 参数 |
| 配置 | int8, 256D, 6Layers, 8b, 1024p |
| 数据 | 8 条序列，每条 ~737M tokens |
| 训练步数 | 1000 步 |
| 训练时间 | ~4 分钟 |
| 硬件 | 零号机 8×H20 |

Baseline 参考：

| 模型 | bits |
|---|---|
| Unigram | 5.0 |
| Bigram | 4.0 |
| Gzip max compression | 2.767 |

训练过程：

| 阶段 | Validation Score | LoRA变化 | Non-LoRA变化 |
|------|-----------------|----------|-------------|
| 初始 (epoch 0) | 6.02 | 1.0 | 1.0 |
| 25% (epoch 250) | ~4.82 | 0.34 | 0.34 |
| 50% (epoch 500) | ~4.63 | 0.12 | 0.12 |
| 75% (epoch 750) | ~4.61 | 0.08 | 0.08 |
| 最终 (epoch 1000) | **4.60-4.62** | 0.064 | 0.067 |

关键指标：

- **最终 Validation Score: ~4.61 bits**
- 训练吞吐量：~1.26M tokens/s (generate) / ~2.3M tokens/s (validation)
- LoRA 参数变化量从 1.0 收敛到 0.064
- Non-LoRA 参数变化量从 1.0 收敛到 0.067

结论：

- 模型从 6.02 bits 收敛到 ~4.61 bits，降低了约 23%
- 性能介于 unigram (5.0) 和 bigram (4.0) 之间
- 训练在 ~500 步后基本收敛，后续变化很小
- LoRA 和 Non-LoRA 参数变化量同步下降，说明整体参数在稳定更新

当前状态：

- 这是方法理解和工程适配，不是 SNN-WAM 科学结果。
- 本地没有发现可引用的 nano-egg 指标日志，因此组会中只能说”完成代码阅读和本地运行适配”，不能说”完成 nano-egg 复现实验”。

Evidence:

- External repo, not registered in SNN-WAM result ledger:
  - `/home/zhan_shaoji/code/nano-egg/README.org`
  - Modified files in that repo: `run.py`, `slurm/multinode_inner.sh`, `sweeps/do_train.sh`, `sweeps/do_wandb.sh`

## 4. Workstream C: SNN-WAM Route Reset

### 4.1 Why the old BC route was frozen

旧路线目标是直接从视觉/状态/action history 预测低层 action。近三周的诊断结果说明，这条路线不适合作为 SNN/ES 的主实验基线。

主要证据：

| Question | Evidence | Interpretation |
|---|---|---|
| closed-loop evaluator 是否有效 | expert replay `27/30 = 90%` | evaluator 能产生成功，不是评估器坏了 |
| WAM-GRU future/no-future 是否成功 | both `0/30` | learned policy 没有优于 zero/random 的 closed-loop 能力 |
| future latent loss 是否带来控制收益 | latent error 下降，但 action MSE/rollout 无改善 | 只能说明 auxiliary latent loss 被优化，不说明控制变好 |
| residual action target 是否有用 | offline split metric 显著改善 | 有工程价值，但不是 closed-loop evidence |
| autoregressive readiness 是否通过 | NOT PASSED | 不适合马上进入 closed-loop 或 ES 后训练 |

Registered evidence:

- `R-G5-DIAGNOSTIC-EVAL-001`
- `R-G9-RESIDUAL-001`
- `R-G10-RESIDUAL-HEAD-001`
- `R-G11-AUTOREG-STAB-001`
- Summary: `docs/WEEKLY_REPORT_2026-06-01.md`

结论：

旧 BC 路线保留为 diagnostic evidence。它暴露了 action representation、gripper metric、自回归误差传播等问题，但不能作为验证 SNN world model 的主路线。

### 4.2 New DINO-WM -> SNN-WAM route

新路线：

```text
observation
  -> frozen DINOv2 patch encoder
  -> z_t: [P, D]
  -> action-conditioned latent dynamics model
  -> z_hat_{t+1:t+H}: [H, P, D]
  -> action sequence optimization / planning
```

已经完成的工程门控：

| Gate | Status | Evidence |
|---|---|---|
| DWM-G1 patch features | PASS | DINOv2 ViT-S/14 patch tokens `[B, 256, 384]`; 18 tests pass |
| DWM-G2 transition dataset | PASS | `z_context`, `future_actions`, `z_target` shape and no-future-leakage tests; 12 tests pass |
| DWM-G3 synthetic ANN baseline | PASS at smoke/gate level | DINOwMTransformer forward/gradient/tiny-train tests; 15 tests pass |
| DWM-G3 real-data baseline | NOT PASSED | Real cached-latent runs do not satisfy persistence/action-use gates |
| DWM-G4 planning sanity | Pending | Current real planning evidence is not acceptable for DWM-G4 |
| DWM-G5 SNN forward | Not started | SNN world model not implemented yet |

Registered evidence:

- `R-DWM-G1-001`
- `R-DWM-G2-001`
- `R-DWM-G3-001`
- `R-DWM-G3-DINOWM-BASELINE-REAL-001`

Recent diagnostic:

- `dinowm_transformer_baseline_rerun_seed0` used explicit `future_actions [B,H,A]` but was not reportable:
  - best val patch cosine error about `0.9996`;
  - multi-horizon H=1/2/4 patch cosine error about `1.0`;
  - audit says `beats_persistence=unsupported`, `uses_action_information=unsupported`, `model_internal_planning_sanity=unsupported`.
- This should be treated as a negative diagnostic: the real-data ANN baseline still needs repair before any SNN/ES claim is meaningful.

Evidence:

- Registered older real diagnostic: `R-DWM-G3-DINOWM-BASELINE-REAL-001`
- Unregistered newer diagnostic, do not cite as formal claim until ledger is updated:
  - `results/runs/dinowm_transformer_baseline_rerun_seed0/summary.json`
  - `results/runs/dinowm_transformer_baseline_rerun_seed0/eval_multihorizon/summary.json`
  - `results/runs/dinowm_transformer_baseline_rerun_seed0/audit_report.json`

## 5. Official DINO-WM Upstream Reproduction

目标：

在本地尽量复现官方 DINO-WM PointMaze 路线，为 SNN-WAM 的 ANN baseline 提供参照。

已完成：

- 官方 repo 已克隆到 `external/dino_wm`。
- 官方 PointMaze 数据已放到 `data/dino_wm/point_maze`。
- 官方 upstream commit: `0a9492fa12044b852ae9e001cc74604b79c8bb0c`。
- 针对本机 RTX 5060 Ti / CUDA `sm_120` 问题，建立了 host-compatible `dino_wm_cu128` 环境。
- DINOv2 torch hub pin 到 `b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8`，绕开 Python 3.9 不兼容语法。
- 安装 MuJoCo 2.1 并修复 `mujoco_py` 依赖。

Smoke results:

| Stage | Setting | Result | Status |
|---|---|---:|---|
| train smoke | official code, PointMaze, 2 rollouts, 1 epoch | train loss `2.5175`, val loss `2.2880` | Smoke runnable |
| plan smoke | 1-epoch smoke checkpoint, `n_evals=1`, `planner.opt_steps=1` | success rate `0.0`, state_dist `1.6831`, plan_loss `2.71846` | Plumbing runnable only |
| full train | official full PointMaze, default batch/workers | WSL restarted after `Loaded 2000 rollouts` | Interrupted, no Python traceback |

Interpretation:

- 官方 train/plan 路径已经能在本机最小规模跑通。
- 但 full DINO-WM reproduction 没有完成；当前不能说官方 DINO-WM 已复现。
- 本机 WSL/显存/内存环境不适合反复跑 strict default full train。
- 已准备 resource-limited full-data preflight，可以作为本地诊断路线，但不能叫 strict upstream reproduction。

Evidence:

- `docs/OFFICIAL_DINOWM_UPSTREAM_REPRO.md`
- `results/upstream/official_dinowm_pointmaze_train_smoke_20260605_cu128_dinov2b483/metrics.csv`
- `results/upstream/official_dinowm_pointmaze_plan_smoke_20260605_cu128_dinov2b483_mujoco_weights/metrics.csv`
- `results/upstream/official_dinowm_pointmaze_full_preflight_20260605_cu128_dinov2b483/wsl_crash_analysis.md`

## 6. What We Learned

1. 不能跳过 ANN/world-model baseline 直接上 SNN/ES。

   ES 的 failure mode 太多：扰动方向、fitness 定义、参数子空间、噪声尺度、batch 方差、JAX sharding 都可能导致不更新。SNN-WAM 必须先有一个可学的 latent world-model objective。

2. 旧 BC policy 失败是有信息量的负结果。

   它说明直接 action regression 会被 gripper metric、action-history leakage、自回归误差放大和 representation bottleneck 混在一起，不适合作为 SNN 主实验。

3. DINO-WM 路线更符合 scientific question。

   它把 SNN 放在 latent dynamics prediction 位置，而不是低层 action head。这样才能问“无梯度训练 SNN 世界模型是否可行”。

4. 当前最大技术瓶颈不是 SNN，而是 real patch-latent ANN baseline 未过门控。

   DWM-G1/G2/G3 synthetic gates pass，但 real baseline 还没有稳定 beat persistence，也没有证明 future actions 被模型有效利用。

## 7. Next Week Plan

优先级按门控顺序：

1. 修复 DWM-G3 real ANN baseline。
   - 使用更可靠 train/val split。
   - 重新跑 persistence、zero action、shuffle future action ablations。
   - 目标：H=1/H=2/H=4 patch cosine error 至少 beat persistence。

2. 补齐 `dinowm_transformer_baseline_rerun_seed0` 的 artifact registry。
   - 如果继续引用该 run，需要登记到 `docs/RESULT_ARTIFACTS.md` 和 `docs/CLAIMS_LEDGER.md`。

3. 为 direct ES 做 toy sanity，不直接上 SNN-WAM。
   - toy quadratic sign convention。
   - small MLP latent predictor ES。
   - fixed seed generation metrics。
   - 确认 updates 非零且 fitness 改善。

4. SNN 只做接口准备，不做科学 claim。
   - LIF/PLIF forward shape。
   - reset policy。
   - spike rate / SynOps proxy logging。
   - 等 ANN baseline 过门控后再接 ES。

## 8. Suggested Slide Outline

1. Title: 三周进展 - 从无梯度 SNN 训练想法到 DINO-WM 世界模型路线
2. Research question: direct-trained SNN latent world model
3. Why old BC route is frozen: evaluator valid, WAM-GRU `0/30`, readiness gate failed
4. ES/EGGROLL work: HyperscaleES can run but optimization evidence not yet positive
5. nano-egg work: int8 ES pretraining code adapted for local resource constraints
6. New SNN-WAM route: DINOv2 patch latents -> latent dynamics -> planning
7. Gates passed: DWM-G1/G2/G3 synthetic tests
8. Gates failed/pending: real DWM-G3 baseline, DWM-G4 planning, SNN not started
9. Official DINO-WM upstream: train/plan smoke runnable, full run blocked by WSL resource crash
10. Next week: fix real ANN baseline, register artifacts, run ES toy sanity, then SNN forward

## 9. Top Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Real DINO-WM baseline does not beat persistence | Without this, SNN/ES results are uninterpretable | Debug split, action conditioning, normalization, model capacity, and official reference |
| ES updates are often zero or fitness-flat | Direct SNN training may fail for optimizer reasons, not SNN reasons | Run toy ES sanity with mandatory nonzero update and fitness-improvement checks |
| Local WSL/GPU environment is unstable for full official DINO-WM | Full reproduction may be blocked locally | Use resource-limited diagnostic locally; use native Linux/remote GPU for strict reproduction |

## 10. Advisor Summary

本周可向导师概括为：

```text
近三周我完成了 ES/EGGROLL 与 nano-egg 的本地复现适配，并把 SNN-WAM 从直接 action-BC 路线收缩到 DINO-WM patch-latent 世界模型路线；证据显示旧 BC 路线和当前 real DINO-WM baseline 都还不能支撑 SNN/ES claim，下一步应先修复真实 patch-latent ANN baseline，再做 ES toy sanity 和 SNN forward 接口。
```
