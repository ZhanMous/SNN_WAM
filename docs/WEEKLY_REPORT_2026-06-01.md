# Weekly Report: 2026-06-01

## 1. This week's goal

本周围绕导师提出的“能否不用梯度训练 VLA/WAM”问题，我先没有直接上 EGGROLL 或 SNN，而是做了一个前置诊断：当前 LIBERO WAM/GRU policy pipeline 是否已经有足够稳定的 supervised warmup，可以作为后续 EGGROLL-style low-rank ES 的初始模型。

原因是 ES 更适合优化非可微系统级目标，例如 closed-loop success、robustness、action smoothness、spike-rate proxy。如果 warmup policy 自己还没有基本 closed-loop 能力，直接用 ES 搜索会很难判断失败来自哪里。

## 2. Data and setup

本周主要使用 LIBERO spatial demonstration 数据和预计算 DINOv2 frozen visual latents。

| Item | Setting |
|---|---|
| Dataset suite | `libero_spatial` |
| Split file | `splits/libero_episode_split_seed20260528.json` |
| Split unit | episode |
| Total episodes | 50 |
| Train/val/test | 40 / 5 / 5 |
| Frozen visual latent | DINOv2 ViT-S/14 CLS token |
| Latent dim | 384 |
| DINOv2 revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| Main trajectory for diagnostics | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5:data/demo_0` |
| Main trajectory length | 103 timesteps |
| Action shape | `[103, 7]` |
| Action semantics | 6 continuous dims + 1 binary gripper dim |
| Proprio state | `robot_states`, 9 dims |
| Full simulator state | `states`, 92 dims, MuJoCo qpos/qvel, decomposition still uncertain |
| Image observations | `agentview_rgb` and `eye_in_hand_rgb`, `[103, 128, 128, 3]` |

The active BC-GRU diagnostic config is `configs/diagnostics/libero_spatial_bc_gru_dinov2s_proprio_task.yaml`. It defines the intended sanity baseline:

```text
input: DINOv2 CLS latent + proprio + task id + action history
history_len: 4
action_horizon: 4
future_horizon: 4
temporal_adapter: bc_gru
hidden_dim: 256
lambda_future: 0.0
optimizer: AdamW, lr=3e-4
```

This config is a diagnostic direction, not a reportable result by itself.

## 3. What I did

### 3.1 Checked whether closed-loop evaluation is valid

流程：

1. 用 LIBERO evaluator 固定 tasks `1,2,3`。
2. 每个 task 跑 10 个 benchmark initial states，即 init states `0-9`。
3. 每个 episode 最大 300 steps，seed=0。
4. 对比 expert replay、WAM-GRU future、WAM-GRU no-future、zero action、random action。
5. Expert replay 使用 HDF5 demo action 序列，目的是确认 evaluator 能不能产生成功，而不是测试 learned policy。

结果：

| Policy | Success |
|---|---:|
| Expert replay | `27/30 = 90%` |
| WAM-GRU future | `0/30 = 0%` |
| WAM-GRU no-future | `0/30 = 0%` |
| Zero action | `0/30 = 0%` |
| Random action | `0/30 = 0%` |

解释：

Evaluator 本身不是主要问题，因为 expert replay 在同一环境、同一 tasks、同一 initial states 下可以成功。当前 learned WAM-GRU policy 在 closed-loop 中没有表现出区别于 zero/random 的能力。

Evidence: `docs/ROLL_OUT_FINDINGS.md`, `R-G5-DIAGNOSTIC-EVAL-001`.

### 3.2 Compared WAM-GRU future vs no-future under offline teacher forcing

流程：

1. 使用 real DINOv2 frozen latents。
2. 两个 WAM-GRU checkpoint 参数量相同，都是 `769,564` trainable parameters。
3. future 版本使用 `lambda_future=1.0`，no-future 版本使用 `lambda_future=0.0`。
4. 在 val split 上做 teacher-forced offline evaluation，共 `6073` samples。

结果：

| Model | Action MSE | Future latent cosine error | Future latent MSE |
|---|---:|---:|---:|
| WAM-GRU future | `0.01910184353` | `0.001129716767` | `0.1030620597` |
| WAM-GRU no-future | `0.01841307318` | `1.003726423` | `5.471436987` |

解释：

future latent loss 确实优化了 latent prediction 指标，但没有带来 action MSE 改善；closed-loop 里两个版本也都是 `0/30`。所以当前不能说 future latent 有控制收益，只能说辅助 latent loss 被优化了。

Evidence:

- `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_offline.csv`
- `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_offline.csv`

### 3.3 Audited action/state contract

流程：

1. 检查 HDF5 schema。
2. 明确 `robot_states`、`states`、`actions`、RGB observations 的含义和 shape。
3. 检查 action 每一维的统计量，确认 gripper 是否应和连续动作混在同一个 MSE 中。

结果：

- `actions` 是 7 维：dims `0-5` 是 continuous delta position / orientation，dim `6` 是 binary gripper。
- `robot_states` 是 9-dim proprio，不是 oracle object state。
- `states` 是 92-dim full MuJoCo state，但还不能可靠分解出 object pose / goal pose。
- Gripper 在 demo_0 中 open/close 分布为 open `64`、close `39`，只有 2 次 transition。

解释：

旧的 global raw action MSE 会把连续控制和二值 gripper 混在一起，不适合作为主要指标。后续必须使用 split objective：continuous normalized MSE + gripper sign accuracy / transition diagnostics。

Evidence:

- `results/g7_state_action_contract/20260529_093805_g7_state_action_contract/summary.json`
- `results/g8_mixed_action_metrics/20260529_094852_g8_mixed_action/summary.json`

### 3.4 Ran H=1 overfit and alignment diagnostics

流程：

1. 在单条 demo 上做 H=1 next-action overfit。
2. 对 target shift 做 sweep：`-1,0,+1,+2`。
3. 检查模型是否真的学到了 next action，而不是复制 action history 中已经出现的动作。
4. 加入 split-gripper head，并对比 timestep embedding upper bound 和 DINO-latent MLP。

结果：

| Variant | Result |
|---|---|
| Raw WAM-GRU, shift `-1` | `eval_mse=0.000066282`, passes old `1e-4` gate |
| Raw WAM-GRU, nominal shift `0` | `eval_mse=0.000103113`, does not pass |
| Split-gripper WAM-GRU on shift `-1` | `eval_mse=0.000228181`, continuous MSE `0.000266211`, gripper MSE `0.0` |
| Timestep embedding MLP | `eval_mse=0.0000000536`, passes |
| DINO-latent MLP | `eval_mse=0.002817895`, fails |

解释：

唯一通过的 WAM-GRU 设置是 `shift=-1`，但这个 target 对应的 action 已经在 history 中出现，不能证明 causal next-action learning。nominal `shift=0` 没有稳健通过，所以当前 policy training pipeline 不能支撑“架构有效”的结论。

Evidence: `docs/ROLL_OUT_FINDINGS.md`, `R-G5-OVERFIT-REPAIR-001`.

### 3.5 Tested representation and action-history bottlenecks

流程：

1. 在同一条 demo 上对比 proprio-only、DINO CLS、action history GRU 等输入。
2. 检查 latent retrieval、latent dynamics、goal-feature planning diagnostic。

结果：

| Variant | Key result |
|---|---|
| proprio-only state | eval MSE `0.0353`, fails |
| DINO CLS MLP | eval MSE `0.0399`, fails |
| action-history GRU | eval MSE `0.00240`, better but still fails old `1e-4` gate |
| DINO CLS nearest timestep retrieval | `0.854` |
| DINO CLS latent-action distance correlation | `0.099` |
| DINO CLS latent dynamics val MSE | `0.238` |

解释：

DINO CLS latent 能区分时间位置，但和 action distance 的相关性很弱。这提示当前视觉 latent 可能没有直接编码足够的 control-relevant state。action history 比单帧 latent 更有用，但仍不足以稳定支持 closed-loop。

Evidence: `results/g6_representation_bottleneck/20260529_091230_g6_repr_bottleneck/summary.json`.

### 3.6 Tried residual action target and residual heads

流程：

1. 将 action prediction 从直接预测 `action[t]` 改为预测 residual action，即近似 `action[t] - action[t-1]`。
2. 对比 full_state/action_history 的不同 capacity。
3. 对比 direct action head vs residual action head。
4. 在 demo_0 训练，并在同 task 的 held-out demos `1,10,11,12,13` 上做 offline evaluation。

结果：

| Experiment | Key result |
|---|---|
| residual action target | continuous_normalized_mse `0.0121` |
| full_state medium capacity | `0.0233` |
| full_state large capacity | `0.0244`, no improvement |
| action_history_gru direct | `0.0321` |
| action_history_gru residual | `0.0133`, 2.42x better |
| full_state_plus_history direct | `0.0183` |
| full_state_plus_history residual | `0.0142`, 1.29x better |
| same-task held-out demo improvement | residual better than direct on demos `1,10,11,12,13`, improvement roughly 7.8x to 22.1x |

解释：

Residual action 是目前最有效的修复方向之一，说明直接预测 absolute action 不合适。但这仍然只是 offline teacher-forced / held-out-demo 指标，不是 closed-loop success。

Evidence:

- `results/g9_residual_action_repair/20260529_100801_g9_residual_repair/summary.json`
- `results/g10_residual_action_head/20260529_101802_g10_residual_head/summary.json`

### 3.7 Tested autoregressive stabilization before ES

流程：

1. 构建三种 evaluation mode：
   - `teacher_forced_h1`: 每一步都用 ground-truth action history。
   - `autoregressive_open_loop`: 把模型预测的 action 回填到 history，但 observation/state 仍来自 recorded sequence。
   - `corrupted_history_robustness`: 给 ground-truth history 加受控噪声。
2. 对比五种 stabilization variants：
   - baseline
   - history noise augmentation
   - history dropout augmentation
   - diagnostic regularization
   - offline multistep loss
3. 设定 closed-loop readiness gate，判断是否值得进入 closed-loop 或 ES。

结果：

| Variant / Gate | Result |
|---|---|
| baseline teacher_forced_mse | `0.0142` |
| history_noise_aug teacher_forced_mse | `0.00991` |
| baseline autoregressive full-sequence MSE | `8.86` |
| history_noise_aug autoregressive full-sequence MSE | `0.99` |
| readiness gate | NOT PASSED |
| max autoregressive MSE | `20.63` |
| min gripper accuracy under autoregressive rollout | `0.517` |
| held-out demo mean MSE | `9.30` |
| worst failure dimension | `delta_pos_z`, total MSE `192.09` |

解释：

history noise augmentation 大幅降低了 autoregressive error，但还没有稳定到可以进入 closed-loop 或 ES 后训练。特别是 gripper accuracy 掉到约 52%，held-out demo error 仍然很高，说明模型在自回归状态下会积累动作历史错误。

Evidence:

- `results/g11_autoregressive_stabilization/20260601_023130_g11_autoregressive_stabilization/summary.json`
- `results/g11_autoregressive_stabilization/20260601_023130_g11_autoregressive_stabilization/closed_loop_readiness_gate.md`
- `results/g11_autoregressive_stabilization/20260601_023130_g11_autoregressive_stabilization/failure_mode_analysis.md`

## 4. Main results table

| Question | Answer from this week | Evidence |
|---|---|---|
| Is the closed-loop evaluator broken? | Probably no. Expert replay succeeds at `27/30`. | `R-G5-DIAGNOSTIC-EVAL-001` |
| Does WAM-GRU currently solve LIBERO spatial closed-loop tasks? | No. Future and no-future are both `0/30`. | `R-G5-DIAGNOSTIC-EVAL-001` |
| Does future latent loss help closed-loop? | Not observed. It improves latent error but both policies are `0/30`. | `eval_offline.csv`, rollout diagnostic |
| Is old raw action MSE reliable? | No. Action should be split into continuous + gripper. | `R-G8-MIXED-001` |
| Is larger model capacity the fix? | Probably no. Medium to large capacity does not improve residual error. | `R-G9-RESIDUAL-001` |
| Is residual action target useful? | Yes for offline metrics. It improves direct prediction substantially. | `R-G9`, `R-G10` |
| Is the current model ready for EGGROLL-style ES? | No. Readiness gate failed. | `R-G11-AUTOREG-STAB-001` |

## 5. Possible explanations

这些是目前的猜测，不是结论。

1. **当前路线可能把 policy learning 问题做得太早。**  
   WAM-GRU 直接从 DINO CLS + action history 预测 action chunk，但 closed-loop 是 `0/30`。可能需要先学习 action-conditioned latent world model，而不是先做 policy。

2. **DINO CLS 可能不是足够好的 control state。**  
   DINO CLS 的 nearest timestep retrieval 可以到 `0.854`，但 latent-action distance correlation 只有 `0.099`。它可能编码了画面相似性和时间位置，但没有稳定编码“哪个 action 会推进任务”。

3. **Action target formulation 可能比 adapter architecture 更关键。**  
   residual action target 和 split head 明显改善 offline 指标，而加 capacity 没有明显帮助。这说明瓶颈可能不是 GRU/SNN 容量，而是 action 表示和训练目标。

4. **Teacher-forced 指标高估了 closed-loop readiness。**  
   在 teacher forcing 下模型可以优于 last-action，但一旦把预测 action 回填到 history，误差会迅速积累。ES 如果直接优化这种 policy，可能主要在对抗历史漂移和 gripper collapse，而不是优化世界模型。

5. **EGGROLL-style ES 的位置应该后移。**  
   更合理的顺序可能是：先用梯度或 surrogate-gradient 训练一个稳定 latent world model / SNN world model，再用 low-rank ES 优化非可微 rollout fitness。当前不适合从一个 `0/30` policy 出发直接做 ES。

6. **下一步可能应该参考 JEPA/V-JEPA。**  
   JEPA 的重点是预测 latent representation 而不是像素或 action 本身。对于本项目，更自然的问题可能是 `z_t, a_t:t+k -> z_{t+k}` 是否可学、是否对目标接近有用，然后再接 policy 或 ES。

## 6. Problems and risks

1. 当前所有 G5-G11 结果主要是 diagnostic，不是 reportable scientific result。
2. 多数结果是 seed 0、小任务范围或 single-demo 诊断，不能写成一般性结论。
3. G11 是 offline autoregressive，不是 environment interaction，不能直接证明 closed-loop failure cause。
4. 92-dim `states` 不能直接称为 true oracle state，因为 MuJoCo state decomposition 尚未确认。
5. 如果不先修正 world-model/representation 任务定义，继续做 SNN/EGGROLL 可能只是在不稳定 pipeline 上叠复杂度。

## 7. Next week's plan

1. 复现一个最小 JEPA/I-JEPA 或 V-JEPA 训练流程，重点看 context encoder、target encoder、predictor、防 collapse 和 latent evaluation。
2. 把机器人任务改写成 action-conditioned latent prediction：`z_t + action history/action chunk -> z_{t+k}`。
3. 在 recorded LIBERO trajectories 上先做 offline latent world-model evaluation，不直接做 closed-loop policy。
4. 设计 SNN 版 temporal latent predictor 的最小接口，但先不做性能 claim。
5. 重新定义 EGGROLL-style ES 的进入条件：必须有一个 surrogate-gradient warmup checkpoint，并且 offline autoregressive/readiness gate 通过后，再考虑 low-rank ES 优化 rollout-level fitness。

## 8. One-sentence advisor summary

本周我围绕“无梯度训练 VLA/WAM”先检查了当前 LIBERO WAM-GRU policy 是否适合作为 EGGROLL-style ES 的 warmup：使用 `libero_spatial` 50 demos、DINOv2-S/14 CLS latents、tasks 1-3 closed-loop evaluator、single-demo H=1、residual action 和 autoregressive readiness diagnostics，结果显示 expert replay `27/30` 但 WAM-GRU future/no-future 均 `0/30`，residual/action-history 改善了 offline 指标但 G11 readiness gate 未通过；我的判断是当前不应直接上 SNN/EGGROLL，而应先复现 JEPA/V-JEPA 并把问题重定义为 action-conditioned latent world model。

## 9. Claim discipline

可以说：

- 本周做了 EGGROLL-style ES 前置条件诊断。
- 当前 WAM-GRU policy pipeline 在 closed-loop 上失败，且 readiness gate 未通过。
- residual action 和 history noise augmentation 是有效的 offline 修复方向，但还不能支撑 closed-loop claim。
- 下周计划先复现 JEPA/V-JEPA，重新定义 latent world-model 任务。

不应说：

- 已经复现 EGGROLL。
- 已经证明无梯度训练可以训练 VLA/WAM。
- 已经验证 SNN world model。
- future latent loss 改善了 closed-loop success。
- WAM-GRU 或 residual policy 已经可用于 closed-loop control。
