# 实验方案反思报告：2026-06-01

## 1. 结论先行

当前实验方案不宜继续沿着“尽快加入 SNN adapter、future latent、鲁棒性、ES 后训练”的完整路线推进。现有证据显示，项目的主要瓶颈还不在 SNN 是否优于 GRU，而在更基础的策略学习链条：因果 next-action 拟合、动作表示、teacher-forced 到 autoregressive 的误差传播，以及 closed-loop readiness。

建议把短期目标从“验证 SNN temporal/world-action adapter 是否提升 closed-loop robustness”收缩为：

```text
先建立一个能通过因果 H=1、multi-demo offline、autoregressive readiness gate、
并在小规模 LIBERO closed-loop 中取得非零成功率的 residual BC/GRU 基线。
```

在这个基线通过前，不建议运行更大的 SNN、future-latent 或 ES 实验；这些实验即使失败，也很难解释是架构问题、表示问题、动作头问题，还是 closed-loop compounding error 问题。

## 2. 本报告依据

本报告只引用已经登记在 `docs/RESULT_ARTIFACTS.md` 和 `docs/CLAIMS_LEDGER.md` 的诊断证据。它不是新的实验结果，也不产生“可投稿结论”。

| 证据 | 关键结果 | 对当前方案的含义 |
|---|---|---|
| `R-G5-DIAGNOSTIC-EVAL-001` | Expert replay 在 tasks 1,2,3 上为 27/30=90%；WAM-GRU future/no-future 均为 0/30；zero/random 也为 0/30。 | closed-loop evaluator 本身可产生成功；learned policy 当前没有 closed-loop 能力。 |
| `R-G5-WAM-GRU-OPEN-LOOP-DIAGNOSTIC-001` | WAM-GRU future/no-future 在 teacher forcing 下 action MSE 优于 zero、random、mean、last-action baseline。 | teacher-forced MSE 改善不足以推出 closed-loop success。 |
| `R-G5-OVERFIT-REPAIR-001` | Raw WAM-GRU 只有 `shift=-1` 通过 `1e-4`，但该目标已在 action history 中；nominal `shift=0` 未通过。 | 当前 policy pipeline 不能支撑架构优劣 claim。 |
| `R-G7-CONTRACT-001` | `robot_states` 是 9-dim proprio，不是 oracle；92-dim `states` 是 full MuJoCo qpos/qvel，但分解未确认。 | 过去的“oracle/proprio”语义需要保守处理；表示瓶颈仍未排除。 |
| `R-G8-MIXED-001` | 7-dim action 应拆成 6 个连续维度和 1 个 binary gripper；旧全局 raw MSE 不适合作为主指标。 | 原始 metric 会混淆 continuous control 和 gripper。 |
| `R-G9-RESIDUAL-001` | Residual action target 将 continuous_normalized_mse 降到 0.0121；capacity 增大无明显收益。 | 直接预测 absolute action 不是最合适的目标；问题更像 residual/action dynamics。 |
| `R-G10-RESIDUAL-HEAD-001` | Residual action 让 action_history_gru 从 0.0321 改善到 0.0133；held-out demos 上仍有 offline 改善。 | residual head 是目前最有希望的修复方向，但仍不是 closed-loop 证据。 |
| `R-G11-AUTOREG-STAB-001` | history_noise_aug 将 autoregressive full-sequence MSE 从 8.86 降到 0.99，但 readiness gate 未通过；gripper accuracy 低至 52%，held-out mean MSE=9.30。 | 自回归稳定性仍不足，不应直接进入大规模 closed-loop 或 SNN 对比。 |

## 3. 核心反思

### 3.1 科学问题提得太靠后，工程前提还没有站稳

原始问题是“能否用 SNN temporal/world-action adapter 改善 future latent/action modeling 和 closed-loop robustness”。这个问题本身合理，但现在问得过早。当前证据说明，GRU/WAM-GRU 基线还没有通过最小策略学习门槛：

- closed-loop diagnostic 中 WAM-GRU future/no-future 均为 0/30；
- nominal causal H=1 next-action 拟合未通过；
- autoregressive readiness gate 未通过；
- future-latent ablation 还没有可观察的 closed-loop 差异，因为两边都是零成功率。

因此，现在继续加 SNN 只能增加变量数量，不能提高结论解释力。

### 3.2 future latent loss 目前没有转化为控制收益

`R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-001/002` 显示，future loss 能显著改善 future latent cosine error，但 action MSE 并没有同步改善，closed-loop 中 future/no-future 又同为 0/30。当前更保守的判断是：

```text
future latent objective 可以被模型优化，但尚未证明它能改善策略动作或 closed-loop success。
```

如果继续把 future latent 作为主线，容易把“可优化的辅助 loss”误当成“有用的 world-action adapter”。

### 3.3 闭环失败不是 evaluator 失败

Expert replay 27/30=90% 是关键证据。它说明 evaluator、task success 判定、fixed initial state 流程和环境交互至少能在正确动作下产生成功。因此当前失败更可能来自 learned policy 输出，而不是 closed-loop 评估器不可用。

这会改变优先级：下一步不应先修 evaluator，而应修 policy 的 action contract、residual head、history conditioning 和 autoregressive stability。

### 3.4 单纯 teacher-forced 指标有误导风险

WAM-GRU 在 teacher forcing 下优于简单动作 baseline，但 closed-loop 和 autoregressive 诊断都显示误差会放大。说明当前离线指标缺少对“把模型预测重新放回 action history 后会发生什么”的惩罚。

后续任何候选模型都应至少经过三层 gate：

1. teacher-forced H=1 或短 horizon 指标；
2. offline autoregressive rollout 指标；
3. matched small closed-loop smoke。

只通过第 1 层不应进入架构 claim。

### 3.5 主要修复方向不应是更大模型

`R-G9-RESIDUAL-001` 显示 capacity 从 medium 到 large 没有带来实质收益，残差具有系统性时间相关性。更大 GRU/SNN 不是首要修复项。更应该优先处理：

- residual action target；
- split continuous/gripper head；
- position/rotation 分头或权重；
- history noise/dropout augmentation；
- autoregressive training objective；
- 更清楚的 object/goal state 或 raw-image conditioning。

### 3.6 SNN 仍可保留，但应后置为公平对照

SNN adapter 不是要删除，而是需要等待一个清晰基线：

```text
BC-GRU/residual-GRU 能在相同数据、相同 action contract、相同 readiness gate 下
取得非零 closed-loop 成功率后，再替换 temporal adapter 为 LIF/PLIF/ALIF。
```

否则无法判断 SNN 的结果是神经动态带来的差异，还是被上游表示和动作目标问题掩盖。

## 4. 建议调整后的实验路线

### Phase A：先建立最小可用 residual BC 基线

目标不是证明 SNN，而是证明当前数据和动作接口能学出一个基本可控策略。

必须固定：

- 输入：current frozen latent、proprio、task id、action history；
- 输出：residual continuous action + binary gripper；
- metric：continuous_normalized_mse、gripper_sign_accuracy、gripper transition 诊断、autoregressive MSE；
- split：至少 single-demo overfit、same-task held-out demos、后续 fixed-init closed-loop；
- gate：H=1 causal fit、offline autoregressive readiness、artifact completeness。

通过标准建议：

- nominal `shift=0` 不能依赖 `shift=-1`；
- residual head 在 single-demo 和 held-out demos 上稳定优于 last-action；
- autoregressive full-sequence error 不出现大范围 phase blowup；
- gripper autoregressive accuracy 不低于 80%；
- readiness gate 通过后才进入 closed-loop。

### Phase B：小规模 matched closed-loop 复评估

只在 Phase A 通过后运行。

建议先用 tasks 1,2,3，init states 0-9，与 expert replay、zero、random 对齐。最小可接受结果不是“高成功率”，而是：

```text
learned policy 至少明显区别于 zero/random，并产生非零成功或可解释的接近成功行为。
```

如果仍为 0/30，则不要进入 SNN/future-latent claim，而应继续 failure taxonomy。

### Phase C：再做 future latent 和 SNN adapter 对照

只有当 residual BC/GRU 基线具备基本 closed-loop 能力后，再做：

- no-future vs future latent；
- GRU vs SNN-LIF；
- GRU vs SNN-PLIF/ALIF；
- robustness under noise/delay/frame drop；
- spike rate/SynOps proxy。

所有对照必须共享 data split、history length、action head、optimizer budget、seed list、closed-loop task list 和 initial states。

## 5. 暂停项与保留项

### 暂停

- 暂停扩大 SNN adapter 实验。
- 暂停 ES/EGGROLL-style post-training。
- 暂停把 future latent loss 作为主要贡献点。
- 暂停 robustness claim，因为基础 closed-loop success 尚未建立。
- 暂停任何“WAM policy 已验证”“SNN 更强”“future latent 改善控制”的表述。

### 保留

- 保留 frozen encoder + temporal adapter 的阶段一边界。
- 保留 DINOv2 latent 作为当前 frozen visual baseline，但不要假设其足够。
- 保留 future latent 作为后续 auxiliary ablation。
- 保留 SNN adapter 作为后续公平架构对照。
- 保留 closed-loop evaluator，因为 expert replay 已支持其有效性。

## 6. 推荐的下一步工作包

1. 实现或整理一个 `residual_bc_gru` 训练配置，统一 residual continuous head、binary gripper head 和 split metrics。
2. 把 G11 readiness gate 固化为进入 closed-loop 前的自动检查。
3. 在同一 task 的 demo_0 训练、demo_1/10/11/12/13 验证上跑 residual BC/GRU，并记录 artifact。
4. 若 gate 通过，运行 tasks 1,2,3 的 matched closed-loop smoke。
5. 若 closed-loop 仍为 0/30，人工标注 failure taxonomy，再决定是否需要 raw image、object-state parse 或 action postprocessing。

## 7. 对导师的一句话总结

当前结果表明，closed-loop evaluator 是可用的，但 WAM-GRU future/no-future 均在 matched LIBERO 评估中 0/30，且 residual 模型虽显著降低 offline/autoregressive error 但 readiness gate 未通过；我建议暂停 SNN/future-latent 扩展，先把 residual BC/GRU 基线修到能通过因果 H=1、自回归稳定性和小规模 closed-loop 非零成功。

## 8. 仍然存在的风险

- 当前多数诊断是 seed 0 和少量任务，不能替代正式 reportable 多 seed 实验。
- G11 是 offline autoregressive，不是环境交互，不能直接解释 closed-loop 失败机制。
- 92-dim full state 的语义分解尚未完全确认，不能称为 true oracle。
- DINOv2 CLS 的不足尚未等价于“视觉表示不可用”，还需要 raw image、patch feature 或 object/goal state 对照。
- 如果 residual BC/GRU 修复后仍无法产生非零 closed-loop success，项目可能需要更大幅度转向 action representation、demo replay alignment 或现成 LIBERO BC baseline，而不是继续比较 adapter 架构。
