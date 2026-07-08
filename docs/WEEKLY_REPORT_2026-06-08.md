# Weekly Report: 2026-06-08

## 1. This week's goal

本周围绕“如何不依赖常规梯度训练 SNN，并最终训练出 SNN 版 latent world model”继续做前置验证。工作分成两条：

1. 阅读并复现陈洛南老师团队的 CSBP 工作，理解它和直接训练 SNN、surrogate gradient、无梯度 ES 之间的关系。
2. 复现 HyperscaleES/EGGROLL 与 nano-egg，评估 ES-style optimizer 目前能否支撑后续 SNN-WAM 路线。

当前最重要的边界是：CSBP、HyperscaleES 和 nano-egg 都是方法理解与外部工程复现，不是 SNN-WAM 的 reportable result；它们只能帮助决定后续实验路线，不能直接证明 SNN world model 已经可训练。

## 2. Work done

### 2.1 阅读并复现 CSBP

阅读对象：

- Paper: *Brain-inspired chaotic spiking backpropagation*, National Science Review, 2024.
- Authors: Zijian Wang, Peng Tao, Luonan Chen.
- Official code: `https://github.com/Wangzj000/CSBP`.

CSBP 的核心思想是：普通 SNN surrogate-gradient training 仍主要沿着梯度动力学更新，容易陷入局部最优或 spike dynamics 过早僵化；CSBP 在常规 BP loss 外加入一个 chaos loss，让神经元输出或膜电位产生类脑 chaotic dynamics，利用混沌动力学的遍历性和伪随机性增加搜索能力。

我的理解：

| 问题 | CSBP 给出的启发 |
|---|---|
| 它是不是无梯度训练？ | 不是。官方实现仍然 `loss.backward()`，并且可以和 SGD/Adam、surrogate gradient 组合。 |
| 它为什么对 SNN 有意义？ | 它没有把 SNN 当普通 ANN 硬套训练，而是利用 spike/membrane dynamics 本身引入训练动力学。 |
| 和本项目的关系是什么？ | 它适合作为 SNN direct-training 的参考 baseline 或训练动力学启发，但不能替代 ES/EGGROLL 这种真正 black-box/no-gradient 路线。 |
| 对后续路线的影响 | 先证明 SNN latent dynamics objective 可学，再比较 surrogate-gradient、CSBP-style chaotic training 和 ES-style black-box training。 |

复现状态：

- 已完成论文机制阅读和官方代码级复现/对照。
- 复现重点是 single-neuron/multi-layer SNN 中 chaos loss 如何接入 BP loss，以及 `z`、`beta` 衰减如何影响训练轨迹。
- 当前 CSBP 复现记录还没有纳入 SNN-WAM 的 result artifacts，因此本周只能写成 literature reproduction / method understanding。

结论：

CSBP 对我的路线有启发，但它不回答“不通过梯度训练 SNN”这个问题。它更像是一个强 surrogate-gradient baseline：如果后续 SNN-WAM 做 SNN latent dynamics predictor，CSBP 可以作为“仍使用反传但加入混沌动力学”的对照组；而真正验证无梯度目标，仍需要 ES/EGGROLL-style training。

### 2.2 复现 HyperscaleES 与 nano-egg

#### HyperscaleES / EGGROLL

本周复现目标是 *Evolution Strategies at the Hyperscale (EGGROLL)* 的训练路径：尝试用 ES 替代梯度下降，微调 RWKV-7 LLM 做数学推理任务。

当前主要配置：

| 参数 | 当前值 |
|---|---|
| 模型 | RWKV-7 1.5B (`7gg1.5B`) |
| `lr_scale` | `1.0` |
| `sigma` | `1e-3` |
| `gen_per_prompt` | `8` |
| `generation_length` | `100` |
| `parallel_gen/gpu` | `512` |

已完成 runs：

| 实验 | 任务 | Batch size |
|---|---|---:|
| Jun4 GSM8K | GSM8K | `4096` |
| Jun8 GSM8K | GSM8K | `4096` |
| Jun4 GSM8K | GSM8K | `128` |
| Jun4 Countdown | Countdown | `4096` |
| 其他 | GSM8K | `256 / 768 / 800` |

最关键的诊断不是 score 上升，而是生成质量退化：

| Epoch | 生成质量 |
|---:|---|
| 0 | 正常英文数学推理，可以读出问题理解和逐步推理。 |
| 50 | 开始重复 token，例如 `Thinking Think< think>...`。 |
| 100 | 变成乱码或重复字符，例如 `ic'ic'owe'o'io...`。 |
| 250 | 重复更严重，基本失去可读推理结构。 |
| Countdown epoch 100 | 同样出现无意义混合输出。 |

Jun4 `bs=4096` 完整实验的退化时间线：

| Epoch | 质量 |
|---:|---|
| 0 | 正常 |
| 100 | 重复字符，如 `-o-`、`{`、`}` |
| 200 | 重复 token 片段 |
| 300 | 重复短语片段 |
| 400 | 完全乱码 |
| 450 | 重复模式 |

解释：

- 代码路径已经能跑到 generation、fitness evaluation 和 ES update。
- 但当前 RWKV-7 1.5B runs 不支持“ES 有效优化 LLM”这个 claim。
- 生成质量 collapse 说明继续单纯扩大 batch 或 epoch 没有意义；应先检查 update magnitude、fitness normalization、reward sparsity、KL/质量约束、`sigma/lr_scale` schedule，以及 update 是否过大或方向错误。
- 对 SNN-WAM 来说，HyperscaleES 仍可作为后续候选 optimizer，但必须先通过更小的 SNN-specific sanity，而不能直接上真实 DINO patch latent world model。

#### nano-egg

nano-egg 的价值在于它比完整 HyperscaleES 更小、更透明，适合理解 EGGROLL-style int8 pretraining 的实际工程形态。

本周完成：

- 阅读 nano-egg single-file implementation。
- 明确它的训练对象是小型 RWKV/minGRU-style language model，而不是 SNN。
- 梳理 int8 权重/激活、ES perturbation、fitness 计算、microbatch update 的实现方式。
- 针对资源做了运行适配，包括 JAX memory fraction、update microbatch、sequence/group-size 整除和 SLURM/sweep 覆盖参数。

当前 nano-egg 复现记录：

| 项目 | 观察 |
|---|---|
| 模型 | RWKV-7 nano-egg, about `4.86M` parameters |
| 配置 | int8, `256D`, `6Layers`, `8b`, `1024p` |
| 数据 | 8 条序列，每条约 `737M` tokens |
| 训练 | 1000 steps, about 4 minutes on `8xH20` |
| 初始 validation | about `6.02` bits |
| 最终 validation | about `4.60-4.62` bits |
| 参考 baseline | unigram `5.0`, bigram `4.0`, gzip max compression `2.767` |

解释：

- nano-egg 给出了一个更清楚的“小模型 + ES-style update + integer/pretraining”的参考实现。
- 从 `6.02` bits 到约 `4.61` bits 的下降说明这个小实现具备工程学习信号。
- 但它不是 SNN，不是 world model，也没有 LIBERO/DINO patch latent 任务，因此只能作为后续 tiny ES sanity 的方法参考。

## 3. Evidence status

| Work item | Evidence | Status |
|---|---|---|
| CSBP literature reproduction | NSR paper + official CSBP GitHub + external reproduction notes | Literature / external engineering, not SNN-WAM artifact |
| HyperscaleES RWKV-7 1.5B runs | User-provided Jun4/Jun8 GSM8K/Countdown notes | External engineering diagnostic |
| HyperscaleES toy and 0.1B audit | `docs/HYPERSCALEES_POTENTIAL_AUDIT_2026-06-08.md` | Diagnostic only |
| nano-egg | `/home/zhan_shaoji/code/nano-egg/README.org` and group-meeting notes | External engineering diagnostic |

No claim from this weekly report should be entered as `supported` in `docs/CLAIMS_LEDGER.md` unless the corresponding result package is registered in `docs/RESULT_ARTIFACTS.md`.

## 4. What I learned

1. CSBP is important, but it is not the same problem as ES. It improves SNN direct training by adding chaotic dynamics to backpropagation; it does not remove gradient training.
2. HyperscaleES has a plausible low-rank ES mechanism, but current RWKV-7 reproduction shows generation collapse, not useful optimization.
3. nano-egg is the cleanest small implementation reference this week. It is more useful for understanding update mechanics than for proving SNN-WAM feasibility.
4. For the main SNN-WAM route, the safest order is still: DINO patch-latent ANN baseline first, then SNN forward/surrogate baseline, then ES/EGGROLL comparison.

## 5. Risks

- If HyperscaleES generation quality collapses, scaling batch size or epoch count will only produce more negative diagnostics. Stop and inspect update scale, reward normalization and quality constraints first.
- CSBP should not be described as no-gradient training. It is a stronger gradient-based SNN training baseline.
- nano-egg should not be described as evidence that ES can train SNN world models.
- Current DINO-WM real baseline has not passed persistence/action-use gates, so it is too early to claim SNN/ES progress on the real world-model objective.

## 6. Next week plan

1. For HyperscaleES: reproduce a tiny controlled sanity with dense reward, fixed seed, update-norm logging and generation-quality guardrail before any larger RWKV run.
2. For SNN-WAM: repair DWM-G3 real patch-latent ANN baseline until it beats persistence and uses future actions.
3. For SNN training comparison: after DWM-G3/G4 pass, implement minimal SNN forward with reset and spike-rate logging, then compare surrogate-gradient, CSBP-style chaotic loss and ES-style update.
4. For reporting: keep CSBP / nano-egg / HyperscaleES as method reproduction unless their artifacts are registered and pass project gates.

## 7. Advisor summary

本周完成了 CSBP 文献阅读与代码级复现理解，并复现/诊断了 HyperscaleES 与 nano-egg；结论是 CSBP 可作为强梯度训练 baseline，nano-egg 可作为小型 ES 实现参考，而当前 HyperscaleES RWKV-7 runs 出现生成质量 collapse，暂不能作为“ES 能有效训练大模型或 SNN world model”的正证据。

## References

- CSBP paper: `https://academic.oup.com/nsr/article/11/6/nwae037/7592018`
- CSBP official code: `https://github.com/Wangzj000/CSBP`
- HyperscaleES audit: `docs/HYPERSCALEES_POTENTIAL_AUDIT_2026-06-08.md`
- Three-week group meeting notes: `docs/GROUP_MEETING_2026-06-08_THREE_WEEK_PROGRESS.md`
