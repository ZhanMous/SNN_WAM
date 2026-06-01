# SNN-WAM Project Plan Summary

本文件是 `SNN_WAM_plan.pdf` 的紧凑工程摘要。原始 PDF 是当前计划来源；本 markdown 用于让仓库内文档和 smoke tests 有稳定入口。

## Route Reset: 2026-06-01

The original direct action BC / WAM-GRU route is now frozen as diagnostic evidence. It showed that direct action BC is not a stable enough base for testing WAM, future-latent objectives, or SNN dynamics. The current route starts from DINO-WM-style latent world modeling:

```text
DINOv2 spatial patch features + action sequence
  -> latent world model
  -> future spatial patch features
  -> action sequence optimization / planning
```

The intended SNN contribution is a directly trained SNN latent world model, not an SNN fine-tuning head, not surrogate-gradient-only training, and not ANN-to-SNN conversion.

See `docs/DINOWM_SNN_WORLDMODEL_PLAN.md`.

## Core Question

在具身基础模型中，SNN 能否作为 `world-action adapter`，承担短期动态记忆、future state/latent prediction 和 action modulation？

更新后的第一阶段不做完整大模型，而是验证冻结 DINOv2 spatial patch features 之后的 latent world model：

```text
DINOv2 patch latent + action sequence
  -> ANN/GRU/SNN world model
  -> future patch latent
  -> planning by action sequence optimization
```

## Stage 1 Boundaries

- 不直接训练完整 VLA/WAM 大模型。
- 不直接上 OpenVLA 7B。
- 不直接上真实 Unitree。
- 不先写论文故事。
- 不从零训练完整 VLA/WAM；但本路线允许在小 SNN latent world model 上研究 direct ES / EGGROLL-style optimization，因为这是用户明确指定的核心目标。
- 不声称 SNN 天然低功耗；第一阶段只报告 spike rate / SynOps proxy。

## Recommended Research Path

1. 冻结 direct action BC 旧路线，保留诊断证据。
2. 复现最小 DINO-WM-style latent world model。
3. 从 CLS latent 切换到 DINOv2 spatial patch features。
4. 建立 patch-latent transition dataset。
5. 训练 ANN/GRU latent world model baseline。
6. 实现直接训练的 SNN latent world model。
7. 研究 EGGROLL-style / low-rank ES 是否能直接优化 SNN world model。
8. 在 latent prediction 通过后，再做 action sequence optimization / planning。
9. 最后再评估 closed-loop planning、robustness 和 spike-rate proxy。

## Required Evidence

每个实验必须可复现、可复评估、可审计。所有科学主张必须能指向结果文件。

最低证据包括：

- config file
- metrics file
- run log
- git commit record
- evaluation output
- notes for anomalies and failure modes

## First Reportable Experiment Shape

第一批新路线可汇报结果应比较：

- copy-last-latent baseline
- ANN/Transformer DINO-WM-style predictor
- GRU latent world model
- directly trained SNN latent world model
- random/untrained SNN baseline

必要指标：

- future patch latent cosine / MSE
- multi-step latent drift
- nearest-neighbor future-frame retrieval
- planning objective reduction
- closed-loop planning success if planning is evaluated
- spike rate / SynOps proxy
- ES population/generation metrics if direct ES is used
