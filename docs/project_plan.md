# SNN-WAM Project Plan Summary

本文件是 `SNN_WAM_plan.pdf` 的紧凑工程摘要。原始 PDF 是当前计划来源；本 markdown 用于让仓库内文档和 smoke tests 有稳定入口。

## Core Question

在具身基础模型中，SNN 能否作为 `world-action adapter`，承担短期动态记忆、future state/latent prediction 和 action modulation？

第一阶段不做完整大模型，而是验证冻结视觉/语言编码器之后的 temporal adapter：

```text
image latent + language latent + action history
  -> temporal adapter: MLP / GRU / SNN
  -> future action chunk + future visual latent chunk
```

## Stage 1 Boundaries

- 不直接训练完整 VLA/WAM 大模型。
- 不直接上 OpenVLA 7B。
- 不直接上真实 Unitree。
- 不先写论文故事。
- 不从零用 ES 训练 SNN-WAM。
- 不声称 SNN 天然低功耗；第一阶段只报告 spike rate / SynOps proxy。

## Recommended Research Path

1. 建立可复现仓库、项目契约和实验协议。
2. 后续安装 LIBERO 主环境，并先复现官方 demo。
3. 后续读取 LIBERO demonstration trajectory，建立 trajectory window dataset。
4. 后续先跑 MLP/GRU baseline。
5. 后续加入 future latent prediction，形成最小 WAM adapter。
6. 后续把 GRU 替换为 SNN-LIF/PLIF adapter。
7. 后续进入 closed-loop evaluation，报告 success rate。
8. 后续加入 noise、delay、frame drop 和 horizon robustness。
9. 后续再考虑 surrogate warmup + EGGROLL-style ES post-training。

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

第一批可汇报结果应比较：

- BC-MLP
- BC-GRU
- WAM-GRU
- WAM-SNN-LIF
- WAM-SNN-PLIF 或 ALIF

必要指标：

- action MSE / L1
- future latent cosine error
- multi-step latent error curve
- closed-loop success rate
- completion steps
- spike rate / SynOps proxy
- inference latency
- robustness under noise, delay, and frame drop
