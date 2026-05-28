# Claims Ledger

所有科学主张必须登记在这里，并且必须指向 `results/` 下的结果文件。被引用的结果必须先登记在 `docs/RESULT_ARTIFACTS.md`。没有结果文件的内容只能写成 hypothesis 或 observation。

## Smoke vs Reportable Claims

**Smoke claims** (engineering observations only):
- Based on smoke artifacts under `results/smoke/`
- Use mock data, stub encoders, or smoke latents
- May have dirty git state
- Status: `observation` (not `supported`)
- Must NOT be cited as scientific evidence

**Reportable claims** (scientific evidence):
- Based on reportable artifacts under `results/runs/`
- Use real LIBERO data and frozen encoders
- Require clean git state (`dirty=False`)
- Status: `supported` (with evidence files)
- Must pass `scripts/preflight_reportable.py` before execution
- Can be cited in papers and reports

| Claim ID | Status | Claim | Artifact IDs | Evidence Files | Evaluation Type | Seeds | Reviewer Notes |
|---|---|---|---|---|---|---|---|
| C-000 | template | 示例：SNN-LIF 在 frame drop 下比 GRU 退化更慢。 | `R-000` | `results/runs/<run_id>/metrics.csv`; `results/tables/<table>.csv` | robustness | `0,1,2` | 示例行，不是当前结论。 |
| C-G3A-001 | observation | The real-data action-only training pipeline runs end-to-end on a small LIBERO subset. | `R-G3A-001` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/metrics.csv`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/summary.json`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/checkpoint.pt` | offline engineering smoke | `0` | Allowed wording only. This is not WAM, VLA, SNN, GRU, closed-loop, generalization, or benchmark evidence. |
| C-G4-WAM-GRU-SMOKE-001 | observation | The future-latent ablation code path runs in deterministic smoke mode and writes train/eval metrics for both `lambda_future=1.0` and `lambda_future=0.0`. | `R-G4-WAM-GRU-FUTURE-SMOKE-001`; `R-G4-WAM-GRU-NO-FUTURE-SMOKE-001` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/metrics.csv`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/eval_offline.csv`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/metrics.csv`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/eval_offline.csv` | offline engineering smoke | `0` | Allowed wording only. This is mock-data smoke evidence for plumbing and metric logging, not real-data WAM, closed-loop success, robustness, or improvement evidence. Smoke latents are not real frozen visual encoder latents. Recorded commit has `dirty=True`. |

## Status Values

- `hypothesis`: 尚未有结果文件支撑。
- `observation`: 有单次运行或人工观察，但不足以写成结论。
- `supported`: 有结果文件和复评估路径支撑。
- `rejected`: 结果文件不支持该主张。

## Rules

- 每条 claim 必须有 `Evidence Files`。
- 每条 claim 必须有 `Artifact IDs`，并且对应 `docs/RESULT_ARTIFACTS.md` 中的登记项。
- `Evidence Files` 必须指向 `results/` 下的具体文件，而不是泛泛目录。
- 汇总表必须能追溯到 per-run 原始结果。
- 不允许把 `spike rate` 写成真实硬件能耗，除非有对应硬件测量结果文件。

## Forbidden Current Claims

- `SNN improves performance.`
- `WAM improves future prediction.`
- `Future latent loss improves closed-loop success.`
- `Future latent loss improves real LIBERO performance.`
- `The model is a validated WAM policy.`
- `The method improves robustness.`
- `Vision-language policy works.`
- `Closed-loop success is improved.`
- `The method generalizes on LIBERO.`
- `The smoke artifacts are reportable scientific evidence.`
- `Smoke latents are real frozen visual encoder latents.`
