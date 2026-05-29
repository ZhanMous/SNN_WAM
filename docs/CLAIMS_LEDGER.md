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
| C-G4-WAM-GRU-DINOV2S-ABLATION-001 | preliminary | WAM-GRU with future latent loss achieves different offline metrics than WAM-GRU without future latent loss when using real frozen DINOv2 latents. | `R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-001`; `R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-002` | `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_offline.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_offline.csv` | offline ablation | `0` | **Preliminary evidence (seed 0 only, git dirty due to untracked output dirs).** Future variant val: action_mse=0.01910, future_latent_cosine_error=0.00113. No-future variant val: action_mse=0.01841, future_latent_cosine_error=1.00373. Allowed: "The ablation compares future vs no-future WAM-GRU using identical real frozen latents and same seed set." Forbidden: "Future latent loss improves closed-loop success." "Future latent loss improves real LIBERO performance." "The model is a validated WAM policy." "The method improves robustness." "The artifacts are reportable scientific evidence." "Smoke latents are real frozen visual encoder latents." |
| C-G5-EVALUATOR-VALIDITY-001 | supported | The closed-loop LIBERO evaluator is a valid test: expert replay achieves 90% success on tasks 1, 2, 3 under identical conditions. | `R-G5-DIAGNOSTIC-EVAL-001` | `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/eval_rollout.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/diagnostic_summary.md` | closed-loop diagnostic | `0` | Expert replay: 27/30=90% on tasks 1,2,3 (demo_0 actions). Zero and random baselines: 0/30 each. The evaluator produces task success when given correct actions. |
| C-G5-WAM-GRU-FAILURE-001 | supported | Both WAM-GRU variants (future and no-future) fail to solve any LIBERO spatial task in closed-loop evaluation, performing no better than zero-action or random-action baselines. | `R-G5-DIAGNOSTIC-EVAL-001` | `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/eval_rollout.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/eval_rollout.csv` | closed-loop diagnostic | `0` | Future: 0/30, no_future: 0/30, zero: 0/30, random: 0/30 on tasks 1,2,3. All failures are max_steps_reached (300-step timeout). Expert replay succeeds on same evaluator (27/30). The failure is conclusive on these tasks. |
| C-G5-WAM-GRU-OPEN-LOOP-001 | supported | Under teacher forcing on the configured val split, both WAM-GRU variants beat zero-action, train-random-action, train-mean-action, and last-action baselines in action MSE. | `R-G5-WAM-GRU-OPEN-LOOP-DIAGNOSTIC-001` | `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/open_loop_diagnostics/open_loop_metrics.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/open_loop_diagnostics/open_loop_metrics.csv`; `results/diagnostics/wam_gru_failure_taxonomy.csv` | teacher-forced open-loop diagnostic | `0` | Future model action_mse=0.019102; no_future action_mse=0.018413; last-action baseline=0.029960; zero=0.222224; random=0.420760; mean=0.200154. This does not imply closed-loop success or future-latent benefit. |
| C-G5-WAM-GRU-OVERFIT-001 | supported | The no-future WAM-GRU did not pass the predeclared near-zero single-demo teacher-forced overfit threshold, so the policy training pipeline is not validated for architecture claims. | `R-G5-WAM-GRU-SINGLE-DEMO-OVERFIT-001` | `results/diagnostics/single_demo_overfit/20260529_wam_gru_no_future_single_demo_overfit_lr0p001_seed0/summary.json`; `results/diagnostics/single_demo_overfit/20260529_wam_gru_no_future_single_demo_overfit_lr0p001_seed0/metrics.csv`; `results/diagnostics/single_demo_overfit/20260529_wam_gru_no_future_single_demo_overfit_lr0p001_seed0/action_trace_diagnostics.csv` | teacher-forced single-demo diagnostic | `0` | Best same-demo action_mse=0.000527 with threshold=0.000100. Closed-loop same initial condition was not run because the evaluator lacks an HDF5 demo-id to benchmark init-state-id mapping. Do not classify the closed-loop failure solely as covariate shift yet. |
| C-G5-OVERFIT-PIPELINE-001 | supported | The training pipeline can memorize a single LIBERO trajectory to near-zero MSE (9.2e-5) when using a direct per-window lookup table, confirming the data loader, loss computation, and optimizer are correct. | `R-G5-OVERFIT-DIAG-001` | `results/diagnostics/overfit_diag/20260529_072619_overfit_diag/summary.json`; `results/diagnostics/overfit_diag/20260529_072619_overfit_diag/lookup_table_metrics.csv` | single-demo pipeline diagnostic | `0` | WindowLookup with 96 parameters, lr=0.1, 62 epochs. Threshold=0.0001. Pipeline is not the bottleneck. |
| C-G5-OVERFIT-H1-FAIL-001 | rejected | The older H=1 WAM-GRU failure value (`3.5e-3`) should not be used as the current diagnosis after the repaired H=1 sweep and split-head diagnostic. | `R-G5-OVERFIT-DIAG-001`; `R-G5-OVERFIT-REPAIR-001` | `results/diagnostics/overfit_diag/20260529_072619_overfit_diag/summary.json`; `results/diagnostics/overfit_repair/20260529_h1_overfit_repair_seed0_v2/timestep_shift_train_sweep.csv`; `results/diagnostics/overfit_repair/20260529_h1_overfit_repair_seed0_v2/split_head_gripper_diagnostics.csv` | single-demo H=1 diagnostic | `0` | Superseded by the repaired diagnostic. The current result is more specific: raw WAM-GRU passes only at target_shift=-1, nominal shift 0 remains above threshold, and split-gripper WAM-GRU still fails. |
| C-G5-OVERFIT-REPAIR-001 | supported | The repaired H=1 diagnostic does not validate WAM-GRU for architecture claims: the only passing raw WAM-GRU shift is `-1`, which predicts `actions[t]` already present in history, and split-gripper WAM-GRU still fails the `1e-4` threshold. | `R-G5-OVERFIT-REPAIR-001` | `results/diagnostics/overfit_repair/20260529_h1_overfit_repair_seed0_v2/summary.json`; `results/diagnostics/overfit_repair/20260529_h1_overfit_repair_seed0_v2/timestep_shift_train_sweep.csv`; `results/diagnostics/overfit_repair/20260529_h1_overfit_repair_seed0_v2/split_head_gripper_diagnostics.csv`; `results/diagnostics/overfit_repair/20260529_h1_overfit_repair_seed0_v2/overfit_debug_curves.csv` | single-demo H=1 repair diagnostic | `0` | Raw sweep: shift -1 `6.63e-5` pass, shift 0 `1.03e-4` fail, +1 `1.46e-4`, +2 `1.68e-4`. Split WAM-GRU: `2.28e-4` fail with gripper MSE 0 and continuous MSE `2.66e-4`. Timestep MLP passes (`5.36e-8`); DINO-latent MLP fails (`2.82e-3`). No future-latent, rollout, robustness, or architecture benefit claim is allowed. |
| C-G5-OVERFIT-GRIPPER-001 | supported | The gripper dimension (index 6) dominates the single-demo overfit residual at 0.029 MSE, 36x the continuous-dims mean (8.2e-4), despite 99.7% sign accuracy. The error is magnitude-based, not direction-based. | `R-G5-OVERFIT-DIAG-001` | `results/diagnostics/overfit_diag/20260529_072619_overfit_diag/summary.json`; `results/diagnostics/overfit_diag/20260529_072619_overfit_diag/overfit_decomposition.csv`; `results/diagnostics/overfit_diag/20260529_072619_overfit_diag/gripper_diagnostics.csv` | single-demo decomposition | `0` | Gripper values are {-1, +1} (binary). The model predicts correct direction but wrong magnitude. Transition F1=0.89, close timing error=1.7 steps. |
| C-G5-CAUSAL-CONTRACT-001 | supported | The causal_next_action_v1 contract invariants pass on the single-demo H=1 dataset: max(action_history_index) < target_action_index, observation_index <= target_action_index, no future latent in input, and future targets are target-only. | `R-G5-CAUSAL-REPAIR-001` | `results/diagnostics/overfit_repair/20260529_081703_h1_overfit_repair_seed0/causal_contract_tests.json` | causal contract test | `0` | 96 samples, 0 failures. The data pipeline correctly separates inputs from targets. |
| C-G5-CAUSAL-H1-LADDER-001 | supported | Under the causal next-action contract (shift=0), only the timestep-embedding MLP passes H=1 single-demo overfit (7.9e-7). All causal baselines using real inputs fail: proprio-only (1.1e-2), action-history GRU (7.9e-4), DINO CLS (4.2e-2), DINO+proprio (2.4e-2), DINO+proprio+history (1.4e-3). | `R-G5-CAUSAL-REPAIR-001` | `results/diagnostics/overfit_repair/20260529_081703_h1_overfit_repair_seed0/causal_h1_baseline_ladder.csv` | causal H=1 baseline ladder | `0` | Threshold=1e-4. Timestep embedding memorizes by timestep ID (not causal). No baseline using real sensor inputs passes. The model architecture cannot fit single-step causal action prediction under current representation. |
| C-G5-LATENT-SANITY-001 | supported | DINOv2 ViT-S/14 CLS latents for the single-demo trajectory are valid: all 103 latents are unique, mean variance 0.072, adjacent cosine distance 0.00064, PCA top-5 concentration 95%. | `R-G5-CAUSAL-REPAIR-001` | `results/diagnostics/overfit_repair/20260529_081703_h1_overfit_repair_seed0/latent_sanity.json`; `results/diagnostics/overfit_repair/20260529_081703_h1_overfit_repair_seed0/latent_sanity_report.md` | latent sanity diagnostic | `0` | Latents change slowly (adjacent cosine 0.00064) but are unique and well-conditioned. |

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
- `WAM-GRU is effective at LIBERO closed-loop control.` (C-G5-WAM-GRU-FAILURE-001: 0/30 on tasks 1,2,3)
- `Future-latent prediction improves WAM-GRU rollout success.` (C-G5-WAM-GRU-FAILURE-001: both variants score 0/30)
- `WAM-GRU performs better than random or zero actions in closed-loop rollout.` (C-G5-WAM-GRU-FAILURE-001: all baselines score 0/30)
- `WAM-GRU learns valid next-action patterns.` (C-G5-OVERFIT-REPAIR-001: H=1 passes only for shift -1, which targets an action already present in history)
- `The WAM-GRU architecture can fit valid single-step next-action prediction.` (C-G5-OVERFIT-REPAIR-001: split-gripper WAM-GRU fails at 2.28e-4 vs threshold 1e-4)
- `A nonzero target shift validates the policy alignment.` (C-G5-OVERFIT-REPAIR-001: best shift -1 is non-causal for next-action prediction)
- `DINOv2 current latent alone can fit the single-demo H=1 action trace.` (C-G5-OVERFIT-REPAIR-001: DINO-latent MLP fails at 2.82e-3)
- `Gripper prediction is accurate.` (C-G5-OVERFIT-GRIPPER-001: 0.029 MSE despite 99.7% sign accuracy)
- `Any causal input representation passes H=1 single-demo overfit.` (C-G5-CAUSAL-H1-LADDER-001: all real-input baselines fail)
- `DINO CLS latents are sufficient for single-step action prediction.` (C-G5-CAUSAL-H1-LADDER-001: DINO CLS only fails at 4.2e-2)
- `Action history alone is sufficient for single-step action prediction.` (C-G5-CAUSAL-H1-LADDER-001: action history GRU fails at 7.9e-4)
- `Proprioceptive state alone is sufficient for single-step action prediction.` (C-G5-CAUSAL-H1-LADDER-001: proprio only fails at 1.1e-2)
- `The WAM-GRU architecture learns valid causal next-action patterns.` (C-G5-CAUSAL-H1-LADDER-001: no causal baseline passes H=1 overfit)
