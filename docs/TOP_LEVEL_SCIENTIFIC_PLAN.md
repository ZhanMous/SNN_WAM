# Top-Level Scientific Plan: DINO-WM -> SNN-WAM

## 1. Direction

The previous direct behavior cloning route is frozen. The old path

```text
DINO CLS / state / action history -> low-level action regression
```

is no longer the main scientific route because it is not a stable base for testing WAM, future-latent prediction, or SNN dynamics.

The new route is:

```text
DINOv2 spatial patch features + action sequence
  -> action-conditioned latent dynamics model
  -> future patch features
  -> action sequence optimization / planning
```

The SNN must model action-conditioned latent dynamics. It must not be introduced as a direct low-level action regressor. ES / EGGROLL-style training is optional and only becomes meaningful after fair surrogate-gradient SNN, GRU, and ANN baselines exist.

## 2. Stage Plan

| Stage | Goal | Implementation Scope | Acceptance Criteria | Decision |
|---|---|---|---|---|
| S0: Route Bootstrap | Freeze old BC route and prepare DINO-WM route | Docs, directory skeleton, artifact contracts, smoke checks | README and project contract identify DINO-WM -> SNN world model route; old route is clearly legacy | Continue if docs/tests pass |
| S1: DINOv2 Patch Features | Extract spatial patch features, not CLS-only latents | Patch extractor, manifest, shape tests | Feature file has explicit `[N, P, D]` or `[N, H_p, W_p, D]`; manifest records model id, revision, preprocessing, image source | Retry if shapes/metadata are ambiguous |
| S2: Patch Transition Dataset | Build no-leakage latent dynamics windows | Dataset returns context patch latents, action sequence, target future patch latents | Tests prove inputs contain no future target latents/actions; all tensors document `[B, T, P, D]`, `[B, T, A]`, `[B, H, P, D]` | Stop if leakage cannot be ruled out |
| S3: DINO-WM ANN Baseline | Reproduce minimal DINO-WM-style latent predictor | Small Transformer/MLP predictor, training loop, metrics | Beats copy-last-latent and random predictor on one-step and multi-step patch latent metrics | Retry architecture/training if copy-last wins |
| S4: Planning Sanity | Test latent-space action sequence optimization | Offline candidate action optimizer over learned world model | Optimized action sequences reduce predicted distance to target latent more than random/action-replay baselines | Stop before closed-loop if planning objective does not improve |
| S5: SNN Dynamics Interface | Replace predictor with SNN world model interface | LIF/PLIF/ALIF predictor forward pass and surrogate-gradient baseline | Shape tests pass; spike stats logged; reset policy tested; surrogate SNN beats random SNN/copy-last on latent prediction | Retry SNN state/reset if unstable |
| S6: Direct SNN Training | Test ES / EGGROLL-style direct SNN world-model training | Low-rank ES over SNN parameters on offline latent fitness | Toy ES improves; offline ES improves future patch latent fitness over random SNN under fixed seed | Stop ES if estimator/sign/noise sanity fails |
| S7: Planning with SNN | Use SNN dynamics model for action optimization | MPC-style latent planning using SNN predictor | Planning objective improves and predicted-vs-actual next-latent agreement is acceptable | Do not claim closed-loop success yet |
| S8: Closed-Loop Planning | Evaluate fixed LIBERO tasks after offline gates | Fixed task/init-state rollout, episode CSV, failure evidence | Closed-loop only runs after S3/S4 and S5/S7 gates pass; success/failure fully logged | Continue only with matched baselines |
| S9: Robustness and Claims | Compare robustness and SNN tradeoffs | Noise/delay/frame drop, spike-rate proxy, latency | SNN claim requires fair ANN/GRU/Transformer baselines and identical budgets | Pivot if SNN adds complexity without measurable benefit |

## 3. Stage Acceptance Details

### S1: Patch Feature Acceptance

Required checks:

- Patch feature shape is recorded exactly.
- Patch order is deterministic and tied to frame index.
- `manifest.json` records:
  - source HDF5 path
  - source observation key
  - frame count
  - DINOv2 model id
  - revision / checkpoint id
  - preprocessing transform
  - patch grid shape
  - latent dtype
  - extraction command
  - git commit

Minimum artifacts:

```text
latents/dinov2_patch/<suite>/<task_or_file>_patch.hdf5
latents/dinov2_patch/<suite>/<task_or_file>_patch_manifest.json
results/smoke/dinowm_patch_extract/<run_id>/summary.json
```

### S2: Dataset Acceptance

Required output contract:

```text
z_context: [B, T_context, P, D]
actions:   [B, T_action, A]
z_target:  [B, H_future, P, D]
mask:      optional [B, H_future]
metadata: trajectory_id, start_index, target_indices
```

Required tests:

- dataset length and shape test
- no-future-leakage test
- episode-boundary exclusion test
- split isolation test
- action/latent index alignment test

### S3: ANN Baseline Acceptance

Required baselines:

- copy-last-latent
- temporal mean latent
- random predictor
- small MLP/Transformer DINO-WM-style predictor
- optional GRU latent dynamics predictor

Primary metrics:

- `patch_latent_mse`
- `patch_latent_cosine_error`
- `multi_step_patch_drift`
- `nearest_neighbor_future_frame_accuracy`
- `per_horizon_patch_error`

Acceptance:

- Learned predictor must beat copy-last-latent on validation for at least one-step and multi-step metrics.
- Metrics must be separated by horizon.
- No planning or SNN claim may be made from S3 alone.

### S4: Planning Sanity Acceptance

Planning is still offline here. It may use action sequences from:

- dataset action replay
- random sampled actions
- optimized action sequence through the learned latent model

Primary metrics:

- initial target latent distance
- optimized target latent distance
- random-action target latent distance
- action smoothness
- predicted-vs-actual next-latent error when ground truth is available

Acceptance:

- Optimized actions reduce predicted target latent distance more than random actions.
- If replay actions are available, compare optimized actions against replay actions but do not claim control success.

### S5: SNN Baseline Acceptance

Required properties:

- SNN predicts future patch latents, not low-level actions.
- State reset behavior is explicit.
- Spike rate is logged as a proxy only.
- Surrogate-gradient SNN is a baseline, not the direct-training target.

Required tests:

- forward shape
- finite loss
- reset behavior
- batch independence
- spike rate in `[0, 1]`
- no hidden `[T, B, ...]` transpose without one explicit conversion point

Acceptance:

- Surrogate-gradient SNN must beat random SNN and copy-last-latent before ES experiments are considered useful.
- No SNN superiority claim is allowed without fair ANN/GRU/Transformer baselines.

### S6: Direct ES / EGGROLL-Style Acceptance

Required before offline SNN ES:

- toy quadratic ES improves under fixed seed
- sign convention test: higher fitness means better model
- perturbation seed is recorded
- parameter subset / low-rank parameterization is documented

Offline fitness:

```text
fitness = - patch_latent_error
          - alpha_drift * multi_step_drift
          - alpha_spike * spike_rate_proxy
          - alpha_smooth * prediction_smoothness_optional
```

Required artifacts:

```text
es_config.yaml
initial_checkpoint.txt or initial_seed.txt
generation_metrics.csv
population_metrics.csv
best_es.pt
fitness_components.csv
command.txt
notes.md
```

Acceptance:

- Direct ES-trained SNN beats random/untrained SNN and copy-last-latent on validation latent prediction.
- Compare against surrogate-gradient SNN and GRU/ANN baselines, but do not require ES to beat them before documenting exploratory results.
- Do not call this a full EGGROLL reproduction.

### S8: Closed-Loop Acceptance

Closed-loop is blocked until:

- S3 ANN baseline passes.
- S4 planning sanity passes.
- S5 or S7 model-specific planning gate passes.
- `scripts/check_result_artifacts.py` passes.

Closed-loop artifacts:

```text
eval_rollout.csv
eval_summary.json
failure_taxonomy.csv
failure_videos/ or failure_frames/
compatibility_report.json
command.txt
environment.txt
git_commit.txt
notes.md
```

Closed-loop claims must name:

- task list
- initial states
- episode count
- seed list
- max steps
- success definition
- compared baselines

## 4. Artifact Structure

Use separate roots for latent extraction, smoke checks, diagnostic experiments, and reportable runs.

```text
latents/
  dinov2_patch/
    <suite>/
      <task_or_file>_patch.hdf5
      <task_or_file>_patch_manifest.json

configs/
  dinowm/
    patch_extract_<suite>.yaml
    patch_transition_dataset_smoke.yaml
    ann_patch_predictor_smoke.yaml
    snn_patch_predictor_smoke.yaml
    es_snn_patch_predictor_smoke.yaml

results/
  smoke/
    dinowm_patch_extract/<run_id>/
    dinowm_patch_dataset/<run_id>/
    dinowm_ann_baseline/<run_id>/
  diagnostics/
    dinowm_planning_sanity/<run_id>/
    snn_patch_predictor/<run_id>/
    es_direct_snn/<run_id>/
  runs/
    <YYYYMMDD_HHMM_suite_model_goal_seed>/
  tables/
  figures/

docs/
  TOP_LEVEL_SCIENTIFIC_PLAN.md
  DINOWM_SNN_WORLDMODEL_PLAN.md
  CLAIMS_LEDGER_DINOWM_SNN.md
  RESULT_ARTIFACTS.md
```

Every train/eval run directory must include:

```text
config.yaml
command.txt
command.sh if applicable
git_commit.txt
environment.txt or environment.json
seeds.txt
split.json
metrics.csv
summary.json or summary.md
notes.md
```

Model-training runs additionally include:

```text
checkpoint.pt
best.pt
normalization_stats.json if normalization is used
```

Planning or closed-loop runs additionally include:

```text
planning_metrics.csv
candidate_action_metrics.csv
eval_rollout.csv when environment rollout is used
failure_taxonomy.csv
failure_videos/ or failure_frames/
```

ES runs additionally include:

```text
es_config.yaml
generation_metrics.csv
population_metrics.csv
fitness_components.csv
best_es.pt
initial_checkpoint.txt or initial_seed.txt
```

## 5. Claim Ledger Schema

Create `docs/CLAIMS_LEDGER_DINOWM_SNN.md` with this table schema:

| Field | Description |
|---|---|
| Claim ID | Stable ID, e.g. `DWM-C-S3-001` |
| Status | `hypothesis`, `observation`, `preliminary`, `supported`, `rejected`, `superseded` |
| Claim | Exact allowed wording |
| Stage | S1-S9 |
| Artifact IDs | IDs registered in `docs/RESULT_ARTIFACTS.md` |
| Evidence Files | Concrete result paths, not directories |
| Evaluation Type | patch extraction, offline latent, planning sanity, ES, closed-loop, robustness |
| Baselines Compared | copy-last, ANN, GRU, SNN, random, etc. |
| Dataset/Split | suite, tasks, train/val/test, held-out split |
| Seeds | all seeds used |
| Main Metrics | key numeric values |
| Limitations | what the result does not show |
| Allowed Wording | sentence that may be used in reports |
| Forbidden Wording | overclaims to avoid |
| Decision | continue, retry, stop, pivot |

Status rules:

- `hypothesis`: no result files yet.
- `observation`: one diagnostic run or smoke evidence.
- `preliminary`: real data, reproducible artifacts, limited seeds/scope.
- `supported`: matched baselines, required seeds/splits, artifact registry complete.
- `rejected`: evidence contradicts claim.
- `superseded`: replaced by stronger or corrected artifact.

Initial forbidden claims:

- DINO-WM is reproduced in this repository.
- Future-latent prediction improves control.
- SNN is better than GRU/Transformer.
- Direct ES trains SNN world models successfully.
- Offline latent metrics imply closed-loop success.
- Spike rate is energy efficiency.
- This is a full VLA/foundation model.

## 6. Stop, Retry, and Pivot Rules

### Global Stop Rules

Stop the current stage if:

- tensor shape contracts are ambiguous or undocumented
- future leakage is detected
- result artifacts cannot be reproduced
- baseline comparison is unfair
- metrics do not map to the claim being made
- closed-loop is requested before offline gates pass

### Retry Rules

Retry within the same stage if:

- implementation bugs are isolated and testable
- a baseline fails because of obvious undertraining or a documented config error
- artifact files are missing but the run can be reproduced
- metric code is incomplete but the scientific question remains valid

### Pivot Rules

Pivot to a different formulation if:

- DINO patch features do not beat copy-last-latent under any simple predictor
- action-conditioned prediction does not improve over action-agnostic prediction
- planning objective improves in model space but not in real next-latent agreement
- SNN adds complexity without beating random/copy-last under matched budgets
- ES fails toy/sign/noise sanity checks

### Closed-Loop Blocker

Do not proceed to closed-loop planning unless:

1. patch transition dataset passes leakage tests
2. ANN/GRU baseline beats copy-last-latent
3. planning sanity improves predicted target latent distance
4. model-specific readiness gate passes
5. artifacts and claim ledger are complete

## 7. Decision Template

Use this after every stage:

```text
Stage:
Artifact IDs:
Required criteria:
Passed:
Failed:
Main metrics:
Baselines:
Claims allowed:
Claims forbidden:
Decision: continue / retry / stop / pivot
Reason:
Next Claude Code prompt:
```

## 8. First Claude Code Prompt: Repository Bootstrap

```text
You are Claude Code working in /home/zhan_shaoji/code/SNN_WAM on branch dinowm_snn_worldmodel.

Goal:
Bootstrap the DINO-WM -> SNN-WAM repository route without implementing training yet.

Project direction:
- The previous direct behavior cloning route is legacy only.
- The new route is DINOv2 spatial patch features -> action-conditioned latent dynamics -> future patch features -> planning.
- SNN will later replace the latent dynamics predictor, not directly regress actions.
- ES / EGGROLL-style training is not part of this bootstrap task.

Files to inspect first:
- README.md
- docs/DINOWM_SNN_WORLDMODEL_PLAN.md
- docs/TOP_LEVEL_SCIENTIFIC_PLAN.md
- docs/PROJECT_CONTRACT.md
- docs/RESULT_ARTIFACTS.md
- docs/CLAIMS_LEDGER.md
- scripts/extract_dinov2_latents.py
- src/data/trajectory_window.py
- tests/test_repository_contract.py

Implementation scope:
1. Create a DINO-WM route skeleton:
   - configs/dinowm/
   - src/data/patch_latent_window.py
   - src/models/patch_predictor.py
   - scripts/extract_dinov2_patch_latents.py
   - tests/test_patch_latent_window.py
   - tests/test_patch_predictor.py
2. Add placeholder configs:
   - configs/dinowm/patch_extract_libero_spatial_smoke.yaml
   - configs/dinowm/patch_transition_dataset_smoke.yaml
   - configs/dinowm/ann_patch_predictor_smoke.yaml
3. Implement only minimal smokeable contracts:
   - patch-latent dataset class can operate on synthetic arrays
   - returns z_context [B,T,P,D], actions [B,T,A], z_target [B,H,P,D]
   - no-future-leakage synthetic test
   - simple patch predictor forward shape test
   - no real DINOv2 network call required unless already locally available
4. Update README or docs only if needed to point to the new skeleton.

Acceptance criteria:
- pytest tests/test_patch_latent_window.py tests/test_patch_predictor.py tests/test_repository_contract.py -q passes.
- No future action or future latent enters model inputs.
- All tensor shapes are documented in docstrings or comments.
- No closed-loop, SNN, ES, or performance claim is added.

Forbidden:
- Do not delete legacy BC diagnostic artifacts.
- Do not claim DINO-WM reproduction.
- Do not implement full training loops.
- Do not add heavy dependencies unless the project contract is updated first.
- Do not run long GPU jobs.

Report back:
- Files changed.
- Commands run.
- Tests passed/failed.
- Remaining risks.
```
