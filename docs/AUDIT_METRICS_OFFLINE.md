# Offline Metric Correctness Audit

**Audit date**: 2026-05-28 (updated)
**Auditor**: automated metric-correctness-audit + scientific-claim-audit
**Status**: **PASS**
**Tests**: 51 passing (`tests/test_metrics.py`)

---

## 1. Metric Definitions

### 1.1 action_mse

| Property | Value |
|---|---|
| File | `src/train/metrics.py:9` |
| Input shape | `pred_actions: [B, H, A]`, `target_actions: [B, H, A]` |
| Mask shape | `[B, H]` or `[B, H, 1]` (optional) |
| Reduction | Mean over batch, horizon, and action dimensions (global scalar) |
| Direction | Lower is better |
| Unit | Same as input tensors (caller's responsibility) |
| Scope | Offline model-level metric, not closed-loop success |
| Dtype | Floating point required; raises `TypeError` otherwise |

**Verdict**: Correct. Shape, reduction, direction, and mask contracts are validated at runtime.

### 1.2 action_mse_per_horizon

| Property | Value |
|---|---|
| File | `src/train/metrics.py:146` |
| Input shape | `pred_actions: [B, H, A]`, `target_actions: [B, H, A]` |
| Mask shape | `[B, H]` or `[B, H, 1]` (optional) |
| Reduction | Mean over A, then mean over B; returns `[H]` |
| Direction | Lower is better |
| Tests | 6 tests |

**Verdict**: Correct. Returns per-horizon MSE as `[H]` tensor.

### 1.3 action_mse_per_dimension

| Property | Value |
|---|---|
| File | `src/train/metrics.py:192` |
| Input shape | `pred_actions: [B, H, A]`, `target_actions: [B, H, A]` |
| Mask shape | `[B, H]` or `[B, H, 1]` (optional) |
| Reduction | Mean over B and H; returns `[A]` |
| Direction | Lower is better |
| Tests | 6 tests |

**Verdict**: Correct. Returns per-action-dimension MSE as `[A]` tensor.

### 1.4 future_latent_cosine_error

| Property | Value |
|---|---|
| File | `src/train/metrics.py:64` |
| Input shape | `pred_latents: [B, H, D]`, `target_latents: [B, H, D]` |
| Mask shape | `[B, H]` or `[B, H, 1]` (optional) |
| Reduction modes | `"mean"` -> scalar, `"per_horizon"` -> `[H]`, `"none"` -> `[B, H]` |
| Direction | Lower is better (0 = perfect, 1 = orthogonal, 2 = opposite) |
| Scope | Offline model-level metric, not closed-loop success |
| Cosine dim | Over latent dimension `D` only |

**Verdict**: Correct. Three reduction modes are tested. Mask validation prevents division by zero.

### 1.5 future_latent_mse

| Property | Value |
|---|---|
| File | `src/train/metrics.py:238` |
| Input shape | `pred_latents: [B, H, D]`, `target_latents: [B, H, D]` |
| Mask shape | `[B, H]` or `[B, H, 1]` (optional) |
| Reduction modes | `"mean"` -> scalar, `"per_horizon"` -> `[H]`, `"none"` -> `[B, H]` |
| Direction | Lower is better |
| Reduction order | Mean over D first, then H, then B |
| Tests | 12 tests |

**Verdict**: Correct. Three reduction modes tested. Consistent with `future_latent_cosine_error` shape contract.

### 1.6 spike_loss (placeholder)

| Property | Value |
|---|---|
| File | `src/train/train_offline.py:1056` |
| Value | Always `0.0` |
| Status | Placeholder; `lambda_spike != 0.0` raises `ValueError` |
| Scope | Reserved for future SNN spike-rate loss |

**Verdict**: Correctly documented as placeholder. No spike rate computation exists in the codebase.

### 1.7 Manual action_mse (denormalized)

| Property | Value |
|---|---|
| File | `src/train/train_offline.py:1001-1008, 1035` |
| Computation | `squared_error_sum / element_count` on denormalized tensors |
| Direction | Lower is better |
| Unit | Always raw action units (denormalized if action transform exists) |
| Purpose | Reported as `action_mse` in CSV; separate from `action_loss` (which may be normalized) |

**Verdict**: Correct. The distinction between `action_loss` (training loss, possibly normalized) and `action_mse` (always raw units) is intentional and documented in the code.

---

## 2. Call Site Audit

### 2.1 Training loop (`src/train/train_offline.py`)

| Call Site | Line | Function | Reduction | Notes |
|---|---|---|---|---|
| Training loss | 1089 | `action_mse(pred, target)` | scalar | Used as `action_loss` in `total_loss` |
| Training loss | 1094 | `future_latent_cosine_error(pred, target)` | scalar (mean) | Used as `future_loss` in `total_loss` |
| Diagnostic | 1127 | `action_mse_per_horizon(pred, target)` | `[H]` | Per-horizon action MSE |
| Diagnostic | 1137 | `action_mse_per_dimension(pred, target)` | `[A]` | Per-dimension action MSE |
| Diagnostic | 1146 | `future_latent_cosine_error(reduction="none")` | `[B, H]` | Manually aggregated |
| Diagnostic | 1163 | `future_latent_mse(reduction="none")` | `[B, H]` | Manually aggregated |
| Raw MSE | 1184 | manual `squared_error_sum / element_count` | scalar | Denormalized |

**Verdict**: All reductions are intentional. Per-horizon aggregation uses batch-weighted sums. Train/val metrics are computed in separate calls and written to separate CSV rows.

### 2.2 Test call sites

| File | Lines | Notes |
|---|---|---|
| `tests/test_metrics.py` | 27 tests | Comprehensive synthetic tests for both metrics |
| `tests/test_temporal_mlp.py` | 42, 50, 76, 79, 82 | `action_mse` used for overfit validation |
| `tests/test_temporal_gru.py` | 116, 119, 122 | `action_mse` used for overfit validation |
| `tests/test_train_offline.py` | 79-107 | End-to-end dry-run metric checks |

**Verdict**: No metric is used with incorrect shapes or reduction modes.

---

## 3. Dimension Reduction Audit

| Metric | Batch | Horizon | Action/Latent | Mask |
|---|---|---|---|---|
| `action_mse` | mean | mean | mean over A | weighted |
| `action_mse_per_horizon` | mean | **kept `[H]`** | mean over A | weighted |
| `action_mse_per_dimension` | mean | mean | **kept `[A]`** | weighted |
| `future_latent_cosine_error` (mean) | mean | mean | cosine over D | weighted |
| `future_latent_cosine_error` (per_horizon) | mean | **kept `[H]`** | cosine over D | weighted |
| `future_latent_cosine_error` (none) | **kept** | **kept** | cosine over D | weighted |
| `future_latent_mse` (mean) | mean | mean | mean over D | weighted |
| `future_latent_mse` (per_horizon) | mean | **kept `[H]`** | mean over D | weighted |
| `future_latent_mse` (none) | **kept** | **kept** | mean over D | weighted |
| Manual action_mse | sum/total | sum/total | sum/total | N/A |

**Verdict**: All reductions are intentional and documented. No silent averaging across unmentioned dimensions.

---

## 4. Direction Verification

| Metric | Direction | Verified by |
|---|---|---|
| `action_mse` | Lower is better | `test_action_mse_direction_lower_is_better` |
| `action_mse_per_horizon` | Lower is better | `test_action_mse_per_horizon_direction_lower_is_better` |
| `action_mse_per_dimension` | Lower is better | `test_action_mse_per_dimension_direction_lower_is_better` |
| `future_latent_cosine_error` | Lower is better | `test_future_latent_cosine_error_direction_lower_is_better` |
| `future_latent_mse` | Lower is better | `test_future_latent_mse_direction_lower_is_better` |
| `total_loss` | Lower is better | Composite of lower-is-better components |
| `spike_loss` | N/A (always 0.0) | Placeholder |

**Verdict**: All metrics are lower-is-better. `lower_is_better` field in CSV is always `"true"`.

---

## 5. Test Coverage

### 5.1 Synthetic metric tests (`tests/test_metrics.py`) — 51 tests

| Test | Metric | Case |
|---|---|---|
| `test_action_mse_perfect_prediction_is_zero` | action_mse | Perfect prediction -> 0 |
| `test_action_mse_known_value` | action_mse | Known MSE value |
| `test_action_mse_masked_padding_does_not_affect_result` | action_mse | Mask excludes padded horizon |
| `test_action_mse_mask_3d_shape` | action_mse | `[B, H, 1]` mask accepted |
| `test_action_mse_batch_reduction` | action_mse | Multi-batch mean |
| `test_action_mse_rejects_2d_shape` | action_mse | Shape validation |
| `test_action_mse_rejects_mismatched_shapes` | action_mse | Shape mismatch |
| `test_action_mse_rejects_non_floating_point` | action_mse | Dtype validation |
| `test_action_mse_rejects_empty_mask` | action_mse | All-zero mask |
| `test_action_mse_rejects_wrong_mask_shape` | action_mse | Mask shape mismatch |
| `test_action_mse_direction_lower_is_better` | action_mse | Direction check |
| `test_future_latent_cosine_error_perfect_prediction_is_zero` | cosine_error | Perfect -> 0 |
| `test_future_latent_cosine_error_known_vector_cases` | cosine_error | Orthogonal=1, opposite=2 |
| `test_future_latent_cosine_error_orthogonal_vectors_give_one` | cosine_error | 3D orthogonal |
| `test_future_latent_cosine_error_opposite_vectors_give_two` | cosine_error | 3D opposite |
| `test_future_latent_cosine_error_masked_padding_does_not_affect_result` | cosine_error | Mask excludes padded |
| `test_future_latent_cosine_error_reduction_none_returns_b_h` | cosine_error | Shape `[B, H]` |
| `test_future_latent_cosine_error_reduction_per_horizon_returns_h` | cosine_error | Shape `[H]` |
| `test_future_latent_cosine_error_batch_reduction` | cosine_error | Multi-batch mean |
| `test_future_latent_cosine_error_mask_with_reduction_none` | cosine_error | Mask + none |
| `test_future_latent_cosine_error_mask_with_reduction_per_horizon` | cosine_error | Mask + per_horizon |
| `test_future_latent_cosine_error_per_horizon_fully_masked_raises` | cosine_error | Fully masked horizon |
| `test_future_latent_cosine_error_rejects_2d_shape` | cosine_error | Shape validation |
| `test_future_latent_cosine_error_rejects_mismatched_shapes` | cosine_error | Shape mismatch |
| `test_future_latent_cosine_error_rejects_non_floating_point` | cosine_error | Dtype validation |
| `test_future_latent_cosine_error_rejects_invalid_reduction` | cosine_error | Invalid reduction mode |
| `test_future_latent_cosine_error_direction_lower_is_better` | cosine_error | Direction check |
| `test_action_mse_per_horizon_perfect_prediction_is_zero` | per_horizon | Perfect -> `[0, 0]` |
| `test_action_mse_per_horizon_known_value` | per_horizon | Known per-horizon values |
| `test_action_mse_per_horizon_masked_padding` | per_horizon | Mask with multi-batch |
| `test_action_mse_per_horizon_batch_averaging` | per_horizon | Mean over B per horizon |
| `test_action_mse_per_horizon_direction_lower_is_better` | per_horizon | Direction check |
| `test_action_mse_per_horizon_rejects_2d` | per_horizon | Shape validation |
| `test_action_mse_per_dimension_perfect_prediction_is_zero` | per_dim | Perfect -> `[0, 0, 0]` |
| `test_action_mse_per_dimension_known_value` | per_dim | Known per-dim values |
| `test_action_mse_per_dimension_masked` | per_dim | Mask excludes padded |
| `test_action_mse_per_dimension_batch_aggregation` | per_dim | Mean over B,H per dim |
| `test_action_mse_per_dimension_direction_lower_is_better` | per_dim | Direction check |
| `test_action_mse_per_dimension_rejects_2d` | per_dim | Shape validation |
| `test_future_latent_mse_perfect_prediction_is_zero` | latent_mse | Perfect -> 0 |
| `test_future_latent_mse_known_value` | latent_mse | Known MSE value |
| `test_future_latent_mse_reduction_none_returns_b_h` | latent_mse | Shape `[B, H]` |
| `test_future_latent_mse_reduction_per_horizon_returns_h` | latent_mse | Shape `[H]` |
| `test_future_latent_mse_masked_padding_does_not_affect_result` | latent_mse | Mask excludes padded |
| `test_future_latent_mse_mask_with_reduction_per_horizon` | latent_mse | Mask + per_horizon |
| `test_future_latent_mse_batch_reduction` | latent_mse | Multi-batch mean |
| `test_future_latent_mse_rejects_2d` | latent_mse | Shape validation |
| `test_future_latent_mse_rejects_mismatched_shapes` | latent_mse | Shape mismatch |
| `test_future_latent_mse_rejects_non_floating_point` | latent_mse | Dtype validation |
| `test_future_latent_mse_rejects_invalid_reduction` | latent_mse | Invalid reduction mode |
| `test_future_latent_mse_direction_lower_is_better` | latent_mse | Direction check |

---

## 6. Scientific Claim Audit

### 6.1 Claims Ledger Review

| Claim ID | Status | Assessment |
|---|---|---|
| C-000 | template | Example only, not cited |
| C-G3A-001 | observation | Correctly marked as "not WAM, VLA, SNN, GRU, closed-loop, generalization, or benchmark evidence" |
| C-G4-WAM-GRU-SMOKE-001 | observation | Correctly marked as "mock-data smoke evidence" |
| C-G4-WAM-GRU-NO-FUTURE-SMOKE-001 | observation | Correctly marked as "mock-data smoke evidence" |

**Verdict**: No overclaiming. All claims are observations with appropriate caveats.

### 6.2 Forbidden Claims Check

The following claims are correctly listed as forbidden in `docs/CLAIMS_LEDGER.md`:
- "SNN improves performance."
- "WAM improves future prediction."
- "Future latent loss improves closed-loop success."
- "Vision-language policy works."
- "Closed-loop success is improved."
- "The method generalizes on LIBERO."

**Verdict**: No forbidden claims found in any doc, README, or code comment.

### 6.3 Metric-Specific Claims

| Statement | Location | Status |
|---|---|---|
| "action error" as evidence type | README.md:12 | Motivation only, not a claim |
| "future latent error" as evidence type | README.md:12 | Motivation only, not a claim |
| "spike rate" as evidence type | README.md:12 | Motivation only, not a claim |
| `spike_loss` in metrics.csv | train_offline.py:1056 | Always 0.0, placeholder documented |
| `lower_is_better` in metrics.csv | train_offline.py:1095 | Correct for all metrics |

**Verdict**: No metric is misrepresented.

---

## 7. Risks

### RISK-1: `spike_loss` placeholder in CSV (Low)

`spike_loss` is always `0.0` in metrics.csv. A reader unfamiliar with the codebase might interpret this as "no spike loss" rather than "not implemented." The field is documented as a placeholder in the code and in this audit.

**Mitigation**: Already documented. No action needed until SNN adapter is implemented.

### RISK-2: `action_loss` vs `action_mse` unit distinction (Low)

In metrics.csv, `action_loss` may be in normalized units (when action transform is active) while `action_mse` is always in raw units. The distinction is documented in the code but not in the CSV header.

**Mitigation**: `action_loss_units` and `action_mse_units` columns are present in CSV. No action needed.

### RISK-3: `future_latent_cosine_error_by_horizon` is JSON in CSV (Low)

The `future_latent_cosine_error_by_horizon` column contains a JSON list (e.g., `[0.5, 0.3, 0.2, 0.1]`). Simple CSV parsers may not handle this correctly.

**Mitigation**: This is a deliberate design choice for per-horizon data. The `json.loads()` call is used in tests and summary generation.

### RISK-4: No per-task or per-seed metric disaggregation (Medium)

Current metrics are aggregated across all trajectories in a split. For reportable experiments, per-task and per-seed disaggregation will be needed.

**Mitigation**: This is a G6/G7 concern, not a metric definition issue. The metric functions support masks that could be used for per-task filtering.

---

## 8. Remaining Work

| Item | Gate | Status |
|---|---|---|
| `action_mse` synthetic tests | G5 | DONE (11 tests) |
| `action_mse_per_horizon` synthetic tests | G5 | DONE (6 tests) |
| `action_mse_per_dimension` synthetic tests | G5 | DONE (6 tests) |
| `future_latent_cosine_error` synthetic tests | G5 | DONE (16 tests) |
| `future_latent_mse` synthetic tests | G5 | DONE (12 tests) |
| Spike rate metric definition | G5 | NOT STARTED (SNN adapter not implemented) |
| SynOps proxy definition | G5 | NOT STARTED |
| Inference latency logging | G3 | NOT STARTED |
| Closed-loop success rate | G6 | NOT STARTED |
| Robustness metrics | G7 | NOT STARTED |

---

## 9. Verdict

**PASS**

All 5 implemented metrics (`action_mse`, `action_mse_per_horizon`, `action_mse_per_dimension`, `future_latent_cosine_error`, `future_latent_mse`) are correctly defined, validated with 51 synthetic tests, and used consistently. No overclaiming detected. The identified risks are low-severity and documented. Spike rate, SynOps, latency, closed-loop, and robustness metrics are not yet implemented — their absence is correctly reflected in the claims ledger.
