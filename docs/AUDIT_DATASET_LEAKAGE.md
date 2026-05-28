# Dataset Leakage Audit

Status: PASS WITH RISKS for `TrajectoryWindowDataset` v1.

This audit covers `src/data/trajectory_window.py`, tests that touch it, and the G2.5 data-policy documents needed before offline training. It validates synthetic dry-run future-latent alignment, but not real frozen visual encoder extraction or rollout.

## Files Inspected

- `src/data/trajectory_window.py`
- `src/data/__init__.py`
- `tests/test_trajectory_window.py`
- `tests/test_repository_contract.py`
- `docs/DATA_CONTRACT.md`
- `docs/LIBERO_DATA_CONTRACT.md`
- `docs/DATA_RISKS.md`
- `docs/EXPERIMENT_PROTOCOL.md`

## Alignment Convention Audited

For a current time index `t`:

- Inputs may include `image_t = images[t]`, `language`, `optional_state_t = states[t]`, current latent `z_t = visual_latents[t]`, and already executed `action_history = actions[t-history_len+1:t+1]`.
- `action_history` is strictly before the current policy decision; under processed LIBERO semantics it includes raw `actions[t]`, which led to `obs[t]`, and excludes future targets at `t+1` and later.
- `target_actions` starts after the current observation: `actions[t+1:t+1+action_horizon]`.
- Future image/frame/latent targets start after the current observation: `images[t+1:t+1+future_horizon]`, frame refs `t+1:t+1+future_horizon`, and `visual_latents[t+1:t+1+future_horizon]`.
- V1 has no padding; edge windows that would need padding are excluded.

This convention follows the G2.5 LIBERO action-semantics audit in `docs/LIBERO_ACTION_SEMANTICS.md`: processed-HDF5 `actions[t]` is treated as the action that led to `obs[t]`, not the action to execute after observing `obs[t]`.

## Field Trace

| Raw field | Dataset output | Input/target role | Time rule | Audit result |
| --- | --- | --- | --- | --- |
| `images: [T, H, W, C]` | `image_t: [H, W, C]` | input | `images[t]` | PASS |
| `language: str` | `language: str` | input | trajectory-level | PASS |
| `actions: [T, action_dim]` | `action_history: [history_len, action_dim]` | input | `actions[t-history_len+1:t+1]` | PASS |
| `states: [T, state_dim]` | `optional_state_t: [state_dim]` or `None` | input when present | `states[t]` | PASS |
| `visual_latents: [T, latent_dim]` | `z_t: [latent_dim]` | input when enabled | `visual_latents[t]` | PASS |
| `actions: [T, action_dim]` | `target_actions: [action_horizon, action_dim]` | target | `actions[t+1:t+1+action_horizon]` | PASS |
| `visual_latents: [T, latent_dim]` | `target_future_latents: [future_horizon, latent_dim]` | target only | `visual_latents[t+1:t+1+future_horizon]` | PASS |
| `images: [T, H, W, C]` | `target_future_images: [future_horizon, H, W, C]` | target only | `images[t+1:t+1+future_horizon]` | PASS |
| `frame_refs: [T]` | `target_future_frame_refs: [future_horizon]` | target reference only | `frame_refs[t+1:t+1+future_horizon]` | PASS |
| rewards/dones/success labels | not returned | none | none | PASS |

## Leakage Checks

| Risk | Evidence | Result |
| --- | --- | --- |
| `action_history` includes future target action `action[t+1]` | `test_time_indexed_mock_sample_shapes_and_causal_alignment` asserts history values `[2, 3, 4, 5]` for `t=5`, and excludes `6`. | PASS |
| Current observation is off by one | Same test asserts `image_t` encodes time `5` for `t=5`. | PASS |
| Optional state uses future state | `test_optional_state_t_is_current_time_not_future_state` asserts `optional_state_t == [5, 5]` and excludes `6`. | PASS |
| Future images start at `t` instead of `t+1` | `test_time_indexed_mock_sample_shapes_and_causal_alignment` asserts future image times `[6, 7, 8, 9]` for `t=5`. | PASS |
| Future latents start at `t` instead of `t+1` | `test_future_latent_targets_align_after_current_time_and_do_not_leak` asserts future latent times `[6, 7, 8, 9]` for `t=5`. | PASS |
| Future targets are included in model inputs | `test_future_targets_are_not_input_keys` asserts `input_keys` is disjoint from `target_keys` and contains no `target` or `future` keys. | PASS |
| Cross-trajectory action history leakage | `test_multiple_trajectories_do_not_share_history_across_boundaries` asserts trajectory B history stays in B-local values. | PASS |
| Rewards/dones leak into sample | `test_future_targets_are_not_input_keys` asserts `rewards` and `dones` are absent. | PASS |

## Regression Tests Added

- `test_optional_state_t_is_current_time_not_future_state`
- `test_multiple_trajectories_do_not_share_history_across_boundaries`
- `test_future_latent_targets_align_after_current_time_and_do_not_leak`

Both tests use synthetic values that encode time index. They would fail under common leakage bugs such as using `states[t+1]` or flattening multiple trajectories before slicing history.

## Scientific Claim Audit

| Claim | Category | Evidence | Allowed wording | Forbidden wording |
| --- | --- | --- | --- | --- |
| V1 action history is causal under the documented window convention. | Supported by current evidence | Synthetic tests in `tests/test_trajectory_window.py` | "Action history includes actions that occurred before the current policy decision." | "The full LIBERO policy input is leak-free." |
| Future images/frame refs are target-only in v1. | Supported by current evidence | `input_keys`/`target_keys` tests | "Future image/frame targets are not listed as inputs." | "No future information can leak in downstream model code." |
| Future latent alignment is correct for synthetic dry-run windows. | Supported with scope limit | `test_future_latent_targets_align_after_current_time_and_do_not_leak` | "Future latent targets start at `t+1` in deterministic dry-run windows." | "Real-data future latent extraction is validated." |
| LIBERO processed-HDF5 `action[t]` semantics are documented. | Supported with residual uncertainty | `docs/LIBERO_ACTION_SEMANTICS.md` and `scripts/check_libero_action_alignment.py` | "`actions[t]` is treated as leading to `obs[t]` for processed LIBERO HDF5." | "`action[t]` is proven to move `obs[t]` to `obs[t+1]`." |
| This dataset supports SNN/GRU training claims. | Too broad for current experiment | Offline dry-run WAM-GRU is smoke-tested only; no closed-loop or real-data WAM evidence. | "This is a causal windowing layer used by offline smoke training." | "SNN/GRU training is validated." |

## Remaining Risks For Next Phase

No critical leakage issue was found in the implemented v1 windowing code.

Offline dry-run training may run with the limits in `docs/SPLIT_POLICY.md` and
`docs/NORMALIZATION_POLICY.md`. Running or claiming real-data WAM training
remains blocked until frozen visual latents and split-aware real-data loading
are materialized in code and artifacts.

Remaining risks:

- Precomputing or adapter-producing real frozen visual latents with recorded metadata before any real-data WAM-style future-latent claim.
- Implementing split-aware real-data loading and proving no train/val/test or normalization leakage in that loader before running real-data training.
- Ensuring future model code consumes only `input_keys`, not the whole sample dictionary.

## G2.5 Status

| Gate Item | Status | Evidence / Blocker |
| --- | --- | --- |
| V1 slicing | PASS | Synthetic tests cover action history, current image/state, future frames, and split boundaries. |
| Action semantics | PASS WITH RISKS | `docs/LIBERO_ACTION_SEMANTICS.md` documents `action_to_current_obs` from local LIBERO source and one real HDF5 demo. |
| Split policy | PASS AS POLICY | `docs/SPLIT_POLICY.md` defines trajectory-level Phase-1 splits; real split-aware loader is not implemented yet. |
| Normalization policy | PASS AS POLICY | `docs/NORMALIZATION_POLICY.md` requires train-only statistics; synthetic tests cover helper behavior. |
| WAM future-latent claims | PARTIAL / REAL-DATA BLOCKED | Synthetic dry-run latent alignment is tested; real frozen latent extraction and closed-loop evidence are not implemented. |
| Training implementation | ALLOWED WITH LIMITS | Offline action-only and WAM-GRU dry-run code may run only if it follows the documented action semantics, split file, and train-only normalization policy. |

## Audit Verdict

PASS WITH RISKS.

The v1 in-memory trajectory-window dataset passes synthetic anti-leakage checks for implemented fields. The residual risks are integration and semantics risks, not observed v1 slicing bugs.
