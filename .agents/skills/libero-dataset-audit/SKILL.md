---
name: libero-dataset-audit
description: Use when implementing or reviewing LIBERO data loading, trajectory inspection, trajectory-window slicing, action history, future action targets, future latent targets, split handling, or data-leakage tests.
---

# LIBERO Dataset Audit

This skill prevents the most damaging failure mode in the project: subtle temporal leakage or misaligned robot trajectories.

## Required inspection before editing

Before modifying dataset code, inspect:

- Raw LIBERO trajectory keys and file format.
- Image, state, action, reward/success, and language fields.
- Time dimension convention.
- Action convention: action at `t` moves from observation `t` to later observation, or dataset-specific equivalent.
- Existing train/val/test split logic.

## Required shape contract

Every dataset item must document these shapes in docstrings and tests:

- `image_t`: `[C, H, W]` or `[H, W, C]`, but choose one and normalize consistently.
- `instruction`: string or encoded tensor.
- `action_history`: `[history_len, action_dim]`.
- `state_t` if used: `[state_dim]`.
- `target_actions`: `[action_horizon, action_dim]`.
- `target_future_latents` or `target_future_images`: `[future_horizon, ...]`.
- Optional masks: `[history_len]`, `[action_horizon]`, `[future_horizon]`.

## Causal alignment rules

Default convention unless explicitly overridden in docs:

- Inputs may include observation at time `t`, instruction, optional state at `t`, and actions strictly before `t`: `actions[t-history_len:t]`.
- Targets may include actions from `t` onward: `actions[t:t+action_horizon]`.
- Future latent/image targets must start after current observation: `images[t+1:t+1+future_horizon]`.
- No input may include future images, future states, target actions, success labels, or rewards from after `t`.

If padding is needed, create masks. Do not silently wrap, repeat the last valid frame, or discard edge windows without documenting it.

## Synthetic anti-leakage test

Add a test with synthetic trajectories where every value encodes its time index. The test must fail if:

- `action_history` includes `action[t]` when it should use only actions before `t`.
- `image_t` is not exactly the current image.
- `target_future_images` starts at `t` instead of `t+1`.
- Any future target appears in the input dictionary.

Example expected checks:

```python
sample = ds[index_for_t_5]
assert sample["image_t"].time_index == 5
assert sample["action_history"].time_indices.tolist() == [1, 2, 3, 4]
assert sample["target_actions"].time_indices.tolist() == [5, 6, 7, 8]
assert sample["target_future_images"].time_indices.tolist() == [6, 7, 8, 9]
```

## Split discipline

- Do not mix trajectories across train/val/test.
- If held-out initial states are used, store them explicitly.
- If using task-level held-out split, document it separately from trajectory-level held-out split.
- Never compute normalization statistics on validation or test data.

## Deliverables

For any dataset change, produce or update:

- `docs/data_contract.md`.
- `tests/test_trajectory_window.py`.
- A small `scripts/inspect_libero_data.py` or notebook-like script.
- Printed example shapes and time-index sanity logs.

## Review output format

Return:

- PASS / FAIL / PASS WITH RISKS.
- Alignment convention used.
- Files inspected.
- Tests added or updated.
- Leakage risks found.
- Commands run.
