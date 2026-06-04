# Data Contract

This document records the raw LIBERO trajectory contract and the G2/G2.5 windowing policy. It is based on `scripts/inspect_libero_data.py`, `scripts/check_libero_action_alignment.py`, and the inspection report at `results/inspections/20260527_093521_libero_data_inspection_real.json`.

## Inspection Commands

Real inspection used for this contract:

```bash
/home/zhan_shaoji/miniconda3/envs/snnwam-libero/bin/python \
  scripts/inspect_libero_data.py \
  --path /home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5 \
  --trajectory data/demo_0 \
  --output-dir results/inspections
```

Mock fallback, when real LIBERO data is unavailable:

```bash
python3 scripts/inspect_libero_data.py --mock
```

Reports are written to `results/inspections/`. These inspection reports are data-contract evidence, not reportable model results.

## Observed Real Trajectory

- Suite: `libero_spatial`
- Demonstration file: `/home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5`
- File format: HDF5
- Trajectory group: `data/demo_0`
- Demo length: `T = 155`
- Raw time axis: axis `0` for image, state, action, reward, and done arrays.
- Language source: HDF5 attribute `data.attrs["problem_info"]["language_instruction"]`.

## Observed Raw Fields

| Field | Shape | Dtype | Role | Time convention |
| --- | --- | --- | --- | --- |
| `actions` | `[155, 7]` | `float64` | action | axis 0 is time |
| `obs/agentview_rgb` | `[155, 128, 128, 3]` | `uint8` | image | axis 0 is time, raw image layout `[T, H, W, C]` |
| `obs/eye_in_hand_rgb` | `[155, 128, 128, 3]` | `uint8` | image | axis 0 is time, raw image layout `[T, H, W, C]` |
| `obs/ee_ori` | `[155, 3]` | `float64` | state/proprio | axis 0 is time |
| `obs/ee_pos` | `[155, 3]` | `float64` | state/proprio | axis 0 is time |
| `obs/ee_states` | `[155, 6]` | `float64` | state/proprio | axis 0 is time |
| `obs/gripper_states` | `[155, 2]` | `float64` | state/proprio | axis 0 is time |
| `obs/joint_states` | `[155, 7]` | `float64` | state/proprio | axis 0 is time |
| `robot_states` | `[155, 9]` | `float64` | state/proprio | axis 0 is time |
| `states` | `[155, 92]` | `float64` | simulator state | axis 0 is time; audit/rollout reset candidate, not a default model input |
| `rewards` | `[155]` | `uint8` | label/evaluation | axis 0 is time; not a model input |
| `dones` | `[155]` | `uint8` | label/evaluation | axis 0 is time; not a model input |
| `attrs/data/problem_info/language_instruction` | scalar string | `str` | language | not time-varying |

The inspected action dimension for this one trajectory is `7`. Do not assume real LIBERO action dimension for other suites, downloaded variants, or future files without printing it first.

## Mock Fields

Mock mode is clearly labeled with `mode=mock`, `mock=True`, and source `synthetic mock trajectory; no real LIBERO data was inspected`. Its example shapes include `[T, H, W, C]`, `[T, state_dim]`, and `[T, action_dim]`, but those values are not evidence about real LIBERO data.

## Trajectory Window Dataset V1

Implemented module: `src/data/trajectory_window.py`.

V1 is an in-memory deterministic windowing dataset for inspected LIBERO-style
trajectories. It does not load all LIBERO files or perform training; the
offline trainer consumes this contract separately. G2.5 split and normalization
policies are defined in `docs/SPLIT_POLICY.md` and
`docs/NORMALIZATION_POLICY.md`.

Raw trajectory inputs use time as the first axis:

- `images`: `[T, H, W, C]`.
- `actions`: `[T, action_dim]`.
- `states`: optional `[T, state_dim]`.
- `visual_latents`: optional `[T, latent_dim]`, produced by a frozen visual
  encoder or deterministic smoke encoder.
- `frame_refs`: optional `[T]`.
- `language`: trajectory-level string.

For current time index `t`, one unbatched sample has:

| Field | Shape | Input or target | Time rule |
| --- | --- | --- | --- |
| `image_t` | `[H, W, C]` | input | exactly `images[t]` |
| `language` | string | input | trajectory-level instruction |
| `action_history` | `[history_len, action_dim]` | input | under processed LIBERO semantics, exactly `actions[t-history_len+1:t+1]`; includes the last executed action that led to `image_t` |
| `optional_state_t` | `[state_dim]` or `None` | input only when present | exactly `states[t]`; no future states |
| `z_t` | `[latent_dim]` or `None` | input only when current latents are enabled | exactly `visual_latents[t]`; current frozen visual latent only |
| `target_actions` | `[action_horizon, action_dim]` | target | starts after `image_t`: `actions[t+1:t+1+action_horizon]` |
| `target_future_latents` | `[future_horizon, latent_dim]` | target only | starts after current observation: `visual_latents[t+1:t+1+future_horizon]` |
| `target_future_images` | `[future_horizon, H, W, C]` | target only | starts after current observation: `images[t+1:t+1+future_horizon]` |
| `target_future_frame_refs` | `[future_horizon]` | target reference only | starts at frame `t+1` |

The offline trainer collate function adds batch axis `B` for action and latent
fields: `action_history: [B, history_len, action_dim]`,
`target_actions: [B, action_horizon, action_dim]`, optional
`z_t: [B, latent_dim]`, and optional
`target_future_latents: [B, future_horizon, latent_dim]`.

V1 performs no padding. Valid windows are restricted to time indices with enough prior actions, target actions, and future targets. Edge windows that would require padding are excluded rather than wrapped, repeated, or silently truncated.

The deterministic mock dataset in `make_mock_trajectory_dataset()` encodes each scalar value with its time index. This is used by `tests/test_trajectory_window.py` to catch off-by-one history or future-target leaks.

## Dataset Item Shape Convention

Any later dataset implementation must preserve and test these shapes:

- `image_t`: raw `[H, W, C]` before final transform, or transformed `[C, H, W]` after a documented transform.
- `instruction`: string or encoded tensor.
- `action_history`: `[history_len, action_dim]`.
- `state_t`: `[state_dim]` if used.
- `target_actions`: `[action_horizon, action_dim]`.
- `z_t`: `[latent_dim]` when frozen current visual latents are enabled.
- `target_future_images`: `[future_horizon, H, W, C]`.
- `target_future_latents`: `[future_horizon, latent_dim]` when frozen latent
  targets are enabled.
- Optional masks: `[history_len]`, `[action_horizon]`, `[future_horizon]`.

## DINO-WM Patch-Latent Transition Dataset

Implemented module: `src/data/patch_latent_dataset.py`.

This dataset is for action-conditioned latent world-model training and planning
over cached DINOv2 patch features. It is separate from direct policy behavior
cloning. One unbatched sample has:

| Field | Shape | Input or target | Time rule |
| --- | --- | --- | --- |
| `z_context` | `[T_ctx, P, D]` | input | patch latents `z[t-T_ctx+1:t+1]` |
| `actions` | `[T_ctx, action_dim]` | input/context | action history strictly before candidate action `t`; left-padded with zeros for the first valid windows |
| `future_actions` | `[H, action_dim]` | input/candidate | candidate actions `actions[t:t+H]`; action `t` is the control for target `z[t+1]` |
| `z_target` | `[H, P, D]` | target only | future patch latents `z[t+1:t+1+H]` |
| `metadata.context_range` | `[T_ctx]` | audit | latent indices in `z_context` |
| `metadata.action_history_range` | `[T_ctx]` | audit | action-history indices before `t`; negative values indicate zero padding |
| `metadata.future_action_range` | `[H]` | audit | candidate future action indices |
| `metadata.target_range` | `[H]` | audit | future latent target indices |

`future_actions` are allowed model inputs because the world model predicts the
effect of a candidate action sequence. Future patch latents, future images,
states, rewards, success labels, and dones remain target/evaluation-only and
must not enter model inputs.

## Time Indexing Convention

Observed raw arrays use time as axis `0`. Default causal convention for the future trajectory-window dataset:

- Inputs at index `t` may include image/observation at `t`, optional state at `t`, instruction, and already executed actions up to the transition into `t`: `actions[t-history_len+1:t+1]`.
- Targets include future actions after `image_t`: `actions[t+1:t+1+action_horizon]`.
- Future image targets start after the current observation:
  `images[t+1:t+1+future_horizon]`.
- Current latent input is `z_t = visual_latents[t]`; future latent targets are
  `visual_latents[t+1:t+1+future_horizon]`.
- Padding must use masks. Do not silently wrap, repeat the last frame, or drop edge windows without documenting it.

Action alignment is documented in `docs/LIBERO_ACTION_SEMANTICS.md`. The final G2.5 convention is `action_to_current_obs`: processed HDF5 `actions[t]` led to `obs[t]`, so policy targets after observing `obs[t]` start at `actions[t+1]`.

## Future Leakage Risks

- `action_history` accidentally includes future target actions such as `action[t+1]`.
- `target_future_images` starts at `t` instead of `t+1`.
- `target_future_latents` starts at `t` instead of `t+1` or appears in
  `input_keys`.
- Future observations or states appear in model input.
- Reward, success, done, or episode outcome fields enter model input.
- Simulator `states` or `init_state` fields are used as model inputs instead of reset/evaluation metadata.
- Normalization statistics are computed on validation/test trajectories.
- Held-out task, initial-state, or success metadata leaks through language/task fields.
- A split policy mixes trajectories across train/val/test or computes statistics outside the train split.

## Unresolved Questions

- Optional stronger validation: replay one processed trajectory and compare returned observations frame-by-frame against stored HDF5 images.
- Choose primary camera inputs and document whether `eye_in_hand_rgb` is used.
- Decide whether robot state/proprioception is a Phase-1 input or audit-only field.
- Implement the real split-aware LIBERO loader that materializes `docs/SPLIT_POLICY.md`.
- Confirm how `init_state` should be stored for closed-loop rollout evaluation.

## Current Implementation Boundary

`TrajectoryWindowDataset` supports optional frozen visual latents for current
input `z_t` and future target `target_future_latents`. The committed trainer
can exercise this path in dry-run WAM-GRU mode with the deterministic
`smoke_time_index` encoder. Real LIBERO WAM training still requires
precomputed frozen latents or a real frozen encoder adapter; no large visual
backbone is fine-tuned in Phase 1.
