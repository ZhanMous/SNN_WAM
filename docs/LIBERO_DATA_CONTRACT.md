# LIBERO Data Contract

## Observed Schema

- Status: observed
- Suite: `libero_spatial`
- Dataset root: `/home/zhan_shaoji/data/libero/datasets`
- Demonstration file: `/home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5`
- Inspection report: `results/inspections/20260528_020959_libero_data_inspection_real.json`
- Trajectory id: `data/demo_0`
- Time axis: `0`
- Observed time lengths: `[103]`
- Action alignment: processed LIBERO HDF5 uses action_to_current_obs convention per docs/LIBERO_ACTION_SEMANTICS.md
- Pre-observation gate: `not observed` / `blocked by G1.5` until a real LIBERO HDF5 demonstration was inspected.

### Actions

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `actions` | `[103, 7]` | `float64` | `0` |

### Images

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `obs/agentview_rgb` | `[103, 128, 128, 3]` | `uint8` | `0` |
| `obs/eye_in_hand_rgb` | `[103, 128, 128, 3]` | `uint8` | `0` |

### State / Proprioception

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `obs/ee_ori` | `[103, 3]` | `float64` | `0` |
| `obs/ee_pos` | `[103, 3]` | `float64` | `0` |
| `obs/ee_states` | `[103, 6]` | `float64` | `0` |
| `obs/gripper_states` | `[103, 2]` | `float64` | `0` |
| `obs/joint_states` | `[103, 7]` | `float64` | `0` |
| `robot_states` | `[103, 9]` | `float64` | `0` |
| `states` | `[103, 92]` | `float64` | `0` |

### Language

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `attrs/data/problem_info/language_instruction` | `None` | `str` | `None` |

## Time Indexing Convention

Use the inspected raw time axis as axis 0. For processed LIBERO HDF5, `docs/LIBERO_ACTION_SEMANTICS.md` defines the G2.5 convention: `actions[t]` led to `obs[t]`, so policy targets after `image_t = obs[t]` start at `actions[t+1]`.

## Future Leakage Rule

Inputs may include observation/state at `t`, instruction, and already executed actions through `actions[t]` under the processed-HDF5 convention. Targets after `image_t = obs[t]` start at `actions[t+1]`; future images/latents also start at `t+1`.
