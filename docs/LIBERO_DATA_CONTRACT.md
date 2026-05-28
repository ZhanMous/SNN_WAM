# LIBERO Data Contract

## Observed Schema

- Status: observed
- Suite: `libero_spatial`
- Dataset root: `/home/zhan_shaoji/data/libero/datasets`
- Demonstration file: `/home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5`
- Inspection report: `results/inspections/20260527_093521_libero_data_inspection_real.json`
- Trajectory id: `data/demo_0`
- Time axis: `0`
- Observed time lengths: `[155]`
- Action alignment: `action_to_current_obs`; see `docs/LIBERO_ACTION_SEMANTICS.md`
- Previous G1.5 state before this inspection: `not observed` / `blocked by G1.5`.

### Actions

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `actions` | `[155, 7]` | `float64` | `0` |

### Images

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `obs/agentview_rgb` | `[155, 128, 128, 3]` | `uint8` | `0` |
| `obs/eye_in_hand_rgb` | `[155, 128, 128, 3]` | `uint8` | `0` |

### State / Proprioception

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `obs/ee_ori` | `[155, 3]` | `float64` | `0` |
| `obs/ee_pos` | `[155, 3]` | `float64` | `0` |
| `obs/ee_states` | `[155, 6]` | `float64` | `0` |
| `obs/gripper_states` | `[155, 2]` | `float64` | `0` |
| `obs/joint_states` | `[155, 7]` | `float64` | `0` |
| `robot_states` | `[155, 9]` | `float64` | `0` |
| `states` | `[155, 92]` | `float64` | `0` |

### Labels / Evaluation Fields

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `rewards` | `[155]` | `uint8` | `0` |
| `dones` | `[155]` | `uint8` | `0` |

### Language

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
| `attrs/data/problem_info/language_instruction` | scalar string | `str` | none |

## Time Indexing Convention

Use the inspected raw time axis as axis 0. For processed LIBERO HDF5, `actions[t]` is treated as the action that led to `obs[t]`; target actions after `image_t = obs[t]` therefore start at `actions[t+1]`.

## Future Leakage Rule

Inputs may include observation/state at `t`, instruction, and already executed actions through `actions[t]` under the processed-HDF5 convention. Targets after `image_t = obs[t]` start at `actions[t+1]`; future images/latents also start at `t+1`.
