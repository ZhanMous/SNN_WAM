# G7 State Schema Audit

Source: `/home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5`
Suite: `libero_spatial`

## Field Inventory

| Field | Shape | Dtype | Category |
|---|---|---|---|
| `data/demo_0/actions` | [103, 7] | float64 | action |
| `data/demo_0/dones` | [103] | uint8 | done_flag |
| `data/demo_0/obs/agentview_rgb` | [103, 128, 128, 3] | uint8 | image |
| `data/demo_0/obs/ee_ori` | [103, 3] | float64 | end_effector_pose |
| `data/demo_0/obs/ee_pos` | [103, 3] | float64 | end_effector_pose |
| `data/demo_0/obs/ee_states` | [103, 6] | float64 | end_effector_states |
| `data/demo_0/obs/eye_in_hand_rgb` | [103, 128, 128, 3] | uint8 | image |
| `data/demo_0/obs/gripper_states` | [103, 2] | float64 | gripper_state |
| `data/demo_0/obs/joint_states` | [103, 7] | float64 | joint_state |
| `data/demo_0/rewards` | [103] | uint8 | reward |
| `data/demo_0/robot_states` | [103, 9] | float64 | robot_state_proprioceptive |
| `data/demo_0/states` | [103, 92] | float64 | full_mujoco_state |

## Availability Summary

- **actions**: ✓
- **ee_ori**: ✓
- **ee_pos**: ✓
- **ee_states**: ✓
- **gripper_states**: ✓
- **image_agentview**: ✓
- **image_eye_in_hand**: ✓
- **joint_states**: ✓
- **robot_states**: ✓
- **states_92d**: ✓

## Object/Goal Fields

- **goal_ori**: ✗ NOT in HDF5 schema
- **goal_pos**: ✗ NOT in HDF5 schema
- **object_ori**: ✗ NOT in HDF5 schema
- **object_pos**: ✗ NOT in HDF5 schema

## Assessment

- **No explicit object pose or goal pose fields exist in the HDF5 schema.**
- The `states` (92-dim) field is the full MuJoCo simulation state (qpos+qvel),
  which theoretically contains object poses but requires knowing the model DOF breakdown.
- The `robot_states` (9-dim) field is proprioceptive only: gripper states + robot joint info.
- The current pipeline loads only `robot_states` as `optional_state_t`.

### Implications for Oracle State Baseline

- A true oracle state with object pose and goal pose is NOT directly available
  as named fields in the HDF5 schema.
- The 92-dim `states` field CAN be used as a full oracle state, since it
  theoretically contains all simulation state including object positions.
- However, its exact decomposition is unknown without the MuJoCo model XML.
- Conservative approach: use the 92-dim `states` as 'full_state_oracle' and
  document that it is the full MuJoCo state, not a hand-crafted oracle.

### Fields NOT Available (Cannot Fabricate)

- Object position/orientation as named fields
- Goal/target position/orientation as named fields
- Object-to-EEF relative vector
- Goal-to-object relative vector
