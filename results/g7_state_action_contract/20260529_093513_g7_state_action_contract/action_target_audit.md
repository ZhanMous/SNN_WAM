# G7 Action Target Audit

Trajectory: `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5:data/demo_0`
Length: 103, Action dim: 7
Convention: action_to_current_obs (confirmed by trajectory_window.py)

## Per-Dimension Statistics

| Dim | Label | Min | Max | Mean | Std | Autocorr |
|---:|---|---:|---:|---:|---:|---:|
| 0 | delta_pos_x | -0.294643 | 0.723214 | 0.169738 | 0.319403 | 0.9936 |
| 1 | delta_pos_y | -0.259821 | 0.744643 | 0.154265 | 0.253025 | 0.9928 |
| 2 | delta_pos_z | -0.937500 | 0.763393 | -0.230851 | 0.514066 | 0.9995 |
| 3 | delta_rot_x | -0.048214 | 0.090000 | 0.005118 | 0.030631 | 0.9749 |
| 4 | delta_rot_y | -0.197143 | 0.184286 | -0.024913 | 0.099857 | 0.9945 |
| 5 | delta_rot_z | -0.093214 | -0.000000 | -0.028741 | 0.030372 | 0.9520 |
| 6 | gripper | -1.000000 | 1.000000 | 0.242718 | 0.970097 | 0.9520 |

## Gripper Analysis
- Unique values: [-1.0, 1.0]
- Is binary: True
- Is sign-coded (±1): True
- Open (>0.5): 64 (62.1%)
- Close (<-0.5): 39 (37.9%)
- Neutral: 0
- Transitions: 2

## Action Convention
- Type: absolute-like
- Continuous dims mean std: 0.207892
- Gripper dim std: 0.970097

## Interpretation

- Actions are 7-dim: 6 continuous (delta position + delta orientation) + 1 gripper.
- Gripper is binary sign-coded: -1 (close) or +1 (open).
- Continuous dims have small magnitude, consistent with delta actions.
- High autocorrelation indicates smooth expert demonstrations.
