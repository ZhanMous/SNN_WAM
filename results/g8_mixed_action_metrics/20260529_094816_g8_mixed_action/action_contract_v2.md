# G8 Action Contract v2

## Action Space
- Total dims: 7
- Continuous dims: [0, 1, 2, 3, 4, 5] (dims 0-5: delta position + delta orientation)
- Gripper dim: 6 (dim 6: binary sign-coded ±1)
- Gripper encoding: binary_sign
- Gripper open command: 1.0
- Gripper close command: -1.0
- Gripper threshold for class: 0.0

## Continuous Action Statistics (from demo)

| Dim | Label | Mean | Std | Min | Max |
|---:|---|---:|---:|---:|---:|
| 0 | delta_pos_x | 0.169738 | 0.319403 | -0.294643 | 0.723214 |
| 1 | delta_pos_y | 0.154265 | 0.253025 | -0.259821 | 0.744643 |
| 2 | delta_pos_z | -0.230851 | 0.514066 | -0.937500 | 0.763393 |
| 3 | delta_rot_x | 0.005118 | 0.030631 | -0.048214 | 0.090000 |
| 4 | delta_rot_y | -0.024913 | 0.099857 | -0.197143 | 0.184286 |
| 5 | delta_rot_z | -0.028741 | 0.030372 | -0.093214 | -0.000000 |

## Gripper Statistics
- Open (>0.5): 64 (62.1%)
- Close (<-0.5): 39 (37.9%)
- Unique values: [-1.0, 1.0]

## Metric Definitions

### Primary Metrics (scientific)
- **continuous_normalized_mse**: MSE of (pred - target) / std_safe, averaged over continuous dims
- **continuous_raw_mse**: Raw MSE of continuous action dims
- **continuous_raw_mae**: MAE of continuous action dims
- **gripper_sign_accuracy**: Fraction of timesteps where predicted sign matches target
- **gripper_transition_f1**: F1 score for detecting open/close transitions

### Diagnostic Metrics
- **global_raw_mse**: MSE over all 7 dims (diagnostic only, not primary)
- **gripper_raw_mse**: MSE of gripper dim (diagnostic only)
- **old_1e4_gate**: Whether global_raw_mse < 1e-4 (engineering overfit gate only)

### Baseline Comparisons
- **beat_last_action_continuous**: continuous_normalized_mse < last_action baseline
- **beat_last_action_gripper_f1**: gripper_transition_f1 > last_action baseline
