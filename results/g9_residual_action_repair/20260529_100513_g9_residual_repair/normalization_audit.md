# G9 Normalization Consistency Audit

## Verification Checklist

- Contract uses train-only statistics: ✓
- Gripper excluded from continuous normalization: ✓
- No test/demo leakage: ✓
- Train-val mean difference: 0.000000

## Train Statistics

| Dim | Mean | Std |
|---:|---:|---:|
| 0 (delta_pos_x) | 0.169738 | 0.319403 |
| 1 (delta_pos_y) | 0.154265 | 0.253025 |
| 2 (delta_pos_z) | -0.230851 | 0.514066 |
| 3 (delta_rot_x) | 0.005118 | 0.030631 |
| 4 (delta_rot_y) | -0.024913 | 0.099857 |
| 5 (delta_rot_z) | -0.028741 | 0.030372 |

## Val Statistics

| Dim | Mean | Std |
|---:|---:|---:|
| 0 (delta_pos_x) | 0.169738 | 0.319403 |
| 1 (delta_pos_y) | 0.154265 | 0.253025 |
| 2 (delta_pos_z) | -0.230851 | 0.514066 |
| 3 (delta_rot_x) | 0.005118 | 0.030631 |
| 4 (delta_rot_y) | -0.024913 | 0.099857 |
| 5 (delta_rot_z) | -0.028741 | 0.030372 |

## Assessment

- Normalization is train-only (no test leakage).
- Gripper is excluded from continuous normalization.
- Loss uses normalized MSE for continuous, BCE for gripper.
- Raw metrics computed after correct denormalization.
