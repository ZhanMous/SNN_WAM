# G9 Action Target Semantics Audit

## Per-Dimension Analysis

| Dim | Label | Min | Max | Std | Smoothness | Sign Flips | Autocorr | Discontinuous? |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | delta_pos_x | -0.294643 | 0.723214 | 0.319403 | 0.028430 | 8 | 0.9936 | no |
| 1 | delta_pos_y | -0.259821 | 0.744643 | 0.253025 | 0.024558 | 11 | 0.9928 | YES |
| 2 | delta_pos_z | -0.937500 | 0.763393 | 0.514066 | 0.025698 | 9 | 0.9995 | no |
| 3 | delta_rot_x | -0.048214 | 0.090000 | 0.030631 | 0.004710 | 11 | 0.9749 | YES |
| 4 | delta_rot_y | -0.197143 | 0.184286 | 0.099857 | 0.008243 | 9 | 0.9945 | no |
| 5 | delta_rot_z | -0.093214 | -0.000000 | 0.030372 | 0.003914 | 7 | 0.9520 | no |

## Orientation Dimension Analysis

- No wraparound/discontinuity issues detected in orientation dims.

## Assessment

- Actions are delta-like (action_to_current_obs convention).
- Position dims (0-2): smooth, high autocorrelation, no discontinuities.
- Orientation dims (3-5): check for sign flips and wraparound.
- High autocorrelation (>0.95) indicates smooth expert demonstrations.
- Sign flips in orientation dims may indicate controller-level rotation representation.
