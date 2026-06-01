# G11 Failure Mode Analysis

## Summary

- Worst demo: libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5:data/demo_2
- Worst continuous dimension: delta_pos_z
- delta_rot_x dominant: False
- Large-motion segments dominant: False
- Gripper errors precede drift: False
- Predicted histories drift to constant: False

## Worst Demo Windows

| Demo | continuous_normalized_mse | error_growth_slope | max_error | gripper_sign_accuracy |
|------|--------------------------|-------------------|-----------|----------------------|
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5:data/demo_2 | 9.181600 | -0.038430 | 22.995104 | 0.567 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5:data/demo_10 | 8.861788 | 0.043889 | 18.363750 | 0.733 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5:data/demo_28 | 7.522357 | 0.047545 | 19.704268 | 0.633 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5:data/demo_35 | 7.519610 | 0.020969 | 20.353184 | 0.550 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5:data/demo_3 | 7.457200 | -0.107574 | 16.180927 | 0.717 |

## Per-Dimension Error Breakdown

| Dimension | Total MSE | Dominant? |
|-----------|----------|-----------|
| delta_pos_x | 121.045100 |  |
| delta_pos_y | 35.625618 |  |
| delta_pos_z | 192.094854 | YES |
| delta_rot_x | 2.139761 |  |
| delta_rot_y | 6.620143 |  |
| delta_rot_z | 2.681347 |  |

## Interpretation

This is an offline failure mode analysis only.
It does not prove closed-loop failure causes.
Gripper error precedence and history drift require per-timestep inspection.
