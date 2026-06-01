# G6 Representation Bottleneck Diagnostic

Dataset: libero_spatial
Trajectory: libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5:data/demo_0 (length=103)
Task: pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate (id=0)
Git commit: b99fbe7
Git dirty: True
Loss threshold: 0.0001

## Causal Contract
- Pass: True

## H=1 Baseline Results

| Variant | Model Type | Feature Dim | Eval MSE | Passed |
|---|---|---:|---:|---|
| proprio_only_state | proprio | 9 | 3.533573e-02 | False |
| raw_image_cnn | cnn_128 | 0 | nan | False |
| dino_cls | mlp_384 | 384 | 3.986751e-02 | False |
| action_history_gru | gru | 256 | 2.399778e-03 | False |

## Pass/Fail Summary
- Passed: 0
- Failed: 3
- Skipped: 1

**Failing variants:**
- proprio_only_state: eval_mse=3.533573e-02
- dino_cls: eval_mse=3.986751e-02
- action_history_gru: eval_mse=2.399778e-03

## Representation-Action Retrieval
See `representation_action_retrieval_report.md` for detailed analysis.

## Latent Dynamics Prediction
See `latent_dynamics_prediction.csv` for details.

## Goal-Feature Planning
See `goal_feature_planning_diagnostic.csv` for details.

## Interpretation Boundaries

- This is a single-demo H=1 overfit diagnostic under strict causal contract.
- It does not measure closed-loop success, generalization, or policy validity.
- Future-latent benefit/harm is not claimed from this diagnostic.
- WAM-GRU architecture validity is not claimed from this diagnostic.
- DINOv2 suitability is not claimed from this diagnostic.
