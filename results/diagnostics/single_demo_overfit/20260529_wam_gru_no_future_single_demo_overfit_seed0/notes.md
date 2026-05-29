# Single-Demo Overfit Diagnostic

Status: single_demo_overfit_fail
Best same-demo action MSE: 0.010339906882672082
Loss threshold: 0.0001
Closed-loop same initial condition: not_run_demo_to_init_state_mapping_not_available

Final rows:
- train_single_demo: action_mse=0.01052373337, action_loss=0.01052373384
- eval_same_demo_teacher_forced: action_mse=0.01033990688, action_loss=0.01033990737

Interpretation boundaries:
- This trains and evaluates on the same demonstration under teacher forcing.
- Passing only validates capacity and basic training mechanics.
- It does not measure closed-loop success or generalization.
