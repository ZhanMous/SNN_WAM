# Single-Demo Overfit Diagnostic

Status: single_demo_overfit_fail
Best same-demo action MSE: 0.0005274823467646326
Loss threshold: 0.0001
Closed-loop same initial condition: not_run_demo_to_init_state_mapping_not_available

Final rows:
- train_single_demo: action_mse=0.0005854283726, action_loss=0.0005854284197
- eval_same_demo_teacher_forced: action_mse=0.0006766273152, action_loss=0.0006766273485

Interpretation boundaries:
- This trains and evaluates on the same demonstration under teacher forcing.
- Passing only validates capacity and basic training mechanics.
- It does not measure closed-loop success or generalization.
