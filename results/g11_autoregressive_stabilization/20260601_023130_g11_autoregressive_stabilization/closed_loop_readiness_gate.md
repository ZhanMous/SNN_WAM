# G11 Closed-Loop Readiness Gate

## Status: NOT PASSED

## Criteria Assessment

| # | Criterion | Status | Details |
|---|-----------|--------|---------|
| 1 | Residual beats last_action on teacher_forced continuous_normalized_mse | PASS | last_action=0.036254312843084335, residual=0.0074495659209787846 |
| 2 | Residual beats last_action on autoregressive full-sequence continuous_normalized_mse | PASS | last_action=0.8604430556297302, residual=0.00901713501662016 |
| 3 | Error growth slope lower than non-stabilized baseline | FAIL | baseline=0.8624122738838196, stabilized=0.9878094792366028 |
| 4 | No severe phase blowup above 0.5 | FAIL | max_ar_mse=20.633235931396484 |
| 5 | Gripper accuracy > 80% under autoregressive rollout | FAIL | min_acc=0.5166666507720947 |
| 6 | Results hold on held-out demos | FAIL | heldout_mean_mse=9.297248148918152 |
| 7 | Artifact registry and claims ledger pass | PASS | checked separately |

## Interpretation

If the gate does NOT pass, offline autoregressive stabilization has not sufficiently reduced compounding error to justify even a limited closed-loop smoke test. If it passes, a limited closed-loop smoke test may be considered but is NOT mandatory and does NOT guarantee closed-loop success.

## Non-Claims

- This gate does NOT prove closed-loop success or failure.
- Passing the gate does NOT mean the model will work in the environment.
- Failing the gate does NOT mean the model cannot work in the environment.
- Offline autoregressive improvement does NOT guarantee closed-loop improvement.
