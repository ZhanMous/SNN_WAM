# Rollout Evaluation Notes

Status: closed_loop_eval
Suite: libero_spatial
Task IDs: [1, 2, 3]
Episodes: 30
Max policy steps: 300
Success rate: 0.0

Failure counts:
- max_steps_reached: 30

Evaluator limitations:
- This rollout is a small smoke run unless summary.json marks it otherwise.
- No model comparison is implied by this single-checkpoint evaluation.
- The policy consumes only action history and the current observation latent; no demonstration actions or future observations are used during rollout.
- Failure episodes remain in eval_rollout.csv and are not filtered.
