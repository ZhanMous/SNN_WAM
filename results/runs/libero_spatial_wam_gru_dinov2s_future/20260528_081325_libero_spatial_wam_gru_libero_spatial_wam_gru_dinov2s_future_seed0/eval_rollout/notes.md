# Rollout Evaluation Notes

Status: closed_loop_smoke
Suite: libero_spatial
Task IDs: [0]
Episodes: 1
Max policy steps: 3
Success rate: 0.0

Failure counts:
- environment/evaluator error: AssertionError: [error] pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl does not exist!: 1

Evaluator limitations:
- This rollout is a small smoke run unless summary.json marks it otherwise.
- No model comparison is implied by this single-checkpoint evaluation.
- The policy consumes only action history and the current observation latent; no demonstration actions or future observations are used during rollout.
- Failure episodes remain in eval_rollout.csv and are not filtered.
