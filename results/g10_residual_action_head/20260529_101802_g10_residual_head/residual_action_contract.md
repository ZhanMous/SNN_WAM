# G10 Residual Action Contract

## Target Variants

### Direct Action
- target = action[t]
- Predicted action = model output

### Residual Action
- target = action[t] - action[t-1] (continuous dims only)
- Predicted action = action[t-1] + predicted_residual
- action[t-1] is available through action_history (last entry)
- Gripper: classification (sign prediction), NOT residual regression

## Causal Contract (preserved)
- Inputs: observation[t], state[t], action_history[t-k:t-1], task_id
- target is action[t] (direct) or action[t]-action[t-1] (residual)
- input includes action[t-1] through action history
- input must NOT include action[t], future actions, future observations

## Reconstruction
- reconstructed_continuous = last_action[..., :6] + predicted_residual
- reconstructed_gripper = sign(predicted_gripper_logits)
- reconstructed_action = cat([reconstructed_continuous, reconstructed_gripper])

## Metrics
- Residual-space MSE: for monitoring training progress only
- Reconstructed-action metrics: primary scientific metrics
- Both direct and residual models evaluated under same split metric contract
