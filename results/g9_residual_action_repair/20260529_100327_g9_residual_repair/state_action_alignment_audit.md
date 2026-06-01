# G9 State-Action Alignment Audit

## Convention
action_to_current_obs: action[t] is the action that led to observation[t].
Target for H=1: action[t+1] (next action after observing state[t]).

## Shift Sanity Table

| Target Shift | Label | Causal? | Leaking? | Notes |
|---:|---|---|---|---|
| -1 | leakage_diagnostic_only | False | True | target is action[t+1+-1] |
| 0 | causal | True | False | target is action[t+1+0] |
| 1 | future_target | False | False | target is action[t+1+1] |

## State-Action Correlation at Different Shifts

- shift=-1 (leaking): mean_correlation=-0.0971
- shift=0 (causal): mean_correlation=-0.0815
- shift=1 (future_state): mean_correlation=-0.0751

## Assessment

- shift=0 is the only valid causal alignment.
- shift=-1 is leakage (action[t] is already in action_history).
- shift=+1 uses future state (not available at decision time).
- If shift=0 shows low state-action correlation, the state may not contain
  sufficient information for the current action at the current timestep.
