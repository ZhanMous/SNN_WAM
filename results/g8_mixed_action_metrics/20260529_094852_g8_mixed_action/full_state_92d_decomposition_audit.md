# G8 Full State 92-dim Decomposition Audit

## Status: UNCERTAIN

The 92-dim `states` field is the full MuJoCo qpos + qvel vector.
Exact decomposition requires the MuJoCo model XML from LIBERO, which is not available in the HDF5 files.

## Known Components

| Component | DOF | Description |
|---|---:|---|
| robot_qpos | variable | 7 DOF (Panda arm joints) |
| gripper_qpos | variable | 2 DOF (Panda gripper joints) |
| object_qpos | variable | 7 DOF per object (position 3 + quaternion 4) |
| robot_qvel | variable | 7 DOF (velocity) |
| gripper_qvel | variable | 2 DOF (velocity) |
| object_qvel | variable | 7 DOF per object (velocity) |

## Estimated Breakdown
qpos: robot(7) + gripper(2) + object(7) = 16; qvel: robot(7) + gripper(2) + object(7) = 16; Total = 32. But we have 92 dims, suggesting multiple objects or additional DOF (e.g., joints, contact forces). Exact decomposition requires the MuJoCo model XML from LIBERO.

## Recommendation
Keep conservative label 'full_state_92d'. Do not claim 'true oracle' or decompose into named features without verifying against the MuJoCo model XML.

## Goal Pose
Unknown. Goal may be task-conditioned, not in state vector.
