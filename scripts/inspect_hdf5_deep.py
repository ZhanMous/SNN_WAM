#!/usr/bin/env python3
"""Deep inspect: states field composition, full 92-dim breakdown, and object pose check."""
import h5py, os, numpy as np

root = "/home/zhan_shaoji/data/libero/datasets"
path = os.path.join(root, "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5")
out_path = "/home/zhan_shaoji/code/SNN_WAM/results/g7_state_action_contract/deep_inspect.txt"

os.makedirs(os.path.dirname(out_path), exist_ok=True)
out = open(out_path, "w")

def log(msg):
    out.write(msg + "\n")

f = h5py.File(path, 'r')
demo = f['data/demo_0']

# 1. Break down robot_states (9-dim)
rs = np.array(demo['robot_states'])
log("=== ROBOT_STATES (9-dim) ===")
log(f"  shape: {rs.shape}")
log(f"  min per dim: {rs.min(axis=0).tolist()}")
log(f"  max per dim: {rs.max(axis=0).tolist()}")
log(f"  mean per dim: {rs.mean(axis=0).tolist()}")
log(f"  std per dim: {rs.std(axis=0).tolist()}")

# 2. Break down obs/ee_pos, ee_ori, gripper_states, ee_states
ee_pos = np.array(demo['obs/ee_pos'])
ee_ori = np.array(demo['obs/ee_ori'])
gripper = np.array(demo['obs/gripper_states'])
ee_states = np.array(demo['obs/ee_states'])
joint = np.array(demo['obs/joint_states'])

log("\n=== OBS/EE_POS (3-dim) ===")
log(f"  min: {ee_pos.min(axis=0).tolist()}")
log(f"  max: {ee_pos.max(axis=0).tolist()}")
log(f"  mean: {ee_pos.mean(axis=0).tolist()}")

log("\n=== OBS/EE_ORI (3-dim) ===")
log(f"  min: {ee_ori.min(axis=0).tolist()}")
log(f"  max: {ee_ori.max(axis=0).tolist()}")
log(f"  mean: {ee_ori.mean(axis=0).tolist()}")

log("\n=== OBS/GRIPPER_STATES (2-dim) ===")
log(f"  min: {gripper.min(axis=0).tolist()}")
log(f"  max: {gripper.max(axis=0).tolist()}")
log(f"  mean: {gripper.mean(axis=0).tolist()}")

log("\n=== OBS/EE_STATES (6-dim) ===")
log(f"  min: {ee_states.min(axis=0).tolist()}")
log(f"  max: {ee_states.max(axis=0).tolist()}")
log(f"  mean: {ee_states.mean(axis=0).tolist()}")

log("\n=== OBS/JOINT_STATES (7-dim) ===")
log(f"  min: {joint.min(axis=0).tolist()}")
log(f"  max: {joint.max(axis=0).tolist()}")
log(f"  mean: {joint.mean(axis=0).tolist()}")

# 3. Check if robot_states = concat(ee_pos, ee_ori, gripper)
log("\n=== ROBOT_STATES COMPOSITION CHECK ===")
reconstructed_8 = np.concatenate([ee_pos, ee_ori, gripper], axis=1)  # 3+3+2=8
diff_8 = np.abs(rs[:, :8] - reconstructed_8).max()
log(f"  ee_pos(3) + ee_ori(3) + gripper(2) = 8 dims, max diff vs robot_states[:8]: {diff_8:.8e}")

# Check what the 9th dim is
log(f"  robot_states[:, 8] (9th dim): min={rs[:, 8].min():.6f}, max={rs[:, 8].max():.6f}, mean={rs[:, 8].mean():.6f}")

# 4. Check the 92-dim states
states = np.array(demo['states'])
log("\n=== STATES (92-dim) ===")
log(f"  shape: {states.shape}")
log(f"  min per dim (first 20): {states.min(axis=0)[:20].tolist()}")
log(f"  max per dim (first 20): {states.max(axis=0)[:20].tolist()}")

# 5. Check for object/goal in problem_info
if 'problem_info' in demo.attrs:
    import json
    pi = demo.attrs['problem_info']
    if isinstance(pi, bytes):
        pi = pi.decode('utf-8')
    info = json.loads(pi)
    log("\n=== PROBLEM_INFO KEYS ===")
    for k in sorted(info.keys()):
        v = info[k]
        if isinstance(v, str) and len(v) > 100:
            log(f"  {k}: {v[:100]}...")
        else:
            log(f"  {k}: {repr(v)[:200]}")

# 6. Check actions
actions = np.array(demo['actions'])
log("\n=== ACTIONS (7-dim) ===")
log(f"  shape: {actions.shape}")
log(f"  min per dim: {actions.min(axis=0).tolist()}")
log(f"  max per dim: {actions.max(axis=0).tolist()}")
log(f"  mean per dim: {actions.mean(axis=0).tolist()}")
log(f"  std per dim: {actions.std(axis=0).tolist()}")
log(f"  unique values dim 6 (gripper): {np.unique(actions[:, 6]).tolist()}")

f.close()
out.close()
print(f"Deep inspect written to {out_path}")
