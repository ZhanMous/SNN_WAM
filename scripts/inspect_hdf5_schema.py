#!/usr/bin/env python3
"""Inspect LIBERO HDF5 schema for G7 state/action contract audit."""
import h5py, os, sys, json

root = "/home/zhan_shaoji/data/libero/datasets"
path = os.path.join(root, "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5")
out_path = "/home/zhan_shaoji/code/SNN_WAM/results/g7_state_action_contract/schema_dump.txt"

os.makedirs(os.path.dirname(out_path), exist_ok=True)
out = open(out_path, "w")

def log(msg):
    print(msg, flush=True)
    out.write(msg + "\n")

log(f"Opening: {path}")
log(f"Exists: {os.path.exists(path)}")

f = h5py.File(path, 'r')
log('\n=== TOP-LEVEL KEYS ===')
for k in f.keys():
    log(f'  {k}: {type(f[k])}')

log('\n=== data/ GROUP ===')
if 'data' in f:
    data = f['data']
    for k in sorted(data.keys()):
        item = data[k]
        if hasattr(item, 'keys'):
            log(f'  {k}/ (group)')
            for sk in sorted(item.keys()):
                sub = item[sk]
                if hasattr(sub, 'shape'):
                    log(f'    {sk}: shape={sub.shape}, dtype={sub.dtype}')
                elif hasattr(sub, 'keys'):
                    log(f'    {sk}/ (group)')
                    for ssk in sorted(sub.keys()):
                        ssub = sub[ssk]
                        if hasattr(ssub, 'shape'):
                            log(f'      {ssk}: shape={ssub.shape}, dtype={ssub.dtype}')
                        else:
                            log(f'      {ssk}: {type(ssub)}')
                else:
                    log(f'    {sk}: {type(sub)}')
        elif hasattr(item, 'shape'):
            log(f'  {k}: shape={item.shape}, dtype={item.dtype}')
        else:
            log(f'  {k}: {type(item)}')

log('\n=== ATTRS on data/demo_0 ===')
if 'data/demo_0' in f:
    demo = f['data/demo_0']
    for ak in demo.attrs:
        val = demo.attrs[ak]
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='replace')
        log(f'  {ak}: {repr(val)[:300]}')

log('\n=== ATTRS on data/demo_0/obs ===')
if 'data/demo_0/obs' in f:
    obs = f['data/demo_0/obs']
    for ak in obs.attrs:
        val = obs.attrs[ak]
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='replace')
        log(f'  {ak}: {repr(val)[:300]}')
    for k in sorted(obs.keys()):
        item = obs[k]
        if hasattr(item, 'shape'):
            log(f'  obs/{k}: shape={item.shape}, dtype={item.dtype}')
        else:
            log(f'  obs/{k}: {type(item)}')

# Check for problem_info
if 'data/demo_0' in f:
    demo = f['data/demo_0']
    if 'problem_info' in demo.attrs:
        pi = demo.attrs['problem_info']
        if isinstance(pi, bytes):
            pi = pi.decode('utf-8')
        log(f'\n=== problem_info ===')
        try:
            info = json.loads(pi)
            for k, v in info.items():
                log(f'  {k}: {repr(v)[:200]}')
        except:
            log(f'  {pi[:500]}')

# Also dump a sample of action values
log('\n=== ACTION SAMPLE (first 5 timesteps) ===')
if 'data/demo_0/actions' in f:
    actions = f['data/demo_0/actions']
    log(f'  shape: {actions.shape}, dtype: {actions.dtype}')
    for t in range(min(5, actions.shape[0])):
        log(f'  t={t}: {actions[t].tolist()}')

# Also dump a sample of state/robot_states values
log('\n=== STATE/ROBOT_STATES SAMPLE (first 5 timesteps) ===')
if 'data/demo_0/robot_states' in f:
    states = f['data/demo_0/robot_states']
    log(f'  shape: {states.shape}, dtype: {states.dtype}')
    for t in range(min(5, states.shape[0])):
        log(f'  t={t}: {states[t].tolist()}')

# Check all keys recursively
log('\n=== ALL KEYS (recursive) ===')
def visit(name, obj):
    if hasattr(obj, 'shape'):
        log(f'  {name}: shape={obj.shape}, dtype={obj.dtype}')
    elif isinstance(obj, h5py.Group):
        log(f'  {name}/ (group)')
f.visititems(visit)

f.close()
out.close()
log(f'\nSchema dump written to {out_path}')
