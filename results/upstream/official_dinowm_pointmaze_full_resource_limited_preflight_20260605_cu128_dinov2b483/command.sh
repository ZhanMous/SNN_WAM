# Official upstream DINO-WM reproduction commands
# repo: https://github.com/gaoyuezhou/dino_wm
# data: https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28

export DATASET_DIR=/home/zhan_shaoji/code/SNN_WAM/data/dino_wm
export WANDB_MODE=offline
export TORCH_HOME=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_resource_limited_preflight_20260605_cu128_dinov2b483/torch_home
export PATH=/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/home/zhan_shaoji/.mujoco/mujoco210/bin:/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/lib:/usr/lib/nvidia:/usr/local/nvidia/lib64
export MUJOCO_GL=osmesa
export D4RL_DATASET_DIR=/tmp/d4rl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd /home/zhan_shaoji/code/SNN_WAM/external/dino_wm

# Train official DINO-WM
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3 num_pred=1 training.seed=0 ckpt_base_path=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_resource_limited_preflight_20260605_cu128_dinov2b483/official_ckpts hydra.run.dir=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_resource_limited_preflight_20260605_cu128_dinov2b483/official_ckpts/outputs/point_maze_official_frameskip5_hist3_seed0 hydra.sweep.dir=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_resource_limited_preflight_20260605_cu128_dinov2b483/official_ckpts/outputs/point_maze_official_frameskip5_hist3_seed0 training.batch_size=1 env.num_workers=0 training.num_reconstruct_samples=1 training.reconstruct_every_x_batch=999999

# Plan with the trained official DINO-WM
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python plan.py model_name=point_maze_official_frameskip5_hist3_seed0 n_evals=5 planner=cem goal_H=5 goal_source=random_state planner.opt_steps=30 ckpt_base_path=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_resource_limited_preflight_20260605_cu128_dinov2b483/official_ckpts seed=0
