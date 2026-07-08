# Official upstream DINO-WM reproduction commands
# repo: https://github.com/gaoyuezhou/dino_wm
# data: https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28

export DATASET_DIR=/home/zhan_shaoji/code/SNN_WAM/data/dino_wm
export WANDB_MODE=offline

# Train official DINO-WM
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3 num_pred=1 training.seed=0 ckpt_base_path=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_preflight_20260605_cu128/official_ckpts hydra.run.dir=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_preflight_20260605_cu128/official_ckpts/outputs/point_maze_official_frameskip5_hist3_seed0 hydra.sweep.dir=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_preflight_20260605_cu128/official_ckpts/outputs/point_maze_official_frameskip5_hist3_seed0

# Plan with the trained official DINO-WM
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python plan.py model_name=point_maze_official_frameskip5_hist3_seed0 n_evals=5 planner=cem goal_H=5 goal_source=random_state planner.opt_steps=30 ckpt_base_path=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_preflight_20260605_cu128/official_ckpts seed=0
