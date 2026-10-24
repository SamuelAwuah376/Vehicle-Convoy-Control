"""Train a MaskablePPO team policy (milestone 3). Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to runs_rl/<RUN_ID>/ (config.json, TensorBoard logs in tb/,
checkpoints/, best_model.zip + best_vecnormalize.pkl, eval_history.csv). Watch with
    .venv/Scripts/tensorboard --logdir runs_rl
Evaluate afterwards with
    .venv/Scripts/python -m nohesi.rl.evaluate runs_rl/<RUN_ID> [--store] [--mpc nonlinear --vehicle dynamic]
"""
from nohesi.rl.config import EnvConfig
from nohesi.rl.train import TrainConfig, train

RUN_ID = 'ppo_v8_e4_dyn_20m_001'       # folder under runs_rl/; reuse with RESUME to continue a run
SCENARIO = 'v5'              # SCENARIOS key (v5: BMW M3s, 45-60 mph traffic, no traffic walls, team starts at 70 mph)
SCORE = 'v8'                 # SCORES key (v8: fast team play: trains above 80 mph, bonus above 120 mph, stronger spread penalty)
PLANT = 'dynamic'            # 'kinematic' or 'dynamic' (dynamic: the M3 with load transfer, grip budget, stability layer)
MPC = 'nonlinear'            # 'linear' or 'nonlinear' (use nonlinear with the dynamic plant)
N_ENVS = 16                  # parallel SubprocVecEnv workers (one CPU thread each)
TOTAL_TIMESTEPS = 20_000_000  # policy decisions summed over envs (240 per 120 s episode); dynamic: ~27/s, ~8.4 days
EVAL_EVERY = 500_000         # decisions between evaluations on held-out seeds 1000-1009 (final: 1000-1019 via evaluate)
CHECKPOINT_EVERY = 1_000_000
SEED = 0
COLLISION_EXTRA = 0.0        # training-only extra multiple of the game's collision term (v7's 50 % wipe replaces the old 4.0)
LEARNING_RATE = 3e-4         # initial LR, decayed linearly to 0 (3e-4 from scratch; lower when fine-tuning)
VF_WARMUP = 0                # fine-tuning: first decisions train only the value network (policy frozen); 0 from scratch
DEVICE = 'cuda'              # learner device: 'cuda' (needs the CUDA torch build) or 'cpu'; env workers always use the CPU
INIT_FROM = None             # e.g. 'models/ppo_v7_e4_6m/final_model.zip' to fine-tune (None: from scratch)
RESUME = None                # e.g. 'runs_rl/ppo_v4_001/checkpoints/model_2000000.zip'

if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import this file on Windows
    env = EnvConfig(scenario=SCENARIO, score=SCORE, vehicle_model=PLANT, mpc=MPC, collision_extra=COLLISION_EXTRA)
    train(TrainConfig(run_id=RUN_ID, env=env, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS, eval_every=EVAL_EVERY,
                      checkpoint_every=CHECKPOINT_EVERY, seed=SEED, learning_rate=LEARNING_RATE, resume=RESUME, init_from=INIT_FROM, vf_warmup=VF_WARMUP,
                      device=DEVICE))
