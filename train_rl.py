"""Train a MaskablePPO team policy (milestone 3). Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to runs_rl/<RUN_ID>/ (config.json, TensorBoard logs in tb/,
checkpoints/, best_model.zip + best_vecnormalize.pkl, eval_history.csv). Watch with
    .venv/Scripts/tensorboard --logdir runs_rl
Evaluate afterwards with
    .venv/Scripts/python -m nohesi.rl.evaluate runs_rl/<RUN_ID> [--store] [--mpc nonlinear --vehicle dynamic]
"""
from nohesi.rl.config import EnvConfig
from nohesi.rl.train import TrainConfig, train

RUN_ID = 'ppo_v7_e4_6m_001'     # folder under runs_rl/; reuse with RESUME to continue a run
SCENARIO = 'v4'              # SCENARIOS key (v4: 20 lane-changing traffic cars, 120 s)
SCORE = 'v7'                 # SCORES key (v7: v6 + team spread penalty; v6: v5 + draft passes worth 3x; v5: 5x drafting)
PLANT = 'kinematic'          # 'kinematic' or 'dynamic'
MPC = 'linear'               # 'linear' or 'nonlinear'
N_ENVS = 16                  # parallel SubprocVecEnv workers (one CPU thread each)
TOTAL_TIMESTEPS = 6_000_000  # policy decisions summed over envs (240 per 120 s episode)
EVAL_EVERY = 500_000         # decisions between evaluations on held-out seeds 1000-1009 (final: 1000-1019 via evaluate)
CHECKPOINT_EVERY = 1_000_000
SEED = 0
COLLISION_EXTRA = 0.0        # training-only extra multiple of the game's collision term (v7's 50 % wipe replaces the old 4.0)
LEARNING_RATE = 1e-4         # initial LR, decayed linearly to 0 (3e-4 from scratch; lower when fine-tuning)
VF_WARMUP = 200_000          # fine-tuning: first decisions train only the value network (policy frozen); 0 from scratch
DEVICE = 'cuda'              # learner device: 'cuda' (needs the CUDA torch build) or 'cpu'; env workers always use the CPU
INIT_FROM = 'models/ppo_v6_e4_5m/final_model.zip'   # new run from this model's weights (None: from scratch)
RESUME = None                # e.g. 'runs_rl/ppo_v4_001/checkpoints/model_2000000.zip'

if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import this file on Windows
    env = EnvConfig(scenario=SCENARIO, score=SCORE, vehicle_model=PLANT, mpc=MPC, collision_extra=COLLISION_EXTRA)
    train(TrainConfig(run_id=RUN_ID, env=env, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS, eval_every=EVAL_EVERY,
                      checkpoint_every=CHECKPOINT_EVERY, seed=SEED, learning_rate=LEARNING_RATE, resume=RESUME, init_from=INIT_FROM, vf_warmup=VF_WARMUP,
                      device=DEVICE))
