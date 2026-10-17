"""Train a MaskablePPO team policy (milestone 3). Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to runs_rl/<RUN_ID>/ (config.json, TensorBoard logs in tb/,
checkpoints/, best_model.zip + best_vecnormalize.pkl, eval_history.csv). Watch with
    .venv/Scripts/tensorboard --logdir runs_rl
Evaluate afterwards with
    .venv/Scripts/python -m nohesi.rl.evaluate runs_rl/<RUN_ID> [--store] [--mpc nonlinear --vehicle dynamic]
"""
from nohesi.rl.config import EnvConfig
from nohesi.rl.train import TrainConfig, train

RUN_ID = 'ppo_v4_e4_5m_001'     # folder under runs_rl/; reuse with RESUME to continue a run
SCENARIO = 'v4'              # SCENARIOS key (v4: 20 lane-changing traffic cars, 120 s)
SCORE = 'v4'                 # SCORES key
PLANT = 'kinematic'          # 'kinematic' or 'dynamic'
MPC = 'linear'               # 'linear' or 'nonlinear'
N_ENVS = 16                  # parallel SubprocVecEnv workers (one CPU thread each)
TOTAL_TIMESTEPS = 5_000_000  # policy decisions summed over envs (240 per 120 s episode)
EVAL_EVERY = 500_000         # decisions between evaluations on held-out seeds 1000-1009 (final: 1000-1019 via evaluate)
CHECKPOINT_EVERY = 1_000_000
SEED = 0
RESUME = None                # e.g. 'runs_rl/ppo_v4_001/checkpoints/model_2000000.zip'

if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import this file on Windows
    env = EnvConfig(scenario=SCENARIO, score=SCORE, vehicle_model=PLANT, mpc=MPC)
    train(TrainConfig(run_id=RUN_ID, env=env, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS, eval_every=EVAL_EVERY,
                      checkpoint_every=CHECKPOINT_EVERY, seed=SEED, resume=RESUME))
