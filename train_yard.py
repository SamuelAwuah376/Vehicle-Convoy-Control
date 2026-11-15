"""Train a MaskablePPO yard-parking policy (nohesi.yard: the followers' HOLD / CREEP / GO / BACK_OFF / REPLAN calls).
Edit the settings below, then run this file (e.g. Run in VS Code). Same as
    python -m nohesi.yard.train --run-id <RUN_ID> --timesteps <TOTAL_TIMESTEPS> --envs <N_ENVS>

Use the project's .venv interpreter. Outputs go to runs_rl/<RUN_ID>/ (config.json, TensorBoard logs in tb/,
checkpoints/, best_model.zip + best_vecnormalize.pkl, final_model.zip, eval_history.csv with one row for the scripted
baseline on the same seeds, baseline.json). Watch with
    .venv/bin/tensorboard --logdir runs_rl
"""
from nohesi.yard.train import YardTrainConfig, train

RUN_ID = 'yard_ppo_001'      # folder under runs_rl/; reuse with RESUME to continue a run
SCENARIO = 'yard1'           # YardEnvConfig.scenario: key into nohesi.yard.scenarios.YARD_SCENARIOS
ENV_VERSION = 'ye1'          # YardEnvConfig.version
N_ENVS = 8                   # parallel SubprocVecEnv workers (one CPU thread each)
SUBPROC = True               # False: all envs in this process (DummyVecEnv), for debugging
TOTAL_TIMESTEPS = 2_000_000  # policy decisions summed over all envs
N_STEPS = 256                # rollout per env per update: 8 x 256 = 2048 decisions per update
BATCH_SIZE = 512
N_EPOCHS = 10
LEARNING_RATE = 3e-4         # decayed linearly to LR_FINAL over the run
LR_FINAL = 1e-5
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
NET = (256, 256)             # policy and value MLPs (tanh)
EVAL_SEEDS = tuple(range(1000, 1020))   # held-out seeds (training draws seeds >= 2000); the scripted baseline runs on them once
EVAL_ENVS = 10               # parallel eval workers (and scripted-baseline processes)
EVAL_EVERY = 50_000          # decisions between evaluations on EVAL_SEEDS
CHECKPOINT_EVERY = 100_000
BASELINE = True              # run the scripted baseline (nohesi.yard.env.scripted_actions) once on EVAL_SEEDS
SEED = 0
DEVICE = 'cpu'               # learner device: 'cpu' or 'cuda'; env workers always use the CPU
RESUME = None                # e.g. 'runs_rl/yard_ppo_001/checkpoints/model_1000000.zip'

if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import this file
    from nohesi.yard.env import YardEnvConfig
    env = YardEnvConfig(version=ENV_VERSION, scenario=SCENARIO)
    train(YardTrainConfig(run_id=RUN_ID, env=env, n_envs=N_ENVS, subproc=SUBPROC, total_timesteps=TOTAL_TIMESTEPS,
                          n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE,
                          lr_final=LR_FINAL, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF, net_pi=NET,
                          net_vf=NET, eval_seeds=EVAL_SEEDS, eval_envs=EVAL_ENVS, eval_every=EVAL_EVERY,
                          checkpoint_every=CHECKPOINT_EVERY, baseline=BASELINE, seed=SEED, device=DEVICE,
                          resume=RESUME))
