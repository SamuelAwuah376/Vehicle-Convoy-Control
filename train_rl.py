"""Train a MaskablePPO team policy (milestone 3). Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to runs_rl/<RUN_ID>/ (config.json, TensorBoard logs in tb/,
checkpoints/, best_model.zip + best_vecnormalize.pkl, eval_history.csv). Watch with
    .venv/Scripts/tensorboard --logdir runs_rl
Evaluate afterwards with
    .venv/Scripts/python -m nohesi.rl.evaluate runs_rl/<RUN_ID> [--store] [--mpc nonlinear --vehicle dynamic]
"""
from nohesi.rl.config import EnvConfig
from nohesi.rl.train import TrainConfig, train

RUN_ID = 'ppo_v8_e5_dyn_b2b_5m_001'    # folder under runs_rl/; reuse with RESUME to continue a run
# Dynamic-car training in four 5M batches (the user's plan): batch 1 (ppo_v8_e4_dyn_b1_5m_001) from scratch; batches
# 2-4 each set INIT_FROM to the previous batch's final model, with a fresh LR cycle. Batch 2 adds the overtake actions
# (env e5: pass the car ahead and return, or stay out), warm-started from batch 1 (rl.train.load_policy_weights).
# Batch 2b (the user's call, 2026-10-27): batch 2 stopped at its 1M checkpoint (46.8k, 0 crashes on 1000-1009) and
# restarts from it with traffic that never seals the road and changes lanes less (scenario v7), a penalty for sitting
# behind traffic and a penalty for braking.
SCENARIO = 'v7'              # SCENARIOS key (v7: v5 = BMW M3s, 45-60 mph traffic, team starts at 70 mph; no traffic
                             # blockades (v6); traffic starts a lane change MOBIL picks with probability 0.1)
SCORE = 'v8'                 # SCORES key (v8: fast team play: trains above 80 mph, bonus above 120 mph, stronger spread penalty)
PLANT = 'dynamic'            # 'kinematic' or 'dynamic' (dynamic: the M3 with load transfer, grip budget, stability layer)
MPC = 'nonlinear'            # 'linear' or 'nonlinear' (use nonlinear with the dynamic plant)
ENV_VERSION = 'e5'           # 'e4' (45 actions per car) or 'e5' (plus the 6 overtakes)
N_ENVS = 16                  # parallel SubprocVecEnv workers (one CPU thread each)
TOTAL_TIMESTEPS = 5_000_000   # policy decisions summed over envs (240 per 120 s episode); dynamic: ~37/s, ~1.6 days
EVAL_EVERY = 500_000         # decisions between evaluations on held-out seeds 1000-1009 (final: 1000-1019 via evaluate)
CHECKPOINT_EVERY = 250_000    # frequent: train_rl_supervised.py resumes from the latest one after a failure
SEED = 0
COLLISION_EXTRA = 0.0        # training-only extra multiple of the game's collision term (v7's 50 % wipe replaces the old 4.0)
CRASH_PENALTY = -500.0       # training-only fixed reward on a crash (= 50,000 game points, ~2 good episodes): the game's crash
                             # cost scales with banked points and the MPC's vehicle clearance is soft (batch 1 had none)
STUCK_PENALTY = -0.25        # training-only reward per second per car stuck behind traffic beyond 2 s (= -25 points/s):
                             # the traffic car ahead in its lane within 100 m, the car less than 2 m/s faster than it
BRAKE_PENALTY = -0.1         # training-only reward per m/s of speed braked away (= -10 points per m/s; the batch 2 policy
                             # brakes away ~87 m/s per car-minute, ~-5k points per episode): smooth driving, little braking
LEARNING_RATE = 1e-3         # initial LR, decayed linearly to LR_FINAL (batch 1: 3e-4 -> 0; batch 2 starts high: new actions)
LR_FINAL = 1e-5
TARGET_KL = 0.05             # PPO stops an update's epochs once the policy has moved this far (guards the high LR); None: off
VF_WARMUP = 100_000          # fine-tuning: first decisions train only the value network (policy frozen); 0 from scratch
                             # (batch 2b: the critic first learns the stuck penalty)
DEVICE = 'cuda'              # learner device: 'cuda' (needs the CUDA torch build) or 'cpu'; env workers always use the CPU
INIT_FROM = 'runs_rl/ppo_v8_e5_dyn_b2_5m_001/checkpoints/model_1000000.zip'   # None: from scratch
RESUME = None                # e.g. 'runs_rl/ppo_v4_001/checkpoints/model_2000000.zip'

if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import this file on Windows
    env = EnvConfig(version=ENV_VERSION, overtakes=ENV_VERSION == 'e5', scenario=SCENARIO, score=SCORE,
                    vehicle_model=PLANT, mpc=MPC, collision_extra=COLLISION_EXTRA, crash_penalty=CRASH_PENALTY,
                    aux_stuck=STUCK_PENALTY, aux_brake=BRAKE_PENALTY)
    train(TrainConfig(run_id=RUN_ID, env=env, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS, eval_every=EVAL_EVERY,
                      checkpoint_every=CHECKPOINT_EVERY, seed=SEED, learning_rate=LEARNING_RATE, lr_final=LR_FINAL,
                      target_kl=TARGET_KL, resume=RESUME, init_from=INIT_FROM, vf_warmup=VF_WARMUP, device=DEVICE))
