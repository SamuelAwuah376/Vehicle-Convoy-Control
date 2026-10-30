"""Train a MaskablePPO team policy (milestone 3). Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to runs_rl/<RUN_ID>/ (config.json, TensorBoard logs in tb/,
checkpoints/, best_model.zip + best_vecnormalize.pkl, eval_history.csv). Watch with
    .venv/Scripts/tensorboard --logdir runs_rl
Evaluate afterwards with
    .venv/Scripts/python -m nohesi.rl.evaluate runs_rl/<RUN_ID> [--store] [--mpc nonlinear --vehicle dynamic]
"""
from nohesi.rl.config import EnvConfig
from nohesi.rl.train import TrainConfig, train

RUN_ID = 'ppo_v10_e5_dyn_b5_5m_001'      # folder under runs_rl/; reuse with RESUME to continue a run
# Dynamic-car training in four 5M batches (the user's plan): batch 1 (ppo_v8_e4_dyn_b1_5m_001) from scratch; batches
# 2-4 each set INIT_FROM to the previous batch's final model, with a fresh LR cycle. Batch 2 adds the overtake actions
# (env e5: pass the car ahead and return, or stay out), warm-started from batch 1 (rl.train.load_policy_weights).
# Batch 2b (the user's call, 2026-10-27): batch 2 stopped at its 1M checkpoint (46.8k, 0 crashes on 1000-1009) and
# restarts from it with traffic that never seals the road and changes lanes less (scenario v7), a penalty for sitting
# behind traffic and a penalty for braking. Batch 2c (2026-10-27): braking and stuck time were flat at 2b's 2M eval
# (approx_kl ~0.07 above target_kl at LR 1e-3), so the remaining 3M warm-start from 2b's 2M checkpoint at a lower LR.
# Batch 3 (2026-10-28): from batch 2c's final model (50 seeds 1000-1049: 56.5k, 0 crashes; 2c's 3M best 54.4k; the 2b
# start 45.5k, 4 % crashes), same settings, a fresh 5M LR cycle. Batch 3b (the user's call, 2026-10-29): batch 3 was flat
# to 2.25M (training stats unchanged since 0.75M; approx_kl ~0.04, 40-54 % clipped: big moves, no gain), so the
# remaining 2.75M warm-start from its 2.25M checkpoint at LR 1e-4 -> 1e-5 (2c's gains came once its LR fell below 1e-4).
# Batch 4 (the user's reshape, 2026-10-29): from batch 3b's final model on score v9 (10x train points from 75 mph,
# 10x draft and train passes, -3 RU for a lane change without a pass, 40x speed points), braking -1 RU per m/s, stuck
# -3 RU per car-second, and the crash penalty scaled above an episode (batch 3b's 1M best under v9: 3,737 RU net per
# episode on 1000-1009, best 5,218), with a value warm-up for the new reward scale. Batch 4b (the user's call,
# 2026-10-30): batch 4 was flat to 1M (train time ~4 s, 61 mph: the team never reached the 75 mph trains), so the
# remaining 4M warm-start from its 1M checkpoint on score v10 (train points ramp from 40 % at 60 mph to 100 % at 80 mph),
# with a value warm-up first (the user asked) and the LR continuing from batch 4's 1M value. Batch 5 (the user's call,
# 2026-10-30): 4b was flat to 0.8M (61 mph, stuck behind 45-60 mph traffic about a third of the time), so traffic gets
# speeds by lane (scenario v8: right and middle lanes 60-65 mph, left lane 70-80 mph) and a fresh 5M run warm-starts
# from 4b's latest checkpoint, value warm-up first. The policy also decides at 5 Hz instead of 2 Hz (the user's call),
# with the per-decision settings rescaled to keep the same time constants (speed slew, discount horizon, rollout length).
SCENARIO = 'v8'              # SCENARIOS key (v7: v5 = BMW M3s, 45-60 mph traffic, team starts at 70 mph; no traffic
                             # blockades (v6); traffic starts a lane change MOBIL picks with probability 0.1; v8: v7 with
                             # traffic at 60-65 mph in the right and middle lanes, 70-80 mph in the left)
SCORE = 'v10'                # SCORES key (v8: fast team play: trains above 80 mph, bonus above 120 mph, stronger spread
                             # penalty; v9: v8 reshaped for team play, see config.score_v9; v10: v9 + a train speed ramp)
PLANT = 'dynamic'            # 'kinematic' or 'dynamic' (dynamic: the M3 with load transfer, grip budget, stability layer)
MPC = 'nonlinear'            # 'linear' or 'nonlinear' (use nonlinear with the dynamic plant)
DECISION_INTERVAL = 0.2      # s between policy decisions (5 Hz; batches 1-4b: 0.5 = 2 Hz); the NMPC replans every 0.1 s
SPEED_STEP = 0.8             # m/s target speed change per speed action (2 Hz: 2.0): 4 m/s per second of commands either way
GAMMA = 0.996                # per decision (2 Hz: 0.99 = 0.996^2.5): the same ~50 s discount horizon
GAE_LAMBDA = 0.98            # per decision (2 Hz: 0.95 = 0.98^2.5)
N_STEPS = 1280               # rollout per env per update (2 Hz: 512): 256 s of driving either way; 24 x 1280 = 30,720
ENV_VERSION = 'e5'           # 'e4' (45 actions per car) or 'e5' (plus the 6 overtakes)
N_ENVS = 24                  # parallel SubprocVecEnv workers (one CPU thread each; 24 = the physical cores)
TOTAL_TIMESTEPS = 5_000_000   # policy decisions summed over envs (600 per 120 s episode at 5 Hz, 240 at 2 Hz); dynamic,
                              # 24 envs at 2 Hz: ~50-60/s, ~25 h
EVAL_ENVS = 10               # parallel eval workers (idle between evals; torch-free workers hold ~0.1 GB each)
EVAL_EVERY = 500_000         # decisions between evaluations on held-out seeds 1000-1009 (final: 1000-1019 via evaluate)
CHECKPOINT_EVERY = 250_000    # frequent: train_rl_supervised.py resumes from the latest one after a failure
SEED = 0
COLLISION_EXTRA = 0.0        # training-only extra multiple of the game's collision term (v7's 50 % wipe replaces the old 4.0)
CRASH_PENALTY = -10_000.0    # training-only fixed reward on a crash (= 1M game points): the game's crash cost scales with
                             # banked points and the MPC's vehicle clearance is soft (batch 1 had none). Kept above a whole
                             # episode: -500 until batch 3b (v8 episodes ~360-840 RU), -10,000 on v9 (~3,700-5,200 RU)
CLIP_REWARD = 50.0           # clip on the normalised reward (SB3 default 10): v9's crash is ~15 return std (v8's -500: 4.5)
STUCK_PENALTY = -3.0         # training-only reward per second per car stuck behind traffic beyond 2 s (= -300 points/s; -0.25 to 3b):
                             # the traffic car ahead in its lane within 100 m, the car less than 2 m/s faster than it
BRAKE_PENALTY = -1.0         # training-only reward per m/s of speed braked away (= -100 points per m/s, -0.1 to 3b; the batch 2 policy
                             # brakes away ~87 m/s per car-minute, ~-5k points per episode): smooth driving, little braking
LEARNING_RATE = 5e-4         # initial LR, decayed linearly to LR_FINAL (batch 1: 3e-4 -> 0; batch 2 1e-3: new actions;
                             # 2c, 3: 3e-4; 3b, 4: 1e-4; 4b: 8.2e-5, batch 4's at 1M; 5: 5e-4, the user's call)
LR_FINAL = 1e-5
TARGET_KL = 0.05             # PPO stops an update's epochs once the policy has moved this far (guards the high LR); None: off
VF_WARMUP = 250_000          # fine-tuning: first decisions train only the value network (policy frozen); 0 from scratch
                             # (batch 2b: 100k, the critic first learned the stuck penalty; 2c, 3, 3b: reward unchanged;
                             # 4: 100k, the reward reshape; 4b: 100k, the train ramp; 5: 250k at 5 Hz = 100k at 2 Hz,
                             # the new traffic and decision rate)
DEVICE = 'cuda'              # learner device: 'cuda' (needs the CUDA torch build) or 'cpu'; env workers always use the CPU
INIT_FROM = 'runs_rl/ppo_v10_e5_dyn_b4b_4m_001/checkpoints/model_750024.zip'   # None: from scratch
RESUME = None                # e.g. 'runs_rl/ppo_v4_001/checkpoints/model_2000000.zip'

if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import this file on Windows
    env = EnvConfig(version=ENV_VERSION, overtakes=ENV_VERSION == 'e5', scenario=SCENARIO, score=SCORE,
                    vehicle_model=PLANT, mpc=MPC, collision_extra=COLLISION_EXTRA, crash_penalty=CRASH_PENALTY,
                    aux_stuck=STUCK_PENALTY, aux_brake=BRAKE_PENALTY, decision_interval=DECISION_INTERVAL,
                    speed_step=SPEED_STEP)
    train(TrainConfig(run_id=RUN_ID, env=env, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS, eval_every=EVAL_EVERY,
                      eval_envs=EVAL_ENVS, checkpoint_every=CHECKPOINT_EVERY, seed=SEED, learning_rate=LEARNING_RATE,
                      lr_final=LR_FINAL, clip_reward=CLIP_REWARD, gamma=GAMMA, gae_lambda=GAE_LAMBDA, n_steps=N_STEPS,
                      target_kl=TARGET_KL, resume=RESUME, init_from=INIT_FROM, vf_warmup=VF_WARMUP, device=DEVICE))
