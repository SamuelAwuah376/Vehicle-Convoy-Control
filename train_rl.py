"""Train a MaskablePPO team policy (milestone 3). Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to runs_rl/<RUN_ID>/ (config.json, TensorBoard logs in tb/,
checkpoints/, best_model.zip + best_vecnormalize.pkl, eval_history.csv). Watch with
    .venv/Scripts/tensorboard --logdir runs_rl
Evaluate afterwards with
    .venv/Scripts/python -m nohesi.rl.evaluate runs_rl/<RUN_ID> [--store] [--mpc nonlinear --vehicle dynamic]
"""
from nohesi.intentions import ACTIONS, FOLLOW
from nohesi.rl.config import EnvConfig
from nohesi.rl.train import TrainConfig, train

RUN_ID = 'ppo_v10_e5_dyn_b11_5m_001'    # folder under runs_rl/; reuse with RESUME to continue a run
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
# Batch 5b (the user's call, 2026-10-30): at LR 5e-4 most of batch 5's updates stopped early on the KL limit (approx_kl
# 0.05-0.07, 40-51 % clipped), which also cut the critic's epochs, and the critic's explained variance fell from ~0.7 at
# 500k to ~0.2 by 700k; so the remaining 4M warm-start from its 1M checkpoint with the actor frozen while the critic
# catches up, then a looser KL limit, the LR continuing from batch 5's 1M value.
# Batch 6 (the user's call, 2026-10-31): 5b gained nothing on 50 seeds (its 1M best 555k vs 541k at the start, final
# 498k) while the policy's entropy rose (5.98 -> 6.28) and team play fell, so the policy goes back to deciding at 2 Hz
# (the 2 Hz per-decision settings again) with a 10x smaller entropy bonus, a fresh 5M run from 5b's 1M best.
# Batch 7 (the user's call, 2026-11-01): batch 6 worked (50 seeds 1000-1049: final 699k vs 457k at its start, +242k +-
# 46k paired, 38/50 wins; train passes 12.3 -> 19.3; 3/50 crashes), so the same settings run 10M from its final model.
# Batch 8 (the user's call, 2026-11-04): batch 7 worked (50 seeds: its 9.0M checkpoint 969k vs batch 6's final 699k,
# +270k +- 58k paired, 39/50 wins; train passes 19.3 -> 26.3), so 10M more from its final model with the stuck penalty
# reshaped: only missed escapes are charged, steeper and sooner (STUCK_*), with a value warm-up for the new reward.
# Batch 8b (the user's call, 2026-11-04): batch 8 stopped at 1.31M (flat; about half its missed-escape charges were cars
# moving into a lane they could pass their leader through), so 10M from batch 7's final model again with: such pass
# lanes counted as ways out (STUCK_PASS_LANES), the planner's retry cooldown 3.0 -> 1.0 s and quicker lane changes
# (PLANNER_OVERRIDES), and a value warm-up for the changed reward and dynamics.
# Batch 9 (the user's call, 2026-11-04): batch 8b stopped at its 1M checkpoint; the remaining 9M continue from it with
# 8b's settings plus: a cost per lean change while holding the lane and per lane change the policy drops while pending
# (LEAN_CHANGE_PENALTY, LC_ABORT_PENALTY: batch 7's policy changes lean ~33 times a minute in a lane, only 38 % of its
# leaning beside a car it passes, and drops ~6 lane changes per car-minute), and the observation's episode clock held
# at 0 (OBS_CLOCK: decide from what the cars see, not the time left), with a value warm-up for the new reward.
# Batch 9b (the user's call, 2026-11-04): batch 9 stopped at step 0 (paused before its first checkpoint) and is replaced:
# the same 9M from 8b's 1M checkpoint with batch 9's settings plus follow mode (FOLLOW: action 45, overtake right while
# slowing, which the policy never used, becomes "follow the teammate directly ahead"; the follower's NMPC tracks the
# leader's path at 0.35 s and joins its lane changes, or breaks off back to the policy when the path closes; while it
# follows it makes no decisions of its own). 90 % of batch 8b's train breaks followed a lane change; with follow forced
# on, 20 seeds: 1,579k vs 1,498k, train 41 vs 32 s, 1 vs 2 crashes, fallbacks 5 vs 16 per episode. The FOLLOW logit
# starts as the mean of the nine lane-keep logits + 8.0 (P(follow | allowed) median 0.12, mean 0.28 on 8b's states;
# its old overtake logit was never picked), with a value warm-up for the new action and observations.
# Batch 9c (the user's call, 2026-11-06): batch 9b stopped at 8.26M (10 seeds at 6.5M: 4.24M, train 80 s, 0 crashes);
# 3M from its 8.25M checkpoint with a fresh learning-rate cycle and the gap search (GAP_SEARCH: a car stuck behind
# traffic whose lane change finds no room at its target speed locks the gap behind or ahead of the blocking car with a
# lower / higher target speed, up to 8 m/s, the smaller change first) plus the MPC's heading allowance on the locked gap
# cars (a yawed car's corners reach past the road-aligned bounds). 20 seeds: the scripted overtaker +26k +- 10k (72k ->
# 98k, stuck 82 -> 52 s, 0 crashes); 9b's 6.5M policy, not trained with it, -0.30M +- 0.41M; with a value warm-up.
# Batch 10 (the user's call, 2026-11-07): 9c gained nothing on 50 seeds 1000-1049 (best, 3.0M: 4,099k vs 9b's 6.5M best
# 4,192k, -93k +- 248k paired; 4.3 fallbacks per episode vs 2.7), so 3M from 9b's 6.5M best (milestone 7's model) with
# the lead-held gap rule (GAP_LEAD: Planner.gap_lead; the overtaker on 50 seeds +20.5k +- 9.5k and stuck -10.4 s vs the
# gap search, which adds nothing on top) and the gap search and the heading allowance off (9b's 6.5M with gap_lead:
# -60k +- 260k; with the allowance as well: -295k +- 274k, fallbacks 2.9 -> 4.3 per episode), a pending lane change
# cancelled after 2 s instead of 4, the LR continuing from 9b's 6.5M value, 25 evaluation seeds instead of 10 (the 10-seed
# best picks noise: 9c's 3.0M best 4,533k on 1000-1009, 4,099k on 50), and a value warm-up for the new gap rule.
# Batch 11 (the user's call, 2026-11-08): 5M from batch 10's final model (= its best, 3.0M; 50 seeds 1000-1049 on v8:
# 4,327k, 0/50 crashes, +136k +- 240k vs 9b's 6.5M, 29/50 better; its 1.5M 4,063k) on denser traffic (scenario v9: 45
# cars, lane speeds 55-62 / 62-68 / 70-80 mph, traffic lane changes 3x as often, spread respawns; 10 seeds 1000-1009:
# batch 10's final 1,050k, 2/10 crashes, 9b's 6.5M 933k), batch 10's other settings, the LR again from 1.34e-4, the
# evaluations on v9, and a value warm-up for the new traffic.
SCENARIO = 'v9'              # SCENARIOS key (v7: v5 = BMW M3s, 45-60 mph traffic, team starts at 70 mph; no traffic
                             # blockades (v6); traffic starts a lane change MOBIL picks with probability 0.1; v8: v7 with
                             # traffic at 60-65 mph in the right and middle lanes, 70-80 mph in the left; v9: v8 with 45
                             # cars, 55-62 / 62-68 / 70-80 mph, 3x traffic lane changes, spread respawns)
SCORE = 'v10'                # SCORES key (v8: fast team play: trains above 80 mph, bonus above 120 mph, stronger spread
                             # penalty; v9: v8 reshaped for team play, see config.score_v9; v10: v9 + a train speed ramp)
PLANT = 'dynamic'            # 'kinematic' or 'dynamic' (dynamic: the M3 with load transfer, grip budget, stability layer)
MPC = 'nonlinear'            # 'linear' or 'nonlinear' (use nonlinear with the dynamic plant)
DECISION_INTERVAL = 0.5      # s between policy decisions (2 Hz; batches 5, 5b: 0.2 = 5 Hz); the NMPC replans every 0.1 s
SPEED_STEP = 2.0             # m/s target speed change per speed action (5 Hz: 0.8): 4 m/s per second of commands either way
GAMMA = 0.99                 # per decision (5 Hz: 0.996, 0.99 = 0.996^2.5): the same ~50 s discount horizon
GAE_LAMBDA = 0.95            # per decision (5 Hz: 0.98, 0.95 = 0.98^2.5)
N_STEPS = 512                # rollout per env per update (5 Hz: 1280): 256 s of driving either way; 24 x 512 = 12,288
ENV_VERSION = 'e5'           # 'e4' (45 actions per car) or 'e5' (plus the 6 overtakes)
N_ENVS = 24                  # parallel SubprocVecEnv workers (one CPU thread each; 24 = the physical cores)
TOTAL_TIMESTEPS = 5_000_000   # policy decisions summed over envs (600 per 120 s episode at 5 Hz, 240 at 2 Hz); dynamic,
                              # 24 envs at 2 Hz: ~50-60/s, ~25 h per 5M (9: the rest of 8b's 10M; 9c: 3M)
EVAL_SEEDS = tuple(range(1000, 1025))   # held-out seeds for the evaluations during training (until 9c: 1000-1009)
EVAL_ENVS = 25               # parallel eval workers (idle between evals; torch-free workers hold ~0.1 GB each; until 9c: 10)
EVAL_EVERY = 500_000         # decisions between evaluations on EVAL_SEEDS (final: 1000-1019 via evaluate)
CHECKPOINT_EVERY = 250_000    # frequent: train_rl_supervised.py resumes from the latest one after a failure
SEED = 0
COLLISION_EXTRA = 0.0        # training-only extra multiple of the game's collision term (v7's 50 % wipe replaces the old 4.0)
CRASH_PENALTY = -10_000.0    # training-only fixed reward on a crash (= 1M game points): the game's crash cost scales with
                             # banked points and the MPC's vehicle clearance is soft (batch 1 had none). Kept above a whole
                             # episode: -500 until batch 3b (v8 episodes ~360-840 RU), -10,000 on v9 (~3,700-5,200 RU)
CLIP_REWARD = 50.0           # clip on the normalised reward (SB3 default 10): v9's crash is ~15 return std (v8's -500: 4.5)
STUCK_PENALTY = -20.0        # training-only reward per second per car stuck behind traffic beyond STUCK_GRACE (-0.25 to 3b,
                             # -3 to 7; 8: -20 = -2,000 points/s, about what a car earns per second from passes, the user's call):
                             # the traffic car ahead in its lane within 100 m, the car less than 2 m/s faster than it
STUCK_ESCAPE_ONLY = True     # charge only missed escapes: an overtake would start, or the mask allows a lane change into a lane
                             # whose leader is >= 2 m/s faster (or none within 100 m), and the car does not take it (until 7: all
                             # stuck time; batch 7's policy: 46 % of it mid-lane-change, 31 % no faster lane, 16.5 % a missed escape)
STUCK_PASS_LANES = True      # 8b: a lane the car can pass its lead traffic car through also counts as a way out (EnvConfig.
                             # stuck_pass_lanes; batch 7's policy: 5.1 s/episode of 8's missed escapes were such moves)
STUCK_GRACE = 0.5            # s stuck before the penalty starts (until 7: 2.0; 8: 0.5, the user's call: openings close fast)
W_LATERAL = 10.0             # 8b: NMPC lateral-error weight (Planner.w_lateral, default 1): lane changes ~2.05 -> ~1.5 s, sideways
                             # force p99 4.2 -> 7.3 m/s2 (grip cap 8.8). Batch 7's final model with cooldown 1.0, 50 seeds:
                             # w 1: 1,048k (2/50 crashes), w 3: 1,184k (2/50), w 10: 1,449k (4/50; +401k +- 98k vs w 1, 34/50 wins);
                             # w 10's crashes: retries into closing gaps ending in cancels and emergency fallbacks
GAP_LEAD = (('gap_lead', True), ('pending_max_time', 2.0))   # 10: Planner.gap_lead (the gap rule's reach stays behind
                             # the car's own lead until it is clear of it sideways) and a pending lane change cancelled after
                             # 2 s (default 4 s; 9b's 6.5M never reached the limit). 9c: (('gap_search', True), ('yaw_margin',
                             # True), ('yaw_margin_scope', 'gap')): the gap search and the heading allowance on the gap cars
PLANNER_OVERRIDES = (('retry_cooldown', 1.0), ('w_lateral', W_LATERAL))+GAP_LEAD   # 8b: Planner settings over the defaults
                             # (EnvConfig.planner_overrides). retry_cooldown 3.0 -> 1.0 s: after a forced cancel the car may
                             # re-request that lane sooner (batch 7's final model, 50 seeds: 1,048k vs 924k, +124k +- 69k);
                             # w_lateral 1 -> W_LATERAL: quicker lane changes
LEAN_CHANGE_PENALTY = -5.0   # 9: training-only reward per lean change while holding the lane (= -500 points; a close pass, which
                             # needs the lean beside a car centred in the next lane, averages ~17k with the team gains)
LC_ABORT_PENALTY = -5.0      # 9: training-only reward per lane change the policy drops while pending (the planner's cancels are free)
OBS_CLOCK = False            # 9: False holds the episode clock input at 0 (and its normalisation mean at 0, so it carries nothing)
BRAKE_PENALTY = -1.0         # training-only reward per m/s of speed braked away (= -100 points per m/s, -0.1 to 3b; the batch 2 policy
                             # brakes away ~87 m/s per car-minute, ~-5k points per episode): smooth driving, little braking
LEARNING_RATE = 1.34e-4      # initial LR, decayed linearly to LR_FINAL (batch 1: 3e-4 -> 0; batch 2 1e-3: new actions;
                             # 2c, 3: 3e-4; 3b, 4: 1e-4; 4b: 8.2e-5, batch 4's at 1M; 5: 5e-4, the user's call;
                             # 5b: 4.02e-4, batch 5's at 1M; 6, 7, 8: 5e-4, batch 5's fresh cycle; 9: 4.51e-4, 8b's at 1M;
                             # 9c: 4.51e-4 again, a fresh cycle, the user's call; 10: 1.34e-4, 9b's at 6.5M: 9c's fresh
                             # cycle moved the policy hard (approx_kl ~0.1) and lost its first 500k; 11: 1.34e-4 again)
LR_FINAL = 1e-5
TARGET_KL = 0.1              # PPO stops an update's epochs once the policy has moved this far (guards the high LR); None: off.
                             # The stop ends the whole update, the critic's epochs too (until 5: 0.05; 5b: 0.1, the user's call)
ENT_COEF = 0.001             # entropy bonus (until 5b: 0.01; 6: 0.001, the user's call: 5b's entropy rose as team play fell)
VF_WARMUP = 100_000          # fine-tuning: first decisions train only the value network (policy frozen); 0 from scratch
                             # (batch 2b: 100k, the critic first learned the stuck penalty; 2c, 3, 3b: reward unchanged;
                             # 4: 100k, the reward reshape; 4b: 100k, the train ramp; 5: 250k at 5 Hz = 100k at 2 Hz,
                             # the new traffic and decision rate; 5b: 250k, the critic had fallen behind, the user asked;
                             # 6: 100k, back to 2 Hz: the critic's per-decision discount and rewards change; 7: 0, the reward, traffic
                             # and decision rate are batch 6's; 8: 100k, the stuck penalty's reshape; 8b: 100k, pass lanes, cooldown, lane changes;
                             # 9: 100k, the lean and abort costs and the clock input; 9b: 100k, follow mode; 9c: 100k, the gap search;
                             # 10: 100k, the lead-held gap rule and the 2 s pending limit; 11: 100k, the v9 traffic)
DEVICE = 'cuda'              # learner device: 'cuda' (needs the CUDA torch build) or 'cpu'; env workers always use the CPU
INIT_FROM = 'runs_rl/ppo_v10_e5_dyn_b10_3m_001/final_model.zip'   # None: from scratch (9b: 8b's 1M; 9c: 9b's 8.25M;
                             # 10: models/ppo_v10_e5_dyn_b9b_6.5m/best_model.zip)
FOLLOW_MODE = True           # 9b: follow mode (EnvConfig.follow: the FOLLOW action, its mask, 6 appended observations)
INIT_COPY_ACTIONS = ()       # 9b: ((FOLLOW, lane-keep actions, 8.0),): FOLLOW's starting logit = mean of the lane-keep
                             # logits + 8.0 (TrainConfig.init_copy_actions); 9c: () (9b's policy has FOLLOW trained)
RESUME = None                # e.g. 'runs_rl/ppo_v4_001/checkpoints/model_2000000.zip'

if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import this file on Windows
    env = EnvConfig(version=ENV_VERSION, overtakes=ENV_VERSION == 'e5', scenario=SCENARIO, score=SCORE,
                    vehicle_model=PLANT, mpc=MPC, collision_extra=COLLISION_EXTRA, crash_penalty=CRASH_PENALTY,
                    aux_stuck=STUCK_PENALTY, stuck_escape_only=STUCK_ESCAPE_ONLY, stuck_grace=STUCK_GRACE,
                    stuck_pass_lanes=STUCK_PASS_LANES, planner_overrides=PLANNER_OVERRIDES,
                    aux_brake=BRAKE_PENALTY, aux_lean_change=LEAN_CHANGE_PENALTY, aux_lc_abort=LC_ABORT_PENALTY,
                    obs_clock=OBS_CLOCK, decision_interval=DECISION_INTERVAL,
                    speed_step=SPEED_STEP, follow=FOLLOW_MODE)
    train(TrainConfig(run_id=RUN_ID, env=env, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS, eval_every=EVAL_EVERY,
                      eval_seeds=EVAL_SEEDS, eval_envs=EVAL_ENVS, checkpoint_every=CHECKPOINT_EVERY, seed=SEED, learning_rate=LEARNING_RATE,
                      lr_final=LR_FINAL, clip_reward=CLIP_REWARD, gamma=GAMMA, gae_lambda=GAE_LAMBDA, n_steps=N_STEPS,
                      target_kl=TARGET_KL, ent_coef=ENT_COEF, resume=RESUME, init_from=INIT_FROM, vf_warmup=VF_WARMUP,
                      init_copy_actions=INIT_COPY_ACTIONS if INIT_FROM else (), device=DEVICE))
