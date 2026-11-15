"""Run one convoy-parking episode in the staging yard. Edit the settings below, then run this file.

The leader parks and scans; its driver picks a spot per follower (unsafe picks are rejected); the followers plan
(Hybrid A*, forward and reverse) and park under the chosen policy. Outputs go to a numbered results folder:
yard.gif, yard.png, metrics.json, events.json. See docs/yard_parking_design.md.
"""
import json
from pathlib import Path
import numpy as np
from nohesi.evaluation.io import next_run_dir
from nohesi.yard.env import YardEnv, YardEnvConfig, scripted_actions
from nohesi.yard.viz import animate, plot

SCENARIO = 'yard1'        # 'yard1' (lidar misses 15 % of low obstacles) or 'yard1_clear' (perfect scan)
POLICY = 'scripted'       # 'scripted' (prioritized baseline) or 'ppo' (a trained run, PPO_RUN)
PPO_RUN = 'models/yard_ppo_ye1_400k'   # POLICY = 'ppo' only: run folder with final_model.zip and final_vecnormalize.pkl (the first trained policy, env ye1)
PPO_MODEL = 'final'       # 'final' or 'best'
SEED = 0                  # yard layout, follower count and driver picks
FOLLOWERS = ()            # (min, max) followers, e.g. (3, 3); () keeps the scenario's (1-3)
ANIMATE = True


def main():
    cfg = YardEnvConfig(scenario=SCENARIO, followers=FOLLOWERS)
    env = YardEnv(cfg)
    obs, _ = env.reset(seed=SEED)
    if POLICY == 'ppo':
        from nohesi.rl.evaluate import load_policy, resolve
        model, normalize = load_policy(*resolve(Path(__file__).parent/PPO_RUN, PPO_MODEL))
        act = lambda o: model.predict(normalize(o[None]), action_masks=env.action_masks()[None], deterministic=True)[0][0]
    else:
        act = lambda o: scripted_actions(env)
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        done = terminated or truncated
    m = info['episode_metrics']
    out = next_run_dir(Path(__file__).parent/'results', f'{SCENARIO}_{POLICY}_seed{SEED}_followers{m["followers"]}')
    out.mkdir(parents=True)
    ep = env.episode()
    (out/'metrics.json').write_text(json.dumps(m, indent=2, default=float)+'\n')
    events = {'events': ep.events, 'picks': [{**p, 'pose': np.round(p['pose'], 3).tolist()} for p in ep.picks]}
    (out/'events.json').write_text(json.dumps(events, indent=2, default=float)+'\n')
    plot(ep, out/'yard.png')
    if ANIMATE:
        animate(ep, out/'yard.gif')
    print(f"{m['end_reason']}: {m['parked']}/{m['followers']} parked in {m['duration_s']:.1f} s | replans {m['replans']} "
          f"| safety stops {m['safety_stops']} | picks rejected {m['picks_rejected']}")
    print(f'outputs in {out}')


if __name__ == '__main__':
    main()
