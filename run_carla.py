"""Run one No Hesi episode in CARLA. Start the CARLA server first (CarlaUE4.exe, e.g. -RenderOffScreen), edit the
settings below, then run this file with the .venv-carla interpreter (Python 3.12, CARLA's client).

Our straight road is loaded as OpenDRIVE; the team cars are CARLA's Tesla Model 3 tuned to the scenario's car and
driven by CARLA's physics; traffic follows our model with the same seeds as run_sim.py. Outputs go to results/ like
run_sim.py (label suffix _carla): team.gif, summary.png, metrics.json, events.json, plus carla.json (physical contacts)
and, with VIDEO, carla.mp4 (chase camera, 1080p at 100 fps; needs RENDER).
"""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from nohesi.carla.control import Calibration
from nohesi.carla.world import CarlaSession
from nohesi.config import SCENARIOS, SCORES, Planner
from nohesi.evaluation.io import append_index, next_run_dir, save
from nohesi.evaluation.runner import run
from nohesi.evaluation.viz import animate, plot

SCENARIO = 'v5'
POLICY = 'ppo'           # 'ppo' (trained team policy) or 'overtaker'
PPO_RUN = 'models/ppo_v8_e4_10m'
PPO_MODEL = 'final'
SEED = 1001
SCORE = 'v8'
MPC = 'nonlinear'        # the CARLA car is dynamic: the nonlinear MPC plans on its calibrated model
CALIBRATION = 'model3_m3'
HOST, PORT = 'localhost', 2000
RENDER = True            # False: CARLA skips rendering (faster); the spectator follows the team when True
ANIMATE = True
VIDEO = True             # chase-camera MP4 of the episode (CARLA renders every sim step: slower)
LABEL = f'{SCENARIO}_{POLICY}'+(f'-{Path(PPO_RUN).name}' if POLICY == 'ppo' else '')+f'_score{SCORE}_seed{SEED}_carla'


def main():
    root = Path(__file__).parent
    scenario = SCENARIOS[SCENARIO]()
    scenario = replace(scenario, vehicle=replace(scenario.vehicle, model='dynamic'), score=SCORES[SCORE]())
    session = CarlaSession(scenario.road, Calibration.load(CALIBRATION), HOST, PORT, render=RENDER)
    video = root/'results'/'_carla_recording.mp4'
    if VIDEO and RENDER:
        from nohesi.carla.video import Recorder
        session.recorder = Recorder(session, video, scenario.road)
        session.recorder.caption = (f'PPO {Path(PPO_RUN).name}' if POLICY == 'ppo' else POLICY)+f'  |  seed {SEED}  |  CARLA'
    try:
        if POLICY == 'ppo':
            from nohesi.rl.env import NoHesiEnv
            from nohesi.rl.evaluate import finished_episode, load_env_cfg, load_policy, resolve
            env_cfg = replace(load_env_cfg(root/PPO_RUN), scenario=SCENARIO, score=SCORE, vehicle_model='dynamic', mpc=MPC)
            model, normalize = load_policy(*resolve(root/PPO_RUN, PPO_MODEL))
            env = NoHesiEnv(env_cfg, world=session)
            obs, _ = env.reset(seed=SEED)
            done = False
            while not done:
                action, _ = model.predict(normalize(obs[None]), action_masks=env.action_masks()[None], deterministic=True)
                obs, _, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                if session.recorder is not None:
                    session.recorder.score = info['team_score']
            episode = finished_episode(env)
            episode.policy = f'ppo-{Path(PPO_RUN).name}'
        else:
            episode = run(scenario, SEED, POLICY, replace(Planner(), mpc=MPC), world=session)
        contacts = list(session.episode.contacts)
    finally:
        if session.recorder is not None:
            session.recorder.close()
        session.close()
    out = next_run_dir(root/'results', LABEL)
    metrics = save(episode, out)
    (out/'carla.json').write_text(json.dumps({'calibration': CALIBRATION, 'contacts': contacts}, indent=1),
                                  encoding='utf-8')
    plot(episode, out/'summary.png')
    if ANIMATE:
        animate(episode, out/'team.gif')
    append_index(out, metrics)
    if video.exists():
        shutil.move(str(video), str(out/'carla.mp4'))
    print(f"team score {metrics['team_score']:.0f} | individual {[round(s) for s in metrics['individual_scores']]} | "
          f"collisions {len(metrics['collisions'])} | overtakes {metrics['overtakes']} | CARLA contacts {len(contacts)}")
    print(f'outputs in {out}')


if __name__ == '__main__':
    main()
