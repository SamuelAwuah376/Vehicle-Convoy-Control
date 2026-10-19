"""Run one No Hesi simulation. Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to OUT: team.gif, summary.png, metrics.json, events.json.
"""
from dataclasses import replace
from pathlib import Path
from nohesi.config import SCENARIOS, SCORES, Planner
from nohesi.evaluation.io import append_index, next_run_dir, save
from nohesi.evaluation.runner import run
from nohesi.evaluation.viz import animate, plot

SCENARIO = 'v4'          # 'v0' 60 s 8 cars, 'v1' 120 s 10 cars, 'v2' 20 cars; 'v3'/'v4' = v1/v2 with lane-changing traffic
POLICY = 'ppo'           # 'overtaker' (lane changes + MPC), 'lane_keeping', or 'ppo' (trained MaskablePPO team policy)
PPO_RUN = 'models/ppo_v6_e4_5m'  # POLICY = 'ppo' only: trained run or models/ folder (its env config must match the current env version)
PPO_MODEL = 'final'      # POLICY = 'ppo' only: 'best' or 'final' model of PPO_RUN
SEED = 2                 # different traffic layout per seed
SCORE = 'v4'             # 'v0' original; 'v1' team combo, 20 m proximity, low speed points; 'v2' graded proximity; 'v3' = v2 + lane-change maneuvers; 'v4' = v3 with drafting for proximity + team close passes
TRAFFIC = None             # number of traffic cars; None keeps the scenario's (v0: 8, v1: 10, v2: 20)
PLANT = 'kinematic'      # simulated vehicle physics: 'kinematic' or 'dynamic' (tyre slip)
MPC = 'linear'           # 'linear' (kinematic OSQP MPC) or 'nonlinear' (acados NMPC on the dynamic bicycle; pair with PLANT = 'dynamic')
ANIMATE = True           # GIF adds a few seconds
# Each run gets the next free number: results/run_NNN_<scenario>_<policy>_score<score>_seed<seed>[...]; see results/runs.csv.
LABEL = f'{SCENARIO}_{POLICY}'+(f'-{Path(PPO_RUN).name}' if POLICY == 'ppo' else '')+f'_score{SCORE}_seed{SEED}'+(f'_traffic{TRAFFIC}' if TRAFFIC else '')+(f'_{PLANT}' if PLANT != 'kinematic' else '')+('_nmpc' if MPC == 'nonlinear' else '')

def run_ppo():
    """Run the trained PPO policy on this file's scenario/score/plant/MPC and save it like the scripted policies."""
    from nohesi.rl.config import EnvConfig
    from nohesi.rl.evaluate import load_env_cfg, load_policy, resolve, store_episode
    if TRAFFIC:
        print('note: TRAFFIC is ignored for POLICY = ppo (the env uses the scenario traffic count)')
    run_dir = Path(__file__).parent/PPO_RUN
    env_cfg = load_env_cfg(run_dir)
    if env_cfg.version != EnvConfig.version:
        raise SystemExit(f'{PPO_RUN} was trained on env {env_cfg.version}, the current env is {EnvConfig.version}; '
                         'its observations no longer match. Pick a run trained on the current env.')
    env_cfg = replace(env_cfg, scenario=SCENARIO, score=SCORE, vehicle_model=PLANT, mpc=MPC)
    model, normalize = load_policy(*resolve(run_dir, PPO_MODEL))
    out, metrics = store_episode(model, normalize, env_cfg, SEED, LABEL, root=Path(__file__).parent/'results',
                                 policy=f'ppo-{run_dir.name}')
    print(f"team score {metrics['team_score']:.0f} | individual {[round(s) for s in metrics['individual_scores']]} | "
          f"collisions {len(metrics['collisions'])} | overtakes {metrics['overtakes']}")
    print(f'outputs in {out}')


if __name__ == '__main__' and POLICY == 'ppo':
    run_ppo()
elif __name__ == '__main__':
    scenario = SCENARIOS[SCENARIO]()
    if TRAFFIC:
        scenario = replace(scenario, traffic=replace(scenario.traffic, count=TRAFFIC))
    scenario = replace(scenario, vehicle=replace(scenario.vehicle, model=PLANT))
    episode = run(scenario, SEED, POLICY, replace(Planner(), mpc=MPC), score=SCORES[SCORE]())
    OUT = next_run_dir(Path(__file__).parent/'results', LABEL)
    metrics = save(episode, OUT)
    plot(episode, OUT/'summary.png')
    if ANIMATE:
        animate(episode, OUT/'team.gif')
    print(f"team score {metrics['team_score']:.0f} | individual {[round(s) for s in metrics['individual_scores']]} | "
          f"collisions {len(metrics['collisions'])} | overtakes {metrics['overtakes']}")
    append_index(OUT, metrics)
    print(f'outputs in {OUT}')
