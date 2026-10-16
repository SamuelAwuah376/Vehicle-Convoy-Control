"""Run one No Hesi simulation. Edit the settings below, then run this file (e.g. Run in VS Code).

Use the project's .venv interpreter. Outputs go to OUT: team.gif, summary.png, metrics.json, events.json.
"""
from dataclasses import replace
from pathlib import Path
from nohesi.config import SCENARIOS, SCORES
from nohesi.evaluation.io import save
from nohesi.evaluation.runner import run
from nohesi.evaluation.viz import animate, plot

SCENARIO = 'v2'          # 'v0' 60 s 8 cars, 'v1' 120 s 10 cars, 'v2' 20 cars; 'v3'/'v4' = v1/v2 with lane-changing traffic
POLICY = 'overtaker'     # 'overtaker' (lane changes + MPC) or 'lane_keeping'
SEED = 2                 # different traffic layout per seed
SCORE = 'v3'             # 'v0' original; 'v1' team combo, 20 m proximity, low speed points; 'v2' graded proximity; 'v3' = v2 + lane-change maneuvers
TRAFFIC = None             # number of traffic cars; None keeps the scenario's (v0: 8, v1: 10, v2: 20)
ANIMATE = True           # GIF adds a few seconds
OUT = Path(__file__).parent/'results'/(f'nohesi_{SCENARIO}_{POLICY}_score{SCORE}_seed{SEED}'+(f'_traffic{TRAFFIC}' if TRAFFIC else ''))

if __name__ == '__main__':
    scenario = SCENARIOS[SCENARIO]()
    if TRAFFIC:
        scenario = replace(scenario, traffic=replace(scenario.traffic, count=TRAFFIC))
    episode = run(scenario, SEED, POLICY, score=SCORES[SCORE]())
    metrics = save(episode, OUT)
    plot(episode, OUT/'summary.png')
    if ANIMATE:
        animate(episode, OUT/'team.gif')
    print(f"team score {metrics['team_score']:.0f} | individual {[round(s) for s in metrics['individual_scores']]} | "
          f"collisions {len(metrics['collisions'])} | overtakes {metrics['overtakes']}")
    print(f'outputs in {OUT}')
