import argparse
import json
from dataclasses import replace
from pathlib import Path
from .config import Scenario, Timing
from .evaluation.io import save
from .evaluation.runner import run

SCENARIOS = {'v0': Scenario}


def main():
    parser = argparse.ArgumentParser(description='No Hesi-style three-car team simulation')
    parser.add_argument('--scenario', choices=sorted(SCENARIOS), default='v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--out', type=Path, default=Path('results/nohesi_v0'))
    parser.add_argument('--animate', action='store_true')
    args = parser.parse_args()
    scenario = SCENARIOS[args.scenario]()
    if args.duration:
        scenario = replace(scenario, timing=Timing(scenario.timing.dt, args.duration))
    episode = run(scenario, args.seed)
    metrics = save(episode, args.out)
    from .evaluation.viz import plot, animate
    plot(episode, args.out/'summary.png')
    if args.animate:
        animate(episode, args.out/'team.gif')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'scenario'}, indent=2))


if __name__ == '__main__':
    main()
