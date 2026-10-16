import argparse
import json
from dataclasses import replace
from pathlib import Path
from .config import SCENARIOS, SCORES, Timing
from .evaluation.io import save
from .evaluation.runner import POLICIES, run


def main():
    parser = argparse.ArgumentParser(description='No Hesi-style three-car team simulation')
    parser.add_argument('--scenario', choices=sorted(SCENARIOS), default='v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', choices=POLICIES, default='lane_keeping')
    parser.add_argument('--score', choices=sorted(SCORES), default='v3')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--animate', action='store_true')
    args = parser.parse_args()
    args.out = args.out or Path(f'results/nohesi_{args.scenario}_{args.policy}_score{args.score}')
    scenario = SCENARIOS[args.scenario]()
    if args.duration:
        scenario = replace(scenario, timing=Timing(scenario.timing.dt, args.duration))
    episode = run(scenario, args.seed, args.policy, score=SCORES[args.score]())
    metrics = save(episode, args.out)
    from .evaluation.viz import plot, animate
    plot(episode, args.out/'summary.png')
    if args.animate:
        animate(episode, args.out/'team.gif')
    print(json.dumps({k: v for k, v in metrics.items() if k not in ('scenario', 'planner_config')}, indent=2))


if __name__ == '__main__':
    main()
