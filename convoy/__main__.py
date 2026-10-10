import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .config import Config
from .simulation import simulate
from .route import Route
from .geometry import safety_metrics


def main():
    parser = argparse.ArgumentParser(description='Deterministic three-vehicle convoy')
    parser.add_argument('--scenario', choices=['nominal', 'braking', 'offset'], default='nominal')
    parser.add_argument('--controller', choices=['baseline', 'mpc'], default='baseline')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--duration', type=float, default=90)
    parser.add_argument('--out', type=Path, default=Path('results/nominal'))
    parser.add_argument('--animate', action='store_true')
    args = parser.parse_args()
    result = simulate(Config(dt=args.dt, duration=args.duration), args.scenario, args.controller)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics() | safety_metrics(result, Route())
    (args.out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(args.out/'trajectory.npz', time=result.time, states=result.states,
                        progress=result.progress, lateral=result.lateral, controls=result.controls, gaps=result.gaps)
    with (args.out/'trajectory.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'vehicle', 'x_m', 'y_m', 'yaw_rad', 'speed_mps', 'progress_m', 'lateral_m'])
        for k, time in enumerate(result.time):
            for i in range(3):
                writer.writerow([time, i, *result.states[k, i], result.progress[k, i], result.lateral[k, i]])
    from .visualization import plot, animate
    plot(result, args.out/'summary.png')
    if args.animate:
        animate(result, args.out/'convoy.gif')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
