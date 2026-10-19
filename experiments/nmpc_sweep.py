"""Controller / plant sweep behind the milestone 2, part 3 claims, re-run with the current code.

Scripted overtaker on scenarios v1 and v4, seeds 0-4, three cases:
    linear MPC + kinematic sim, linear MPC + dynamic sim, nonlinear MPC (acados) + dynamic sim.
Writes one results folder (results/run_NNN_nmpc_sweep_...) with episodes.csv (one row per episode), summary.csv
(per scenario and case), summary.json and NOTE.md. Needs the `nmpc` extra and acados (see pyproject.toml).

    .venv/Scripts/python experiments/nmpc_sweep.py [--score v3] [--seeds 0-4] [--workers 10]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import numpy as np   # noqa: E402

CASES = (('linear_kinematic', 'linear', 'kinematic'), ('linear_dynamic', 'linear', 'dynamic'),
         ('nmpc_dynamic', 'nonlinear', 'dynamic'))
SCENARIOS = ('v1', 'v4')


def episode(job):
    """One episode -> flat metrics row (runs in a worker process)."""
    from nohesi.config import SCENARIOS as SC, SCORES, Planner
    from nohesi.evaluation.metrics import metrics
    from nohesi.evaluation.runner import run
    scenario_name, case, mpc, plant, seed, score = job
    scenario = SC[scenario_name]()
    scenario = replace(scenario, vehicle=replace(scenario.vehicle, model=plant))
    start = time.perf_counter()
    ep = run(scenario, seed, 'overtaker', replace(Planner(), mpc=mpc), score=SCORES[score]())
    wall = time.perf_counter()-start
    m = metrics(ep)
    p = m['planner']
    fallbacks = p['fallbacks']
    err = p['plan_error_next_step_m'] or {}
    return {
        'scenario': scenario_name, 'case': case, 'mpc': mpc, 'plant': plant, 'seed': seed, 'score_version': score,
        'team_score': round(m['team_score'], 1), 'end_reason': m['end_reason'], 'duration_s': m['duration_s'],
        'collisions': len(m['collisions']),
        'collision_kinds': ';'.join(f"{c['vehicle']}:{c['kind']}" for c in m['collisions']),
        'close_passes': m['close_passes_scored'], 'lane_changes': p['lane_changes_completed'],
        'fallbacks': sum(fallbacks.values()), 'fallback_shifted': fallbacks.get('shifted', 0),
        'fallback_lane_keep': fallbacks.get('lane_keep', 0), 'fallback_emergency': fallbacks.get('emergency', 0),
        'solve_failures': p['qp_failures'], 'solves': p['qp_solves'],
        'solve_ms_mean': round(p['solve_ms']['mean'], 3), 'solve_ms_p95': round(p['solve_ms']['p95'], 3),
        'solve_ms_max': round(p['solve_ms']['max'], 3),
        'plan_err_s_mean_mm': round(1e3*err.get('s_mean', np.nan), 3), 'plan_err_s_max_mm': round(1e3*err.get('s_max', np.nan), 3),
        'plan_err_d_mean_mm': round(1e3*err.get('d_mean', np.nan), 3), 'plan_err_d_max_mm': round(1e3*err.get('d_max', np.nan), 3),
        'min_clearance_traffic_m': m['min_clearance_traffic_m'], 'min_clearance_teammates_m': m['min_clearance_teammates_m'],
        'planner_version': (m.get('planner_config') or {}).get('version'), 'wall_s': round(wall, 1),
    }


def summarize(rows):
    out = []
    for scenario in SCENARIOS:
        for case, _, _ in CASES:
            r = [x for x in rows if x['scenario'] == scenario and x['case'] == case]
            if not r:
                continue
            col = lambda k: np.array([x[k] for x in r], float)   # noqa: E731
            finite = lambda a: a[np.isfinite(a)]                 # noqa: E731
            out.append({
                'scenario': scenario, 'case': case, 'episodes': len(r),
                'crashed_episodes': int(sum(x['collisions'] > 0 for x in r)),
                'team_score_mean': round(float(col('team_score').mean()), 1),
                'team_score_min': round(float(col('team_score').min()), 1),
                'team_score_max': round(float(col('team_score').max()), 1),
                'mean_duration_s': round(float(col('duration_s').mean()), 1),
                'close_passes_mean': round(float(col('close_passes').mean()), 1),
                'lane_changes_mean': round(float(col('lane_changes').mean()), 1),
                'fallbacks_mean': round(float(col('fallbacks').mean()), 1),
                'emergency_mean': round(float(col('fallback_emergency').mean()), 1),
                'solve_failures_total': int(col('solve_failures').sum()),
                'solve_ms_mean': round(float(col('solve_ms_mean').mean()), 2),
                'solve_ms_p95_max': round(float(col('solve_ms_p95').max()), 2),
                'plan_err_d_mean_mm': round(float(col('plan_err_d_mean_mm').mean()), 2),
                'plan_err_d_max_mm': round(float(col('plan_err_d_max_mm').max()), 2),
                'plan_err_s_mean_mm': round(float(col('plan_err_s_mean_mm').mean()), 2),
                'min_clearance_traffic_m': round(float(finite(col('min_clearance_traffic_m')).min()), 3),
                'min_clearance_teammates_m': round(float(finite(col('min_clearance_teammates_m')).min()), 3),
            })
    return out


def git_commit():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'nohesi'], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip())
        return commit, dirty
    except OSError:
        return None, None


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--score', default='v3', help='score version (v3: the one the milestone 2 numbers used)')
    p.add_argument('--seeds', default='0-4')
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args(argv)
    lo, hi = map(int, a.seeds.split('-'))
    seeds = range(lo, hi+1)
    jobs = [(s, case, mpc, plant, seed, a.score) for s in SCENARIOS for case, mpc, plant in CASES for seed in seeds]
    # Build the acados solver once before the pool (workers then load the cached build).
    print(f'{len(jobs)} episodes; warming up the NMPC build ...', flush=True)
    first = episode(next(j for j in jobs if j[2] == 'nonlinear'))
    rows = [first]
    rest = [j for j in jobs if not (j[0] == first['scenario'] and j[1] == first['case'] and j[4] == first['seed'])]
    start = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as pool:
        for row in pool.map(episode, rest):
            rows.append(row)
            print(f"{row['scenario']} {row['case']:16s} seed {row['seed']}: {row['team_score']:9.1f} | "
                  f"{row['end_reason']:9s} at {row['duration_s']:5.1f} s | solve {row['solve_ms_mean']:.2f} ms | "
                  f"plan d err {row['plan_err_d_mean_mm']:.2f} mm", flush=True)
    order = {c: n for n, (c, _, _) in enumerate(CASES)}
    rows.sort(key=lambda r: (SCENARIOS.index(r['scenario']), order[r['case']], r['seed']))
    summary = summarize(rows)
    commit, dirty = git_commit()

    from nohesi.config import Planner
    from nohesi.evaluation.io import next_run_dir
    out = next_run_dir(ROOT/'results', f'nmpc_sweep_v1_v4_seeds{a.seeds}_score{a.score}')
    out.mkdir(parents=True)
    write_csv(out/'episodes.csv', rows)
    write_csv(out/'summary.csv', summary)
    meta = {'created': datetime.now().isoformat(timespec='seconds'), 'git_commit': commit, 'git_dirty_nohesi': dirty,
            'planner_version': Planner().version, 'policy': 'overtaker', 'scenarios': list(SCENARIOS),
            'seeds': list(seeds), 'score': a.score, 'cases': [dict(zip(('case', 'mpc', 'plant'), c)) for c in CASES],
            'wall_s': round(time.perf_counter()-start, 1), 'summary': summary}
    (out/'summary.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    print(f'outputs in {out}')
    return out


if __name__ == '__main__':
    main()
