import csv
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from .metrics import metrics

INDEX_FIELDS = ('run', 'time', 'folder', 'scenario', 'policy', 'score', 'seed', 'planner', 'vehicle_model',
                'team_score', 'collisions', 'end_reason')


def next_run_dir(root, label):
    """results/run_NNN_<label>: one past the highest existing run number, so runs never overwrite."""
    root = Path(root)
    taken = [int(m.group(1)) for p in root.glob('run_*') if (m := re.match(r'run_(\d+)', p.name))] if root.exists() else []
    return root/f'run_{max(taken, default=0)+1:03d}_{label}'


def append_index(out, m):
    """Add one row per run to <results>/runs.csv."""
    out = Path(out); index = out.parent/'runs.csv'
    new = not index.exists()
    with index.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(INDEX_FIELDS)
        number = re.match(r'run_(\d+)', out.name)
        writer.writerow([number.group(1) if number else '', datetime.now().isoformat(timespec='seconds'), out.name,
                         m['scenario_version'], m['policy'], m['score_version'], m['seed'],
                         (m.get('planner_config') or {}).get('version', ''), m['scenario']['vehicle'].get('model', 'kinematic'),
                         round(m['team_score'], 1), len(m['collisions']), m['end_reason']])


def _plans(ep):
    """Executed MPC/fallback plans per replan: plan_time (M,), plan_vehicle (M,), plan_source (M,), plan_z (M, N+1, 4)."""
    plans = ep.planner_log.plans if ep.planner_log else []
    if not plans:
        return {}
    return {'plan_time': np.array([p[0] for p in plans]), 'plan_vehicle': np.array([p[1] for p in plans]),
            'plan_source': np.array([p[2] for p in plans]), 'plan_z': np.array([p[3] for p in plans])}


def save(ep, out):
    """Write metrics.json (with seed, versions and full scenario), events.json, NPZ and CSV histories."""
    out.mkdir(parents=True, exist_ok=True)
    m = metrics(ep)
    (out/'metrics.json').write_text(json.dumps(m, indent=2)+'\n')
    events = {'overtakes': [asdict(o) for o in ep.overtakes],
              'close_passes': [asdict(o) | {'points': p} for o, p in ep.awarded],
              'maneuvers': list(getattr(ep, 'maneuvers', [])),
              'collisions': [asdict(c) for c in ep.collisions],
              'intentions': [asdict(r) for r in ep.intention_records],
              'fallbacks': [dict(zip(('time', 'vehicle', 'level'), f)) for f in ep.planner_log.fallbacks]
              if ep.planner_log else [],
              'plan_conflicts': [dict(zip(('time', 'vehicle', 'other', 'step', 'resolution'), (c[0], c[1], list(c[2]), c[3], c[4])))
                                 for c in ep.planner_log.conflicts] if ep.planner_log else []}
    (out/'events.json').write_text(json.dumps(events, indent=2)+'\n')
    np.savez_compressed(out/'trajectory.npz', time=ep.time, team=ep.team, team_s=ep.team_s, team_d=ep.team_d,
                        traffic_s=ep.traffic_s, traffic_d=ep.traffic_d, traffic_v=ep.traffic_v,
                        traffic_id=ep.traffic_id, controls=ep.controls, pair_clearance=ep.pair_clearance,
                        traffic_clearance=ep.traffic_clearance, compute_time=ep.compute_time,
                        **{f'score_{t}': v for t, v in ep.score.items()}, **_plans(ep),
                        **({'combo_multiplier': ep.combo_multiplier} if ep.combo_multiplier is not None else {}))
    with (out/'team.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'vehicle', 'x_m', 'y_m', 'yaw_rad', 'speed_mps', 's_m', 'd_m', 'cumulative_score'])
        cumulative = np.vstack((np.zeros((1, 3)), np.cumsum(sum(ep.score.values()), axis=0)))
        for k, time in enumerate(ep.time):
            for i in range(3):
                writer.writerow([time, i, *ep.team[k, i], ep.team_s[k, i], ep.team_d[k, i], cumulative[k, i]])
    return m
