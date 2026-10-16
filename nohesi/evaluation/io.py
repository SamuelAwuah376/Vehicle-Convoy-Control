import csv
import json
from dataclasses import asdict
import numpy as np
from .metrics import metrics


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
