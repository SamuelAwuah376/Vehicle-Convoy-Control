from collections import Counter
from dataclasses import asdict
import numpy as np
from ..geometry import body_centre
from ..scoring.events import PAIRS
from ..scoring.score import team_pass_multiplier


def _finite_min(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.min()) if values.size else None


def metrics(ep):
    cfg = ep.scenario.score
    individual = sum(ep.score[t].sum(axis=0) for t in ep.score)
    dt = ep.scenario.timing.dt
    planner = planner_metrics(ep)
    return {
        'scenario_version': ep.scenario.version, 'score_version': cfg.version,
        'seed': ep.seed, 'policy': ep.policy, 'end_reason': ep.end_reason,
        'duration_s': float(ep.time[-1]),
        'team_score': float(individual.sum()),
        'individual_scores': individual.tolist(),
        'score_by_term': {t: ep.score[t].sum(axis=0).tolist() for t in ep.score},
        'collisions': [asdict(c) for c in ep.collisions],
        'overtakes': len(ep.overtakes),
        'close_passes_scored': len(ep.awarded),
        'draft_passes_scored': int(sum(getattr(ep, 'draft_pass', None) or ())),
        'train_passes_scored': int(sum(getattr(ep, 'train_pass', None) or ())),
        'train_time_s': float(getattr(ep, 'train_time', 0.0)),
        'mean_speed_mps': float(ep.team[:, :, 3].mean()),
        'top_speed_time_s': [float(np.sum(ep.team[1:, i, 3] > cfg.top_speed_threshold)*dt) for i in range(3)],
        'team_pass': team_pass_metrics(ep),
        'combo': combo_metrics(ep),
        'maneuvers': maneuver_metrics(ep),
        'proximity_time_s': {f'{i}-{j}': float(np.sum(ep.score['proximity'][:, i] > 0)*dt) for i, j in PAIRS},
        'draft_time_s': [float(np.sum(ep.score['draft'][:, i] > 0)*dt) for i in range(3)] if 'draft' in ep.score else None,
        'min_clearance_traffic_m': _finite_min(ep.traffic_clearance),
        'min_clearance_teammates_m': _finite_min(ep.pair_clearance),
        'mpc_failures': planner['qp_failures'] if planner else None,
        'policy_compute_ms': {'mean': float(1e3*ep.compute_time.mean()), 'max': float(1e3*ep.compute_time.max())},
        'planner': planner,
        'scenario': asdict(ep.scenario),
        'planner_config': asdict(ep.planner) if ep.planner else None,
    }


def combo_metrics(ep):
    m = getattr(ep, 'combo_multiplier', None)
    if m is None or not len(m):
        return None
    return {'max_multiplier': float(m.max()), 'time_above_1_s': float(np.sum(m > 1)*ep.scenario.timing.dt),
            'max_count': int(ep.max_combo)}


def team_pass_metrics(ep):
    """Team multipliers of scored passes, from body-centre distances at each pass step."""
    cfg = ep.scenario.score
    m = []
    for o, _ in ep.awarded:
        c = body_centre(ep.team[int(np.argmin(np.abs(ep.time-o.time)))], ep.scenario.vehicle)
        m.append(team_pass_multiplier(o.vehicle, [np.linalg.norm(c[i]-c[j]) for i, j in PAIRS], cfg))
    m = np.array(m)
    return {'count': int(m.size), 'boosted': int(np.sum(m > 1)), 'mean_multiplier': float(m.mean()) if m.size else None,
            'max_multiplier': float(m.max()) if m.size else None}


def maneuver_metrics(ep):
    awards = getattr(ep, 'maneuvers', [])
    return {'count': dict(Counter(a['kind'] for a in awards)), 'points': float(sum(a['points'] for a in awards))}


def _gap_margins(records):
    """Predicted bumper margin (m) of accepted lane changes to the nearer car of the locked gap; open gaps excluded."""
    m = np.array([r.gap_margin for r in records if r.kind == 'accepted'])
    m = m[np.isfinite(m)]
    return {'count': int(m.size), 'min': float(m.min()), 'median': float(np.median(m))} if m.size else None


def double_lane_changes(records, lane):
    """Accepted/completed lane changes spanning two lanes. Records carry only the destination lane, so
    each vehicle's lane is tracked from `lane` (the scenario team lane), updated on 'completed' and on 'cancelled'
    (whose record lane is the lane the car keeps, e.g. the middle lane after a stalled two-lane move)."""
    last, out = {}, Counter()
    for r in records:
        if r.kind in ('accepted', 'completed'):
            if abs(r.lane-last.get(r.vehicle, lane)) == 2:
                out[r.kind] += 1
            if r.kind == 'completed':
                last[r.vehicle] = r.lane
        elif r.kind == 'cancelled':
            last[r.vehicle] = r.lane
    return {'accepted': out['accepted'], 'completed': out['completed']}


def planner_metrics(ep):
    log = ep.planner_log
    if log is None:
        return None
    kinds = Counter(r.kind for r in ep.intention_records)
    times = np.array([s[4] for s in log.solves])
    iterations = np.array([s[3] for s in log.solves])
    errors = []
    for due, i, s, d in log.predictions:
        k = int(round(due/ep.scenario.timing.dt))
        if k < len(ep.time):
            errors.append((abs(ep.team_s[k, i]-s), abs(ep.team_d[k, i]-d)))
    errors = np.array(errors).reshape(-1, 2)
    return {
        'lane_changes_accepted': kinds['accepted'], 'lane_changes_completed': kinds['completed'],
        'double_lane_changes': double_lane_changes(ep.intention_records, ep.scenario.team_lane),
        'locked_gap_margin_m': _gap_margins(ep.intention_records),
        'lane_changes_aborted': kinds['aborted'], 'lane_changes_cancelled': kinds['cancelled'],
        'masked_lane_actions': kinds['masked'],
        'rejections': dict(Counter(r.reason for r in ep.intention_records if r.kind == 'rejected')),
        'qp_solves': len(log.solves),
        'qp_failures': int(sum(s[2] != 'solved' for s in log.solves)),
        'qp_status': dict(Counter(s[2] for s in log.solves)),
        'fallbacks': dict(Counter(f[2] for f in log.fallbacks)),
        'plan_conflicts': dict(Counter(c[4] for c in log.conflicts)),
        'solve_ms': {'mean': float(1e3*times.mean()), 'p95': float(1e3*np.percentile(times, 95)),
                     'max': float(1e3*times.max())} if times.size else None,
        'iterations': {'mean': float(iterations.mean()), 'max': int(iterations.max())} if iterations.size else None,
        'plan_error_next_step_m': {'s_mean': float(errors[:, 0].mean()), 's_max': float(errors[:, 0].max()),
                            'd_mean': float(errors[:, 1].mean()), 'd_max': float(errors[:, 1].max())} if len(errors) else None,
        'missed_opportunity_s': log.missed.tolist(),
    }
