from dataclasses import asdict
import numpy as np
from ..scoring.events import PAIRS


def _finite_min(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.min()) if values.size else None


def metrics(ep):
    cfg = ep.scenario.score
    individual = sum(ep.score[t].sum(axis=0) for t in ep.score)
    dt = ep.scenario.timing.dt
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
        'proximity_time_s': {f'{i}-{j}': float(np.sum(ep.score['proximity'][:, i] > 0)*dt) for i, j in PAIRS},
        'min_clearance_traffic_m': _finite_min(ep.traffic_clearance),
        'min_clearance_teammates_m': _finite_min(ep.pair_clearance),
        'mpc_failures': None,   # no MPC in milestone 1
        'policy_compute_ms': {'mean': float(1e3*ep.compute_time.mean()), 'max': float(1e3*ep.compute_time.max())},
        'scenario': asdict(ep.scenario),
    }
