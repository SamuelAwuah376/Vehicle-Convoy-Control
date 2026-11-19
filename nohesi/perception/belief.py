"""The belief adapter (design sections 7 and 8): the tracker's Belief as the perceived Snapshot that the planner, the
intentions and the policy read, and the belief's prediction over a planning horizon.

- perceived_snapshot(truth, belief, road, confirmed_only=True) -> Snapshot: the true snapshot with its traffic replaced
  by the belief's confirmed tracks: traffic_s, traffic_d, traffic_v, traffic_vd the track means (CG, road frame),
  traffic_id the track ids, traffic_target_lane the most likely lane of lane_prob (the IMM's estimate of the lane each
  car is heading to, replacing the true intent), traffic_sig_s, traffic_sig_d, traffic_sig_v the square roots of the
  covariance diagonals and traffic_p_exist the existence probabilities. The team fields, time, collisions and done are
  the truth's (each car knows its own state, its teammates' over V2V; collisions and the episode end are physical).
- belief_prediction(belief, road, times, cfg=None, model='keep') -> (s, d, v, psi, sig_s, sig_d), each
  (len(times), n) like prediction.traffic_prediction: every track's moment-matched mean and covariance propagated
  `times` seconds ahead (offsets from belief.time) through one IMM motion model (tracking.models.motion, with its
  process noise) by the unscented transform. psi is the heading atan2(v_d, v_s) of the predicted mean. The default,
  lane keep, is the 'keeps its lane' alternative of design section 8; 'left' / 'right' give the changing one.
"""
from dataclasses import replace
import numpy as np
from .config import TrackerConfig
from .tracking.models import V_MIN, motion, process_noise
from .tracking.ukf import ukf_predict

__all__ = ['perceived_snapshot', 'belief_prediction']


def perceived_snapshot(truth, belief, road, confirmed_only=True):
    """The perceived Snapshot of `belief` (module docstring), built on the true snapshot `truth` (not modified)."""
    b = belief.select(np.asarray(belief.confirmed, bool)) if confirmed_only else belief
    n = len(b)
    mean = np.asarray(b.mean, float).reshape(n, 4)
    var = np.diagonal(np.asarray(b.cov, float).reshape(n, 4, 4), axis1=1, axis2=2)
    sig = np.sqrt(np.maximum(var, 0.0))
    lanes = road.cfg.lanes
    prob = np.asarray(b.lane_prob, float)
    if n and prob.ndim == 2 and prob.shape[1]:
        target = np.clip(np.argmax(prob, axis=1), 0, lanes-1)
    else:
        target = road.lane_of(mean[:, 1])
    return replace(truth, traffic_s=mean[:, 0].copy(), traffic_d=mean[:, 1].copy(), traffic_v=mean[:, 2].copy(),
                   traffic_id=np.asarray(b.id, int).copy(), traffic_vd=mean[:, 3].copy(),
                   traffic_target_lane=np.asarray(target, int), traffic_sig_s=sig[:, 0].copy(),
                   traffic_sig_d=sig[:, 1].copy(), traffic_sig_v=sig[:, 2].copy(),
                   traffic_p_exist=np.asarray(b.p_exist, float).copy())


def belief_prediction(belief, road, times, cfg=None, model='keep'):
    """Predicted (s, d, v, psi, sig_s, sig_d) of every track at `times` (module docstring). cfg: the TrackerConfig
    whose motion model and process noise are used (default TrackerConfig())."""
    cfg = TrackerConfig() if cfg is None else cfg
    t = np.asarray(times, float).reshape(-1)
    if np.any(t < 0):
        raise ValueError('prediction times are offsets from belief.time and must be >= 0')
    n = len(belief)
    out = np.zeros((6, len(t), n))
    if n == 0 or len(t) == 0:
        return tuple(out)
    f = motion(model, cfg, road)
    m, P = np.asarray(belief.mean, float).copy(), np.asarray(belief.cov, float).copy()
    order = np.argsort(t, kind='stable')
    prev = 0.0
    for k in order:
        dt = float(t[k])-prev
        if dt > 0:
            m, P = ukf_predict(m, P, lambda X, dt=dt: f(X, dt), process_noise(model, cfg, dt))
            prev = float(t[k])
        var = np.diagonal(P, axis1=1, axis2=2)
        out[:, k] = (m[:, 0], m[:, 1], m[:, 2], np.arctan2(m[:, 3], np.maximum(m[:, 2], V_MIN)),
                     np.sqrt(np.maximum(var[:, 0], 0.0)), np.sqrt(np.maximum(var[:, 1], 0.0)))
    return tuple(out)
