"""The centralized team tracker (design section 6): one IMM-UKF belief over the traffic from every team car's
detections, with association, existence, track management and the replay of late measurements.

API:
- TeamTracker(cfg, road, params, specs, car_height=1.45, lanes=None, *, cycle=0.05, hold=0.0, record_nis=True,
  clutter=None, exist_max=existence.EXIST_MAX). cfg a TrackerConfig; road a nohesi.road.Road (or its config); params
  the vehicle (every car's size, Scenario.vehicle); specs {sensor name: LidarSpec | RadarSpec | CameraSpec} (or a
  sequence of specs); lanes the road's lane count (default from the road). cycle: the fusion cycle in seconds; hold:
  scans are applied only once they are this old (0: on arrival; see Late measurements); clutter: {sensor name: false
  detections per unit measurement volume}, default existence.clutter_density of each spec.
  - process(detections, now, team, scans=()): consumes types.Detection lists that may arrive late and out of order.
    team (k, >= 4): the team cars' CG states [x, y, yaw, v] at `now` (exact, V2V), indexed by Detection.car.
    scans: ScanInfo(sensor, car, time, sensor_pose) of scans that ran, including those that saw nothing (a
    Detection list cannot say a scan was empty, and without that a car leaving a sensor's view is never missed by it).
  - belief(time) -> types.Belief: every live track predicted to `time` (the moment-matched IMM mean and covariance,
    p_exist, lane_prob from imm.lane_probability, mode_prob, last_seen, confirmed), ordered by id. Ids are unique ints,
    stable for a track's life (merging keeps the confirmed, then older, track's id). The planner reads
    belief.select(belief.confirmed).
  - nis_log: list of NIS_DTYPE record arrays, one per fusion cycle with updates; nis_records() concatenates them. One
    record per associated detection: time (the detection's), track id, kind, sensor, car, dof (the measured dimensions;
    a camera edge cut by the image border carries none) and nis, the detection's NIS against its track's predicted
    measurement, moment-matched over the IMM models (step 2; chi-square with dof degrees of freedom when the filter is
    consistent). Records of cycles that a late scan replays are replaced, so the log is final only for cycles older
    than history + hold.
  - stats: counters (cycles, replays, births, merges, deleted, late_dropped).
  - reset(): forget everything (a new episode; also done automatically when `now` goes back).
- ScanInfo: the scan announcement above (sensor_pose (x, y, yaw, z) at `time`).
- predict_rows(means, covs, mu, dt, cfg, motions) -> IMMState: imm.imm_predict with its own dt per track (dt < 0 runs
  the motion models backward). transition_rows(rate, dt), noise_rows(cfg, dt): the closed-form IMM transition matrices
  (equal to imm.imm_transition) and models.process_noise per row.
- joint_update(state, X, wm, start, nu, Pzz, D, R) -> IMMState: the IMM update of tracks with several detections each
  (step 4); exactly imm.imm_update for one.

Fusion cycles. Detections with the same (sensor, car, meta['scan']) form one scan; scans are binned by scan time into
fusion cycles of `cycle` seconds (the simulation step), and cycles are applied in time order at their latest scan time
T. One cycle is:
1. Expected detection probability of every track for every scan (existence.expected_pd: field of view, range, the
   sensor's nominal pd, and the visible fraction of the track's box from the scan's sensor at T against the other
   likely tracks' boxes and the teammates; a radar's detections of hidden cars are left out, see expected_pd's
   through), times for a radar existence.doppler_pass of the track's box point nearest it (a car abeam of a side
   radar has no range rate, and the processing's stationary test removes the very point the radar model predicts;
   counted as misses, those scans deleted a car held alongside a team car within a few cycles). Sensors of one car
   within ORIGIN_MERGE of each other (its front lidar,
   radar and camera) share one visibility evaluation, done only for the tracks roughly in their view.
2. Candidate pairs by a coarse position gate (association.coarse_candidates). Each candidate's track is predicted to T
   (one IMM prediction per track) and its sigma points go through the detection's measurement model, each point
   first moved to the detection's own time by its own model's motion (a lidar box carries its mean beam time, up to a
   sweep before T; that offset's position process noise joins its R). Every IMM model's unscented predicted
   measurement gives the IMM gate (association.imm_gate): chi-square (cfg.gate) on the NIS of the moment-matched
   prediction, and the mixture likelihood of the informative dims, which gives both the existence update and the
   assignment cost, -log(likelihood / clutter density) (so detections of different R and dof compete fairly).
3. Global nearest neighbour per scan (association.gnn); a pair whose track and detection are in no other gated pair of
   its scan is taken directly.
4. One joint IMM update at T per assigned track with all its detections (joint_update): each detection statistically
   linearized on the gate's own sigma points and moments (no second pass through the measurement models), the
   detections fused in information form. The same as the UKF update for one detection; for several it treats their
   linearization errors as independent, as sequential UKF updates would.
   Simplifications: every scan of a cycle is gated against the tracks as they were at the cycle's start, so the gate of
   a later scan in the same cycle does not see the earlier scans' updates; a radar or camera detection off T (not in
   Tier A, whose sensors fire at the simulation steps) gets no process noise for its offset. This is what lets a cycle
   cost a handful of batched calls instead of a few per scan.
5. Existence (existence module): survival to T, then per scan a covered, unassigned track is missed with that scan's
   expected pd and an assigned one gains by its likelihood ratio against the sensor's clutter density (of the
   informative dims). The factors multiply the odds, so their order does not matter; the result is capped at
   exist_max. The misses of one cycle together lower the odds by at most a factor 1 - MISS_CAP: they are not
   independent, since every sensor shares the same predicted visibility and coverage (when an occluder pulls out a
   little earlier than its track says, a car's lidar, radar and camera all miss at once, and counted as three
   independent misses that would delete a car tracked for seconds in one cycle). A car that really vanished is still
   deleted within a few cycles.
6. Confirmation: confirm_m detections in the last confirm_n counted cycles (cycles where some sensor was expected to
   see the track, or did); confirmed stays set.
7. Births from the unassigned detections that gated with no track, except those whose rough position puts their car
   inside the box of a likely track (two cars cannot overlap: such a detection measures a tracked car its gate
   missed, typically one whose lane change its IMM is still catching up with, and a birth would split its track).
   Ordered lidar box (>= dense_points points), lidar box, radar, camera; a detection becomes a new track's seed unless
   it is compatible (3D Mahalanobis on s, d, v_s of the two single-detection states at T, and not from the same scan)
   with an earlier seed. Seeds give the initial state (design section 6 table: measured parts mapped through an
   unscented transform, speed from the lane's confirmed tracks' median +- init_speed_sigma_lane, else the nearest team
   car's +- init_speed_sigma_own, radar range rate when |cos bearing| >= radar_speed_min_cos, lateral speed
   0 +- init_vd_sigma; mu = cfg.imm_init) and the initial existence by source. The other detections are then gated
   and assigned to the new tracks as in steps 2-4.
8. Merging: two tracks are duplicates in a cycle when their positions are within each other's gate (chi-square on
   the combined position covariance) or their boxes share MERGE_OVERLAP of their area, and their speeds agree (within
   the gate on v_s plus MERGE_DV: one car's duplicates move together, cars passing each other do not); duplicates for
   merge_cycles cycles in a row become one (the confirmed, then older, survives). Deletion: existence below
   delete_exist; outside the team window (roi_behind behind the last team car to roi_ahead ahead of the first); an
   unconfirmed track not detected for confirm_n cycles (the M-of-N rule: it cannot be confirmed in its window, and a
   clutter track hidden behind a real car would otherwise never be missed); along-road sigma above MAX_SIGMA_S.

Compiled path. With numba on (perception/_jit.py) the numeric work of a cycle runs in _tracking_jit's kernels: the
predictions, the gate, the joint update, the coarse gate, coverage and visibility, existence, the lidar and radar
births, the occupied test and the assignment's cost matrices; the bookkeeping (scans, the replay history, merging,
scipy's assignment) stays here. The numpy code stays as the reference (predict_rows_numpy, joint_update_numpy, the
numpy branches of the methods) and is the path with NOHESI_JIT=0; the two agree to rounding (the tests compare them
call by call and over whole replays), and a kernel that meets a matrix only ukf.robust_cholesky or floor_eigenvalues
can take hands the call back to numpy.

Late measurements. The tracker keeps the scans and a snapshot of the tracks (with the id counter, merge counters and
NIS log length) before each cycle of the last history + hold seconds. A scan for an already applied cycle restores the
snapshot before that cycle and re-applies the buffered cycles in order, so the belief depends only on the set of scans
received, not on their delivery order. Scans older than that window are dropped (stats['late_dropped']). With hold 0,
scans are applied on arrival, and sensors with different latencies make most cycles run more than once (each run
costs a cycle); hold = the largest delivery latency applies each cycle once, complete, at the cost of that much extra
delay.
"""
from bisect import bisect_right
from math import floor
from types import SimpleNamespace
from typing import NamedTuple
import numpy as np
from ...geometry import cg_offsets
from .. import _jit
from ..geometry import car_boxes, focal, wrap
from ..types import Belief
from . import _tracking_jit as K
from . import association as A
from . import existence as E
from .imm import IMMState, MU_FLOOR, init_state, imm_combine, lane_probability
from .models import MODELS, NEAR, V_MIN, heading, lane_of, motion
from .ukf import ALPHA, BETA, KAPPA, _weights, floor_eigenvalues, sigma_points, symmetrize

__all__ = ['TeamTracker', 'ScanInfo', 'NIS_DTYPE', 'predict_rows', 'predict_rows_numpy', 'joint_update',
           'joint_update_numpy', 'transition_rows',
           'noise_rows']

KIND_ORDER = {'lidar_box': 0, 'radar': 1, 'camera_box': 2}
COUNT_PD = 0.5            # a cycle counts for confirmation when some sensor was this likely to see the track
RADAR_BIRTH_SIGMA = 0.7   # m added to a radar birth's position: the near-surface to centre step is approximate
REACH_SLACK = (0.1, 10.0)  # rad, m beyond a sensor's field of view and range where visibility is still evaluated
ORIGIN_MERGE = 0.5        # m: a car's sensors this close share one visibility evaluation
OCCLUDER_EXIST = 0.5      # tracks at least this likely to exist (or confirmed) occlude others
MERGE_OVERLAP = 0.5       # tracks whose boxes share this much of their area count as duplicates ...
MERGE_DV = 1.0            # ... and so do tracks within each other's position gate, if their speeds agree to within
                          # the gate on v_s plus this (m/s): one car's duplicates move together, cars passing do not
MISS_CAP = 0.9            # a cycle's misses lower a track's odds by at most 1 - MISS_CAP (see Fusion cycles, step 5)
MAX_SIGMA_S = 50.0        # m: a track this uncertain along the road is deleted
NIS_DTYPE = np.dtype([('time', 'f8'), ('track', 'i8'), ('kind', 'U10'), ('sensor', 'U24'), ('car', 'i8'),
                      ('dof', 'i8'), ('nis', 'f8')])
_KINDS = ('lidar_box', 'radar', 'camera_box')
_C, _WM, _WC = _weights(4, ALPHA, BETA, KAPPA)       # the track state's sigma-point scale and weights, as writable
_WM, _WC, _SC = np.array(_WM), np.array(_WC), float(np.sqrt(_C))    # arrays for the kernels
_C3, _WM3, _WC3 = _weights(3, ALPHA, BETA, KAPPA)    # the same for a lidar box's [x, y, heading] (births)
_WM3, _WC3, _SC3 = np.array(_WM3), np.array(_WC3), float(np.sqrt(_C3))


class _Gate(NamedTuple):
    """A gating pass (TeamTracker._gate): per pair ll, nis, gated, cost, its detection dj, moments mo, R (with the
    time offset's noise) and row, its track's index in trk; per track in trk the IMM prediction pred to the cycle time
    and its sigma points X (weights wm)."""
    ll: np.ndarray
    nis: np.ndarray
    gated: np.ndarray
    cost: np.ndarray
    dj: np.ndarray
    trk: np.ndarray
    row: np.ndarray
    pred: IMMState
    X: np.ndarray
    wm: np.ndarray
    mo: A.Moments
    R: np.ndarray


class ScanInfo(NamedTuple):
    """A scan that ran (with or without detections): sensor name, carrying car, scan time (meta['scan'] of its
    detections) and the sensor pose (x, y, yaw, z) at that time."""
    sensor: str
    car: int
    time: float
    sensor_pose: tuple


# ------------------------------------------------------------------------------------------------ batched predict
def transition_rows(rate, dt):
    """(P, 3, 3) IMM transition matrices over |dt| (P,), the closed form of imm.imm_transition's chain (keep -> each
    change at `rate`, change -> keep at 10 x rate): with b = 10 rate and l = 2 rate + b, P(keep -> keep) =
    b/l + (1 - b/l) e^{-l t}, and from a change the probability of keep is b/l (1 - e^{-l t}) while the difference
    between the two changes decays as e^{-b t}."""
    dt = np.abs(np.asarray(dt, float)).reshape(-1)
    out = np.zeros((len(dt), 3, 3))
    a, b = float(rate), 10*float(rate)
    if a <= 0:
        out[:] = np.eye(3)
        return out
    lam = 2*a+b
    pk, pl = b/lam, a/lam
    e2, e1 = np.exp(-lam*dt), np.exp(-b*dt)
    out[:, 0, 0] = pk+(1-pk)*e2
    out[:, 0, 1] = out[:, 0, 2] = pl*(1-e2)
    back = pk*(1-e2)
    out[:, 1, 0] = out[:, 2, 0] = back
    out[:, 1, 1] = out[:, 2, 2] = (1-back+e1)/2
    out[:, 1, 2] = out[:, 2, 1] = (1-back-e1)/2
    return out


def noise_rows(cfg, dt):
    """models.process_noise over |dt| for every row: (P, 4, 4)."""
    dt = np.abs(np.asarray(dt, float)).reshape(-1)
    a, b, c = dt**3/3, dt**2/2, dt
    ql, qd = cfg.q_long**2, cfg.q_lat**2
    Q = np.zeros((len(dt), 4, 4))
    Q[:, 0, 0], Q[:, 0, 2], Q[:, 2, 0], Q[:, 2, 2] = ql*a, ql*b, ql*b, ql*c
    Q[:, 1, 1], Q[:, 1, 3], Q[:, 3, 1], Q[:, 3, 3] = qd*a, qd*b, qd*b, qd*c
    return Q


_MOTION_ROWS = {}


def _motion_row(cfg, motions):
    """(steps (J,), the MOTION row of _tracking_jit) of motions made by models.motion with one cfg and road, else
    None (the numpy path then runs them as given). Cached per (cfg, motions) objects."""
    key = (id(cfg), tuple(id(f) for f in motions))
    got = _MOTION_ROWS.get(key)
    if got is None or got[0] is not cfg or any(a is not b for a, b in zip(got[1], motions)):
        if len(_MOTION_ROWS) > 64:
            _MOTION_ROWS.clear()
        got = _MOTION_ROWS[key] = (cfg, tuple(motions), _make_motion_row(cfg, motions))
    return got[2]


def _make_motion_row(cfg, motions):
    pars = [getattr(f, 'params', None) for f in motions]
    if not pars or any(p is None or p != pars[0] for p in pars):
        return None
    k, speed, lanes, w, half = pars[0]
    mp = np.array([k, speed, lanes, w, half, float(cfg.imm_switch), float(cfg.q_long)**2, float(cfg.q_lat)**2,
                   MU_FLOOR])
    return np.array([f.step for f in motions], np.int64), mp


def predict_rows(means, covs, mu, dt, cfg, motions):
    """imm.imm_predict of tracks (P, 3, 4), (P, 3, 4, 4), (P, 3) by their own dt (P,): mixing, then each model's
    unscented prediction (motions: the models.motion functions in MODELS order). Rows with dt = 0 come back unchanged;
    dt < 0 runs the motions backward with the transition and process noise of |dt|. Compiled
    (_tracking_jit.predict_rows) when numba is on and the motions are models.motion's, else predict_rows_numpy (also
    for a covariance without a plain Cholesky factor, which ukf.robust_cholesky handles)."""
    if _jit.ENABLED:
        mr = _motion_row(cfg, motions)
        means, covs, mu = (np.ascontiguousarray(a, float) for a in (means, covs, mu))
        if mr is not None and means.ndim == 3 and means.shape[1:] == (len(motions), 4):
            m, P, u, ok = K.predict_rows(means, covs, mu, np.ascontiguousarray(np.asarray(dt, float).reshape(-1)),
                                         mr[0], mr[1], _WM, _WC, _SC)
            if ok:
                return IMMState(m, P, u)
    return predict_rows_numpy(means, covs, mu, dt, cfg, motions)


def predict_rows_numpy(means, covs, mu, dt, cfg, motions):
    """predict_rows in numpy (the reference of the compiled kernel)."""
    dt = np.asarray(dt, float).reshape(-1)
    out = IMMState(np.array(means, float), np.array(covs, float), np.array(mu, float))
    idx = np.flatnonzero(dt != 0)
    if idx.size == 0:
        return out
    whole = idx.size == len(dt)
    m_, c_, u_, d_ = (out.means, out.covs, out.mu, dt) if whole else (out.means[idx], out.covs[idx], out.mu[idx],
                                                                       dt[idx])
    Pi = transition_rows(cfg.imm_switch, d_)
    cbar = np.maximum((u_[:, :, None]*Pi).sum(1), MU_FLOOR)
    w = u_[:, :, None]*Pi/cbar[:, None, :]                                  # mixing weights mu_{i|j}
    wt = np.swapaxes(w, 1, 2)
    m0 = wt @ m_
    dm = np.swapaxes(m_[:, :, None, :]-m0[:, None, :, :], 1, 2)           # (p, j, i, n)
    P0 = (wt @ c_.reshape(len(c_), 3, -1)).reshape(c_.shape)+np.swapaxes(dm*wt[..., None], -1, -2) @ dm
    X, wm, wc = sigma_points(m0, P0)
    k = X.shape[-2]
    dtr = np.repeat(d_, k)
    Y = np.empty_like(X)
    for j, f in enumerate(motions):
        Y[:, j] = f(X[:, j].reshape(-1, 4), dtr).reshape(X[:, j].shape)
    m = wm @ Y
    dY = Y-m[..., None, :]
    P = symmetrize(np.swapaxes(dY*wc[:, None], -1, -2) @ dY+noise_rows(cfg, d_)[:, None])
    mu_new = cbar/cbar.sum(-1, keepdims=True)
    if whole:
        return IMMState(m, P, mu_new)
    out.means[idx], out.covs[idx], out.mu[idx] = m, P, mu_new
    return out


def joint_update(state, X, wm, start, nu, Pzz, D, R):
    """IMM update of tracks with several detections each (module docstring, step 4). state: the tracks' predicted
    IMMState (U, J, ...) and X (U, J, 2n+1, n) its sigma points (weights wm); per detection, sorted by track with
    start (U,) the first of each track's: its association.moments nu, Pzz, D (P, J, ...) and R (P, M, M). Compiled
    (_tracking_jit.joint_update) when numba is on, else (and when some R* has no plain Cholesky factor)
    joint_update_numpy."""
    if _jit.ENABLED:
        c = lambda a: np.ascontiguousarray(a, float)
        m, P, mu, ok, healthy = K.joint_update(c(state.means), c(state.covs), c(state.mu), c(X), float(wm[1]),
                                               np.ascontiguousarray(start, np.int64), c(nu), c(Pzz), c(D), c(R),
                                               MU_FLOOR)
        if ok:
            return IMMState(m, P if healthy else floor_eigenvalues(P), mu)
    return joint_update_numpy(state, X, wm, start, nu, Pzz, D, R)


def joint_update_numpy(state, X, wm, start, nu, Pzz, D, R):
    """joint_update in numpy (the reference of the compiled kernel).

    Each detection is statistically linearized on its track's sigma points, z ~ zhat + D^T xi + e with
    x = mean + L xi and cov(e) = R* = R + Pzz - D^T D, and a track's detections are fused in information form in xi:
    Lam = I + sum D R*^-1 D^T, b = sum D R*^-1 nu, mean + L Lam^-1 b, cov - L (I - Lam^-1) L^T. For one detection
    that is exactly ukf.ukf_update (Woodbury); for several it treats their linearization errors as independent, as
    sequential UKF updates do, where a stacked joint update would correlate them. Each model's likelihood is the
    joint Gaussian's from the same quantities (NIS = sum nu R*^-1 nu - b Lam^-1 b, log det S = sum log det R*
    + log det Lam), and mu is reweighted as in imm.imm_update. The cost is a few (P, J, M, M) and (U, J, n, n)
    operations, whatever the number of detections per track."""
    Dt = np.swapaxes(D, -1, -2)
    L = A.cholesky(R[:, None]+Pzz-Dt @ D)
    Aw = A.forward(L, Dt)
    w = A.forward(L, nu)
    Y = np.add.reduceat(np.swapaxes(Aw, -1, -2) @ Aw, start, axis=0)
    b = np.add.reduceat((Aw*w[..., None]).sum(-2), start, axis=0)
    q = np.add.reduceat((w*w).sum(-1)+A.logdet(L), start, axis=0)
    n = X.shape[-1]
    Lc = A.cholesky(np.eye(n)+Y)
    Lci = A.forward(Lc, np.broadcast_to(np.eye(n), Lc.shape))
    inv = np.swapaxes(Lci, -1, -2) @ Lci
    xi = (inv @ b[..., None])[..., 0]
    a = np.log(np.maximum(state.mu, MU_FLOOR))-0.5*(q-(b*xi).sum(-1)+A.logdet(Lc))
    e = np.exp(a-a.max(-1, keepdims=True))
    mu = np.maximum(e/e.sum(-1, keepdims=True), MU_FLOOR)
    G = (X[..., 1:1+n, :]-X[..., :1, :])*np.sqrt(2*wm[1])            # rows: the columns of L (1 / sqrt(c))
    mean = state.means+(G*xi[..., None]).sum(-2)
    return IMMState(mean, _healthy(state.covs-np.swapaxes(G, -1, -2) @ (np.eye(n)-inv) @ G),
                    mu/mu.sum(-1, keepdims=True))


def _healthy(P):
    """P symmetrized, and eigenvalue-floored (ukf.floor_eigenvalues) only when some matrix is not positive definite:
    a batched Cholesky test costs a fraction of the eigendecomposition."""
    P = symmetrize(P)
    try:
        np.linalg.cholesky(P)
        return P
    except np.linalg.LinAlgError:
        return floor_eigenvalues(P)


def _cv(mean, cov, dt, cfg=None):
    """Constant-velocity prediction of combined states (n, 4), (n, 4, 4) by dt (n,), with cfg's process noise
    (noise_rows; none without cfg). Compiled (_tracking_jit.cv_rows) when numba is on."""
    dt = np.asarray(dt, float)
    if _jit.ENABLED:
        ql, qd = (0.0, 0.0) if cfg is None else (cfg.q_long**2, cfg.q_lat**2)
        return K.cv_rows(np.ascontiguousarray(mean, float), np.ascontiguousarray(cov, float),
                         np.ascontiguousarray(dt.reshape(-1)), float(ql), float(qd), cfg is not None)
    F = np.tile(np.eye(4), (len(dt), 1, 1))
    F[:, 0, 2] = F[:, 1, 3] = dt
    P = F @ cov @ np.swapaxes(F, -1, -2)
    return np.einsum('nij,nj->ni', F, mean), P if cfg is None else P+noise_rows(cfg, dt)


def _spec_row(spec, reach):
    """A sensor's SPEC row for _tracking_jit.coverage: its reach (existence.reach) and the numeric spec fields the
    pd models read (NaN where the spec has none)."""
    row = np.full(len(K.SPEC_FIELDS), np.nan)
    row[:2] = reach
    for i, f in enumerate(K.SPEC_FIELDS[2:], 2):
        v = getattr(spec, f, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            row[i] = float(v)
    if E.kind_of(spec) == 'camera_box':
        row[K.S_FOCAL] = focal(spec)
    return row


def _median(v):
    """np.median of a short 1-D array (the same value: the middle element, or the mean of the two middle ones),
    without np.median's per-call overhead."""
    if np.isnan(v).any():
        return float(np.median(v))
    s = sorted(v.tolist())
    h = len(s)//2
    return s[h] if len(s) % 2 else (s[h-1]+s[h])/2


def _unique_sorted(ci):
    """np.unique(ci, return_inverse=True) (values, inverse) of an index array, by run lengths when ci is already
    sorted (np.nonzero's row-major pairs are)."""
    if ci.size and np.all(ci[1:] >= ci[:-1]):
        first = np.empty(ci.size, bool)
        first[0] = True
        np.not_equal(ci[1:], ci[:-1], out=first[1:])
        return ci[first], np.cumsum(first)-1
    trk, row = np.unique(ci, return_inverse=True)
    return trk, row.reshape(-1)


def _popcount(x, bits):
    return sum((x >> k) & 1 for k in range(bits))


# ------------------------------------------------------------------------------------------------ track storage
class _Tracks:
    """The tracks as arrays: id, t (time of the filter state), means, covs, mu (IMM), p (existence at time tp),
    last_seen, hist (detection bits of the last confirm_n counted cycles, bit 0 the latest), confirmed."""
    FIELDS = ('id', 't', 'means', 'covs', 'mu', 'p', 'tp', 'last_seen', 'hist', 'confirmed')

    def __init__(self, **a):
        for k in self.FIELDS:
            setattr(self, k, a[k])

    @classmethod
    def empty(cls):
        return cls(id=np.zeros(0, int), t=np.zeros(0), means=np.zeros((0, 3, 4)), covs=np.zeros((0, 3, 4, 4)),
                   mu=np.zeros((0, 3)), p=np.zeros(0), tp=np.zeros(0), last_seen=np.zeros(0), hist=np.zeros(0, int),
                   confirmed=np.zeros(0, bool))

    def __len__(self):
        return len(self.id)

    def copy(self):
        return _Tracks(**{k: getattr(self, k).copy() for k in self.FIELDS})

    def take(self, idx):
        return _Tracks(**{k: getattr(self, k)[idx] for k in self.FIELDS})

    def extend(self, other):
        return _Tracks(**{k: np.concatenate((getattr(self, k), getattr(other, k))) for k in self.FIELDS})


class _Scan:
    """One scan: its detections in a deterministic order and the pose and time its coverage is evaluated at."""
    def __init__(self, time, sensor, car, kind, spec, dets, pose=None, pose_time=None):
        self.time, self.sensor, self.car, self.kind, self.spec = float(time), sensor, int(car), kind, spec
        self.dets = (sorted(dets, key=lambda d: (float(d.time), np.asarray(d.z, float).tobytes())) if len(dets) > 1
                     else list(dets))
        if pose is None and self.dets:
            last = max(self.dets, key=lambda d: float(d.time))
            pose, pose_time = last.sensor_pose, last.time
        self.pose = None if pose is None else tuple(float(v) for v in pose)
        self.pose_time = self.time if pose_time is None else float(pose_time)
        self.key = (self.time, KIND_ORDER[kind], self.car, sensor)

    def merged(self, other):
        pose = self.pose if self.pose is not None else other.pose
        pt = self.pose_time if self.pose is not None else other.pose_time
        return _Scan(self.time, self.sensor, self.car, self.kind, self.spec, self.dets+other.dets, pose, pt)


# ------------------------------------------------------------------------------------------------ the tracker
class TeamTracker:
    """The centralized IMM-UKF team tracker (module docstring)."""

    def __init__(self, cfg, road, params, specs, car_height=1.45, lanes=None, *, cycle=0.05, hold=0.0,
                 record_nis=True, clutter=None, exist_max=E.EXIST_MAX):
        self.cfg, self.road, self.params = cfg, road, params
        if not isinstance(specs, dict):
            specs = {s.name: s for s in specs}
        self.specs = dict(specs)
        self.car_height = float(car_height)
        rc = getattr(road, 'cfg', road)
        self.lanes = int(rc.lanes if lanes is None else lanes)
        self._half_width = rc.lanes*rc.lane_width/2
        self.cycle, self.hold = float(cycle), float(hold)
        if self.cycle <= 0 or self.hold < 0:
            raise ValueError('cycle must be > 0 and hold >= 0')
        self.record_nis, self.exist_max = bool(record_nis), float(exist_max)
        self._motions = tuple(motion(m, cfg, road) for m in MODELS)
        self._kind = {name: E.kind_of(spec) for name, spec in self.specs.items()}
        self._reach = {name: E.reach(spec, self.car_height) for name, spec in self.specs.items()}
        clutter = dict(clutter or {})
        self._log_clutter = {name: float(np.log(clutter.get(name, E.clutter_density(spec))))
                             for name, spec in self.specs.items()}
        self._thr = np.array([A.gate_threshold(float(cfg.gate), d) for d in range(5)])
        self._thr3 = A.gate_threshold(float(cfg.gate), 3)
        self._thr1 = A.gate_threshold(float(cfg.gate), 1)
        self._thr2 = A.gate_threshold(float(cfg.gate), 2)
        rear, front = cg_offsets(params)
        self._shift = (rear+front)/2
        self._mrow = _motion_row(cfg, self._motions)
        self._geom = np.array([sum(cg_offsets(params))/2, params.length/2, params.width/2, V_MIN, NEAR,
                               self.car_height])     # _tracking_jit's GEOM row (association._body's shift)
        self._cover = np.array([REACH_SLACK[0], REACH_SLACK[1], ORIGIN_MERGE, E.VIS_SAMPLES, E.STATIONARY_SIGMAS,
                                E.LIDAR_MIN_POINTS, rear, front, params.length, params.width, self.car_height])
        self._srow = {name: _spec_row(spec, self._reach[name]) for name, spec in self.specs.items()}
        self._birth = np.array([self._shift, params.length, params.width, float(cfg.init_vd_sigma)**2,
                                float(cfg.radar_speed_min_cos), RADAR_BIRTH_SIGMA**2])     # _tracking_jit's BIRTH row
        self.reset()

    # ---------------------------------------------------------------------------------------------- public
    def reset(self):
        self._tr = _Tracks.empty()
        self._next_id = 1
        self._merge = {}
        self.nis_log = []
        self._pending = {}         # cycle -> {scan key: _Scan}
        self._snaps = {}           # cycle -> snapshot before it
        self._done = None          # last applied cycle
        self._floor = None         # cycles below this are no longer kept
        self._team = []            # [(time, (k, 4) team states)]
        self._now = None
        self._cache = None
        self.stats = dict(cycles=0, replays=0, births=0, merges=0, deleted=0, late_dropped=0)

    def process(self, detections, now, team, scans=()):
        """Consume a batch of detections (any delivery order) at time `now` (module docstring)."""
        now = float(now)
        if self._now is not None and now < self._now-1e-9:
            self.reset()
        self._now = now
        self._push_team(now, team)
        c_lo = self._cycle_of(now-self.hold-self.cfg.history)
        self._floor = c_lo if self._floor is None else max(self._floor, c_lo)
        first = None
        for sc in self._group(detections, scans):
            c = self._cycle_of(sc.time)
            if c < self._floor:
                self.stats['late_dropped'] += max(len(sc.dets), 1)
                continue
            bucket = self._pending.setdefault(c, {})
            bucket[sc.key] = bucket[sc.key].merged(sc) if sc.key in bucket else sc
            first = c if first is None else min(first, c)
        if first is not None and self._done is not None and first <= self._done:
            self._restore(first)
        ready = self._cycle_of(now-self.hold) if self.hold > 0 else None
        for c in sorted(self._pending):
            if (self._done is not None and c <= self._done) or (ready is not None and c > ready):
                continue
            self._snaps[c] = self._snapshot()
            self._run_cycle(sorted(self._pending[c].values(), key=lambda s: s.key))
            self._done = c
            self._cache = None
        for c in [c for c in self._pending if c < self._floor]:
            del self._pending[c]
        for c in [c for c in self._snaps if c < self._floor]:
            del self._snaps[c]
        keep = self.cfg.history+self.hold+2*self.cycle+1.0
        while len(self._team) > 2 and self._team[1][0] < now-keep:
            self._team.pop(0)

    def belief(self, time):
        """types.Belief of every live track predicted to `time`."""
        time = float(time)
        if self._cache is not None and self._cache[0] == time:
            return self._cache[1]
        S = self._tr
        if len(S) == 0:
            b = Belief.empty(time, self.lanes)
        else:
            order = np.argsort(S.id, kind='stable')
            st = predict_rows(S.means[order], S.covs[order], S.mu[order], time-S.t[order], self.cfg, self._motions)
            m, P = imm_combine(st)
            b = Belief(time, S.id[order].copy(), m, P, E.survive(S.p[order], self.cfg, time-S.tp[order]),
                       lane_probability(st, self.road), st.mu, S.last_seen[order].copy(), S.confirmed[order].copy())
        self._cache = (time, b)
        return b

    def nis_records(self):
        """Every NIS record as one NIS_DTYPE array."""
        return np.concatenate(self.nis_log) if self.nis_log else np.zeros(0, NIS_DTYPE)

    def __len__(self):
        return len(self._tr)

    # ---------------------------------------------------------------------------------------------- bookkeeping
    def _cycle_of(self, t):
        return int(floor(t/self.cycle+1e-6))

    def _group(self, detections, scans):
        groups, infos = {}, {}
        for det in detections:
            st = det.meta.get('scan', det.time) if det.meta else det.time
            groups.setdefault((float(st), det.sensor, int(det.car)), []).append(det)
        for info in scans:
            key = (float(info.time), info.sensor, int(info.car))
            groups.setdefault(key, [])
            infos[key] = info
        out = []
        for key, dets in groups.items():
            if key[1] not in self.specs:
                raise KeyError(f'no sensor spec named {key[1]!r}')
            info = infos.get(key)
            if not dets and info is None:
                continue
            kind = self._kind[key[1]]
            if any(d.kind != kind for d in dets):
                raise ValueError(f'sensor {key[1]!r} gives {kind!r} detections')
            out.append(_Scan(key[0], key[1], key[2], kind, self.specs[key[1]], dets,
                             None if info is None else info.sensor_pose, None if info is None else info.time))
        return out

    def _snapshot(self):
        return self._tr.copy(), self._next_id, dict(self._merge), len(self.nis_log)

    def _restore(self, first):
        k0 = min(k for k in self._snaps if k >= first)
        tr, nid, merge, nlog = self._snaps[k0]
        self._tr, self._next_id, self._merge = tr.copy(), nid, dict(merge)
        del self.nis_log[nlog:]
        for k in [k for k in self._snaps if k >= first]:
            del self._snaps[k]
        self._done = first-1
        self._cache = None
        self.stats['replays'] += 1

    def _push_team(self, now, team):
        if team is None:
            return
        team = np.array(team, float).reshape(len(team), -1)
        if self._team and abs(self._team[-1][0]-now) < 1e-12:
            self._team[-1] = (now, team)
        elif not self._team or now > self._team[-1][0]:
            self._team.append((now, team))

    def _team_at(self, t):
        """Team CG states [x, y, yaw, v, ...] at t: interpolated between the V2V states received, extrapolated at
        constant speed and heading outside them. None when no team state was given."""
        if not self._team:
            return None
        times = [h[0] for h in self._team]
        k = bisect_right(times, t)
        if 0 < k < len(times):
            (ta, a), (tb, b) = self._team[k-1], self._team[k]
            if a.shape == b.shape:
                w = (t-ta)/(tb-ta)
                out = a+w*(b-a)
                out[:, 2] = a[:, 2]+w*wrap(b[:, 2]-a[:, 2])
                return out
        t0, base = self._team[min(max(k-1, 0), len(times)-1)]
        out = base.copy()
        dt = t-t0
        out[:, 0] += base[:, 3]*np.cos(base[:, 2])*dt
        out[:, 1] += base[:, 3]*np.sin(base[:, 2])*dt
        return out

    # ---------------------------------------------------------------------------------------------- one cycle
    def _run_cycle(self, scans):
        self.stats['cycles'] += 1
        cfg = self.cfg
        T = max(s.time for s in scans)
        team = self._team_at(T)
        tab = A.DetTable.build([(s.dets, s.spec, self._log_clutter[s.sensor]) for s in scans], self.params)
        D = len(tab)
        S = self._tr
        n = len(S)
        gated_det = np.zeros(D, bool)
        records = []
        mc = np.zeros((0, 4))
        if n:
            mc, Pc = imm_combine(IMMState(S.means, S.covs, S.mu))
            pd = self._coverage(scans, mc, S, team, T)
            ci, cj = A.coarse_candidates(tab, mc, Pc, S.t, cfg)
            chosen = np.zeros(0, int)
            if ci.size:
                g = self._gate(ci, cj, tab, T)
                gated_det[cj[g.gated]] = True
                chosen = self._assign(ci, cj, g.gated, g.cost, tab)
            hit_scan, hit_tr = tab.scan[cj[chosen]], ci[chosen]
            log_lr = g.ll[chosen]-tab.log_clutter[cj[chosen]] if chosen.size else np.zeros(0)
            self._existence(S, T, pd, hit_scan, hit_tr, log_lr, pd[hit_scan, hit_tr])
            if chosen.size:
                records.append(self._records(S.id[ci[chosen]], cj[chosen], tab, g.nis[chosen]))
                self._apply(g, chosen, tab, T)
            detected = np.zeros(n, bool)
            detected[hit_tr] = True
            counted = detected | (1-np.prod(1-pd, axis=0) >= COUNT_PD)
            mask = (1 << int(cfg.confirm_n))-1
            S.hist = np.where(counted, ((S.hist << 1) | detected.astype(int)) & mask, S.hist)
            S.confirmed |= _popcount(S.hist, int(cfg.confirm_n)) >= int(cfg.confirm_m)
        unused = np.flatnonzero(~gated_det & (tab.dof[:D] > 0))
        if unused.size and n:
            unused = unused[~self._occupied(tab, unused)]
        if unused.size:
            rec = self._births(tab, unused, T, team, mc, S.confirmed[:n].copy() if n else np.zeros(0, bool))
            if rec is not None:
                records.append(rec)
        if records and self.record_nis:
            self.nis_log.append(np.concatenate(records))
        self._manage(T, team)

    def _coverage(self, scans, mc, S, team, T):
        """Expected detection probability (scans, tracks) of every track for every scan: one existence.expected_pd
        call per sensor type, every scan of that type at once (compiled in one call, _tracking_jit.coverage, when
        numba is on)."""
        n = len(mc)
        pd = np.zeros((len(scans), n))
        ks = np.array([k for k, sc in enumerate(scans) if sc.pose is not None], int)
        if ks.size == 0 or n == 0:
            return pd
        if _jit.ENABLED:
            sc_ = [scans[k] for k in ks]
            tp = np.array([sc.pose_time for sc in sc_])
            at = {t: self._team_at(t) for t in np.unique(tp).tolist()}
            mate = np.zeros((len(sc_), 4))
            for q, (t, sc) in enumerate(zip(tp.tolist(), sc_)):
                if at[t] is not None and sc.car < len(at[t]):
                    mate[q] = at[t][sc.car, :4]
            likely = (S.p >= OCCLUDER_EXIST) | S.confirmed
            tm = np.zeros((0, 4)) if team is None else np.ascontiguousarray(team[:, :4], float)
            pd[ks] = K.coverage(np.array([sc.pose for sc in sc_]), tp, np.array([sc.car for sc in sc_], np.int64),
                                np.array([KIND_ORDER[sc.kind] for sc in sc_], np.int64),
                                np.array([self._srow[sc.sensor] for sc in sc_]), mate, np.ascontiguousarray(mc),
                                np.ascontiguousarray(S.t, float), likely, tm, team is not None, float(T),
                                self._cover, self._geom)
            return pd
        poses = np.array([scans[k].pose for k in ks])
        tp = np.array([scans[k].pose_time for k in ks])
        pos = mc[None, :, :2]+mc[None, :, 2:4]*(tp[:, None]-S.t[None, :])[..., None]
        rel = pos-poses[:, None, :2]
        ang = np.arctan2(rel[..., 1], rel[..., 0])
        rng = np.hypot(rel[..., 0], rel[..., 1])
        az, los = A.wrap_angle(ang-poses[:, None, 2]), ang
        lim = np.array([self._reach[scans[k].sensor] for k in ks])
        need = (np.abs(az) <= lim[:, :1]+REACH_SLACK[0]) & (rng <= lim[:, 1:]+REACH_SLACK[1])
        v = self._visibility([scans[k].car for k in ks], poses, tp, need, mc, S, team, T)
        kinds = np.array([KIND_ORDER[scans[k].kind] for k in ks])
        for code in np.unique(kinds):
            g = np.flatnonzero(kinds == code)
            rows = E.spec_rows(tuple(scans[k].spec for k in ks[g]))
            pd[ks[g]], _ = E.expected_pd(rows, rng[g], az[g], los[g], v[g], self.params, self.car_height,
                                         through=False)
            if code == KIND_ORDER['radar']:
                pd[ks[g]] *= self._doppler(rows, [scans[k] for k in ks[g]], poses[g], tp[g], pos[g], mc,
                                           pd[ks[g]] > 0)
        return pd

    def _doppler(self, rows, scans, poses, tp, pos, mc, mask):
        """(scans, tracks) existence.doppler_pass of each track's box point nearest each radar (where mask), with the
        carrying car's speed along its heading at the scan time as the sensor's velocity (1 elsewhere)."""
        out = np.ones(mask.shape)
        q, j = np.nonzero(mask)
        if q.size == 0:
            return out
        X = np.column_stack((pos[q, j], mc[j, 2:4]))
        origin = poses[q, :2]
        rr = A.h_radar_rows(X, origin, poses[q, 2], np.zeros((len(q), 2)), self.params)
        team = {t: self._team_at(t) for t in np.unique(tp).tolist()}
        st = np.array([team[t][sc.car, :4] if team[t] is not None and sc.car < len(team[t]) else np.zeros(4)
                       for t, sc in zip(tp.tolist(), scans)])[q]
        a = poses[q, 2]+rr[:, 1]
        slope = st[:, 3]*(np.sin(a)*np.cos(st[:, 2])-np.cos(a)*np.sin(st[:, 2]))
        noise = SimpleNamespace(sigma_range_rate=rows.sigma_range_rate[q, 0],
                                sigma_azimuth=rows.sigma_azimuth[q, 0])
        out[q, j] = E.doppler_pass(noise, rr[:, 2], slope)
        return out

    def _visibility(self, cars, poses, tp, need, mc, S, team, T):
        """(scans, tracks) visible fraction of each track's box at T from each scan's sensor, against the likely
        tracks and the teammates (existence.visibility; the carrying car's own body skipped), for the tracks in need
        (scans, tracks) (roughly in the sensor's field of view; the others get 1, they are not covered anyway). A
        sensor's position at T is its offset from its car at the scan's pose time added to the car's position at T, and
        sensors of one car within ORIGIN_MERGE of each other share one evaluation (a car's front lidar, radar and
        camera)."""
        cars = np.asarray(cars, int)
        n = len(mc)
        vis = np.ones((len(cars), n))
        if team is None or not np.any(cars < len(team)):
            return vis
        k = len(team)
        at = {t: self._team_at(t) for t in np.unique(tp).tolist()}
        ok = np.flatnonzero(cars < k)
        off = poses[ok, :2]-np.array([at[t][c, :2] for t, c in zip(tp[ok].tolist(), cars[ok].tolist())])
        origin = team[cars[ok], :2]+off
        gap = np.hypot(origin[:, None, 0]-origin[None, :, 0], origin[:, None, 1]-origin[None, :, 1])
        close = ((gap <= ORIGIN_MERGE) & (cars[ok][:, None] == cars[ok][None, :])).tolist()
        group, reps = np.zeros(len(ok), int), []
        for q in range(len(ok)):
            for g, r in enumerate(reps):
                if close[r][q]:
                    group[q] = g
                    break
            else:
                group[q] = len(reps)
                reps.append(q)
        reps = np.array(reps, int)
        pos = mc[:, :2]+mc[:, 2:4]*(T-S.t)[:, None]
        boxes = np.concatenate((car_boxes(np.column_stack((pos, heading(mc))), self.params),
                                car_boxes(team[:, :3], self.params)))
        skip = np.zeros((len(reps), n+k), bool)
        skip[:, :n] = ~((S.p >= OCCLUDER_EXIST) | S.confirmed)
        skip[np.arange(len(reps)), n+cars[ok[reps]]] = True
        want = np.zeros((len(reps), n))
        np.add.at(want, group, need[ok])
        vis[ok] = np.where(need[ok], E.visibility(origin[reps], boxes, np.arange(n), skip, want > 0)[group], 1.0)
        return vis

    def _gate(self, ci, cj, tab, T):
        """The gating pass of candidate pairs (track ci, detection cj) at cycle time T: each track's IMM prediction
        to T and its sigma points, each detection's measurement model on them, moved first to the detection's own time
        by each point's own motion model (a lidar box stamped at its beam time; that offset's position process noise
        joins its R), every model's predicted measurement (association.moments) and the IMM gate
        (association.imm_gate): chi-square on the NIS of the moment-matched predicted measurement, the mixture
        log-likelihood of the informative dims (the existence update) and the cost, minus its log-likelihood ratio
        against the sensor's clutter (detections of different R and dof compare on their likelihood)."""
        cfg, S = self.cfg, self._tr
        trk, row = _unique_sorted(ci)
        pred = predict_rows(S.means[trk], S.covs[trk], S.mu[trk], T-S.t[trk], cfg, self._motions)
        dt = tab.time[cj]-T
        if _jit.ENABLED and self._mrow is not None:
            g = self._gate_jit(cj, tab, row, trk, pred, dt)
            if g is not None:
                return g
        X, wm, wc = sigma_points(pred.means, pred.covs)
        mo = A.moments(tab, cj, A.evaluate(tab, cj, X[row], self.params, self.car_height, dt, self._motions), wm, wc)
        R = tab.R_fit[cj]
        lid = np.flatnonzero((tab.kind[cj] == A.LIDAR) & (dt != 0))
        if lid.size:
            R[lid, 0, 0] += cfg.q_long**2*np.abs(dt[lid])**3/3
            R[lid, 1, 1] += cfg.q_lat**2*np.abs(dt[lid])**3/3
        ll, nis, _ = A.imm_gate(mo, R, pred.mu[row], tab, cj)
        gated = nis <= self._thr[np.minimum(tab.dof[cj], 4)]
        ll = ll+tab.ll_pad[cj]
        return _Gate(ll, nis, gated, tab.log_clutter[cj]-ll, cj, trk, row, pred, X, wm, mo, R)

    def _gate_jit(self, cj, tab, row, trk, pred, dt):
        """_gate's measurement part compiled (_tracking_jit.sigma_rows and gate_pairs); None when a matrix needs the
        numpy path's safeguards (ukf.robust_cholesky)."""
        U, J, n = pred.means.shape
        X, ok = K.sigma_rows(np.ascontiguousarray(pred.means.reshape(-1, n)),
                             np.ascontiguousarray(pred.covs.reshape(-1, n, n)), _SC)
        if not ok:
            return None
        X = X.reshape(U, J, 2*n+1, n)
        steps, mp = self._mrow
        zhat, nu, Pzz, D, R, ll, nis, ok = K.gate_pairs(
            X, np.ascontiguousarray(row, np.int64), np.ascontiguousarray(cj, np.int64), dt, pred.mu[row],
            np.ascontiguousarray(tab.kind, np.int64), tab.z, tab.R_fit, tab.dz_dv, tab.v_prior, tab.pose, tab.vs,
            tab.cam, tab.wrap, steps, mp, self._geom, _WM, _WC)
        if not ok:
            return None
        gated = nis <= self._thr[np.minimum(tab.dof[cj], 4)]
        ll = ll+tab.ll_pad[cj]
        return _Gate(ll, nis, gated, tab.log_clutter[cj]-ll, cj, trk, row, pred, X, _WM, A.Moments(zhat, nu, Pzz, D),
                     R)

    @staticmethod
    def _assign(ci, cj, gated, cost, tab):
        """Indices of the chosen pairs: global nearest neighbour within each scan. A pair whose track and detection
        are in no other gated pair of its scan is its own assignment problem and is taken as it is; the others are
        solved per scan."""
        sel = np.flatnonzero(gated)
        if sel.size == 0:
            return sel
        sc = tab.scan[cj[sel]]
        key = sc*(int(ci.max())+1)+ci[sel]                  # (scan, track): pairs sharing one, and a detection
        free = (np.bincount(key)[key] == 1) & (np.bincount(cj[sel])[cj[sel]] == 1)
        out = [sel[free]]
        rest, sc = sel[~free], sc[~free]
        order = np.argsort(sc, kind='stable')
        rest, sc = rest[order], sc[order]
        if _jit.ENABLED and rest.size:          # the same matrices, built compiled
            bounds = np.concatenate(([0], np.flatnonzero(np.diff(sc))+1, [rest.size]))
            C, I, shapes = K.assign_parts(ci[rest], cj[rest], cost[rest], rest, bounds)
            o = 0
            for r_, c_ in shapes.tolist():
                Cm, Im = C[o:o+r_*c_].reshape(r_, c_), I[o:o+r_*c_].reshape(r_, c_)
                r, c = A.gnn(Cm)
                out.append(Im[r, c])
                o += r_*c_
            return np.sort(np.concatenate(out))
        for part in np.split(rest, np.flatnonzero(np.diff(sc))+1) if rest.size else ():
            ui, uj = np.unique(ci[part]), np.unique(cj[part])
            ri, rj = np.searchsorted(ui, ci[part]), np.searchsorted(uj, cj[part])     # np.unique's inverse
            C = np.full((len(uj), len(ui)), np.inf)
            I = np.full(C.shape, -1)
            C[rj, ri], I[rj, ri] = cost[part], part
            r, c = A.gnn(C)
            out.append(I[r, c])
        return np.sort(np.concatenate(out))

    def _existence(self, S, T, pd, hit_scan, hit_tr, log_lr, hit_pd):
        """Existence of every track after the cycle at T (module docstring, step 5); compiled
        (_tracking_jit.existence) when numba is on."""
        gate = float(self.cfg.gate)
        hit = np.zeros(pd.shape, bool)
        hit[hit_scan, hit_tr] = True
        if _jit.ENABLED:
            ep = np.array([gate, np.log1p(-MISS_CAP), float(self.cfg.exist_survive), E.EXIST_MIN, self.exist_max,
                           E.PD_DETECTED_MIN])
            S.p = K.existence(np.ascontiguousarray(S.p, float), np.ascontiguousarray(T-S.tp, float), pd, hit,
                              np.ascontiguousarray(hit_tr, np.int64), np.ascontiguousarray(log_lr, float),
                              np.ascontiguousarray(hit_pd, float), ep)
            S.tp = np.full(len(S), T)
            return
        L = np.maximum(np.where((pd > 0) & ~hit, E.miss_log_factor(pd, gate), 0.0).sum(0), np.log1p(-MISS_CAP))
        np.add.at(L, hit_tr, E.hit_log_factor(hit_pd, gate, log_lr))
        S.p = E.apply_log_odds(E.survive(S.p, self.cfg, T-S.tp), L, self.exist_max)
        S.tp = np.full(len(S), T)

    def _apply(self, g, pick, tab, T):
        """One joint IMM update (joint_update) at the cycle time T of every track with detections among the gating
        pass g's pairs pick, from that pass's predictions and moments."""
        S = self._tr
        order = np.lexsort((pick, g.row[pick]))
        pick = pick[order]
        rows = g.row[pick]
        first = np.r_[True, rows[1:] != rows[:-1]]
        start = np.flatnonzero(first)
        u = rows[start]
        st = IMMState(g.pred.means[u], g.pred.covs[u], g.pred.mu[u])
        post = joint_update(st, g.X[u], g.wm, start, g.mo.nu[pick], g.mo.Pzz[pick], g.mo.D[pick], g.R[pick])
        k = g.trk[u]
        S.means[k], S.covs[k], S.mu[k], S.t[k] = post.means, post.covs, post.mu, T
        np.maximum.at(S.last_seen, g.trk[rows], tab.time[g.dj[pick]])

    def _records(self, ids, dj, tab, nis):
        """NIS_DTYPE records of the pairs (track ids, detections dj); None when the tracker keeps no NIS log."""
        if not self.record_nis:
            return None
        rec = np.zeros(len(ids), NIS_DTYPE)
        rec['time'], rec['track'], rec['dof'], rec['nis'] = tab.time[dj], ids, tab.dof[dj], nis
        rec['kind'] = [_KINDS[k] for k in tab.kind[dj]]
        rec['sensor'] = [tab.spec[s].name for s in tab.scan[dj]]
        rec['car'] = [tab.dets[j].car for j in dj]
        return rec

    # ---------------------------------------------------------------------------------------------- births
    def _occupied(self, tab, U):
        """Detections U whose rough CG position (DetTable.xy) puts their car inside the box of a likely track
        (confirmed, or p_exist >= OCCLUDER_EXIST) at the detection's time. Two cars cannot overlap, so these are
        measurements of a tracked car that its gate missed (a lane change its IMM is still catching up with), and a
        birth from them would split the track."""
        S = self._tr
        likely = np.flatnonzero(S.confirmed | (S.p >= OCCLUDER_EXIST))
        if likely.size == 0:
            return np.zeros(len(U), bool)
        if _jit.ENABLED:
            return K.occupied(tab.time[U], tab.xy[U], tab.has_xy[U], S.means[likely], S.mu[likely], S.t[likely],
                              float(self.params.length), float(self.params.width))
        m = (S.mu[likely, :, None]*S.means[likely]).sum(1)
        dt = tab.time[U][:, None]-S.t[likely][None, :]
        ds = tab.xy[U, 0][:, None]-(m[:, 0]+m[:, 2]*dt)
        dd = tab.xy[U, 1][:, None]-(m[:, 1]+m[:, 3]*dt)
        return tab.has_xy[U] & ((np.abs(ds) < self.params.length) & (np.abs(dd) < self.params.width)).any(1)

    def _births(self, tab, U, T, team, mc, confirmed):
        """New tracks from the unassigned detections (module docstring, step 7); returns their NIS records."""
        cfg = self.cfg
        kind = tab.kind[U]
        prio = np.where(kind == A.LIDAR, np.where(tab.n_points[U] >= cfg.dense_points, 0, 1),
                        np.where(kind == A.RADAR, 2, 3))
        order = np.lexsort((U, prio))
        U, prio = U[order], prio[order]
        mean, cov, ok = self._birth_states(tab, U, team, mc, confirmed)
        if team is not None:
            ok &= (mean[:, 0] >= team[:, 0].min()-cfg.roi_behind) & (mean[:, 0] <= team[:, 0].max()+cfg.roi_ahead)
        ok &= np.abs(mean[:, 1]) <= self._half_width+1.0
        U, prio, mean, cov = U[ok], prio[ok], mean[ok], cov[ok]
        if U.size == 0:
            return None
        mT, PT = _cv(mean, cov, T-tab.time[U])
        sub = np.ix_(range(len(U)), [0, 1, 2], [0, 1, 2])
        e = mT[:, None, :3]-mT[None, :, :3]
        Cs = PT[sub][:, None]+PT[sub][None, :]
        m2 = np.einsum('abi,abi->ab', e, np.linalg.solve(Cs, e[..., None])[..., 0])
        sc = tab.scan[U]
        compat = (m2 <= self._thr3) & (sc[:, None] != sc[None, :])
        seed = np.zeros(len(U), bool)
        for u in range(len(U)):
            seed[u] = not compat[u, seed].any()
        s_idx = np.flatnonzero(seed)
        k = len(s_idx)
        init = np.array([cfg.init_exist_lidar_dense, cfg.init_exist_lidar, cfg.init_exist_radar,
                         cfg.init_exist_camera])[prio[s_idx]]
        st = init_state(mean[s_idx], cov[s_idx], cfg.imm_init)
        t0 = tab.time[U[s_idx]]
        n0 = len(self._tr)
        new = _Tracks(id=np.arange(self._next_id, self._next_id+k), t=t0.copy(), means=st.means, covs=st.covs,
                      mu=st.mu, p=np.asarray(init, float), tp=np.full(k, T), last_seen=t0.copy(),
                      hist=np.ones(k, int), confirmed=np.full(k, int(cfg.confirm_m) <= 1))
        self._next_id += k
        self._tr = self._tr.extend(new)
        self.stats['births'] += k
        members = np.flatnonzero(~seed)
        if members.size == 0:
            return None
        a, b = np.nonzero(compat[np.ix_(s_idx, members)])
        if a.size == 0:
            return None
        ci, cj = n0+a, U[members[b]]
        g = self._gate(ci, cj, tab, T)
        chosen = self._assign(ci, cj, g.gated, g.cost, tab)
        if chosen.size == 0:
            return None
        S = self._tr
        L = np.zeros(len(S))
        np.add.at(L, ci[chosen], E.hit_log_factor(E.PD_DETECTED_MIN, float(cfg.gate),
                                                  g.ll[chosen]-tab.log_clutter[cj[chosen]]))
        S.p = E.apply_log_odds(S.p, L, self.exist_max)
        rec = self._records(S.id[ci[chosen]], cj[chosen], tab, g.nis[chosen])
        self._apply(g, chosen, tab, T)
        return rec

    def _birth_states(self, tab, U, team, mc, confirmed):
        """(mean (u, 4), cov (u, 4, 4), ok (u,)) of single-detection track states (design section 6 table)."""
        cfg, prm = self.cfg, self.params
        u = len(U)
        mean, cov = np.zeros((u, 4)), np.zeros((u, 4, 4))
        ok = np.ones(u, bool)
        have_v = np.zeros(u, bool)
        kind = tab.kind[U]
        if _jit.ENABLED and self._birth_jit(tab, U, kind, mean, cov, have_v):
            L = Rd = np.zeros(0, int)        # done compiled
        else:
            L, Rd = np.flatnonzero(kind == A.LIDAR), np.flatnonzero(kind == A.RADAR)
        if L.size:
            X, wm, wc = sigma_points(tab.z[U[L], :3], tab.R[U[L], :3, :3])
            Y = X[..., :2]-self._shift*np.stack((np.cos(X[..., 2]), np.sin(X[..., 2])), -1)
            mean[L, :2], cov[L, :2, :2] = _moments(Y, wm, wc)
        if Rd.size:
            j = U[Rd]
            m0 = np.column_stack((tab.z[j, :3], np.zeros(len(j))))
            P0 = np.zeros((len(j), 4, 4))
            P0[:, :3, :3] = tab.R[j, :3, :3]
            P0[:, 3, 3] = cfg.init_vd_sigma**2
            X, wm, wc = sigma_points(m0, P0)
            phi = tab.pose[j, 2][:, None]+X[..., 1]
            c, s = np.cos(phi), np.sin(phi)
            reach = X[..., 0]+np.abs(c)*prm.length/2+np.abs(s)*prm.width/2
            vs = tab.vs[j][:, None, :]
            speed = np.abs(np.cos(tab.pose[j, 2]+tab.z[j, 1])) >= cfg.radar_speed_min_cos
            cs = np.where(np.abs(c) < 1e-3, 1e-3, c)
            v_s = np.where(speed[:, None], vs[..., 0]+(X[..., 2]-(X[..., 3]-vs[..., 1])*s)/cs, 0.0)
            Y = np.stack((tab.pose[j, :2][:, None, 0]+reach*c-self._shift, tab.pose[j, :2][:, None, 1]+reach*s, v_s,
                          X[..., 3]), -1)
            m, P = _moments(Y, wm, wc)
            P[:, :2, :2] += RADAR_BIRTH_SIGMA**2*np.eye(2)
            P[~speed, 2, :] = 0.0
            P[~speed, :, 2] = 0.0
            mean[Rd], cov[Rd] = m, P
            have_v[Rd] = speed
        Cm = np.flatnonzero(kind == A.CAMERA)
        for q in Cm:
            init = tab.dets[U[q]].meta.get('init') if tab.dets[U[q]].meta else None
            if init is None:
                ok[q] = False
                continue
            mean[q, :2], cov[q, :2, :2] = np.asarray(init[0], float), np.asarray(init[1], float)
        lat = kind != A.RADAR
        cov[lat, 3, 3] = cfg.init_vd_sigma**2
        need = ~have_v
        if need.any():
            v0, s0 = self._speed_prior(mean[need, :2], team, mc, confirmed)
            mean[need, 2], cov[need, 2, 2] = v0, s0**2
        cov = symmetrize(cov)
        ok &= np.isfinite(mean).all(1) & np.isfinite(cov).all((1, 2))
        return mean, cov, ok

    def _birth_jit(self, tab, U, kind, mean, cov, have_v):
        """_birth_states' lidar and radar rows compiled (_tracking_jit.birth_rows), written into mean, cov, have_v;
        False when a covariance needs the numpy path's safeguards (nothing written)."""
        sel = np.flatnonzero((kind == A.LIDAR) | (kind == A.RADAR))
        if sel.size == 0:
            return True
        j = U[sel]
        m, P, hv, ok = K.birth_rows(np.ascontiguousarray(kind[sel], np.int64), tab.z[j], tab.R[j], tab.pose[j],
                                    tab.vs[j], self._birth, _WM3, _WC3, _SC3, _WM, _WC, _SC)
        if ok:
            mean[sel], cov[sel], have_v[sel] = m, P, hv
        return ok

    def _speed_prior(self, xy, team, mc, confirmed):
        """(v0, sigma) of the speed along the road for new tracks at xy: the median of the confirmed tracks in the
        lane, else the nearest team car's speed (design section 6)."""
        cfg = self.cfg
        lane = lane_of(xy[:, 1], self.road)
        v0 = np.zeros(len(xy))
        sig = np.full(len(xy), float(cfg.init_speed_sigma_own))
        if team is not None and len(team):
            near = np.argmin(np.abs(xy[:, 0][:, None]-team[None, :, 0]), axis=1)
            v0 = team[near, 3].astype(float)
        if len(mc) and confirmed.any():
            lanes = lane_of(mc[confirmed, 1], self.road)
            vs = mc[confirmed, 2]
            med = np.array([_median(vs[lanes == l]) if np.any(lanes == l) else np.nan for l in range(self.lanes)])
            have = np.isfinite(med[lane])
            v0[have], sig[have] = med[lane[have]], cfg.init_speed_sigma_lane
        return v0, sig

    # ---------------------------------------------------------------------------------------------- management
    def _manage(self, T, team):
        """Merging of mutual duplicates and deletion (module docstring, step 8)."""
        cfg = self.cfg
        S = self._tr
        n = len(S)
        if n == 0:
            self._merge = {}
            return
        mc, Pc = imm_combine(IMMState(S.means, S.covs, S.mu))
        mT, PT = _cv(mc, Pc, T-S.t, cfg)         # with the process noise: the deletion test reads the belief's sigma
        drop = ~(np.isfinite(mT).all(1) & np.isfinite(PT).all((1, 2)))
        merge = {}
        if n > 1:
            L, W = self.params.length, self.params.width
            i, j = np.nonzero(np.triu((np.abs(mT[:, None, 0]-mT[None, :, 0]) < L)
                                      & (np.abs(mT[:, None, 1]-mT[None, :, 1]) < W), 1))
            if i.size:
                e = mT[i, :2]-mT[j, :2]
                m2 = np.einsum('ki,ki->k', e, np.linalg.solve(PT[i, :2, :2]+PT[j, :2, :2], e[..., None])[..., 0])
                share = (1-np.abs(e[:, 0])/L)*(1-np.abs(e[:, 1])/W)
                dv = mT[i, 2]-mT[j, 2]
                pace = dv*dv <= self._thr1*(PT[i, 2, 2]+PT[j, 2, 2])+MERGE_DV**2
                same = ((m2 <= self._thr2) | (share >= MERGE_OVERLAP)) & pace
                for a, b in zip(i[same], j[same]):
                    key = (int(min(S.id[a], S.id[b])), int(max(S.id[a], S.id[b])))
                    merge[key] = self._merge.get(key, 0)+1
        index = {int(k): n_ for n_, k in enumerate(S.id)}
        for key in sorted(k for k, v in merge.items() if v >= cfg.merge_cycles):
            a, b = index[key[0]], index[key[1]]
            if drop[a] or drop[b]:
                continue
            keep, lose = (b, a) if (S.confirmed[b] and not S.confirmed[a]) else (a, b)
            S.p[keep] = max(S.p[keep], S.p[lose])
            S.last_seen[keep] = max(S.last_seen[keep], S.last_seen[lose])
            S.confirmed[keep] |= S.confirmed[lose]
            S.hist[keep] |= S.hist[lose]
            drop[lose] = True
            self.stats['merges'] += 1
        drop |= (S.p < cfg.delete_exist) | (PT[:, 0, 0] > MAX_SIGMA_S**2)
        drop |= ~S.confirmed & (T-S.last_seen > int(cfg.confirm_n)*self.cycle+1e-9)
        if team is not None and len(team):
            drop |= (mT[:, 0] < team[:, 0].min()-cfg.roi_behind) | (mT[:, 0] > team[:, 0].max()+cfg.roi_ahead)
        if drop.any():
            gone = set(int(k) for k in S.id[drop])
            merge = {k: v for k, v in merge.items() if k[0] not in gone and k[1] not in gone}
            self.stats['deleted'] += int(drop.sum())
            self._tr = S.take(~drop)
        self._merge = merge


def _moments(Y, wm, wc):
    """Weighted mean (b, n) and covariance (b, n, n) of sigma-point images Y (b, k, n)."""
    m = wm @ Y
    dY = Y-m[:, None, :]
    return m, symmetrize(np.swapaxes(dY*wc[:, None], -1, -2) @ dY)
