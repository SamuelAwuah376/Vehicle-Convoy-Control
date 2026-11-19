"""Perception evaluation (design section 12, phase P2 acceptance): how well the team's belief matches the truth.

Everything is evaluated inside the observation window, WINDOW = 80 m behind to 250 m ahead of the team centre (the
mean of the team cars' s), on the confirmed tracks (what the planner reads), at every recorded belief time t >= 0.

- gospa(est, truth, c, p, alpha=2): GOSPA (Rahmathullah, Garcia-Fernandez and Svensson 2017) on the (s, d) positions,
  with its alpha = 2 decomposition into localization (the sum of d^p over the assigned pairs, d < c), missed and false
  (c^p / 2 per car or track left over). Its optimal assignment is also the track-to-car matching that NEES and
  identity switches use.
- match_frames(beliefs, snaps): per belief time a FRAME_DTYPE record (counts, the GOSPA parts, the NEES sums), per
  matched pair a PAIR_DTYPE record (errors, NEES of the 4-state [s, d, v_s, v_d] and of the position, per-component
  normalized errors, the time since the track was last seen, the car's true lateral speed), per false track a
  FALSE_DTYPE record (distance to the nearest true car, whether its id is ever matched in the episode), each confirmed
  track's span (first and last confirmed time) and coverage counts by offset from the team centre. A false track
  within c of a car is a duplicate (the car is matched to another track); beyond 3 c it is a ghost.
- nees(err, cov), consistency(values, dof): the 95 % chi-square tests. The average over N independent samples of a
  consistent filter lies in chi2.ppf([0.025, 0.975], N dof) / N; one sample lies in chi2.ppf([0.025, 0.975], dof)
  with probability 0.95. Successive samples of one track are correlated, so over a whole episode the bound on the
  average is far tighter than the data's real spread: summarize also gives Bar-Shalom's per-time test (the average
  over every pair of every episode at one time step, against that step's bounds) and the fraction of single samples
  inside their interval.
- nis_stats(records, gate): the tracker's NIS records (tracking.tracker.NIS_DTYPE) pooled per sensor and measured
  dimension. Every recorded NIS passed the association gate, so a consistent filter's NIS is chi-square truncated at
  the gate's quantile: the expected average and its bounds use the truncated moments (k F_{k+2}(t) / F_k(t) and
  k (k + 2) F_{k+4}(t) / F_k(t)), and the PIT histogram of F_k(nis) / gate is flat when the filter is consistent.
- id_switches(pairs, spans): identity switches (a true car's matched track id changes, the CLEAR MOT count) and track
  swaps (a track's matched car changes), over the GOSPA matches; a switch is a takeover when the old track is still
  confirmed at the switch (a duplicate or a neighbour's track won the assignment), else a rebirth (the old track was
  lost and a new one confirmed).
- DetectionCounter: per sensor, the detection probability against range after processing. Each scan's detections are
  matched one to one (Hungarian) to the true cars around the sensor: a lidar box by its centre (LIDAR_GATE), a radar
  cluster by its point's distance to a car's box (RADAR_GATE plus 3 sigma of its position), a camera box by its IoU
  with the car's projected 3D box (CAMERA_IOU). The opportunities are the traffic cars whose body centre is inside the
  sensor's field of view and nominal range (a camera's: where a car is min_height_px tall); a car is also counted as
  visible when at least VISIBLE of its facing outline is in the sensor's line of sight past the other cars. Detections
  matched to no car are false, those matched to a teammate leaked through the teammate removal.
- evaluate_episode(...) gathers one episode's results; summarize(evals) pools any number of them into a JSON-ready
  dict.

The truth here is what a metric needs and nothing reads it back into the pipeline.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from ..geometry import body_centre
from .config import CameraSpec, LidarSpec, RadarSpec
from .geometry import box_corners_3d, car_boxes, focal, project, raycast, sensor_pose, to_sensor, to_world

__all__ = ['WINDOW', 'GOSPA_C', 'GOSPA_P', 'GOSPA_ALPHA', 'OFFSET_EDGES', 'RANGE_EDGES', 'FRAME_DTYPE', 'PAIR_DTYPE',
           'FALSE_DTYPE', 'SPAN_DTYPE', 'Gospa', 'gospa', 'window_mask', 'nees', 'consistency', 'match_frames',
           'nis_stats', 'nis_summary', 'id_switches', 'DetectionCounter', 'nominal_reach', 'evaluate_episode',
           'summarize']

WINDOW = (80.0, 250.0)          # m behind and ahead of the team centre
GOSPA_C, GOSPA_P, GOSPA_ALPHA = 5.0, 2.0, 2.0
OFFSET_EDGES = (-80.0, -40.0, -10.0, 10.0, 40.0, 100.0, 175.0, 250.0)   # along-road offset bins from the team centre
RANGE_EDGES = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0)  # sensor range bins (m)
AGE_EDGES = (0.0, 0.3, 1.0, np.inf)   # s since a track was last seen
PROB = 0.95
PIT_BINS = 10
LIDAR_GATE = 3.0      # m between a lidar box centre and a car's body centre
RADAR_GATE = 1.0      # m from a radar point to a car's box, plus 3 sigma of the point's position
CAMERA_IOU = 0.3      # IoU of a camera box with a car's projected box
VISIBLE = 0.3         # fraction of a car's facing outline in sight for it to count as visible (the camera's rule)
OUTLINE_SAMPLES = 6   # outline points per box edge for the visibility test
REACH_SLACK = 20.0    # m beyond a sensor's nominal range where cars are still matching candidates
LANE_CHANGE_VD = 0.3  # m/s: a car moving across the road faster than this is changing lane (NEES split)

FRAME_DTYPE = np.dtype([('time', 'f8'), ('truth', 'i4'), ('tracks', 'i4'), ('assigned', 'i4'), ('missed', 'i4'),
                        ('false', 'i4'), ('localization', 'f8'), ('gospa', 'f8'), ('nees4', 'f8'), ('nees2', 'f8')])
PAIR_DTYPE = np.dtype([('time', 'f8'), ('track', 'i8'), ('truth', 'i8'), ('dist', 'f4'), ('offset', 'f4'),
                       ('nees4', 'f4'), ('nees2', 'f4'), ('err', 'f4', (4,)), ('z', 'f4', (4,)), ('age', 'f4'),
                       ('p_exist', 'f4'), ('vd', 'f4')])
FALSE_DTYPE = np.dtype([('time', 'f8'), ('track', 'i8'), ('nearest', 'f4'), ('offset', 'f4'), ('p_exist', 'f4'),
                        ('ever', '?')])
SPAN_DTYPE = np.dtype([('track', 'i8'), ('first', 'f8'), ('last', 'f8')])
STATE = ('s', 'd', 'v_s', 'v_d')


# ---------------------------------------------------------------------------------------------------------- GOSPA
@dataclass(frozen=True)
class Gospa:
    """GOSPA and its parts (each a p-th power contribution, so distance**p = localization + missed + false);
    est_index, truth_index: the assigned pairs (d < c)."""
    distance: float
    localization: float
    missed: float
    false: float
    est_index: np.ndarray
    truth_index: np.ndarray


def gospa(est, truth, c=GOSPA_C, p=GOSPA_P, alpha=GOSPA_ALPHA):
    """GOSPA between estimated points est (n, k) and true points truth (m, k) (module docstring). Only alpha = 2 has
    the localization / missed / false decomposition, so only it is accepted."""
    if alpha != 2:
        raise ValueError('only alpha = 2 decomposes into localization, missed and false')
    X = np.asarray(est, float).reshape(len(est), -1) if len(est) else np.zeros((0, 2))
    Y = np.asarray(truth, float).reshape(len(truth), -1) if len(truth) else np.zeros((0, 2))
    cp = float(c)**p
    rows = cols = np.zeros(0, int)
    loc = 0.0
    if len(X) and len(Y):
        D = np.linalg.norm(X[:, None, :]-Y[None, :, :], axis=2)
        cost = np.minimum(D, c)**p
        r, k = linear_sum_assignment(cost)
        ok = D[r, k] < c
        rows, cols = r[ok], k[ok]
        loc = float(np.sum(D[rows, cols]**p))
    missed, false = cp/alpha*(len(Y)-len(rows)), cp/alpha*(len(X)-len(rows))
    return Gospa(float((loc+missed+false)**(1/p)), loc, missed, false, rows, cols)


def window_mask(s, centre, window=WINDOW):
    """Positions s inside the observation window around the team centre."""
    s = np.asarray(s, float)
    return (s >= centre-window[0]) & (s <= centre+window[1])


# ---------------------------------------------------------------------------------------------------------- NEES
def nees(err, cov):
    """(n,) normalized estimation error squared err^T cov^-1 err of errors (n, k) with covariances (n, k, k)."""
    err = np.asarray(err, float)
    cov = np.asarray(cov, float)
    if len(err) == 0:
        return np.zeros(0)
    try:
        sol = np.linalg.solve(cov, err[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.einsum('nij,nj->ni', np.linalg.pinv(cov), err)
    return np.einsum('ni,ni->n', err, sol)


def consistency(values, dof, prob=PROB):
    """The chi-square consistency test of NEES or NIS samples with `dof` degrees of freedom (module docstring)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    out = {'n': n, 'dof': dof}
    if n == 0:
        return out
    a, b = (1-prob)/2, (1+prob)/2
    lo, hi = chi2.ppf([a, b], n*dof)/n
    q_lo, q_hi = chi2.ppf([a, b], dof)
    mean = float(v.mean())
    out.update(mean=mean, bounds=[float(lo), float(hi)], within=bool(lo <= mean <= hi), median=float(np.median(v)),
               median_expected=float(chi2.median(dof)), inside=float(np.mean((v >= q_lo) & (v <= q_hi))),
               inside_expected=prob, above_999=float(np.mean(v > chi2.ppf(0.999, dof))))
    return out


# ---------------------------------------------------------------------------------------------------------- matching
def _truth_index(snaps):
    return {round(float(s.time), 6): s for s in snaps}


def match_frames(beliefs, snaps, window=WINDOW, c=GOSPA_C, p=GOSPA_P, t_min=0.0, confirmed_only=True,
                 edges=OFFSET_EDGES):
    """GOSPA, matches and NEES per belief time (module docstring). beliefs: [(time, Belief)]; snaps: the true
    Snapshots (a belief without a snapshot at its time is skipped). Returns (frames FRAME_DTYPE, pairs PAIR_DTYPE,
    bins, false FALSE_DTYPE, spans SPAN_DTYPE) with bins the per-offset-bin counts {'edges', 'truth', 'matched',
    'tracks', 'false'}."""
    truth = _truth_index(snaps)
    nb = len(edges)-1
    bins = {'edges': list(edges), 'truth': np.zeros(nb, int), 'matched': np.zeros(nb, int),
            'tracks': np.zeros(nb, int), 'false': np.zeros(nb, int)}
    frames, pairs, falses, spans = [], [], [], {}
    for time, b in beliefs:
        time = float(time)
        snap = truth.get(round(time, 6))
        if time < t_min-1e-9 or snap is None:
            continue
        if confirmed_only:
            b = b.select(np.asarray(b.confirmed, bool))
        for i in np.asarray(b.id).tolist():
            spans.setdefault(i, [time, time])[1] = time
        centre = float(np.mean(snap.team_s))
        vd = np.zeros(len(snap.traffic_s)) if snap.traffic_vd is None else np.asarray(snap.traffic_vd, float)
        true = np.column_stack((snap.traffic_s, snap.traffic_d, snap.traffic_v, vd))
        tin = np.flatnonzero(window_mask(true[:, 0], centre, window))
        mean = np.asarray(b.mean, float).reshape(-1, 4)
        bin_ = np.flatnonzero(window_mask(mean[:, 0], centre, window))
        g = gospa(mean[bin_, :2], true[tin, :2], c, p)
        tr, ca = bin_[g.est_index], tin[g.truth_index]
        err = mean[tr]-true[ca]
        cov = np.asarray(b.cov, float).reshape(-1, 4, 4)[tr]
        n4, n2 = nees(err, cov), nees(err[:, :2], cov[:, :2, :2])
        sig = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-300))
        rec = np.zeros(len(tr), PAIR_DTYPE)
        rec['time'], rec['track'], rec['truth'] = time, np.asarray(b.id)[tr], np.asarray(snap.traffic_id)[ca]
        rec['dist'] = np.hypot(err[:, 0], err[:, 1])
        rec['offset'] = true[ca, 0]-centre
        rec['nees4'], rec['nees2'], rec['err'], rec['z'] = n4, n2, err, err/sig
        rec['age'] = time-np.asarray(b.last_seen, float)[tr]
        rec['p_exist'] = np.asarray(b.p_exist, float)[tr]
        rec['vd'] = true[ca, 3]
        pairs.append(rec)
        n_true, n_est, k = len(tin), len(bin_), len(tr)
        frames.append((time, n_true, n_est, k, n_true-k, n_est-k, g.localization, g.distance, float(n4.sum()),
                       float(n2.sum())))
        _bin_add(bins['truth'], true[tin, 0]-centre, edges)
        _bin_add(bins['matched'], true[ca, 0]-centre, edges)
        _bin_add(bins['tracks'], mean[bin_, 0]-centre, edges)
        unmatched = np.setdiff1d(bin_, tr)
        _bin_add(bins['false'], mean[unmatched, 0]-centre, edges)
        f = np.zeros(len(unmatched), FALSE_DTYPE)
        f['time'], f['track'] = time, np.asarray(b.id)[unmatched]
        gap = np.linalg.norm(mean[unmatched, None, :2]-true[None, :, :2], axis=2)
        f['nearest'] = gap.min(axis=1) if len(true) else np.inf
        f['offset'], f['p_exist'] = mean[unmatched, 0]-centre, np.asarray(b.p_exist, float)[unmatched]
        falses.append(f)
    frames = np.array(frames, FRAME_DTYPE) if frames else np.zeros(0, FRAME_DTYPE)
    pairs = np.concatenate(pairs) if pairs else np.zeros(0, PAIR_DTYPE)
    falses = np.concatenate(falses) if falses else np.zeros(0, FALSE_DTYPE)
    falses['ever'] = np.isin(falses['track'], pairs['track'])
    spans = np.array([(k, a, z) for k, (a, z) in spans.items()], SPAN_DTYPE) if spans else np.zeros(0, SPAN_DTYPE)
    return frames, pairs, bins, falses, spans


def _bin_add(acc, x, edges):
    k = np.searchsorted(edges, np.asarray(x, float), side='right')-1
    k = k[(k >= 0) & (k < len(acc))]
    np.add.at(acc, k, 1)


def id_switches(pairs, spans=None):
    """Identity switches and track swaps over matched pairs (PAIR_DTYPE): {'switches', 'takeovers', 'rebirths',
    'track_swaps', 'pairs', 'truths', 'tracks'}. A car that loses its track and is found again by the same track id does
    not switch. spans (SPAN_DTYPE, match_frames): a switch whose old track is confirmed until at least the switch time
    is a takeover, else a rebirth (without spans both are None)."""
    out = {'switches': 0, 'takeovers': None, 'rebirths': None, 'track_swaps': 0, 'pairs': int(len(pairs)),
           'truths': 0, 'tracks': 0}
    if len(pairs) == 0:
        if spans is not None:
            out['takeovers'] = out['rebirths'] = 0
        return out
    for key, other, name in (('truth', 'track', 'switches'), ('track', 'truth', 'track_swaps')):
        o = np.lexsort((pairs['time'], pairs[key]))
        a, b = pairs[key][o], pairs[other][o]
        sw = np.flatnonzero((a[1:] == a[:-1]) & (b[1:] != b[:-1]))
        out[name] = int(len(sw))
        if name == 'switches' and spans is not None:
            last = dict(zip(spans['track'].tolist(), spans['last'].tolist()))
            t = pairs['time'][o][sw+1]
            alive = np.array([last.get(k, -np.inf) >= u-1e-9 for k, u in zip(b[sw].tolist(), t)], bool)
            out['takeovers'], out['rebirths'] = int(alive.sum()), int((~alive).sum())
    out['truths'], out['tracks'] = int(len(np.unique(pairs['truth']))), int(len(np.unique(pairs['track'])))
    return out


# ---------------------------------------------------------------------------------------------------------- NIS
def _truncated(dof, gate):
    """Mean and variance of chi2(dof) truncated at its `gate` quantile."""
    t = chi2.ppf(gate, dof)
    m = dof*chi2.cdf(t, dof+2)/gate
    m2 = dof*(dof+2)*chi2.cdf(t, dof+4)/gate
    return float(m), float(m2-m*m)


def nis_stats(records, gate=0.99, t_min=0.0, prob=PROB, bins=PIT_BINS):
    """The tracker's NIS records (NIS_DTYPE) at times >= t_min pooled per (sensor, dof): {'sensor:dof': {'sensor',
    'kind', 'dof', 'n', 'sum', 'sumsq', 'inside', 'pit'}}; inside counts the samples in the central `prob` interval of
    the gate-truncated chi-square, pit the histogram of F_dof(nis) / gate. Additive over episodes (nis_summary)."""
    out = {}
    if records is None or len(records) == 0:
        return out
    r = records[(records['time'] >= t_min-1e-9) & (records['dof'] > 0) & np.isfinite(records['nis'])]
    keys = np.char.add(np.char.add(r['sensor'].astype(str), ':'), r['dof'].astype(str))
    a, b = (1-prob)/2, (1+prob)/2
    for key in np.unique(keys):
        sel = keys == key
        v = r['nis'][sel].astype(float)
        dof = int(r['dof'][sel][0])
        lo, hi = chi2.ppf([a*gate, b*gate], dof)
        u = np.clip(chi2.cdf(v, dof)/gate, 0.0, 1.0)
        out[str(key)] = {'sensor': str(r['sensor'][sel][0]), 'kind': str(r['kind'][sel][0]), 'dof': dof,
                         'n': int(len(v)), 'sum': float(v.sum()), 'sumsq': float(np.sum(v*v)),
                         'inside': int(np.sum((v >= lo) & (v <= hi))),
                         'pit': np.histogram(u, bins=bins, range=(0.0, 1.0))[0].astype(int).tolist()}
    return out


def _merge_nis(stats):
    out = {}
    for st in stats:
        for key, g in st.items():
            if key not in out:
                out[key] = {**g, 'pit': list(g['pit'])}
                continue
            o = out[key]
            for f in ('n', 'sum', 'sumsq', 'inside'):
                o[f] += g[f]
            o['pit'] = [x+y for x, y in zip(o['pit'], g['pit'])]
    return out


def nis_summary(stats, gate=0.99, prob=PROB):
    """Pooled NIS statistics (from nis_stats, merged) per sensor, per kind and over all: n, the average NIS and dof,
    the expected average under the gate's truncation with its `prob` bounds (normal approximation: the samples are
    many), whether the average is inside them, the ratio average / expected, the fraction inside the interval
    (expected `prob`) and the PIT histogram."""
    z = float(np.sqrt(chi2.ppf(prob, 1)))
    groups = {}
    for g in stats.values():
        for name in (f"sensor:{g['sensor']}", f"kind:{g['kind']}", 'all'):
            groups.setdefault(name, []).append(g)
    out = {}
    for name, gs in sorted(groups.items()):
        n = sum(g['n'] for g in gs)
        if n == 0:
            continue
        mom = [(g, *_truncated(g['dof'], gate)) for g in gs]
        expected = sum(g['n']*m for g, m, _ in mom)/n
        var = sum(g['n']*v for g, _, v in mom)/n**2
        mean = sum(g['sum'] for g in gs)/n
        lo, hi = expected-z*np.sqrt(var), expected+z*np.sqrt(var)
        pit = np.sum([g['pit'] for g in gs], axis=0)
        out[name] = {'n': n, 'mean': mean, 'mean_dof': sum(g['n']*g['dof'] for g in gs)/n, 'expected': expected,
                     'bounds': [float(lo), float(hi)], 'within': bool(lo <= mean <= hi), 'ratio': mean/expected,
                     'inside': sum(g['inside'] for g in gs)/n, 'inside_expected': prob,
                     'pit': (pit/max(pit.sum(), 1)).round(4).tolist()}
    return out


# ---------------------------------------------------------------------------------------------------------- detection
def nominal_reach(spec, car_height=1.45):
    """(half field of view at short range, half beyond near_range, near_range, range) of a sensor spec."""
    if isinstance(spec, LidarSpec):
        return spec.fov/2, spec.fov/2, spec.range_max, spec.range_max
    if isinstance(spec, RadarSpec):
        return spec.fov/2, spec.fov_far/2, spec.near_range, spec.range_max
    if isinstance(spec, CameraSpec):
        r = float(focal(spec)*car_height/spec.min_height_px)
        return spec.hfov/2, spec.hfov/2, r, r
    raise TypeError(f'unknown sensor spec {type(spec).__name__}')


class DetectionCounter:
    """Per-sensor detection probability against range (module docstring). specs: the suite (a sequence or {name:
    spec}); params: the vehicle (every car's size); add() one scan at a time, result() the counts (additive over
    episodes, summarize pools them)."""

    def __init__(self, specs, params, car_height=1.45, edges=RANGE_EDGES, visible=VISIBLE):
        self.specs = dict(specs) if isinstance(specs, dict) else {s.name: s for s in specs}
        self.params, self.car_height = params, float(car_height)
        self.edges, self.visible = np.asarray(edges, float), float(visible)
        self.counts = {}
        self._cache = None

    def _truth(self, truth):
        """Per-time arrays of the truth: CG states [x, y, heading] and velocities of traffic then team, their boxes
        and body centres (kept while the truth is the same: every sensor of a time step shares them)."""
        tr = np.asarray(truth.traffic, float).reshape(-1, 5)
        team = np.asarray(truth.team, float).reshape(-1, 4)
        key = (float(truth.time), tr.tobytes(), team.tobytes())
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        states = np.vstack((tr[:, :3], team[:, :3]))
        vel = np.vstack((tr[:, 3:5], np.column_stack((team[:, 3]*np.cos(team[:, 2]), team[:, 3]*np.sin(team[:, 2])))))
        data = (states, vel, car_boxes(states, self.params), body_centre(states, self.params), len(tr))
        self._cache = (key, data)
        return data

    def _counts(self, name):
        if name not in self.counts:
            nb = len(self.edges)-1
            self.counts[name] = {'kind': type(self.specs[name]).__name__, 'edges': self.edges.tolist(),
                                 'opp': np.zeros(nb, int), 'hit': np.zeros(nb, int), 'opp_visible': np.zeros(nb, int),
                                 'hit_visible': np.zeros(nb, int), 'scans': 0, 'detections': 0, 'false': 0,
                                 'mate': 0}
        return self.counts[name]

    def add(self, sensor, car, time, dets, truth):
        """One scan of `sensor` on team car `car` at `time` (a lidar's sweep end) with its Detections, against the
        truth at that time (types.TruthState)."""
        spec = self.specs[sensor]
        acc = self._counts(sensor)
        states, vel, boxes, centres, n_tr = self._truth(truth)
        own = n_tr+int(car)
        pose = sensor_pose(spec.mount, np.asarray(truth.team, float)[int(car)], self.params)
        half_near, half_far, near, reach = nominal_reach(spec, self.car_height)
        local = to_sensor(centres, pose)
        rng, az = np.hypot(local[:, 0], local[:, 1]), np.arctan2(local[:, 1], local[:, 0])
        half = np.where(rng <= near, half_near, half_far)
        cand = np.flatnonzero((rng <= reach+REACH_SLACK) & (np.arange(len(states)) != own))
        traffic = np.arange(len(states)) < n_tr
        opp = np.flatnonzero((np.abs(az) <= half) & (rng <= min(reach, self.edges[-1])) & traffic)
        hit = np.zeros(len(states), bool)
        acc['scans'] += 1
        acc['detections'] += len(dets)
        if len(dets) and len(cand):
            cost = self._cost(spec, dets, cand, states, vel, boxes, centres, float(time))
            r, k = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9))
            ok = np.isfinite(cost[r, k])
            matched = cand[k[ok]]
            hit[matched] = True
            acc['mate'] += int(np.sum(matched >= n_tr))
            acc['false'] += len(dets)-int(ok.sum())
        else:
            acc['false'] += len(dets)
        if len(opp) == 0:
            return
        vis = self._visible(pose, boxes, opp, own) >= self.visible
        k = np.searchsorted(self.edges, rng[opp], side='right')-1
        good = (k >= 0) & (k < len(self.edges)-1)
        k, h, vis = k[good], hit[opp][good], vis[good]
        np.add.at(acc['opp'], k, 1)
        np.add.at(acc['hit'], k[h], 1)
        np.add.at(acc['opp_visible'], k[vis], 1)
        np.add.at(acc['hit_visible'], k[vis & h], 1)

    def _cost(self, spec, dets, cand, states, vel, boxes, centres, time):
        """(dets, candidates) matching cost, inf outside the gate."""
        kind = dets[0].kind
        if kind == 'lidar_box':
            z = np.array([d.z[:2] for d in dets], float)
            dt = np.array([d.time for d in dets], float)-time
            c = centres[cand][None]+dt[:, None, None]*vel[cand][None]
            cost = np.linalg.norm(c-z[:, None, :], axis=2)
            return np.where(cost <= LIDAR_GATE, cost, np.inf)
        if kind == 'radar':
            pts = np.array([to_world(np.array([d.z[0]*np.cos(d.z[1]), d.z[0]*np.sin(d.z[1])]), d.sensor_pose)
                            for d in dets])
            r = np.array([d.z[0] for d in dets], float)
            cost = _box_distance(pts, states[cand], self.params)
            gate = RADAR_GATE+3*np.hypot(spec.sigma_range, r*spec.sigma_azimuth)
            return np.where(cost <= gate[:, None], cost, np.inf)
        if kind == 'camera_box':
            d0 = dets[0]
            corners = box_corners_3d(states[cand], self.params, self.car_height)
            u, v, depth = project(spec, d0.sensor_pose, corners, d0.ego.pitch if d0.ego is not None else 0.0)
            front = np.all(depth > 0.1, axis=1)
            W, H = float(spec.width_px), float(spec.height_px)
            with np.errstate(invalid='ignore'):
                box = np.column_stack((np.clip(u.min(1), 0, W), np.clip(u.max(1), 0, W), np.clip(v.min(1), 0, H),
                                       np.clip(v.max(1), 0, H)))
            z = np.array([d.z for d in dets], float)
            iou = _iou(z, box)
            iou[:, ~front] = 0.0
            return np.where(iou >= CAMERA_IOU, 1.0-iou, np.inf)
        raise ValueError(f'unknown detection kind {kind!r}')

    def _visible(self, pose, boxes, targets, own):
        """Visible fraction of each target box's facing outline (OUTLINE_SAMPLES points per edge, those nearer than
        its centre, as geometry.visible_fraction) from the sensor at pose, against every box but the carrying car's."""
        o = np.asarray(pose[:2], float)
        B = boxes[targets]
        f = np.linspace(0.0, 1.0, OUTLINE_SAMPLES, endpoint=False)[None, None, :, None]
        pts = (B[:, :, None]+f*(np.roll(B, -1, axis=1)-B)[:, :, None]).reshape(len(B), -1, 2)
        dist = np.linalg.norm(pts-o, axis=2)
        near = dist <= np.linalg.norm(B.mean(1)-o, axis=1)[:, None]+1e-9
        ti, pi = np.nonzero(near)
        if len(ti) == 0:
            return np.zeros(len(targets))
        p = pts[ti, pi]
        ang = np.arctan2(p[:, 1]-o[1], p[:, 0]-o[0])
        d = dist[ti, pi]
        r, idx = raycast(o, ang, boxes, float(d.max())+1.0, skip=(own,))
        seen = (idx == targets[ti]) | (r >= d-1e-6)
        num = np.bincount(ti, seen.astype(float), len(targets))
        den = np.bincount(ti, minlength=len(targets))
        return num/np.maximum(den, 1)

    def result(self):
        return self.counts


def _box_distance(points, states, params):
    """(n points, M cars) distance from points (n, 2) to the cars' body rectangles (CG states (M, 3)), 0 inside."""
    c = body_centre(states, params)
    rel = points[:, None, :]-c[None]
    ch, sh = np.cos(states[:, 2]), np.sin(states[:, 2])
    lx = rel[..., 0]*ch+rel[..., 1]*sh
    ly = -rel[..., 0]*sh+rel[..., 1]*ch
    dx = np.maximum(np.abs(lx)-params.length/2, 0.0)
    dy = np.maximum(np.abs(ly)-params.width/2, 0.0)
    return np.hypot(dx, dy)


def _iou(a, b):
    """(n, m) IoU of image boxes a (n, 4) and b (m, 4), [u_left, u_right, v_top, v_bottom]."""
    w = np.minimum(a[:, None, 1], b[None, :, 1])-np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3])-np.maximum(a[:, None, 2], b[None, :, 2])
    inter = np.maximum(w, 0.0)*np.maximum(h, 0.0)
    area = lambda x: np.maximum(x[:, 1]-x[:, 0], 0.0)*np.maximum(x[:, 3]-x[:, 2], 0.0)   # noqa: E731
    union = area(a)[:, None]+area(b)[None]-inter
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(union > 0, inter/union, 0.0)
    return np.nan_to_num(out)


# ---------------------------------------------------------------------------------------------------------- episodes
def evaluate_episode(beliefs, snaps, nis_records=None, detections=None, *, window=WINDOW, c=GOSPA_C, p=GOSPA_P,
                     gate=0.99, t_min=0.0):
    """One episode's evaluation: beliefs [(time, Belief)] against the true snaps, the tracker's NIS records and a
    DetectionCounter's result. Returns {'frames', 'pairs', 'bins', 'false', 'ids', 'nis', 'detection'} for
    summarize."""
    frames, pairs, bins, false, spans = match_frames(beliefs, snaps, window, c, p, t_min)
    return {'frames': frames, 'pairs': pairs, 'bins': bins, 'false': false, 'ids': id_switches(pairs, spans),
            'nis': nis_stats(nis_records, gate, t_min), 'detection': detections or {}}


def _ratio(a, b):
    return float(a/b) if b else None


def summarize(evals, gate=0.99, prob=PROB, window=WINDOW, c=GOSPA_C, p=GOSPA_P):
    """Pool episode evaluations (evaluate_episode) into one JSON-ready summary: GOSPA and its parts per frame,
    coverage and false tracks by offset, NEES (state and position: overall, per time step across episodes, by offset
    and by time since last seen, normalized error mean and spread per state component), NIS (nis_summary), identity
    switches and the per-sensor detection probability against range."""
    evals = list(evals)
    F = np.concatenate([e['frames'] for e in evals]) if evals else np.zeros(0, FRAME_DTYPE)
    P = np.concatenate([e['pairs'] for e in evals]) if evals else np.zeros(0, PAIR_DTYPE)
    out = {'episodes': len(evals), 'frames': int(len(F)), 'window': [-window[0], window[1]],
           'gospa': {'c': c, 'p': p, 'alpha': 2.0}}
    if len(F):
        nt, na = int(F['truth'].sum()), int(F['assigned'].sum())
        cp2 = c**p/2
        out['gospa'].update(
            mean=float(F['gospa'].mean()), median=float(np.median(F['gospa'])),
            localization=float(F['localization'].mean()), missed=float(cp2*F['missed'].mean()),
            false=float(cp2*F['false'].mean()), rms_position_error=float(np.sqrt(F['localization'].sum()/max(na, 1))),
            truth_per_frame=float(F['truth'].mean()), tracks_per_frame=float(F['tracks'].mean()),
            missed_per_frame=float(F['missed'].mean()), false_per_frame=float(F['false'].mean()),
            coverage=_ratio(na, nt), false_fraction=_ratio(int(F['false'].sum()), int(F['tracks'].sum())))
    edges = evals[0]['bins']['edges'] if evals else list(OFFSET_EDGES)
    acc = {k: np.sum([e['bins'][k] for e in evals], axis=0) for k in ('truth', 'matched', 'tracks', 'false')} \
        if evals else {}
    out['by_offset'] = [{'from': edges[i], 'to': edges[i+1], 'truth': int(acc['truth'][i]),
                         'matched': int(acc['matched'][i]), 'coverage': _ratio(acc['matched'][i], acc['truth'][i]),
                         'tracks': int(acc['tracks'][i]), 'false': int(acc['false'][i]),
                         'false_fraction': _ratio(acc['false'][i], acc['tracks'][i])}
                        for i in range(len(edges)-1)] if evals else []
    dt = float(np.median(np.diff(evals[0]['frames']['time']))) if evals and len(evals[0]['frames']) > 1 else 0.05
    out['false_tracks'] = _false_summary([e.get('false', np.zeros(0, FALSE_DTYPE)) for e in evals], c, dt)
    if len(P):
        out['false_tracks']['p_exist_matched_mean'] = float(P['p_exist'].mean())
    out['nees'] = _nees_summary(P, prob)
    out['nis'] = nis_summary(_merge_nis([e['nis'] for e in evals]), gate, prob)
    ids = [e['ids'] for e in evals]
    minutes = sum(i['pairs'] for i in ids)*dt/60
    split = all(i.get('takeovers') is not None for i in ids)
    out['id_switches'] = {'switches': sum(i['switches'] for i in ids),
                          'takeovers': sum(i['takeovers'] for i in ids) if split else None,
                          'rebirths': sum(i['rebirths'] for i in ids) if split else None,
                          'track_swaps': sum(i['track_swaps'] for i in ids),
                          'tracked_car_minutes': minutes,
                          'switches_per_car_minute': _ratio(sum(i['switches'] for i in ids), minutes),
                          'tracks_per_car': _ratio(sum(i['tracks'] for i in ids), sum(i['truths'] for i in ids))}
    out['detection'] = _detection_summary([e['detection'] for e in evals])
    return out


def _nees_summary(P, prob):
    out = {}
    if len(P) == 0:
        return out
    for name, col, dof in (('state', 'nees4', 4), ('position', 'nees2', 2)):
        v = P[col].astype(float)
        res = consistency(v, dof, prob)
        t = np.round(P['time'], 6)
        _, inv = np.unique(t, return_inverse=True)
        n = np.bincount(inv)
        m = np.bincount(inv, v)/n
        a, b = (1-prob)/2, (1+prob)/2
        lo, hi = chi2.ppf(a, n*dof)/n, chi2.ppf(b, n*dof)/n
        res['per_time'] = {'steps': int(len(n)), 'pairs_per_step': float(n.mean()),
                           'inside': float(np.mean((m >= lo) & (m <= hi))), 'inside_expected': prob,
                           'below': float(np.mean(m < lo)), 'above': float(np.mean(m > hi))}
        res['by_offset'] = [{'from': OFFSET_EDGES[i], 'to': OFFSET_EDGES[i+1],
                             **_short(consistency(v[(P['offset'] >= OFFSET_EDGES[i]) &
                                                    (P['offset'] < OFFSET_EDGES[i+1])], dof, prob))}
                            for i in range(len(OFFSET_EDGES)-1)]
        res['by_age'] = [{'from': AGE_EDGES[i], 'to': None if np.isinf(AGE_EDGES[i+1]) else AGE_EDGES[i+1],
                          **_short(consistency(v[(P['age'] >= AGE_EDGES[i]) & (P['age'] < AGE_EDGES[i+1])], dof,
                                               prob))}
                         for i in range(len(AGE_EDGES)-1)]
        moving = np.abs(P['vd']) > LANE_CHANGE_VD
        res['by_lane_change'] = [{'changing': bool(k), **_short(consistency(v[moving == k], dof, prob))}
                                 for k in (False, True)]
        out[name] = res
    z = P['z'].astype(float)
    e = P['err'].astype(float)
    out['components'] = {k: {'z_mean': float(z[:, i].mean()), 'z_std': float(z[:, i].std()),
                             'err_mean': float(e[:, i].mean()), 'err_rms': float(np.sqrt(np.mean(e[:, i]**2)))}
                         for i, k in enumerate(STATE)}
    return out


def _short(res):
    return {k: res[k] for k in ('n', 'mean', 'median', 'inside') if k in res}


def _false_summary(falses, c, dt):
    """False confirmed tracks by cause: per track-frame the share that are duplicates (nearest car within c, so matched
    to another track), misplaced (c to 3 c) and ghosts (beyond 3 c), the share whose id is never matched in its
    episode, and how long each never-matched id stays false (seconds; ids are per episode)."""
    n = sum(len(f) for f in falses)
    out = {'track_frames': int(n)}
    if n == 0:
        return out
    near = np.concatenate([f['nearest'] for f in falses]).astype(float)
    ever = np.concatenate([f['ever'] for f in falses])
    life = np.concatenate([np.unique(f['track'][~f['ever']], return_counts=True)[1] for f in falses])*dt
    out.update(duplicate=float(np.mean(near < c)), misplaced=float(np.mean((near >= c) & (near < 3*c))),
               ghost=float(np.mean(near >= 3*c)), never_matched=float(np.mean(~ever)),
               p_exist_mean=float(np.mean(np.concatenate([f['p_exist'] for f in falses]))),
               never_matched_ids=int(len(life)))
    if len(life):
        out.update(never_matched_life_median_s=float(np.median(life)), never_matched_life_p90_s=float(
            np.percentile(life, 90)), never_matched_life_max_s=float(life.max()))
    return out


def _detection_summary(dets):
    pooled = {}
    for d in dets:
        for name, c in d.items():
            if name not in pooled:
                pooled[name] = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in c.items()}
                continue
            o = pooled[name]
            for k in ('opp', 'hit', 'opp_visible', 'hit_visible'):
                o[k] = o[k]+c[k]
            for k in ('scans', 'detections', 'false', 'mate'):
                o[k] += c[k]
    out = {}
    for name, c in sorted(pooled.items()):
        pd = [_ratio(h, o) for h, o in zip(c['hit'], c['opp'])]
        pdv = [_ratio(h, o) for h, o in zip(c['hit_visible'], c['opp_visible'])]
        out[name] = {'kind': c['kind'], 'edges': list(c['edges']), 'opp': np.asarray(c['opp']).tolist(),
                     'pd': pd, 'opp_visible': np.asarray(c['opp_visible']).tolist(), 'pd_visible': pdv,
                     'pd_all': _ratio(int(np.sum(c['hit'])), int(np.sum(c['opp']))),
                     'pd_visible_all': _ratio(int(np.sum(c['hit_visible'])), int(np.sum(c['opp_visible']))),
                     'scans': c['scans'], 'detections_per_scan': _ratio(c['detections'], c['scans']),
                     'false_per_scan': _ratio(c['false'], c['scans']), 'mate_per_scan': _ratio(c['mate'], c['scans'])}
    return out
