"""Data passed between the perception stages (the shared contract).

Frames (geometry.py has the transforms):
- world: the road frame on the straight road, x = s along the road, y = d to the left (Road.to_xy is the identity);
- car: origin at the CG, x along the car's heading, y to its left;
- sensor: origin at the sensor, x along its boresight, y to its left, z up. Azimuth is measured in the sensor's x-y
  plane, counter-clockwise (left) positive, 0 on the boresight.

Car states are CG states, as everywhere in nohesi: the body overhangs the axles (geometry.cg_offsets), so the body
centre is the CG shifted along the heading (nohesi.geometry.body_centre). Every car, team and traffic, has the
scenario's vehicle size (Scenario.vehicle.length, width) and Perception.car_height.
"""
from collections import deque
from dataclasses import dataclass, field
import numpy as np

__all__ = ['LidarScan', 'RadarScan', 'CameraFrame', 'EgoState', 'Detection', 'DetectionBatch', 'Belief', 'TruthHistory',
           'TruthState', 'DETECTION_KINDS']


@dataclass(frozen=True)
class LidarScan:
    """One sweep. Tier A: planar beams (points None). Tier B (CARLA): points (n, 3) in the sensor frame, beams'
    azimuth/range derived from them."""
    sensor: str
    car: int                 # team index of the carrying car
    t_start: float
    t_end: float
    azimuth: np.ndarray      # (n,) sensor-frame beam angle
    range: np.ndarray        # (n,) metres, np.nan where the beam returned nothing
    time: np.ndarray         # (n,) fire time of each beam, t_start..t_end
    points: np.ndarray = None


@dataclass(frozen=True)
class RadarScan:
    """Radar detections of one scan, in the sensor frame."""
    sensor: str
    car: int
    time: float
    range: np.ndarray        # (n,)
    azimuth: np.ndarray      # (n,)
    range_rate: np.ndarray   # (n,) d(range)/dt in m/s: negative when closing
    snr: np.ndarray = None   # (n,) dB, optional
    truth_id: np.ndarray = None   # (n,) traffic id or -1 (clutter) / -2 - team index; for evaluation only, never read
                                  # by processing or tracking


@dataclass(frozen=True)
class CameraFrame:
    """2D detections of one frame: boxes [u_left, u_right, v_top, v_bottom] in pixels (u right, v down)."""
    sensor: str
    car: int
    time: float
    boxes: np.ndarray        # (n, 4)
    score: np.ndarray        # (n,)
    pitch: float = 0.0       # the body pitch the frame was taken at (rad, nose down positive): Tier A truth, used
                             # only by evaluation; processing uses EgoState.pitch (an estimate)
    truth_id: np.ndarray = None   # as RadarScan.truth_id


@dataclass(frozen=True)
class EgoState:
    """A team car's own state at `time` (CG, world frame): position, heading, body-frame velocity, yaw rate and
    longitudinal acceleration. Exact in the first version (design section 11, question 3)."""
    time: float
    x: float
    y: float
    yaw: float
    vx: float               # body forward speed
    vy: float = 0.0         # body lateral speed
    r: float = 0.0          # yaw rate
    ax: float = 0.0         # longitudinal acceleration (camera pitch)
    pitch: float = 0.0      # estimated body pitch (nose down positive)


DETECTION_KINDS = ('lidar_box', 'radar', 'camera_box')


@dataclass(frozen=True)
class Detection:
    """One processed measurement for the tracker.

    z layouts by kind:
    - 'lidar_box':  [x, y, heading] of the BODY CENTRE in the world frame; R (3, 3). meta: n_points, sides (1 or 2).
    - 'radar':      [range, azimuth, range_rate] of the cluster centre in the sensor frame (sensor_pose); R (3, 3).
                    meta: n_points.
    - 'camera_box': [u_left, u_right, v_top, v_bottom] pixels; R (4, 4). meta: 'init' = (xy (2,), P (2, 2)), the
                    single-frame geometric position of the car's CG-level ground contact (world frame) for births and
                    gating; 'score'.
    sensor_pose: (x, y, yaw, z) of the sensor in the world frame at `time`. ego: the carrying car's EgoState at
    `time` (radar range rate needs the sensor's own velocity; the camera model needs the pitch)."""
    kind: str
    sensor: str
    car: int
    time: float
    z: np.ndarray
    R: np.ndarray
    sensor_pose: tuple
    ego: EgoState
    meta: dict = field(default_factory=dict)



class DetectionBatch:
    """One scan's detections as stacked arrays: what processing hands the tracker (the pipeline's interface; the
    per-scan processing functions still return Detection lists). It is a sequence of Detections too: len, iteration
    and indexing give Detection objects equal to the list processing returns for the scan (built on first use, kept).

    Every detection has kind, sensor and car, and meta['scan'] = scan (the scan time; Detection lists of one scan share
    them). Per detection: time (n,), z (n, d), R (n, d, d), pose (n, 4) its sensor_pose, ego (n,) its EgoState (list),
    n_points (n,) int (meta 'n_points', 0 without). Kind-specific fields (None when the kind has none):
    - 'lidar_box': fitted (n,) bool (meta has the velocity model), v_prior (n, 2), dz_dv (n, 3, 2), R_fit (n, 3, 3)
      (zeros where not fitted);
    - 'camera_box': trunc_mode (n,) int (meta 'truncated': 0 none, 1 the bool True, 3 the bool False, 2 four flags),
      truncated (n, 4) bool (the flags where trunc_mode 2), has_init (n,) bool, init_xy (n, 2), init_P (n, 2, 2)
      (meta 'init', zeros without).
    make(batch) -> [Detection] builds the Detections (processing's own meta), or dets gives them; from_detections
    builds a batch from any Detection list of one scan (the tracker's path for Detection input)."""
    __slots__ = ('kind', 'sensor', 'car', 'scan', 'time', 'z', 'R', 'pose', 'ego', 'n_points', 'fitted', 'v_prior',
                 'dz_dv', 'R_fit', 'trunc_mode', 'truncated', 'has_init', 'init_xy', 'init_P', 'extra', '_make',
                 '_dets')

    def __init__(self, kind, sensor, car, scan, time, z, R, pose, ego, n_points, *, fitted=None, v_prior=None,
                 dz_dv=None, R_fit=None, trunc_mode=None, truncated=None, has_init=None, init_xy=None, init_P=None,
                 extra=None, make=None, dets=None):
        self.kind, self.sensor, self.car, self.scan = kind, sensor, car, scan
        self.time, self.z, self.R, self.pose, self.ego, self.n_points = time, z, R, pose, ego, n_points
        self.fitted, self.v_prior, self.dz_dv, self.R_fit = fitted, v_prior, dz_dv, R_fit
        self.trunc_mode, self.truncated, self.has_init, self.init_xy, self.init_P = (trunc_mode, truncated, has_init,
                                                                                    init_xy, init_P)
        self.extra = extra           # processing's own arrays for make
        self._make, self._dets = make, dets

    @classmethod
    def empty(cls, kind, sensor, car, scan):
        d = {'lidar_box': 3, 'radar': 3, 'camera_box': 4}.get(kind, 3)
        return cls(kind, sensor, car, scan, np.zeros(0), np.zeros((0, d)), np.zeros((0, d, d)), np.zeros((0, 4)), [],
                   np.zeros(0, np.int64), dets=[])

    @classmethod
    def from_detections(cls, dets, kind=None, sensor=None, car=None, scan=None):
        """The batch of Detections `dets` (one scan's; kind, sensor, car and scan from the first when not given, for an
        empty list they are required). The Detections are kept as the batch's items."""
        dets = list(dets)
        if not dets:
            return cls.empty(kind, sensor, car, scan)
        d0 = dets[0]
        kind = d0.kind if kind is None else kind
        sensor = d0.sensor if sensor is None else sensor
        car = d0.car if car is None else car
        if scan is None:
            scan = d0.meta.get('scan', d0.time) if d0.meta else d0.time
        n = len(dets)
        metas = [d.meta or {} for d in dets]
        out = cls(kind, sensor, car, scan, np.array([float(d.time) for d in dets]), np.array([d.z for d in dets], float),
                  np.array([d.R for d in dets], float), np.array([d.sensor_pose for d in dets], float),
                  [d.ego for d in dets], np.array([int(m.get('n_points', 0)) for m in metas], np.int64), dets=dets)
        if kind == 'lidar_box':
            out.fitted = np.array(['dz_dv' in m for m in metas], bool)
            out.v_prior, out.dz_dv, out.R_fit = np.zeros((n, 2)), np.zeros((n, 3, 2)), np.zeros((n, 3, 3))
            for k in np.flatnonzero(out.fitted).tolist():
                m = metas[k]
                out.v_prior[k] = np.asarray(m['v_prior'], float).reshape(2)
                out.dz_dv[k] = np.asarray(m['dz_dv'], float).reshape(3, 2)
                out.R_fit[k] = np.asarray(m['R_fit'], float).reshape(3, 3)
        elif kind == 'camera_box':
            out.trunc_mode, out.truncated = np.zeros(n, np.int64), np.zeros((n, 4), bool)
            out.has_init, out.init_xy, out.init_P = np.zeros(n, bool), np.zeros((n, 2)), np.zeros((n, 2, 2))
            for k, m in enumerate(metas):
                f = m.get('truncated')
                if f is not None:
                    if np.ndim(f) == 0:
                        out.trunc_mode[k] = 1 if bool(f) else 3
                    else:
                        out.trunc_mode[k] = 2
                        out.truncated[k] = np.asarray(f, bool).reshape(4)
                init = m.get('init')
                if init is not None:
                    out.has_init[k] = True
                    out.init_xy[k] = np.asarray(init[0], float).reshape(2)
                    out.init_P[k] = np.asarray(init[1], float).reshape(2, 2)
        return out

    def detections(self):
        """The batch's Detections (a list, built once)."""
        if self._dets is None:
            self._dets = self._make(self)
        return self._dets

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        return iter(self.detections())

    def __getitem__(self, k):
        return self.detections()[k]

@dataclass(frozen=True)
class Belief:
    """The team's belief over the traffic at `time` (tracker output; design section 7). Track state [s, d, v_s, v_d]
    of the CG in road coordinates (v_s = ds/dt, as Snapshot.traffic_v; v_d = dd/dt, as Snapshot.traffic_vd)."""
    time: float
    id: np.ndarray          # (n,) int, stable while a track lives
    mean: np.ndarray        # (n, 4)
    cov: np.ndarray         # (n, 4, 4)
    p_exist: np.ndarray     # (n,)
    lane_prob: np.ndarray   # (n, lanes): probability over the lane each car is heading to
    mode_prob: np.ndarray   # (n, 3): IMM model probabilities (lane keep, change left, change right)
    last_seen: np.ndarray   # (n,)
    confirmed: np.ndarray   # (n,) bool

    @staticmethod
    def empty(time, lanes=3):
        return Belief(time, np.zeros(0, int), np.zeros((0, 4)), np.zeros((0, 4, 4)), np.zeros(0), np.zeros((0, lanes)),
                      np.zeros((0, 3)), np.zeros(0), np.zeros(0, bool))

    def __len__(self):
        return len(self.id)

    def select(self, mask):
        return Belief(self.time, *(getattr(self, k)[mask] for k in ('id', 'mean', 'cov', 'p_exist', 'lane_prob',
                                                                    'mode_prob', 'last_seen', 'confirmed')))


@dataclass(frozen=True)
class TruthState:
    """The true state at one time (from TruthHistory.at). traffic: (N, 5) [x, y, heading, v_s, v_d] of the CG, with
    ids (N,); team: (3, 4) [x, y, yaw, v], team_vy (3,), team_r (3,), team_ax (3,)."""
    time: float
    traffic: np.ndarray
    traffic_id: np.ndarray
    team: np.ndarray
    team_vy: np.ndarray
    team_r: np.ndarray
    team_ax: np.ndarray


class TruthHistory:
    """The last few true snapshots, interpolated to any time between them (the simulators fire beams and frames at
    times between sim steps). Traffic is matched by id: a car that is new or was recycled (new id) since the earlier
    snapshot takes its later state as is. Times outside the stored window clamp to the nearest end.
    Only the sensor simulators read this; processing and tracking never see the truth.

    Each snapshot's TruthState is built once, when it is pushed, with read-only arrays (at() returns it as is at the
    ends of the window, so a caller cannot change the history through it)."""
    def __init__(self, maxlen=12):
        self.snaps = deque(maxlen=maxlen)
        self._states = deque(maxlen=maxlen)

    def push(self, snap, team_ax=None):
        ax = np.zeros(len(snap.team_s)) if team_ax is None else np.asarray(team_ax, float)
        if self.snaps and snap.time <= self.snaps[-1][0].time:
            self.snaps.clear()
            self._states.clear()
        self.snaps.append((snap, ax))
        state = self._state(snap, ax)
        for name in ('traffic', 'traffic_id', 'team', 'team_vy', 'team_r', 'team_ax'):
            a = np.array(getattr(state, name))      # a copy: the snapshot's and the caller's arrays stay writeable
            a.flags.writeable = False
            object.__setattr__(state, name, a)
        self._states.append(state)

    @staticmethod
    def _state(snap, ax):
        vd = snap.traffic_vd if snap.traffic_vd is not None else np.zeros_like(snap.traffic_d)
        heading = np.arctan2(vd, np.maximum(snap.traffic_v, 1e-3))
        traffic = np.column_stack((snap.traffic_s, snap.traffic_d, heading, snap.traffic_v, vd))
        z = lambda a: np.zeros(len(snap.team_s)) if a is None else np.asarray(a, float)
        return TruthState(snap.time, traffic, snap.traffic_id.copy(), snap.team.copy(), z(snap.team_vy), z(snap.team_r),
                          np.asarray(ax, float))

    def at(self, t):
        if not self.snaps:
            raise ValueError('empty truth history')
        snaps, states = self.snaps, self._states
        if t <= snaps[0][0].time:
            return states[0]
        if t >= snaps[-1][0].time:
            return states[-1]
        k = next(k for k in range(1, len(snaps)) if snaps[k][0].time >= t)
        a, b = states[k-1], states[k]
        w = (t-a.time)/(b.time-a.time)
        traffic = b.traffic.copy()
        if len(a.traffic_id) and len(b.traffic_id):      # each of b's cars matched by id to a's (the last of equal ids)
            order = np.argsort(a.traffic_id, kind='stable')
            ids = a.traffic_id[order]
            pos = np.searchsorted(ids, b.traffic_id, side='right')-1
            found = (pos >= 0) & (ids[np.maximum(pos, 0)] == b.traffic_id)
            m = order[pos[found]]
            traffic[found] = a.traffic[m]+w*(b.traffic[found]-a.traffic[m])
        team = a.team+w*(b.team-a.team)
        team[:, 2] = a.team[:, 2]+w*((b.team[:, 2]-a.team[:, 2]+np.pi) % (2*np.pi)-np.pi)
        lerp = lambda p, q: p+w*(q-p)
        return TruthState(t, traffic, b.traffic_id, team, lerp(a.team_vy, b.team_vy), lerp(a.team_r, b.team_r),
                          lerp(a.team_ax, b.team_ax))
