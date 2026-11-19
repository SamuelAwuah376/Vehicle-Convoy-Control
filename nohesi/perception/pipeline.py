"""The end-to-end perception pipeline (design sections 3, 5-7): the system the seam calls (perception.system).

PerceptionSystem(cfg, scenario, road, params, seed, *, record=False, hold=None) implements reset(snap) and
step(snap, accel), both returning the perceived Snapshot. Its RNG is default_rng([seed, cfg.seed_tag]), separate from
the world's, so the traffic is the same with perception on or off.

One call, at the true snapshot's time `now`:
1. The snapshot and the applied accelerations join a types.TruthHistory (what the simulators read) and the team's
   own history (each car's state is exact, its teammates' come exactly over V2V: the approved first version).
2. Every sensor of every team car fires at each of its scheduled times up to `now`: every spec at its rate, all cars
   in phase from the start of the run (a lidar's first sweep ends one period in, so a whole sweep lies within the
   history; a time within 1e-6 s of `now` is `now` itself). Each scan is simulated (nohesi.perception.sim, Tier A)
   and processed (nohesi.perception.processing) with the carrying car's EgoState and the teammates' states from the
   team history (the camera's pitch estimate is pitch_per_accel times the car's own braking); the radar scans and the
   camera frames of one time are processed in one batch each (process_radars, process_cameras: the same detections
   as one by one, for a fraction of the cost). Its detections and a
   tracking.ScanInfo (so an empty scan still counts as a miss) are queued for delivery at the scan time plus the
   sensor's latency plus cfg.v2v_latency.
3. Everything delivered by `now` goes to the tracker (TeamTracker.process with the team's states at `now`), which
   applies a fusion cycle (one sim step) once all of its data can have arrived: hold = the largest delivery latency,
   so every cycle runs once, complete, without replays. The belief is predicted to `now`.
4. The perceived snapshot: belief.perceived_snapshot of the confirmed tracks.

Modes (cfg.mode): 'sensors' as above; 'shadow' runs the same but returns the true snapshot (the beliefs are kept, with
record=True, for evaluation); 'ideal' (experiment E0) runs no simulators: every call, every traffic car gives one
perfect 'lidar_box' detection (the true body centre and heading at `now`, R = diag(IDEAL_SIGMA)^2) from the all-seeing
IDEAL_LIDAR on car 0, delivered at once (no latency, hold 0), into one tracker.

Sharing (cfg.sharing): 'team', one centralized TeamTracker fed by all three cars; 'own' (the ablation), one tracker per
car fed only by its own sensors, without the V2V latency (a car's own sensors do not use the link), and the view is
car 0's belief (all of them are in self.trackers, and recorded with record=True).

Pre-roll (design section 6, in reset): cfg.preroll seconds of synthetic history at the sim step before snap.time, every
traffic car and team car run backwards at constant velocity from snap (traffic [v, v_d]; a team car its world CG
velocity, heading held), without the world's RNG. The pipeline runs over it as over an episode (accelerations zero)
and then on snap itself, so reset returns the perceived snapshot at snap.time with the tracks confirmed during the
pre-roll; data still in flight then arrives during the episode.

Evaluation (record=True): self.beliefs, [(time, Belief)] of the view's tracker after every call (all live tracks,
confirmed or not; pre-roll calls have negative times), self.car_beliefs, [(time, (Belief, ...))] of every car's
tracker with sharing 'own', and scan_records(), one SCAN_DTYPE record per scan (raw measurement count, how many came
from real cars by the simulators' truth ids, detections after processing).

Profiling: self.profile[stage] sums the seconds spent per stage over every call ('sim_lidar', 'sim_radar',
'sim_camera', 'proc_lidar', 'proc_radar', 'proc_camera', 'ideal', 'tracker', 'snapshot'); self.calls counts the
calls; timing() gives the milliseconds per call of each stage and the total.

Processing settings the specs do not hold are module constants, with the measurements behind them: RADAR_SCALE (radar
clustering), LIDAR_REFINE and CAMERA_REFINE (the optional refinements).

Simplifications: the fusion centre gets the true team states at `now` (V2V exact); sensors fire in phase on all cars;
the teammates' states used by processing are exact at the scan's own times.
"""
from collections import deque
from dataclasses import replace
from time import perf_counter
import numpy as np
from ..geometry import body_centre
from ..world import TEAM
from .belief import perceived_snapshot
from .config import CameraSpec, LidarSpec, Mount, RadarSpec
from .geometry import sensor_pose, wrap
from .processing.camera import process_cameras
from .processing.lidar import process_lidar
from .processing.radar import process_radars
from .sim.camera import simulate_cameras
from .sim.lidar import simulate_lidar
from .sim.radar import simulate_radars
from .tracking.tracker import ScanInfo, TeamTracker
from .types import Detection, EgoState, TruthHistory

__all__ = ['PerceptionSystem', 'IDEAL_LIDAR', 'IDEAL_SIGMA', 'SCAN_DTYPE', 'STAGES', 'RADAR_SCALE', 'LIDAR_REFINE',
           'CAMERA_REFINE']

IDEAL_LIDAR = LidarSpec(name='lidar_ideal', mount=Mount(z=0.6), fov=2*np.pi, range_max=1000.0, beam_spacing=1e-5,
                        range_sigma=0.0, dropout=0.0, far_dropout=0.0, latency=0.0)
IDEAL_SIGMA = (0.01, 0.01, 1e-3)   # m, m, rad: the ideal detections' R (exact z; R only keeps the filter regular)
IDEAL_POINTS = 1000                # n_points of an ideal detection (a dense lidar box: the highest initial existence)
STAGES = ('sim_lidar', 'sim_radar', 'sim_camera', 'proc_lidar', 'proc_radar', 'proc_camera', 'ideal', 'tracker',
          'snapshot')
SCAN_DTYPE = np.dtype([('time', 'f8'), ('car', 'i8'), ('sensor', 'U24'), ('kind', 'U10'), ('raw', 'i8'),
                       ('true', 'i8'), ('detections', 'i8')])
RADAR_SCALE = (3.0, 1.5, 1.0)   # radar DBSCAN scales (along the road, across it, range rate): the simulator spreads
                                # a car's points ~1 m along its outline, and the processing default (1.5 along)
                                # splits ~9 % of cars in two; on v9 seeds 1000-1002 (10 s, shadow) this cut false
                                # confirmed tracks within 150 m by 13-27 % and id switches by about half, no misses
LIDAR_REFINE = True     # processing's optional refinements (lidar ICP, camera Gauss-Newton init), the processing
CAMERA_REFINE = True    # defaults. Both off cut the pipeline's time by ~19 % (lidar processing -29 %, camera -58 %)
                        # with no consistent change in coverage, errors or false tracks on v9 seeds 1000-1001 (10 s,
                        # shadow); kept on, as validated by their unit tests. Since the lidar refines only clusters of
                        # processing.lidar.REFINE_MIN points and the camera frames are batched, off saves ~22 % of the
                        # lidar processing (v9 seed 1001, 4 s) and little of the camera's
TEAM_HISTORY = 4.0   # s of team states kept for processing (a sweep and the latencies need well under a second)
SNAP_TOL = 1e-6      # s: a scheduled time this close to the snapshot's is the snapshot's
_LIDAR, _RADAR, _CAMERA = 0, 1, 2
_KIND = ('lidar_box', 'radar', 'camera_box')


class _TeamHistory:
    """The team's exact states over the last TEAM_HISTORY seconds, linearly interpolated (as types.TruthHistory
    interpolates the team, so processing sees exactly the motion the simulators used): per car [x, y, yaw, v, vy, r,
    ax], yaw unwrapped along the history; times outside it clamp to the nearest end."""
    def __init__(self):
        self.times, self.states = deque(), deque()
        self._cache = None

    def push(self, snap, accel):
        k = len(snap.team)
        z = lambda a: np.zeros(k) if a is None else np.asarray(a, float)
        st = np.column_stack((snap.team, z(snap.team_vy), z(snap.team_r), z(accel)))
        if self.times and snap.time <= self.times[-1]:
            self.times.clear()
            self.states.clear()
        if self.states:
            prev = self.states[-1][:, 2]
            st[:, 2] = prev+wrap(st[:, 2]-prev)
        self.times.append(float(snap.time))
        self.states.append(st)
        while len(self.times) > 2 and self.times[1] < snap.time-TEAM_HISTORY:
            self.times.popleft()
            self.states.popleft()
        self._cache = None

    def at(self, t):
        if self._cache is None:
            self._cache = np.array(self.times), np.array(self.states)
        times, states = self._cache
        k = int(np.searchsorted(times, t))
        if k <= 0:
            return states[0].copy()
        if k >= len(times):
            return states[-1].copy()
        w = (t-times[k-1])/(times[k]-times[k-1])
        return states[k-1]+w*(states[k]-states[k-1])

    def ego(self, car, t, pitch_per_accel=0.0):
        """EgoState of team car `car` at t; pitch the braking-pitch estimate pitch_per_accel x (-ax)."""
        x, y, yaw, v, vy, r, ax = (float(a) for a in self.at(t)[car])
        vx = float(np.sqrt(max(v*v-vy*vy, 0.0)))       # team v is the CG speed (DynamicPlant: hypot(vx, vy))
        return EgoState(float(t), x, y, float(wrap(yaw)), vx, vy, r, ax, pitch_per_accel*(-ax))

    def mates(self, car, t):
        """(k - 1, 3) CG states [x, y, yaw] of the other team cars at t."""
        st = self.at(t)
        return np.delete(st[:, :3], car, axis=0)


class PerceptionSystem:
    """The Tier A perception pipeline behind the seam (module docstring). cfg: perception.config.Perception; scenario
    the nohesi Scenario (the sim step); road a nohesi.road.Road; params world.plant.params (every car's size, the
    sensor mounts); seed the episode seed. record: keep the beliefs and scan counts for evaluation; hold: the
    tracker's hold in seconds (None: the largest delivery latency of its sensors)."""

    def __init__(self, cfg, scenario, road, params, seed, *, record=False, hold=None):
        self.cfg, self.scenario, self.road, self.params = cfg, scenario, road, params
        self.seed, self.record = seed, bool(record)
        self.dt = float(scenario.timing.dt)
        self.mode, self.ideal, self.shadow = cfg.mode, cfg.mode == 'ideal', cfg.mode == 'shadow'
        self.team_size = TEAM
        if self.ideal:
            self.sensors = (IDEAL_LIDAR,)
            self.own = False
        else:
            self.sensors = (*cfg.lidars, *cfg.radars, *cfg.cameras)
            self.own = cfg.sharing == 'own'
        self.link = 0.0 if self.own or self.ideal else float(cfg.v2v_latency)
        latency = max((float(s.latency) for s in self.sensors), default=0.0)+self.link
        self.hold = (0.0 if self.ideal else latency) if hold is None else float(hold)
        self.reset_state()

    # ---------------------------------------------------------------------------------------------- the protocol
    def reset(self, snap):
        """Pre-roll from the initial true snapshot (module docstring); returns the perceived snapshot at snap.time."""
        self.reset_state()
        steps = int(round(float(self.cfg.preroll)/self.dt))
        self._start = float(snap.time)-steps*self.dt
        for k in range(steps):
            self._advance(self._backward(snap, (steps-k)*self.dt), np.zeros(len(snap.team)))
        return self._advance(snap, np.zeros(len(snap.team)))

    def step(self, snap, accel):
        """The perceived snapshot after a world step to `snap` with the (k,) applied accelerations `accel` (a time
        before the last one starts over, without a pre-roll)."""
        if self._now is not None and snap.time < self._now-1e-9:
            self.reset_state()
        if self._start is None:
            self._start = float(snap.time)
        return self._advance(snap, np.asarray(accel, float))

    # ---------------------------------------------------------------------------------------------- public state
    def reset_state(self):
        """Forget everything: histories, schedules, queued detections, trackers, records and the RNG."""
        self.rng = np.random.default_rng([int(self.seed), int(self.cfg.seed_tag)])
        self.truth = TruthHistory()
        self.team = _TeamHistory()
        n = self.team_size if self.own else 1
        specs = {s.name: s for s in self.sensors}
        self.trackers = [TeamTracker(self.cfg.tracker, self.road, self.params, specs, self.cfg.car_height,
                                     cycle=self.dt, hold=self.hold, record_nis=self.record) for _ in range(n)]
        self._start = self._now = None
        self._fired = {s.name: 0 for s in self.sensors}
        self._queue = []
        self._seq = 0
        self.belief = None
        self.beliefs, self.car_beliefs, self._scans = [], [], []
        self.profile = dict.fromkeys(STAGES, 0.0)
        self.calls = 0

    @property
    def tracker(self):
        """The tracker the view comes from (the team's, or car 0's with sharing 'own')."""
        return self.trackers[0]

    def scan_records(self):
        """Every scan's SCAN_DTYPE record (record=True)."""
        return np.array(self._scans, SCAN_DTYPE) if self._scans else np.zeros(0, SCAN_DTYPE)

    def timing(self):
        """Milliseconds per call of each stage and in total."""
        n = max(self.calls, 1)
        out = {k: 1e3*v/n for k, v in self.profile.items()}
        out['total'] = sum(out.values())
        return out

    # ---------------------------------------------------------------------------------------------- one call
    def _advance(self, snap, accel):
        self.calls += 1
        now = self._now = float(snap.time)
        self.truth.push(snap, accel)
        self.team.push(snap, accel)
        if self.ideal:
            self._ideal_scan(snap)
        else:
            self._fire(now)
        tic = perf_counter()
        due = [q for q in self._queue if q[0] <= now+1e-9]
        self._queue = [q for q in self._queue if q[0] > now+1e-9]
        for i, tracker in enumerate(self.trackers):
            mine = [q for q in due if not self.own or q[2] == i]
            tracker.process([d for q in mine for d in q[3]], now, snap.team, [q[4] for q in mine])
        beliefs = tuple(t.belief(now) for t in self.trackers)
        self.profile['tracker'] += perf_counter()-tic
        self.belief = beliefs[0]
        if self.record:
            self.beliefs.append((now, beliefs[0]))
            if self.own:
                self.car_beliefs.append((now, beliefs))
        if self.shadow:
            return snap
        tic = perf_counter()
        view = perceived_snapshot(snap, beliefs[0], self.road)
        self.profile['snapshot'] += perf_counter()-tic
        return view

    def _backward(self, snap, tau):
        """snap's cars run tau seconds backwards at constant velocity (the pre-roll's synthetic history)."""
        team = np.array(snap.team, float)
        vy = np.zeros(len(team)) if snap.team_vy is None else np.asarray(snap.team_vy, float)
        vx = np.sqrt(np.maximum(team[:, 3]**2-vy**2, 0.0))
        c, s = np.cos(team[:, 2]), np.sin(team[:, 2])
        team[:, 0] -= tau*(c*vx-s*vy)
        team[:, 1] -= tau*(s*vx+c*vy)
        team_s, team_d = self.road.to_frenet(team[:, :2])
        vd = np.zeros(len(snap.traffic_s)) if snap.traffic_vd is None else np.asarray(snap.traffic_vd, float)
        return replace(snap, time=float(snap.time)-tau, team=team, team_s=team_s, team_d=team_d,
                       traffic_s=snap.traffic_s-tau*snap.traffic_v, traffic_d=snap.traffic_d-tau*vd,
                       collisions=(), done=False)

    # ---------------------------------------------------------------------------------------------- the sensors
    def _due(self, now):
        """The scheduled scans up to now: [(time, type, spec)] in time, then type and suite order."""
        out = []
        for order, spec in enumerate(self.sensors):
            period = 1.0/float(spec.rate)
            first = 1 if isinstance(spec, LidarSpec) else 0
            while True:
                n = self._fired[spec.name]+first
                t = self._start+n*period
                if t > now+SNAP_TOL:
                    break
                self._fired[spec.name] += 1
                out.append((now if abs(t-now) <= SNAP_TOL else t, _type(spec), order, spec))
        out.sort(key=lambda e: (e[0], e[1], e[2]))
        return out

    def _fire(self, now):
        events = self._due(now)
        cars = range(self.team_size)
        k = 0
        while k < len(events):
            t, kind = events[k][0], events[k][1]
            j = k
            while j < len(events) and events[j][0] == t and events[j][1] == kind:
                j += 1
            specs = [e[3] for e in events[k:j]]
            k = j
            if kind == _LIDAR:
                for car in cars:
                    for spec in specs:
                        tic = perf_counter()
                        scan = simulate_lidar(spec, car, self.truth, t, self.params, self.rng)
                        toc = perf_counter()
                        dets = process_lidar(scan, spec, lambda u, c=car: self.team.ego(c, u),
                                             lambda u, c=car: self.team.mates(c, u), self.params, self.road,
                                             refine=LIDAR_REFINE)
                        self._tally('sim_lidar', 'proc_lidar', tic, toc)
                        self._queue_scan(spec, car, float(scan.t_end), dets,
                                         int(np.count_nonzero(np.isfinite(scan.range))), -1)
            elif kind == _RADAR:
                tic = perf_counter()
                scans = simulate_radars(specs, list(cars), self.truth, t, self.params, self.rng)
                toc = perf_counter()
                self.profile['sim_radar'] += toc-tic
                jobs = [(c, s) for c in cars for s in specs]
                egos = {c: self.team.ego(c, t) for c in cars}
                mates = {c: self.team.mates(c, t) for c in cars}
                dets = process_radars(scans, [s for _, s in jobs], [egos[c] for c, _ in jobs],
                                      [mates[c] for c, _ in jobs], self.params, scale=RADAR_SCALE)
                self.profile['proc_radar'] += perf_counter()-toc
                for scan, (car, spec), d in zip(scans, jobs, dets):
                    self._queue_scan(spec, car, float(scan.time), d, len(scan.range),
                                     int(np.count_nonzero(np.asarray(scan.truth_id) >= 0)))
            else:
                tic = perf_counter()
                frames = [fr for car in cars for fr in simulate_cameras(specs, car, self.truth, t, self.params,
                                                                        self.rng, self.cfg.car_height)]
                toc = perf_counter()
                jobs = [(car, spec) for car in cars for spec in specs]
                mates = {car: self.team.mates(car, t) for car in cars}
                dets = process_cameras(frames, [s for _, s in jobs],
                                       [self.team.ego(c, t, s.pitch_per_accel) for c, s in jobs],
                                       [mates[c] for c, _ in jobs], self.params, self.cfg.car_height,
                                       refine=CAMERA_REFINE)
                self._tally('sim_camera', 'proc_camera', tic, toc)
                for frame, (car, spec), d in zip(frames, jobs, dets):
                    self._queue_scan(spec, car, float(frame.time), d, len(frame.boxes),
                                     int(np.count_nonzero(np.asarray(frame.truth_id) >= 0)))

    def _tally(self, sim, proc, tic, toc):
        self.profile[sim] += toc-tic
        self.profile[proc] += perf_counter()-toc

    def _queue_scan(self, spec, car, time, dets, raw, true):
        """Queue one scan's detections and its ScanInfo for delivery at time + latency (+ the V2V link)."""
        pose = sensor_pose(spec.mount, self.team.at(time)[car], self.params)
        info = ScanInfo(spec.name, int(car), time, pose)
        self._queue.append((time+float(spec.latency)+self.link, self._seq, int(car), dets, info))
        self._seq += 1
        if self.record:
            self._scans.append((time, car, spec.name, _KIND[_type(spec)], raw, true, len(dets)))

    def _ideal_scan(self, snap):
        """Experiment E0's perfect detections at snap.time (module docstring), delivered at once."""
        tic = perf_counter()
        t = float(snap.time)
        vd = np.zeros(len(snap.traffic_s)) if snap.traffic_vd is None else np.asarray(snap.traffic_vd, float)
        heading = np.arctan2(vd, np.maximum(snap.traffic_v, 1e-3))
        states = np.column_stack((snap.traffic_s, snap.traffic_d, heading))
        centre = body_centre(states, self.params)
        ego = self.team.ego(0, t)
        pose = sensor_pose(IDEAL_LIDAR.mount, (ego.x, ego.y, ego.yaw), self.params)
        R = np.diag(np.square(IDEAL_SIGMA))
        dets = [Detection('lidar_box', IDEAL_LIDAR.name, 0, t, np.array([x, y, h]), R.copy(), pose, ego,
                          {'n_points': IDEAL_POINTS, 'sides': 2, 'scan': t})
                for (x, y), h in zip(centre, heading)]
        self.profile['ideal'] += perf_counter()-tic
        self._queue_scan(IDEAL_LIDAR, 0, t, dets, len(dets), len(dets))


def _type(spec):
    if isinstance(spec, LidarSpec):
        return _LIDAR
    if isinstance(spec, RadarSpec):
        return _RADAR
    if isinstance(spec, CameraSpec):
        return _CAMERA
    raise TypeError(f'unknown sensor spec {type(spec).__name__}')
