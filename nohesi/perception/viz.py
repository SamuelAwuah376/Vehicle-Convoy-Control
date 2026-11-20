"""Viewer overlay for perception (design phase P1): what the team's sensors saw and what the tracker believes, drawn
over the true traffic, so misses, false tracks and the uncertainty show at a glance.

A frame has two panels in the road frame (s along, d across; the d axis is stretched so covariances are readable,
and everything is drawn in data coordinates, so shapes stay exact): the tracker's window around the team
(TrackerConfig.roi_behind behind the last car to roi_ahead beyond the first) and a zoom around the team. Each shows
- the team's sensor fields of view as light wedges, tinted by modality (FOV_COLORS): per modality the union of every
  car's wedges, so overlaps do not darken and blind spots stay white (the front radar's narrow far beam included; a
  camera's reaches to where a car is min_height_px tall);
- each car's latest data in the car's colour: lidar points (its last sweep, every beam from the sensor pose at its
  own fire time, so the scan skew shows), radar points (clutter hollow) and camera rays from the camera to each box's
  single-frame position (meta['init'] of its detection; false boxes dotted). Clutter and false boxes come from the
  simulators' truth ids, which only the viewer reads;
- the belief at the frame's time: confirmed tracks as boxes at their mean with 2-sigma position ellipses (ids in the
  zoom), tentative tracks faded and dashed;
- the truth: traffic outlines and the team cars filled. A true car with no confirmed track within GATE (Hungarian on
  the CG position) is outlined red (missed), a confirmed track with no true car is red (false); the title counts
  both within NEAR of the nearest team car, with the matched tracks' mean position error.

The data are those measured at the frame's time, while the belief there holds only what has been delivered (each
sensor's latency plus the V2V link: the tracker's hold), predicted to that time.

PerceptionRecorder is the pipeline with record=True that also keeps every scan in the world frame (ScanView): during
each call's sensor firing it wraps the pipeline's process_* functions, which receive every raw scan with its ego state
(lidar: the ego-at-time function) and return its detections, and passes their results through unchanged, so its
beliefs are an unrecorded run's. render_shadow_episode() renders scenario v9, seed 1000, 15 s in shadow mode (the
planner drives on the true state) to results/perception_viz_v9_seed1000_15s/: a GIF and a PNG of its densest moment
(the most true cars in the zoom). python -m nohesi.perception.viz runs it.
"""
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ..evaluation.viz import COLORS
from ..geometry import footprints
from . import pipeline as _pipeline
from .config import CameraSpec, LidarSpec, Perception, RadarSpec
from .geometry import focal, mount_offset, sensor_pose, wrap
from .pipeline import PerceptionSystem
from .processing.radar import ego_at

__all__ = ['ScanView', 'PerceptionRecorder', 'recorder', 'record_episode', 'fov_shape', 'coverage', 'match_tracks',
           'frame_stats', 'draw_frame', 'animate_perception', 'plot_perception', 'dense_time', 'render_shadow_episode',
           'GATE', 'NEAR', 'ZOOM', 'MODALITIES', 'FOV_COLORS', 'TRACK_COLOR', 'MISS_COLOR', 'OUT_DIR']

GATE = 2.0             # m: a confirmed track within this of a true car's CG (Hungarian) is that car's track
NEAR = 100.0           # m along the road from the nearest team car: the title's counts (where the planner decides)
ZOOM = (50.0, 100.0)   # m behind and ahead of the team's mean s: the zoom panel
MODALITIES = ('lidar', 'radar', 'camera')
FOV_COLORS = {'lidar': '#55a868', 'radar': '#dd8452', 'camera': '#4c72b0'}
FOV_ALPHA = 0.09       # tint of one modality's coverage
GRID = (800, 64)       # coverage raster columns and rows per panel
TRACK_COLOR = '#2b6cb0'
MISS_COLOR = '#d62728'
TRUE_COLOR = '#222222'
RAY_ALPHA = 0.4
FIGSIZE = (12.0, 5.4)
KNOTS = 9              # ego samples over a lidar sweep (the team history is linear between sim steps, which they hit)
OUT_DIR = Path(__file__).resolve().parents[2]/'results'/'perception_viz_v9_seed1000_15s'


class ScanView(NamedTuple):
    """One scan in the world frame, for drawing. points: lidar hits (each from its beam's pose), radar points, or the
    camera boxes' single-frame positions (ray ends); real: from a real car by the simulator's truth ids (lidar, and
    data without truth ids: all True). origin: the sensor's position at `time` (lidar: the sweep end)."""
    kind: str
    sensor: str
    car: int
    time: float
    origin: np.ndarray
    points: np.ndarray
    real: np.ndarray


def _pose(ego, spec, params):
    return np.array(sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), params)[:3])


def _lidar_view(scan, spec, args, dets, params):
    ego = args[0]
    hit = np.isfinite(scan.range)
    knots = np.linspace(float(scan.t_start), float(scan.t_end), KNOTS)
    egos = [ego(float(t)) for t in knots]
    x, y = np.array([e.x for e in egos]), np.array([e.y for e in egos])
    yaw = np.unwrap([e.yaw for e in egos])
    t = np.asarray(scan.time, float)[hit]
    ex, ey, eyaw = (np.interp(t, knots, a) for a in (x, y, yaw))
    ox, oy = mount_offset(spec.mount, params)
    c, s = np.cos(eyaw), np.sin(eyaw)
    a, r = eyaw+spec.mount.yaw+np.asarray(scan.azimuth, float)[hit], np.asarray(scan.range, float)[hit]
    pts = np.column_stack((ex+c*ox-s*oy+r*np.cos(a), ey+s*ox+c*oy+r*np.sin(a))).astype(np.float32)
    return ScanView('lidar', scan.sensor, int(scan.car), float(scan.t_end), _pose(egos[-1], spec, params)[:2], pts,
                    np.ones(len(pts), bool))


def _radar_view(scan, spec, args, dets, params):
    x, y, yaw = _pose(ego_at(args[0], scan.time), spec, params)
    r, a = np.asarray(scan.range, float), yaw+np.asarray(scan.azimuth, float)
    pts = np.column_stack((x+r*np.cos(a), y+r*np.sin(a))).astype(np.float32)
    real = np.ones(len(r), bool) if scan.truth_id is None else np.asarray(scan.truth_id) != -1
    return ScanView('radar', scan.sensor, int(scan.car), float(scan.time), np.array([x, y]), pts, real)


def _camera_view(frame, spec, args, dets, params):
    x, y, _ = _pose(ego_at(args[0], frame.time), spec, params)
    pts = np.array([d.meta['init'][0] for d in dets], np.float32).reshape(-1, 2)
    real = np.ones(len(dets), bool)
    if frame.truth_id is not None and len(dets):
        boxes, truth = np.asarray(frame.boxes, float).reshape(-1, 4), np.asarray(frame.truth_id)
        for k, d in enumerate(dets):
            row = np.flatnonzero((boxes == d.z).all(1))
            real[k] = not (len(row) and truth[row[0]] == -1)
    return ScanView('camera', frame.sensor, int(frame.car), float(frame.time), np.array([x, y]), pts, real)


_VIEWS = {'process_lidar_batch': _lidar_view, 'process_radars_batch': _radar_view,
          'process_cameras_batch': _camera_view}
_BATCHED = ('process_radars_batch', 'process_cameras_batch')   # a step's radar scans and frames go at once


class PerceptionRecorder(PerceptionSystem):
    """PerceptionSystem with record=True that also keeps every scan as a ScanView in self.scans (module docstring;
    reset clears them with the rest of the state, then the pre-roll's scans join at negative times)."""

    def __init__(self, cfg, scenario, road, params, seed, *, hold=None):
        super().__init__(cfg, scenario, road, params, seed, record=True, hold=hold)

    def reset_state(self):
        super().reset_state()
        self.scans = []

    def _fire(self, now):
        saved = {name: getattr(_pipeline, name) for name in _VIEWS}
        try:
            for name, view in _VIEWS.items():
                setattr(_pipeline, name, (self._logged_batch if name in _BATCHED else self._logged)(saved[name], view))
            super()._fire(now)
        finally:
            for name, process in saved.items():
                setattr(_pipeline, name, process)

    def _logged(self, process, view):
        def run(scan, spec, *args, **kwargs):
            dets = process(scan, spec, *args, **kwargs)
            self.scans.append(view(scan, spec, args, dets, self.params))
            return dets
        return run

    def _logged_batch(self, process, view):
        def run(scans, specs, egos, *args, **kwargs):
            out = process(scans, specs, egos, *args, **kwargs)
            self.scans.extend(view(scan, spec, (ego,), dets, self.params)
                              for scan, spec, ego, dets in zip(scans, specs, egos, out))
            return out
        return run


def recorder(scenario, world, seed):
    """EpisodeRun's perception factory for a PerceptionRecorder on the scenario's perception settings."""
    return PerceptionRecorder(scenario.perception, scenario, world.road, world.plant.params, seed)


def record_episode(scenario=None, seed=1000, duration=15.0, policy='overtaker', planner=None, mode='shadow'):
    """Run an episode of `scenario` (default v9; its perception settings if on, else the defaults) for `duration` s
    with perception on in `mode` and a PerceptionRecorder (run.perception); returns the finished EpisodeRun
    (run.snaps holds the truth)."""
    from ..config import Planner, scenario_v9
    from ..evaluation.runner import EpisodeRun
    s = scenario or scenario_v9()
    cfg = replace(s.perception if s.perception.enabled else Perception(), enabled=True, mode=mode)
    s = replace(s, timing=replace(s.timing, duration=float(duration)), perception=cfg)
    run = EpisodeRun(s, seed, policy, planner or Planner(), perception=recorder)
    while not run.done:
        run.step()
    return run


# -------------------------------------------------------------------------------------------------- geometry
def fov_shape(spec, car_height):
    """(modality, mount, (fov, reach, fov_far, range_max)) of a sensor spec: fov wide out to reach, fov_far wide from
    there to range_max (a lidar's and a camera's are one wedge; a camera reaches to where a car of car_height is
    min_height_px tall)."""
    if isinstance(spec, LidarSpec):
        return 'lidar', spec.mount, (spec.fov, spec.range_max, spec.fov, spec.range_max)
    if isinstance(spec, RadarSpec):
        reach = min(spec.near_range, spec.range_max)
        return 'radar', spec.mount, (spec.fov, reach, min(spec.fov_far, spec.fov), spec.range_max)
    if isinstance(spec, CameraSpec):
        reach = focal(spec)*car_height/spec.min_height_px
        return 'camera', spec.mount, (spec.hfov, reach, spec.hfov, reach)
    raise TypeError(f'unknown sensor spec {type(spec).__name__}')


def coverage(shapes, team, params, xs, ys):
    """(len(ys), len(xs), len(MODALITIES)) bool: the road-frame grid points each modality covers, the union of the
    team cars' fields of view (shapes from fov_shape; team (k, >= 3) CG states)."""
    X, Y = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    out = np.zeros(X.shape+(len(MODALITIES),), bool)
    for state in team:
        for kind, mount, (fov, reach, fov_far, far) in shapes:
            x, y, yaw, _ = sensor_pose(mount, state, params)
            cols = np.flatnonzero(np.abs(X[0]-x) <= far)
            if cols.size == 0:
                continue
            c = slice(cols[0], cols[-1]+1)
            dx, dy = X[:, c]-x, Y[:, c]-y
            r, az = np.hypot(dx, dy), np.abs(wrap(np.arctan2(dy, dx)-yaw))
            out[:, c, MODALITIES.index(kind)] |= ((r <= reach) & (az <= fov/2)) | ((r <= far) & (az <= fov_far/2))
    return out


def _ellipses(mean, cov, k=2.0, n=24):
    """(m, n, 2) k-sigma ellipses of 2D Gaussians (means (m, 2), covariances (m, 2, 2))."""
    w, V = np.linalg.eigh(np.asarray(cov, float))
    L = V*np.sqrt(np.maximum(w, 0.0))[:, None, :]
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.asarray(mean, float)[:, None, :]+k*np.einsum('mij,jn->mni', L, np.stack((np.cos(a), np.sin(a))))


def _states(s, d, vs, vd):
    vd = np.zeros_like(s) if vd is None else vd
    return np.column_stack((s, d, np.arctan2(vd, np.maximum(vs, 1e-3))))


def match_tracks(track_sd, car_sd, gate=GATE):
    """Hungarian matching of tracks to true cars on position within `gate`: (track matched (n,), car matched (N,),
    distance of each matched track (n,), nan where unmatched)."""
    track_sd, car_sd = np.asarray(track_sd, float).reshape(-1, 2), np.asarray(car_sd, float).reshape(-1, 2)
    tm, cm, dist = np.zeros(len(track_sd), bool), np.zeros(len(car_sd), bool), np.full(len(track_sd), np.nan)
    if len(track_sd) and len(car_sd):
        D = np.hypot(track_sd[:, None, 0]-car_sd[None, :, 0], track_sd[:, None, 1]-car_sd[None, :, 1])
        rows, cols = linear_sum_assignment(np.where(D < gate, D, 1e6))
        ok = D[rows, cols] < gate
        tm[rows[ok]], cm[cols[ok]], dist[rows[ok]] = True, True, D[rows[ok], cols[ok]]
    return tm, cm, dist


# -------------------------------------------------------------------------------------------------- the scene
class _Scene:
    """A recorded run indexed by frame (an index into run.snaps): truth, belief and the latest scan of every
    (car, sensor)."""

    def __init__(self, run):
        system = run.perception
        if system is None or not getattr(system, 'record', False):
            raise ValueError('the run needs a recording perception system (record_episode, or perception=recorder)')
        self.run, self.system = run, system
        self.params, self.road = run.world.plant.params, run.world.road
        self.tracker = system.cfg.tracker
        self.times = np.array([s.time for s in run.snaps])
        self.beliefs = {round(float(t), 6): b for t, b in system.beliefs}
        index = {}
        for v in getattr(system, 'scans', ()):
            index.setdefault((v.car, v.sensor), []).append(v)
        self.index = [(np.array([v.time for v in vs]), vs) for _, vs in sorted(index.items())]
        self.shapes = [] if system.ideal else [fov_shape(s, system.cfg.car_height) for s in system.sensors]
        self._frames = {}

    def scans(self, k):
        """The latest scan of every (car, sensor) at frame k."""
        t = self.times[k]+1e-6
        out = []
        for times, vs in self.index:
            i = int(np.searchsorted(times, t))-1
            if i >= 0:
                out.append(vs[i])
        return out

    def windows(self, k):
        """(tracker window (lo, hi), zoom (lo, hi)) along the road at frame k."""
        s = self.run.snaps[k].team_s
        return ((s.min()-self.tracker.roi_behind, s.max()+self.tracker.roi_ahead),
                (s.mean()-ZOOM[0], s.mean()+ZOOM[1]))

    def frame(self, k):
        """Frame k's truth, belief and matching (cached)."""
        if k in self._frames:
            return self._frames[k]
        snap, b = self.run.snaps[k], self.beliefs.get(round(float(self.times[k]), 6))
        cars = _states(snap.traffic_s, snap.traffic_d, snap.traffic_v, snap.traffic_vd)
        if b is None or len(b) == 0:
            mean, cov, conf, ids = np.zeros((0, 4)), np.zeros((0, 4, 4)), np.zeros(0, bool), np.zeros(0, int)
        else:
            mean, cov, conf, ids = np.asarray(b.mean), np.asarray(b.cov), np.asarray(b.confirmed, bool), b.id
        tracks = _states(mean[:, 0], mean[:, 1], mean[:, 2], mean[:, 3])
        tm, cm, dist = match_tracks(tracks[conf, :2], cars[:, :2])
        matched, error = np.zeros(len(tracks), bool), np.full(len(tracks), np.nan)
        matched[conf], error[conf] = tm, dist
        near = lambda s: np.min(np.abs(s[:, None]-snap.team_s[None]), axis=1) <= NEAR if len(s) else np.zeros(0, bool)
        self._frames[k] = out = dict(snap=snap, cars=cars, missed=~cm, tracks=tracks, mean=mean, cov=cov, conf=conf,
                                     ids=np.asarray(ids, int), false=conf & ~matched, error=error,
                                     cars_near=near(cars[:, 0]), tracks_near=near(tracks[:, 0]))
        return out


def _scene(run):
    return run if isinstance(run, _Scene) else _Scene(run)


def frame_stats(run, k):
    """Frame k (an index into run.snaps) of a recorded run: its time, the true cars within NEAR of the team ('cars'),
    how many of them have no confirmed track ('missed'), the confirmed tracks within NEAR with no true car ('false'),
    all confirmed and tentative tracks, and the mean position error of the matched tracks within NEAR ('error', m;
    nan when there are none)."""
    f = _scene(run).frame(k)
    err = f['error'][f['conf'] & f['tracks_near'] & np.isfinite(f['error'])]
    return dict(time=float(f['snap'].time), cars=int(f['cars_near'].sum()),
                missed=int((f['missed'] & f['cars_near']).sum()), false=int((f['false'] & f['tracks_near']).sum()),
                confirmed=int(f['conf'].sum()), tentative=int((~f['conf']).sum()),
                error=float(err.mean()) if len(err) else float('nan'))


# -------------------------------------------------------------------------------------------------- drawing
def _coverage_image(ax, scene, snap, lo, hi, ylo, yhi):
    if not scene.shapes:
        return
    nx, ny = GRID
    xs, ys = lo+(np.arange(nx)+0.5)*(hi-lo)/nx, ylo+(np.arange(ny)+0.5)*(yhi-ylo)/ny
    cover = coverage(scene.shapes, snap.team, scene.params, xs, ys)
    rgb = np.ones((ny, nx, 3))
    for m, kind in enumerate(MODALITIES):
        a = FOV_ALPHA*cover[..., m, None]
        rgb = rgb*(1-a)+np.array(to_rgb(FOV_COLORS[kind]))*a
    ax.imshow(rgb, extent=(lo, hi, ylo, yhi), origin='lower', aspect='auto', interpolation='nearest', zorder=0)


def _panel(ax, scene, k, lo, hi, ids, title):
    ax.clear()
    f, road, params = scene.frame(k), scene.road, scene.params
    snap, half = f['snap'], road.width/2
    ylo, yhi = -half-1.0, half+1.0
    _coverage_image(ax, scene, snap, lo, hi, ylo, yhi)
    ax.set_xlim(lo, hi); ax.set_ylim(ylo, yhi); ax.set_aspect('auto'); ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=7, pad=1)
    ax.set_title(title, fontsize=8, loc='left', pad=2)
    for y in np.linspace(-half, half, road.cfg.lanes+1):
        edge = abs(abs(y)-half) < 1e-9
        ax.axhline(y, color='gray', lw=1.0 if edge else 0.6, ls='-' if edge else '--', zorder=1)
    points, rays, ray_colors, ray_styles = {}, [], [], []
    for v in scene.scans(k):
        color = COLORS[v.car % len(COLORS)]
        if v.kind == 'camera':
            for end, real in zip(v.points, v.real):
                rays.append((v.origin, end)); ray_colors.append(to_rgba(color, RAY_ALPHA))
                ray_styles.append('-' if real else ':')
        else:
            for real in (True, False):
                points.setdefault((v.kind, v.car, real), []).append(v.points[v.real == real])
    for (kind, car, real), p in points.items():
        p, color = np.concatenate(p), COLORS[car % len(COLORS)]
        if kind == 'lidar':
            ax.plot(p[:, 0], p[:, 1], ls='', marker='.', ms=1.6, mew=0, color=color, zorder=4)
        elif len(p):
            ax.plot(p[:, 0], p[:, 1], ls='', marker='^', ms=3.2, mfc=color if real else 'none', mec=color, mew=0.6,
                    zorder=5)
    if rays:
        ax.add_collection(LineCollection(rays, colors=ray_colors, linestyles=ray_styles, linewidths=0.5, zorder=3))
        ends = np.array([r[1] for r in rays])
        ax.scatter(ends[:, 0], ends[:, 1], s=3, c=ray_colors, linewidths=0, zorder=3)
    inside = lambda s: (s > lo-10) & (s < hi+10)
    tracks, conf = f['tracks'], f['conf']
    show = inside(tracks[:, 0])
    for confirmed, alpha, ls in ((False, 0.12, '--'), (True, 0.35, '-')):
        m = show & (conf == confirmed)
        if not m.any():
            continue
        edge = np.where(f['false'][m, None], to_rgba(MISS_COLOR), to_rgba(TRACK_COLOR, min(1.0, 3*alpha)))
        ax.add_collection(PolyCollection(footprints(tracks[m], params), facecolors=to_rgba(TRACK_COLOR, alpha),
                                         edgecolors=edge, linewidths=0.9 if confirmed else 0.6, linestyles=ls,
                                         zorder=6))
        ax.add_collection(PolyCollection(_ellipses(f['mean'][m, :2], f['cov'][m, :2, :2]), facecolors='none',
                                         edgecolors=edge, linewidths=0.8 if confirmed else 0.5, linestyles=ls,
                                         zorder=6.5))
        if confirmed and ids:
            for (s, d, _), i, bad in zip(tracks[m], f['ids'][m], f['false'][m]):
                ax.text(s, d+params.width/2+0.15, str(i), fontsize=6, ha='center', va='bottom', zorder=9,
                        clip_on=True, color=MISS_COLOR if bad else TRACK_COLOR)
    cars, missed = f['cars'], f['missed']
    m = inside(cars[:, 0])
    if m.any():
        ax.add_collection(PolyCollection(footprints(cars[m], params), facecolors='none',
                                         edgecolors=np.where(missed[m, None], to_rgba(MISS_COLOR), to_rgba(TRUE_COLOR)),
                                         linewidths=np.where(missed[m], 1.4, 0.7), zorder=7))
    ax.add_collection(PolyCollection(footprints(snap.team[:, :3], params), facecolors=COLORS[:len(snap.team)],
                                     edgecolors='black', linewidths=0.4, zorder=8))


def _legend(fig):
    gray = '#666666'
    handles = [Patch(facecolor=c, edgecolor='black', lw=0.4, label=f'team car {i}') for i, c in enumerate(COLORS)]
    handles += [Patch(facecolor=to_rgba(FOV_COLORS[k], 3*FOV_ALPHA), label=f'{k} field of view') for k in MODALITIES]
    handles += [Line2D([], [], ls='', marker='.', ms=5, mew=0, color=gray, label='lidar points'),
                Line2D([], [], ls='', marker='^', ms=4, color=gray, label='radar points'),
                Line2D([], [], ls='', marker='^', ms=4, mfc='none', mec=gray, label='radar clutter'),
                Line2D([], [], color=gray, lw=0.8, label='camera rays (false box dotted)'),
                Patch(facecolor=to_rgba(TRACK_COLOR, 0.35), edgecolor=TRACK_COLOR, label='confirmed track, 2σ'),
                Patch(facecolor=to_rgba(TRACK_COLOR, 0.12), edgecolor=to_rgba(TRACK_COLOR, 0.36), ls='--',
                      label='tentative track'),
                Patch(facecolor='none', edgecolor=TRUE_COLOR, label='true car'),
                Patch(facecolor='none', edgecolor=MISS_COLOR, lw=1.4, label='missed car / false track')]
    fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=7, frameon=False, handlelength=1.5,
               columnspacing=1.1, borderaxespad=0.2)


def _figure(dpi):
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, dpi=dpi, gridspec_kw=dict(height_ratios=(1.0, 1.8)))
    fig.subplots_adjust(left=0.015, right=0.99, top=0.885, bottom=0.135, hspace=0.3)
    _legend(fig)
    return fig, axes, fig.suptitle('', fontsize=8.5, x=0.015, y=0.995, ha='left', va='top', linespacing=1.4)


def _draw(scene, k, axes, title):
    (lo, hi), (zlo, zhi) = scene.windows(k)
    st, system, sc = frame_stats(scene, k), scene.system, scene.run.scenario
    err = f'{st["error"]:.2f} m' if np.isfinite(st['error']) else '-'
    title.set_text(f'No Hesi {sc.version} | seed {scene.run.seed} | perception {system.mode}, sharing '
                   f'{system.cfg.sharing} | t = {st["time"]:.1f} s\n{st["confirmed"]} confirmed and '
                   f'{st["tentative"]} tentative tracks | within {NEAR:.0f} m of the team: {st["cars"]} cars, '
                   f'{st["missed"]} missed, {st["false"]} false tracks, mean position error {err}')
    _panel(axes[0], scene, k, lo, hi, False, f'tracker window ({scene.tracker.roi_behind:.0f} m behind to '
                                             f'{scene.tracker.roi_ahead:.0f} m ahead of the team)')
    _panel(axes[1], scene, k, zlo, zhi, True, 'around the team (track ids)')


def draw_frame(run, k, axes=None, dpi=100):
    """Draw frame k (an index into run.snaps) of a recorded run on two axes (None: a new figure with the legend);
    returns the figure."""
    if axes is None:
        fig, axes, title = _figure(dpi)
    else:
        fig = axes[0].figure
        title = fig.suptitle('', fontsize=8.5, x=0.015, y=0.995, ha='left', va='top', linespacing=1.4)
    _draw(_scene(run), k, axes, title)
    return fig


def _frames(scene, start, end, stride):
    step = max(1, int(round(stride/float(scene.run.scenario.timing.dt))))
    return [k for k in range(0, len(scene.times), step) if start-1e-9 <= scene.times[k] <= end+1e-9]


def animate_perception(run, path, start=0.0, end=None, stride=0.2, fps=5, dpi=80, colors=128):
    """GIF of a recorded run from `start` to `end` s (None: its end), one frame every `stride` s at `fps` (the
    defaults play in real time); frames quantized to an adaptive palette of `colors` colours without dithering, which
    keeps 15 s to a few MB. Returns the frame count."""
    scene = _scene(run)
    end = scene.times[-1] if end is None else end
    fig, axes, title = _figure(dpi)
    images = []
    try:
        for k in _frames(scene, start, end, stride):
            _draw(scene, k, axes, title)
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            images.append(rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    finally:
        plt.close(fig)
    if not images:
        raise ValueError('no frames between start and end')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000/fps)), loop=0,
                   optimize=True)
    return len(images)


def plot_perception(run, path, time, dpi=150):
    """PNG of a recorded run's frame at `time` s (the nearest snapshot); returns that snapshot's time."""
    scene = _scene(run)
    k = int(np.argmin(np.abs(scene.times-time)))
    fig = draw_frame(scene, k, dpi=dpi)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return float(scene.times[k])


def dense_time(run, stride=0.2):
    """The time, on frames every `stride` s, with the most true traffic cars in the zoom panel (the first on ties)."""
    scene = _scene(run)
    frames = _frames(scene, scene.times[0], scene.times[-1], stride)
    count = []
    for k in frames:
        lo, hi = scene.windows(k)[1]
        s = scene.run.snaps[k].traffic_s
        count.append(int(np.count_nonzero((s > lo) & (s < hi))))
    return float(scene.times[frames[int(np.argmax(count))]])


def render_shadow_episode(out_dir=OUT_DIR, seed=1000, duration=15.0, policy='overtaker', run=None, **gif):
    """Record scenario v9, `seed`, `duration` s in shadow mode with the scripted `policy` (or take a recorded `run`)
    and write perception.gif (animate_perception, options `gif`) and dense_t<time>s.png (its densest moment,
    dense_time) to out_dir; returns {'gif', 'png', 'frames', 'dense_time', 'run'}."""
    run = run or record_episode(seed=seed, duration=duration, policy=policy)
    scene, out = _Scene(run), Path(out_dir)
    frames = animate_perception(scene, out/'perception.gif', **gif)
    t = dense_time(scene, gif.get('stride', 0.2))
    png = out/f'dense_t{t:.1f}s.png'
    plot_perception(scene, png, t)
    return dict(gif=out/'perception.gif', png=png, frames=frames, dense_time=t, run=run)


if __name__ == '__main__':
    result = render_shadow_episode()
    print(result['gif'], f"{result['frames']} frames, {result['gif'].stat().st_size/1e6:.1f} MB")
    print(result['png'], f"t = {result['dense_time']:.1f} s")
