"""The perception viewer overlay (nohesi.perception.viz; design phase P1): the recorder is transparent and puts every
scan in the world frame correctly, the overlay renders a short recorded shadow episode to a GIF and a PNG, and with
perception off (or in shadow mode) the existing episode viewer's output is unchanged."""
from dataclasses import replace
from math import radians
import matplotlib
import numpy as np
import pytest
from PIL import Image
from nohesi.config import Planner, scenario_v9
from nohesi.evaluation import viz as episode_viz
from nohesi.evaluation.runner import EpisodeRun
from nohesi.geometry import footprints
from nohesi.perception import viz
from nohesi.perception.config import CameraSpec, LidarSpec, Mount, Perception, RadarSpec
from nohesi.perception.geometry import sensor_pose
from nohesi.perception.pipeline import PerceptionSystem
from nohesi.perception.types import EgoState, LidarScan, RadarScan

SEED = 1000
DURATION = 1.0
PARAMS = scenario_v9().vehicle


def short(perception=None, duration=DURATION):
    s = scenario_v9()
    return replace(s, timing=replace(s.timing, duration=duration), perception=perception or Perception())


def drive(scenario, policy='overtaker', **kw):
    r = EpisodeRun(scenario, SEED, policy, Planner(), **kw)
    while not r.done:
        r.step()
    return r


@pytest.fixture(scope='module')
def recorded():
    return viz.record_episode(seed=SEED, duration=DURATION)


def test_recorder_is_transparent(recorded):
    """The recorder's beliefs and scan counts are those of the plain recording pipeline on the same episode."""
    plain = lambda sc, world, seed: PerceptionSystem(sc.perception, sc, world.road, world.plant.params, seed,
                                                     record=True)
    other = drive(recorded.scenario, perception=plain).perception
    assert len(other.beliefs) == len(recorded.perception.beliefs) > int(DURATION/0.05)
    for (t, b), (u, c) in zip(recorded.perception.beliefs, other.beliefs):
        assert t == u
        for k in ('id', 'mean', 'cov', 'p_exist', 'confirmed'):
            np.testing.assert_array_equal(getattr(b, k), getattr(c, k))
    np.testing.assert_array_equal(recorded.perception.scan_records(), other.scan_records())
    rec = recorded.perception.scan_records()
    assert len(recorded.perception.scans) == len(rec)
    kinds = {'lidar_box': 'lidar', 'radar': 'radar', 'camera_box': 'camera'}
    assert [(v.time, v.car, v.sensor, v.kind) for v in recorded.perception.scans] == \
        [(r['time'], r['car'], r['sensor'], kinds[r['kind']]) for r in rec]
    assert min(v.time for v in recorded.perception.scans) < 0 < max(v.time for v in recorded.perception.scans)


def test_lidar_view_places_each_beam_from_its_own_pose():
    """Every beam leaves from the sensor's pose at its fire time (scan skew), for a moving, yawed car."""
    ego = lambda t: EgoState(t, 10.0+30.0*t, 1.0-2.0*t, 0.2, 30.0)
    for spec in (LidarSpec(), LidarSpec(name='rear', mount=Mount(x=0.05, yaw=np.pi, edge='rear'), fov=np.pi)):
        az, rng = np.array([-0.3, 0.0, 0.25, 0.4]), np.array([20.0, np.nan, 35.5, 8.0])
        tm = np.array([0.0, 0.02, 0.05, 0.1])
        scan = LidarScan(spec.name, 1, 0.0, 0.1, az, rng, tm)
        v = viz._lidar_view(scan, spec, (ego,), [], PARAMS)
        hit = np.isfinite(rng)
        want = []
        for a, r, t in zip(az[hit], rng[hit], tm[hit]):
            e = ego(t)
            x, y, yaw, _ = sensor_pose(spec.mount, (e.x, e.y, e.yaw), PARAMS)
            want.append((x+r*np.cos(yaw+a), y+r*np.sin(yaw+a)))
        np.testing.assert_allclose(v.points, want, atol=1e-4)
        e = ego(0.1)
        np.testing.assert_allclose(v.origin, sensor_pose(spec.mount, (e.x, e.y, e.yaw), PARAMS)[:2])
        assert (v.kind, v.car, v.time) == ('lidar', 1, 0.1) and v.real.all()


def test_radar_view_flags_clutter():
    spec = RadarSpec(name='radar_left', mount=Mount(x=0.8, y=0.95, z=1.0, yaw=radians(90.0)))
    ego = EgoState(0.5, 100.0, -3.6, 0.05, 30.0)
    scan = RadarScan(spec.name, 2, 0.5, np.array([10.0, 20.0, 5.0]), np.array([0.0, 0.3, -0.2]), np.zeros(3),
                     truth_id=np.array([3, -1, -4]))
    v = viz._radar_view(scan, spec, (ego,), [], PARAMS)
    x, y, yaw, _ = sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), PARAMS)
    a = yaw+scan.azimuth
    np.testing.assert_allclose(v.points, np.column_stack((x+scan.range*np.cos(a), y+scan.range*np.sin(a))), atol=1e-4)
    np.testing.assert_array_equal(v.real, [True, False, True])   # -4: a teammate, a real car


def test_recorded_scans_land_on_the_cars(recorded):
    """End to end: lidar points lie on a car (within its travel over a sweep along the road), real radar points near
    one, and every sensor rides on its car."""
    snaps = {round(s.time, 6): s for s in recorded.snaps}
    lidar_ok, radar_d, n_lidar = [], [], 0
    for v in recorded.perception.scans:
        snap = snaps.get(round(v.time, 6))
        if snap is None:
            continue
        cars = np.vstack((np.column_stack((snap.traffic_s, snap.traffic_d, np.zeros(len(snap.traffic_s)))),
                          snap.team[:, :3]))
        assert np.hypot(*(v.origin-snap.team[v.car, :2])) < PARAMS.length
        if v.kind == 'lidar' and len(v.points):
            n_lidar += 1
            ds, dd = v.points[:, None, 0]-cars[None, :, 0], v.points[:, None, 1]-cars[None, :, 1]
            along = (ds > -PARAMS.length/2-4.0) & (ds < PARAMS.length/2+0.3)
            lidar_ok.append((along & (np.abs(dd) < PARAMS.width/2+0.3)).any(1))
        elif v.kind == 'radar' and v.real.any():
            boxes = footprints(cars, PARAMS)
            for p in v.points[v.real]:
                inside = (p[0] > boxes[:, :, 0].min(1)-0.5) & (p[0] < boxes[:, :, 0].max(1)+0.5)
                radar_d.append(np.min(np.abs(p[1]-cars[inside, 1])) if inside.any() else np.inf)
    assert n_lidar > 10 and np.concatenate(lidar_ok).mean() > 0.97
    assert np.median(np.array(radar_d)-PARAMS.width/2) < 0.5


def test_coverage_of_the_default_suite():
    """Front lidar to 200 m, the front radar's narrow far beam to 250 m, the rear lidar to 120 m behind, side radars
    and cameras (not the lidars) right beside the car."""
    shapes = [viz.fov_shape(s, 1.45) for s in (*Perception().lidars, *Perception().radars, *Perception().cameras)]
    team = np.array([[0.0, 0.0, 0.0, 30.0]])
    pts = np.array([[100.0, 0.0], [210.0, 0.0], [240.0, 0.0], [100.0, 20.0], [60.0, 20.0], [-100.0, 0.0],
                    [-130.0, 0.0], [0.0, 3.6], [0.0, -3.6], [300.0, 0.0]])
    cover = np.array([viz.coverage(shapes, team, PARAMS, [x], [y])[0, 0] for x, y in pts])   # (n, lidar/radar/camera)
    want = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1],
                     [0, 1, 1], [0, 0, 0]], bool)
    np.testing.assert_array_equal(cover, want)
    cam = CameraSpec()
    assert viz.fov_shape(cam, 1.45)[2][1] == pytest.approx(1662.77*1.45/15.0, rel=1e-4)


def test_match_tracks():
    tracks = np.array([[0.0, 0.0], [50.0, 3.6], [100.0, 0.0], [10.5, 0.0]])
    cars = np.array([[0.4, 0.1], [10.0, 0.0], [80.0, -3.6], [50.0, 2.0]])
    tm, cm, dist = viz.match_tracks(tracks, cars)
    np.testing.assert_array_equal(tm, [True, True, False, True])
    np.testing.assert_array_equal(cm, [True, True, False, True])
    np.testing.assert_allclose(dist, [np.hypot(0.4, 0.1), 1.6, np.nan, 0.5])
    tm, cm, dist = viz.match_tracks(np.zeros((0, 2)), cars)
    assert tm.shape == (0,) and not cm.any()


def test_overlay_draws_tracks_truth_and_sensors(recorded):
    k = len(recorded.snaps)-1
    fig = viz.draw_frame(recorded, k)
    try:
        scene = viz._Scene(recorded)
        f, st = scene.frame(k), viz.frame_stats(scene, k)
        b = recorded.perception.beliefs[-1][1]
        assert st['confirmed']+st['tentative'] == len(b) and st['confirmed'] == int(b.confirmed.sum()) > 10
        assert 0 <= st['missed'] <= st['cars'] and st['false'] <= st['confirmed']
        top, zoom = fig.axes[:2]
        lo, hi = scene.windows(k)[1]
        inside = f['conf'] & (f['tracks'][:, 0] > lo-10) & (f['tracks'][:, 0] < hi+10)
        assert sorted(t.get_text() for t in zoom.texts) == sorted(str(i) for i in f['ids'][inside])
        assert not top.texts and len(top.images) == len(zoom.images) == 1
        markers = {line.get_marker() for line in zoom.lines}
        assert {'.', '^'} <= markers
        assert any(len(c.get_segments()) for c in zoom.collections if hasattr(c, 'get_segments'))
        assert f'{st["confirmed"]} confirmed' in fig.get_suptitle()
    finally:
        matplotlib.pyplot.close(fig)


def test_render_gif_and_png(recorded, tmp_path):
    out = viz.render_shadow_episode(out_dir=tmp_path/'viz', run=recorded, stride=0.5)
    assert out['frames'] == 3 and out['gif'].exists() and out['png'].exists()
    with Image.open(out['gif']) as gif:
        assert gif.n_frames == 3 and gif.size == (960, 432)
    with Image.open(out['png']) as png:
        assert png.size == (1800, 810)
    assert out['dense_time'] in (0.0, 0.5, 1.0) and out['png'].name == f"dense_t{out['dense_time']:.1f}s.png"
    with pytest.raises(ValueError):
        viz.animate_perception(recorded, tmp_path/'none.gif', start=5.0)
    with pytest.raises(ValueError):
        viz.draw_frame(drive(short(duration=0.1), 'lane_keeping'), 0)


def _episode_outputs(ep, tmp_path, name):
    episode_viz.animate(ep, tmp_path/f'{name}.gif')
    episode_viz.plot(ep, tmp_path/f'{name}.png')
    return (tmp_path/f'{name}.gif').read_bytes(), (tmp_path/f'{name}.png').read_bytes()


def test_disabled_perception_leaves_the_episode_viewer_unchanged(recorded, tmp_path):
    """Perception off builds no system and gives the same Episode and viewer files as no perception at all; the
    overlay leaves matplotlib's global state and the viewer's output alone; a shadow run draws as an unperceived one."""
    off = drive(short(), 'lane_keeping')
    none = drive(short(), 'lane_keeping', perception=None)
    assert off.perception is None
    before = _episode_outputs(off.episode(), tmp_path, 'off')
    assert before == _episode_outputs(none.episode(), tmp_path, 'none')
    params = dict(matplotlib.rcParams)
    matplotlib.pyplot.close(viz.draw_frame(recorded, 0))
    viz.animate_perception(recorded, tmp_path/'p.gif', stride=1.0)
    assert dict(matplotlib.rcParams) == params and matplotlib.get_backend().lower() == 'agg'
    assert _episode_outputs(off.episode(), tmp_path, 'after') == before
    truth = drive(replace(recorded.scenario, perception=Perception()))
    assert _episode_outputs(recorded.episode(), tmp_path, 'shadow') == _episode_outputs(truth.episode(), tmp_path,
                                                                                          'truth')
