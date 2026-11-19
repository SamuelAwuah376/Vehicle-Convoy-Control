"""Tier A radar simulator (nohesi.perception.sim.radar): detection curve, noise, range rate, field of view,
occlusion, resolution merging, clutter, truth ids, determinism and speed."""
from dataclasses import replace
import time
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.geometry import cg_offsets
from nohesi.world import Snapshot
from nohesi.perception import geometry as G
from nohesi.perception.config import default_radars
from nohesi.perception.types import TruthHistory
from nohesi.perception.sim import radar as R
from nohesi.perception.sim.radar import (detection_probability, nearest_points, sensor_velocity, simulate_radar,
                                         simulate_radars, visible_fractions)

VEH = scenario_v9().vehicle
RADARS = default_radars()
FRONT, REAR, LEFT, RIGHT = RADARS
LANE = 3.6
CLEAN = dict(pd_ref=1.0, false_alarms=0.0, points_max=1)        # a visible car always gives one point; no clutter
EXACT = dict(sigma_range=0.0, sigma_azimuth=0.0, sigma_range_rate=0.0)
EGO = (0.0, LANE, 0.0, 30.0)
FAR_MATES = ((-1000.0, 0.0, 30.0), (-2000.0, 0.0, 30.0))
names = lambda spec: spec.name


def truth(traffic=(), ego=EGO, mates=FAR_MATES, vy=0.0, r=0.0):
    """TruthHistory of one snapshot: traffic rows (s, d, v_s[, v_d]) with ids 100, 101, ...; team car 0 at ego
    (x, y, yaw, v), its teammates at (s, d, v) along the road."""
    rows = np.array([tuple(row)+(0.0,)*(4-len(row)) for row in traffic], float).reshape(-1, 4)
    team = np.array([ego]+[(s, d, 0.0, v) for s, d, v in mates], float)
    snap = Snapshot(time=0.0, team=team, team_steer=np.zeros(3), team_s=team[:, 0], team_d=team[:, 1],
                    traffic_s=rows[:, 0], traffic_d=rows[:, 1], traffic_v=rows[:, 2],
                    traffic_id=np.arange(100, 100+len(rows)), collisions=(), done=False, traffic_vd=rows[:, 3],
                    team_vy=np.array([vy, 0.0, 0.0]), team_r=np.array([r, 0.0, 0.0]))
    history = TruthHistory()
    history.push(snap)
    return history


def highway(seed, count=45):
    """A v9-like scene: up to `count` cars on lane centres from 150 m behind to 400 m ahead, gaps >= 15 m, the team
    spread over the three lanes."""
    rng = np.random.default_rng(seed)
    mates = ((-20.0, 0.0, 30.0), (20.0, 2*LANE, 30.0))
    team = [(EGO[0], EGO[1])]+[(s, d) for s, d, _ in mates]
    cars = []
    for lane in range(3):
        s = -150+rng.uniform(0, 20)
        while s < 400:
            if all(abs(s-ts) > 12 or abs(lane*LANE-td) > 1 for ts, td in team):
                cars.append((s, lane*LANE, rng.uniform(25, 35)))
            s += rng.uniform(15, 60)
    cars = [cars[k] for k in np.sort(rng.choice(len(cars), min(count, len(cars)), replace=False))]
    return truth(cars, mates=mates)


def sensor(spec, history, car=0):
    return G.sensor_pose(spec.mount, history.at(0.0).team[car], VEH)


def boxes_of(history):
    state = history.at(0.0)
    return G.car_boxes(np.vstack((state.traffic[:, :3], state.team[:, :3])), VEH)


def polar(pose, point):
    d = np.asarray(point)-pose[:2]
    return np.hypot(*d), float(G.wrap(np.arctan2(d[1], d[0])-pose[2]))


def car_at(pose, rho, bearing, v=30.0):
    """truth() with one car whose CG is at range rho and sensor-frame bearing from the sensor pose."""
    cg = np.array(pose[:2])+rho*np.array([np.cos(pose[2]+bearing), np.sin(pose[2]+bearing)])
    return truth([(cg[0], cg[1], v)])


def scans(spec, history, n, rng, car=0):
    """n independent scans of one radar, in one batched call."""
    return simulate_radars([spec]*n, [car], history, 0.0, VEH, rng)


@pytest.mark.parametrize('spec', RADARS, ids=names)
def test_detection_probability_curve(spec):
    r = np.linspace(1.0, spec.range_max, 500)
    pd = detection_probability(spec, r)
    assert detection_probability(spec, spec.ref_range) == pytest.approx(spec.pd_ref)
    assert detection_probability(spec, spec.range_max) == pytest.approx(0.5)
    assert detection_probability(spec, spec.range_max+0.1) == 0
    np.testing.assert_allclose(pd[r <= spec.ref_range], spec.pd_ref)
    assert (np.diff(pd) <= 1e-15).all() and (pd[r > spec.ref_range] < spec.pd_ref).all()
    far = r > spec.ref_range                     # logistic in SNR (dB): the log odds are linear in 40 log10(r)
    logit = np.log(pd[far]/(1-pd[far]))
    fit = np.polyfit(40*np.log10(r[far]), logit, 1)
    np.testing.assert_allclose(np.polyval(fit, 40*np.log10(r[far])), logit, atol=1e-9)
    k = 10**(10/40)                              # a target 10 dB brighter reaches 10^(10/40) times as far
    ok = r*k <= spec.range_max
    np.testing.assert_allclose(detection_probability(spec, r[ok]*k, spec.rcs_dbsm+10), pd[ok])
    np.testing.assert_allclose(detection_probability(replace(spec, ref_range=spec.range_max), r), spec.pd_ref)
    np.testing.assert_allclose(detection_probability(replace(spec, pd_ref=1.0), r), 1.0)


def test_detection_rate_follows_the_curve():
    spec = replace(FRONT, false_alarms=0.0, points_max=1)
    rng = np.random.default_rng(8)
    for s in (120.0, 205.0, 245.0):
        history = truth([(s, LANE, 30.0)])
        pose = sensor(spec, history)
        p = float(detection_probability(spec, polar(pose, G.nearest_box_point(pose[:2], boxes_of(history)[0]))[0]))
        n = 2000
        rate = np.mean([len(x.range) for x in scans(spec, history, n, rng)])
        assert rate == pytest.approx(p, abs=4*np.sqrt(p*(1-p)/n))


def test_vectorized_geometry_matches_the_contract():
    rng = np.random.default_rng(3)
    states = np.column_stack((rng.uniform(-60, 60, 30), rng.uniform(-8, 8, 30), rng.normal(0, 0.1, 30)))
    boxes = G.car_boxes(states, VEH)
    for origin in rng.uniform(-50, 50, (6, 2)):
        p, d = nearest_points(origin, boxes)
        ref = np.array([G.nearest_box_point(origin, b) for b in boxes])
        np.testing.assert_allclose(p, ref, atol=1e-12)
        np.testing.assert_allclose(d, np.linalg.norm(ref-origin, axis=1), atol=1e-12)
        vis = visible_fractions(origin, boxes, np.arange(1, 30), skip=(0,))
        np.testing.assert_array_equal(vis, [G.visible_fraction(origin, boxes, k, skip=(0,)) for k in range(1, 30)])
    for seed in range(4):                        # lane-aligned cars: grazing rays from the mirror-mounted radars
        history = highway(seed)
        boxes, own = boxes_of(history), len(history.at(0.0).traffic)
        for spec in RADARS:
            o = sensor(spec, history)[:2]
            targets = np.delete(np.arange(len(boxes)), own)
            np.testing.assert_array_equal(visible_fractions(o, boxes, targets, skip=(own,)),
                                          [G.visible_fraction(o, boxes, k, skip=(own,)) for k in targets])


def test_noise_statistics():
    spec = replace(FRONT, **CLEAN)
    history = truth([(62.0, LANE, 25.0)])
    pose = sensor(spec, history)
    r_true, az_true = polar(pose, G.nearest_box_point(pose[:2], boxes_of(history)[0]))
    out = scans(spec, history, 2000, np.random.default_rng(7))
    assert all(len(s.range) == 1 and s.truth_id[0] == 100 for s in out)
    z = np.array([(s.range[0], s.azimuth[0], s.range_rate[0]) for s in out])
    for k, (true, sigma) in enumerate(((r_true, spec.sigma_range), (az_true, spec.sigma_azimuth),
                                       (25.0-30.0, spec.sigma_range_rate))):
        err = z[:, k]-true
        assert abs(err.mean()) < 4*sigma/np.sqrt(len(err))
        assert err.std() == pytest.approx(sigma, rel=0.08)
        assert abs(np.mean(err**3))/sigma**3 < 0.25           # symmetric
    assert az_true == 0 and r_true == pytest.approx(62.0-cg_offsets(VEH)[1]+cg_offsets(VEH)[0]+0.05)


@pytest.mark.parametrize('spec, s, v, closing', [(FRONT, 60.0, 25.0, True), (FRONT, 60.0, 35.0, False),
                                                 (REAR, -60.0, 35.0, True), (REAR, -60.0, 25.0, False)],
                         ids=['ahead slower', 'ahead faster', 'behind faster', 'behind slower'])
def test_range_rate_closing_and_opening(spec, s, v, closing):
    spec = replace(spec, pd_ref=1.0, false_alarms=0.0)
    rr = np.concatenate([x.range_rate for x in scans(spec, truth([(s, LANE, v)]), 300, np.random.default_rng(1))])
    expect = -5.0 if closing else 5.0
    assert len(rr) >= 300 and (np.sign(rr) == np.sign(expect)).all()
    assert rr.mean() == pytest.approx(expect, abs=0.02)


def test_range_rate_of_a_car_merging_towards_the_side_radar():
    spec = replace(LEFT, pd_ref=1.0, false_alarms=0.0)
    out = scans(spec, truth([(0.8, 2*LANE, 30.0, -1.0)]), 300, np.random.default_rng(2))
    rr, az = (np.concatenate([getattr(x, k) for x in out]) for k in ('range_rate', 'azimuth'))
    assert len(rr) >= 300 and (rr < 0).all()
    np.testing.assert_allclose(rr, -np.cos(az), atol=4*spec.sigma_range_rate+0.05)


def test_sensor_velocity_includes_the_lever_arm():
    yaw, vx, vy, r, dt = 0.3, 30.0, 0.5, 0.25, 1e-6
    for spec in RADARS:
        a = np.array(G.sensor_pose(spec.mount, np.array([0.0, 0.0, yaw]), VEH)[:2])
        moved = np.array([vx*np.cos(yaw)-vy*np.sin(yaw), vx*np.sin(yaw)+vy*np.cos(yaw)])*dt
        b = np.array(G.sensor_pose(spec.mount, np.r_[moved, yaw+r*dt], VEH)[:2])
        np.testing.assert_allclose(sensor_velocity(spec.mount, yaw, vx, vy, r, VEH), (b-a)/dt, atol=1e-4)


@pytest.mark.parametrize('spec', RADARS, ids=names)
def test_stationary_clutter_range_rate(spec):
    spec = replace(spec, false_alarms=10.0, res_range=0.0, **EXACT)
    ego, vy, r = (0.0, LANE, 0.05, 30.0), 0.4, 0.2
    history = truth([], ego=ego, vy=vy, r=r)
    v_sensor = sensor_velocity(spec.mount, ego[2], np.sqrt(ego[3]**2-vy**2), vy, r, VEH)
    yaw = sensor(spec, history)[2]
    out = scans(spec, history, 200, np.random.default_rng(11))
    rr, az, tid = (np.concatenate([getattr(x, k) for x in out]) for k in ('range_rate', 'azimuth', 'truth_id'))
    assert (tid == -1).all()
    still = np.abs(rr+np.cos(yaw+az)*v_sensor[0]+np.sin(yaw+az)*v_sensor[1]) < 1e-9
    assert still.mean() == pytest.approx(R.CLUTTER_STATIONARY, abs=0.04)
    lo, hi = R.CLUTTER_RANGE_RATE
    assert ((rr[~still] >= lo) & (rr[~still] <= hi)).all()
    ahead = np.abs(G.wrap(yaw+az)) < np.radians(20)            # a fixed point ahead closes at about the car's speed
    if ahead.any():
        assert (rr[still & ahead] < -0.9*ego[3]).all()


@pytest.mark.parametrize('spec', RADARS, ids=names)
def test_field_of_view(spec):
    spec = replace(spec, **CLEAN)
    history = truth()
    pose = sensor(spec, history)
    rng = np.random.default_rng(5)
    sweeps = [(rho, edge) for rho in (15.0, 0.9*spec.near_range) for edge in (spec.fov/2, -spec.fov/2)]
    if spec.fov_far != spec.fov:
        sweeps += [(rho, edge) for rho in (spec.near_range+15, spec.range_max-10) for edge in (spec.fov_far/2,
                                                                                               -spec.fov_far/2)]
    places = [(rho, edge+np.radians(b)) for rho, edge in sweeps for b in np.arange(-5, 5.01, 0.25)]
    places += [(rho, 0.0) for rho in np.arange(spec.range_max-3, spec.range_max+6, 0.25)]
    seen = []
    for rho, bearing in places:
        h = car_at(pose, rho, bearing)
        r, az = polar(pose, G.nearest_box_point(pose[:2], boxes_of(h)[0]))
        expect = r <= spec.range_max and abs(az) <= (spec.fov if r <= spec.near_range else spec.fov_far)/2
        scan = simulate_radar(spec, 0, h, 0.0, VEH, rng)
        assert (len(scan.range) == 1) == expect, (rho, np.degrees(bearing), r, np.degrees(az))
        seen.append(expect)
    assert any(seen) and not all(seen)
    if spec.fov_far < spec.fov:                  # the wide near field is cut down beyond near_range
        wide = (spec.fov+spec.fov_far)/4
        assert [len(simulate_radar(spec, 0, car_at(pose, rho, wide), 0.0, VEH, rng).range)
                for rho in (0.5*spec.near_range, 1.3*spec.near_range)] == [1, 0]


def test_occlusion_lowers_detections():
    spec = replace(FRONT, false_alarms=0.0, points_max=1)
    rng = np.random.default_rng(9)
    far = (75.0, LANE, 25.0)
    for blocker, hidden in ((None, 'none'), ((35.0, LANE, 25.0), 'all'), ((35.0, LANE+1.1, 25.0), 'part')):
        history = truth([far]+([blocker] if blocker else []))
        boxes, pose = boxes_of(history), sensor(spec, history)
        v = visible_fractions(pose[:2], boxes, [0], skip=(len(boxes)-3,))[0]
        assert {'none': v == 1, 'all': v == 0, 'part': 0.2 < v < 0.8}[hidden]
        r = G.nearest_box_point(pose[:2], boxes[0])-pose[:2]
        p = detection_probability(spec, np.hypot(*r))*(spec.occluded_pd+(1-spec.occluded_pd)*v)
        n = 3000
        rate = np.mean([100 in s.truth_id for s in scans(spec, history, n, rng)])
        assert rate == pytest.approx(p, abs=4*np.sqrt(p*(1-p)/n))
        if hidden == 'all':
            assert p == pytest.approx(spec.pd_ref*spec.occluded_pd)


def test_merge_rule():
    res_r, res_a = np.full(8, 0.4), np.full(8, np.radians(4.0))
    r = np.array([50.0, 50.3, 50.0, 51.0, 80.0, 80.2, 80.45, 50.0])
    az = np.radians([0.0, 3.0, 10.0, 0.0, 0.0, 1.0, 2.0, 0.0])
    rr = np.arange(8.0)
    tid = np.array([5, -1, 6, 7, -1, 8, 9, 10])
    sensor = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    mr, maz, mrr, mtid, ms = R._merge(r, az, rr, tid, sensor, res_r, res_a)
    groups = {int(i): (a, b, c, int(s)) for a, b, c, i, s in zip(mr, maz, mrr, mtid, ms)}
    assert sorted(groups) == [5, 6, 7, 8, 10]                  # {0, 1} merge, the chain {4, 5, 6} merges
    np.testing.assert_allclose(groups[5][:3], (50.15, np.radians(1.5), 0.5))
    np.testing.assert_allclose(groups[8][:3], (np.mean(r[4:7]), np.radians(1.0), 5.0))
    np.testing.assert_allclose(groups[6][:3], (50.0, np.radians(10.0), 2.0))
    assert groups[10][3] == 1 and groups[7][:3] == (51.0, 0.0, 3.0)
    same = R._merge(r, az, rr, tid, sensor, np.zeros(8), res_a)
    assert all(np.array_equal(a, b) for a, b in zip(same, (r, az, rr, tid, sensor)))


def test_unresolved_cars_merge_resolved_ones_do_not():
    spec = replace(FRONT, occluded_pd=1.0, **CLEAN, **EXACT)
    rng = np.random.default_rng(0)
    for s, count in ((150.0, 1), (15.0, 2)):    # side by side: 0.7 deg apart at 150 m, 10 deg at 15 m (res 4 deg)
        scan = simulate_radar(spec, 0, truth([(s, LANE-1.85, 30.0), (s, LANE+1.85, 30.0)]), 0.0, VEH, rng)
        assert len(scan.range) == count and set(scan.truth_id) <= {100, 101}
        assert abs(scan.azimuth).max() < 1e-9 if count == 1 else abs(scan.azimuth).min() > spec.res_azimuth/2


def test_reflection_points_lie_on_the_facing_outline():
    rng = np.random.default_rng(4)
    D, shift = 300, sum(cg_offsets(VEH))/2
    centre, heading = rng.uniform(-30, 30, (D, 2)), rng.normal(0, 0.4, D)
    centre = centre[np.hypot(*centre.T) > 6]
    heading, D = heading[:len(centre)], len(centre)
    origin = np.zeros((D, 2))
    cg = centre-shift*np.column_stack((np.cos(heading), np.sin(heading)))
    boxes = G.car_boxes(np.column_stack((cg, heading)), VEH)
    pts = R._reflection_points(origin, centre, heading, (VEH.length/2, VEH.width/2), rng.normal(0, 2.0, (D, 3)))
    np.testing.assert_allclose(pts[:, 0], [G.nearest_box_point(o, b) for o, b in zip(origin, boxes)], atol=1e-9)
    a, e = boxes[:, None], (boxes[:, R.NEXT]-boxes)[:, None]                 # corners and edges (D, 1, 4, 2)
    w = pts[:, :, None]-a
    t = np.clip((w*e).sum(-1)/(e*e).sum(-1), 0, 1)
    on_edge = np.hypot(*np.moveaxis(w-t[..., None]*e, -1, 0)) < 1e-9          # (D, 1+m, 4)
    facing = (e[..., 0]*(origin[:, None, None, 1]-a[..., 1])-e[..., 1]*(origin[:, None, None, 0]-a[..., 0])) < 1e-9
    assert (on_edge & facing).any(-1).all()


def test_points_per_car():
    spec = replace(FRONT, pd_ref=1.0, false_alarms=0.0, **EXACT)
    history = truth([(12.0, LANE+1.0, 30.0)])
    out = scans(spec, history, 400, np.random.default_rng(6))
    count = np.array([len(s.range) for s in out])
    assert count.min() >= 1 and count.max() <= spec.points_max and count.mean() > 1.3
    pose = sensor(spec, history)
    box = boxes_of(history)[0]
    r_near, r_far = polar(pose, G.nearest_box_point(pose[:2], box))[0], np.hypot(*(box-pose[:2]).T).max()
    r = np.concatenate([s.range for s in out])
    assert (r >= r_near-1e-9).all() and (r <= r_far).all()
    assert np.mean([np.isclose(s.range[0], r_near) for s in out]) > 0.5      # the nearest point leads most scans


def test_truth_ids_for_team_and_traffic():
    spec = replace(FRONT, **CLEAN)
    history = truth([(40.0, 0.0, 30.0)], mates=((30.0, LANE, 31.0), (-2000.0, 0.0, 30.0)))
    rng = np.random.default_rng(3)
    scan = simulate_radar(spec, 0, history, 0.0, VEH, rng)
    assert sorted(scan.truth_id) == [-3, 100]
    np.testing.assert_allclose(scan.range_rate[scan.truth_id == -3], 1.0, atol=0.5)
    scan = simulate_radar(replace(REAR, **CLEAN), 1, history, 0.0, VEH, rng)
    assert list(scan.truth_id) == [-2] and scan.car == 1 and scan.sensor == 'radar_rear'
    scan = simulate_radar(replace(FRONT, pd_ref=1.0, points_max=1, false_alarms=30.0), 0, history, 0.0, VEH, rng)
    assert (scan.truth_id == -1).sum() > 10 and np.all(np.diff(scan.range) >= 0)


def test_deterministic_given_the_generator():
    history = highway(1)
    run = lambda seed: simulate_radars(RADARS, range(3), history, 0.0, VEH, np.random.default_rng(seed))
    a, b, c = run(42), run(42), run(43)
    assert [(x.sensor, x.car) for x in a] == [(s.name, car) for car in range(3) for s in RADARS]
    for x, y in zip(a, b):
        for k in ('range', 'azimuth', 'range_rate', 'truth_id'):
            np.testing.assert_array_equal(getattr(x, k), getattr(y, k))
    assert any(len(x.range) != len(y.range) or not np.array_equal(x.range, y.range) for x, y in zip(a, c))
    one = lambda: simulate_radar(FRONT, 0, history, 0.0, VEH, np.random.default_rng(5))
    np.testing.assert_array_equal(one().range, one().range)


def test_batched_scans_match_single_scans():
    specs = [replace(s, occluded_pd=1.0, **CLEAN, **EXACT) for s in RADARS]
    history = highway(2)
    rng = np.random.default_rng(0)
    batch = simulate_radars(specs, range(3), history, 0.0, VEH, rng)
    single = [simulate_radar(s, car, history, 0.0, VEH, rng) for car in range(3) for s in specs]
    assert sum(len(x.range) for x in batch) > 20
    for x, y in zip(batch, single):
        assert (x.sensor, x.car) == (y.sensor, y.car)
        np.testing.assert_array_equal(x.truth_id, y.truth_id)
        for k in ('range', 'azimuth', 'range_rate'):
            np.testing.assert_allclose(getattr(x, k), getattr(y, k), atol=1e-12)


@pytest.mark.parametrize('spec', RADARS, ids=names)
def test_clutter_rate_and_spread(spec):
    rng = np.random.default_rng(2)
    out = [x for _ in range(20) for x in scans(spec, truth(), 100, rng)]
    n = np.array([len(x.range) for x in out])
    lam = spec.false_alarms
    assert n.mean() == pytest.approx(lam, abs=4*np.sqrt(lam/len(n)))
    assert n.var() == pytest.approx(lam, rel=0.15)                   # Poisson
    r, az, tid = (np.concatenate([getattr(x, k) for x in out]) for k in ('range', 'azimuth', 'truth_id'))
    assert (tid == -1).all()
    tol_r, tol_a = 4*spec.sigma_range, 4*spec.sigma_azimuth
    assert r.min() > R.CLUTTER_RANGE_MIN-tol_r and r.max() < spec.range_max+tol_r
    assert (np.abs(az) <= spec.fov/2+tol_a).all()
    assert (np.abs(az[r > spec.near_range+tol_r]) <= spec.fov_far/2+tol_a).all()
    near = (min(spec.near_range, spec.range_max)-R.CLUTTER_RANGE_MIN)*spec.fov
    far = max(spec.range_max-spec.near_range, 0.0)*spec.fov_far
    assert np.mean(r > spec.near_range) == pytest.approx(far/(near+far), abs=0.03)
    top = min(spec.near_range, spec.range_max)                       # uniform in range within the near field
    inner = r < top-tol_r
    assert np.mean(r[inner] < (R.CLUTTER_RANGE_MIN+top)/2) == pytest.approx(0.5, abs=0.04)


def test_speed():
    history = highway(1)
    rng = np.random.default_rng(0)
    simulate_radars(RADARS, range(3), history, 0.0, VEH, rng)
    t0 = time.perf_counter()
    for _ in range(20):
        out = simulate_radars(RADARS, range(3), history, 0.0, VEH, rng)
    step = (time.perf_counter()-t0)/20
    t0 = time.perf_counter()
    for spec in RADARS*3:
        simulate_radar(spec, 0, history, 0.0, VEH, rng)
    scan = (time.perf_counter()-t0)/12
    print(f'radar: {step*1e3:.2f} ms per step (12 scans, {sum(len(x.range) for x in out)} points), '
          f'{scan*1e3:.2f} ms per single scan')
    assert step < 0.02 and scan < 0.01
