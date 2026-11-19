"""Radar processing (nohesi/perception/processing/radar.py): ego motion, stationary rejection with the lever arm,
teammate removal, DBSCAN, one Detection per car with the cluster centre and an R that holds the spread, speed.
Scans are synthesized here from the contract geometry (not from the simulator)."""
import time
from dataclasses import replace
from math import radians
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception.config import default_radars
from nohesi.perception.geometry import car_boxes, mount_offset, rot, sensor_pose, wrap
from nohesi.perception.processing.radar import (MATE_MARGIN, box_distance, dbscan, ego_at, process_radar,
                                                sensor_velocity, stationary_range_rate)
from nohesi.perception.types import EgoState, RadarScan

VEH = scenario_v9().vehicle
FRONT, REAR, LEFT, RIGHT = default_radars()
LANE = (1.8, 5.4, 9.0)
EGO = EgoState(time=1.0, x=0.0, y=LANE[1], yaw=0.0, vx=30.0)


def v_sensor(spec, ego):
    """The sensor's world velocity, written out independently of the module."""
    ox, oy = mount_offset(spec.mount, VEH)
    v = np.array([ego.vx, ego.vy])+ego.r*np.array([-oy, ox])
    return rot(ego.yaw) @ v


def scan(spec, ego, points, vel, rng=None, ids=None):
    """RadarScan of world points (n, 2) moving at world velocities (n, 2), with the spec's noise when rng is given."""
    pose = sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), VEH)
    rel = np.asarray(points, float)-np.array(pose[:2])
    r = np.linalg.norm(rel, axis=1)
    los = rel/r[:, None]
    az = wrap(np.arctan2(rel[:, 1], rel[:, 0])-pose[2])
    rr = np.einsum('ij,ij->i', np.asarray(vel, float)-v_sensor(spec, ego), los)
    if rng is not None:
        r = r+rng.normal(0, spec.sigma_range, len(r))
        az = wrap(az+rng.normal(0, spec.sigma_azimuth, len(r)))
        rr = rr+rng.normal(0, spec.sigma_range_rate, len(r))
    return RadarScan(spec.name, 0, ego.time, r, az, rr, truth_id=ids)


def car_points(spec, ego, car, k=3, spacing=0.9):
    """k points on the outline of the car (x, y, heading) facing the sensor: the nearest one and neighbours along the
    outline every `spacing` m (as a radar sees a car's near corner and edges)."""
    box = car_boxes(np.array([car], float), VEH)[0]
    o = np.array(sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), VEH)[:2])
    ds = 0.01
    edges = [box[i]+np.linspace(0, 1, int(np.linalg.norm(box[(i+1) % 4]-box[i])/ds), endpoint=False)[:, None] *
             (box[(i+1) % 4]-box[i]) for i in range(4)]
    outline = np.concatenate(edges)
    i0 = np.argmin(np.linalg.norm(outline-o, axis=1))
    offs = (np.arange(k)-(k-1)/2)*spacing/ds
    return outline[(i0+offs.astype(int)) % len(outline)]


def cars_scan(spec, ego, cars, speeds, rng=None, k=3):
    pts = [car_points(spec, ego, c, k) for c in cars]
    vel = [np.tile([v, 0.0], (len(p), 1)) for p, v in zip(pts, speeds)]
    ids = np.repeat(np.arange(len(cars)), [len(p) for p in pts])
    return scan(spec, ego, np.concatenate(pts), np.concatenate(vel), rng, ids), pts


def test_ego_at_matches_integrated_twist():
    ego = EgoState(0.0, 10.0, 2.0, 0.1, vx=30.0, vy=0.5, r=0.2, ax=-3.0, pitch=0.01)
    moved = ego_at(ego, 0.3)
    x, y, yaw, dt = ego.x, ego.y, ego.yaw, 1e-5
    for _ in range(int(round(0.3/dt))):
        c, s = np.cos(yaw), np.sin(yaw)
        x, y, yaw = x+(c*ego.vx-s*ego.vy)*dt, y+(s*ego.vx+c*ego.vy)*dt, yaw+ego.r*dt
    assert (moved.x, moved.y, moved.yaw) == pytest.approx((x, y, yaw), abs=1e-4)
    assert moved.time == 0.3 and (moved.vx, moved.vy, moved.r, moved.ax, moved.pitch) == (30.0, 0.5, 0.2, -3.0, 0.01)
    straight = ego_at(replace(ego, r=0.0), 0.3)
    assert (straight.x, straight.y) == pytest.approx((10.0+0.3*(30*np.cos(0.1)-0.5*np.sin(0.1)),
                                                      2.0+0.3*(30*np.sin(0.1)+0.5*np.cos(0.1))))
    assert ego_at(ego, 0.0) is ego


def test_sensor_velocity_includes_lever_arm():
    ego = EgoState(0.0, 0.0, 0.0, radians(30.0), vx=0.0, r=0.5)
    for spec in (FRONT, REAR, LEFT, RIGHT):
        ox, oy = mount_offset(spec.mount, VEH)
        arm = rot(ego.yaw) @ np.array([ox, oy])
        assert sensor_velocity(spec.mount, ego, VEH) == pytest.approx(0.5*np.array([-arm[1], arm[0]]))
    assert sensor_velocity(FRONT.mount, EGO, VEH) == pytest.approx([30.0, 0.0])


@pytest.mark.parametrize('spec', [FRONT, REAR, LEFT, RIGHT], ids=lambda s: s.name)
def test_stationary_points_are_rejected_while_turning(spec):
    rng = np.random.default_rng(1)
    ego = EgoState(0.0, 0.0, LANE[1], radians(3.0), vx=30.0, vy=0.4, r=0.3)
    pose = sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), VEH)
    n = 200
    az = rng.uniform(-0.45, 0.45, n)*min(spec.fov, radians(150.0))
    r = rng.uniform(5.0, min(spec.range_max, 70.0), n)
    pts = np.array(pose[:2])+r[:, None]*np.column_stack((np.cos(pose[2]+az), np.sin(pose[2]+az)))
    fixed = scan(spec, ego, pts, np.zeros((n, 2)), rng)
    assert len(process_radar(fixed, spec, ego, None, VEH)) <= 3          # a 3 sigma test passes 0.3 % (0.5 of 200)
    noiseless = scan(spec, ego, pts, np.zeros((n, 2)))
    assert process_radar(noiseless, spec, ego, None, VEH) == []
    # The lever arm matters: the same scan read with the yaw rate ignored lets fixed points through.
    if spec is FRONT:
        assert len(process_radar(fixed, spec, replace(ego, r=0.0), None, VEH)) > 10
    # A car moving along the road is not stationary (seen 40 deg toward the front by a side radar: straight abeam, its
    # own velocity is across the line of sight, the Doppler blind band).
    a = pose[2]+{LEFT.name: -0.7, RIGHT.name: 0.7}.get(spec.name, 0.0)
    cars, _ = cars_scan(spec, ego, [(pose[0]+20*np.cos(a), pose[1]+20*np.sin(a), 0.0)], [24.0], rng)
    assert len(process_radar(cars, spec, ego, None, VEH)) == 1


def test_stationary_range_rate_formula():
    pose = (0.0, 0.0, radians(90.0), 0.5)
    v = np.array([30.0, 0.0])
    assert stationary_range_rate([0.0, -np.pi/2, np.pi/2], pose, v) == pytest.approx([0.0, -30.0, 30.0], abs=1e-9)


def test_three_points_per_car_give_one_detection_each():
    rng = np.random.default_rng(2)
    cars = [(40.0, LANE[1], 0.0), (25.0, LANE[2], 0.0), (60.0, LANE[0], 0.0), (12.0, LANE[0], 0.0)]
    speeds = [25.0, 34.0, 27.0, 29.0]
    sc, pts = cars_scan(FRONT, EGO, cars, speeds, rng)
    dets = process_radar(sc, FRONT, EGO, None, VEH)
    assert len(dets) == 4 and all(d.meta['n_points'] == 3 for d in dets)
    pose = sensor_pose(FRONT.mount, (EGO.x, EGO.y, EGO.yaw), VEH)
    for d in dets:
        centre = d.sensor_pose[:2]+d.z[0]*np.array([np.cos(pose[2]+d.z[1]), np.sin(pose[2]+d.z[1])])
        k = int(np.argmin([np.linalg.norm(p.mean(0)-centre) for p in pts]))
        assert np.linalg.norm(pts[k].mean(0)-centre) < 0.5
        assert d.z[2] == pytest.approx(speeds[k]-30.0, abs=0.3)


def test_adjacent_cars_at_different_speeds_stay_apart():
    """Two cars side by side 80 m ahead, 2.6 m apart (one mid lane change), so their facing points are within the
    1.5 m position scale: 7 m/s of speed difference keeps them apart; at the same speed they merge."""
    cars = [(80.0, LANE[1], 0.0), (80.0, LANE[1]+2.6, 0.0)]
    pts = [car_points(FRONT, EGO, c) for c in cars]
    gap = min(np.linalg.norm(a-b) for a in pts[0] for b in pts[1])
    assert gap < 1.5
    for speeds, n in (((26.0, 33.0), 2), ((29.0, 29.0), 1)):
        sc, _ = cars_scan(FRONT, EGO, cars, speeds)
        dets = process_radar(sc, FRONT, EGO, None, VEH)
        assert len(dets) == n
        if n == 2:
            assert sorted(d.z[2] for d in dets) == pytest.approx([-4.0, 3.0], abs=0.05)


def test_car_beside_a_side_radar_is_one_cluster():
    """A car beside the left radar 6 m/s faster: four points spanning 70 deg of bearing have range rates 2 m/s apart,
    which one rigid car shows; the plain (x, y, range_rate) metric splits it, the rigid-car allowance keeps it whole.
    Its point exactly abeam would match a fixed point's range rate (the Doppler blind band), so none is put there."""
    car = (EGO.x+0.8, LANE[2], 0.0)
    sc, _ = cars_scan(LEFT, EGO, [car], [36.0], k=4)
    assert np.ptp(sc.range_rate) > 6.0 and np.min(np.abs(np.diff(sc.range_rate))) > 1.5
    assert len(process_radar(sc, LEFT, EGO, None, VEH, rel_speed=0.0)) == 4
    (d,) = process_radar(sc, LEFT, EGO, None, VEH)
    assert d.meta['n_points'] == 4
    abeam = scan(LEFT, EGO, [[mount_offset(LEFT.mount, VEH)[0], LANE[2]-VEH.width/2]], [[36.0, 0.0]])
    assert process_radar(abeam, LEFT, EGO, None, VEH) == []


def test_cluster_centre_and_covariance():
    sc, pts = cars_scan(FRONT, EGO, [(30.0, LANE[2], 0.0)], [24.0])
    (d,) = process_radar(sc, FRONT, EGO, None, VEH)
    pose = sensor_pose(FRONT.mount, (EGO.x, EGO.y, EGO.yaw), VEH)
    local = (pts[0]-np.array(pose[:2])) @ rot(pose[2])      # world -> sensor frame (rows times R)
    c = local.mean(0)
    assert d.z[:2] == pytest.approx([np.hypot(*c), np.arctan2(c[1], c[0])], abs=1e-9)
    assert d.z[2] == pytest.approx(sc.range_rate.mean(), abs=1e-12)
    dev = np.column_stack((sc.range-d.z[0], wrap(sc.azimuth-d.z[1]), sc.range_rate-d.z[2]))
    noise = np.diag([FRONT.sigma_range**2, FRONT.sigma_azimuth**2, FRONT.sigma_range_rate**2])
    assert d.R == pytest.approx(noise/3+dev.T @ dev/3, rel=1e-9, abs=1e-15)
    assert np.allclose(d.R, d.R.T) and np.all(np.linalg.eigvalsh(d.R) > 0)
    assert d.R[0, 0] > noise[0, 0]/3+0.1            # the points spread over ~1.8 m of the car's corner

    single = scan(FRONT, EGO, pts[0][:1], [[24.0, 0.0]])
    (d1,) = process_radar(single, FRONT, EGO, None, VEH)
    assert d1.meta['n_points'] == 1 and d1.R == pytest.approx(noise)
    assert d1.z == pytest.approx([single.range[0], single.azimuth[0], single.range_rate[0]])


def test_detection_fields_and_ego_moved_to_scan_time():
    early = EgoState(0.9, -3.0, LANE[1], 0.0, vx=30.0)
    sc, _ = cars_scan(LEFT, EGO, [(4.0, LANE[2]+3.6, 0.0)], [33.0])
    dets = process_radar(sc, LEFT, early, None, VEH)
    assert len(dets) == 1
    d = dets[0]
    assert (d.kind, d.sensor, d.car, d.time) == ('radar', LEFT.name, 0, 1.0)
    assert d.ego.time == 1.0 and d.ego.x == pytest.approx(0.0)
    assert d.sensor_pose == pytest.approx(sensor_pose(LEFT.mount, (0.0, LANE[1], 0.0), VEH))
    assert d.meta == {'n_points': 3, 'scan': 1.0}
    assert d.z.shape == (3,) and d.R.shape == (3, 3)


def test_teammate_points_are_dropped():
    mate = np.array([[20.0, LANE[1], 0.0], [-200.0, LANE[1], 0.0]])
    box = car_boxes(mate[:1], VEH)[0]
    rear = box[[0, 3]].mean(0)
    pts = np.array([rear, rear+[0.0, 0.6], rear-[0.8, 0.0], rear-[MATE_MARGIN+0.3, 0.0]])
    vel = np.tile([31.0, 0.0], (4, 1))
    traffic, _ = cars_scan(FRONT, EGO, [(45.0, LANE[2], 0.0)], [26.0])
    sc = scan(FRONT, EGO, np.concatenate((pts, car_points(FRONT, EGO, (45.0, LANE[2], 0.0)))),
              np.concatenate((vel, np.tile([26.0, 0.0], (3, 1)))))
    dets = process_radar(sc, FRONT, EGO, mate, VEH)
    assert sorted(d.meta['n_points'] for d in dets) == [1, 3]       # only the point 1.3 m behind the mate stays
    assert len(process_radar(sc, FRONT, EGO, None, VEH)) == 2
    assert len(process_radar(sc, FRONT, EGO, mate[:, :3].tolist(), VEH)) == 2


def test_box_distance():
    box = car_boxes(np.array([[0.0, 0.0, 0.0]]), VEH)
    front = box[0, 1, 0]
    d = box_distance(np.array([[0.0, 0.0], [front+0.5, 0.0], [0.0, VEH.width/2+2.0], [front+3.0, VEH.width/2+4.0]]), box)
    assert d[:, 0] == pytest.approx([0.0, 0.5, 2.0, 5.0])
    assert box_distance(np.zeros((0, 2)), box).shape == (0, 1)


def test_dbscan():
    chain = np.column_stack((np.arange(5)*0.9, np.zeros(5)))
    pts = np.vstack((chain, [[10.0, 0.0]], [[20.0, 0.0], [20.5, 0.0], [21.0, 0.0], [21.5, 0.0]]))
    assert dbscan(pts, 1.0, 1).tolist() == [0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
    labels = dbscan(pts, 0.6, 3)      # cores: points with 2 neighbours; chain ends 0.9 apart are too far at eps 0.6
    assert labels.tolist() == [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0]
    assert dbscan(np.zeros((0, 3))).shape == (0,)
    assert dbscan(np.zeros((3, 3))).tolist() == [0, 0, 0]


def test_empty_and_invalid_entries():
    empty = RadarScan(FRONT.name, 0, 1.0, np.zeros(0), np.zeros(0), np.zeros(0))
    assert process_radar(empty, FRONT, EGO, None, VEH) == []
    sc, _ = cars_scan(FRONT, EGO, [(40.0, LANE[1], 0.0)], [25.0])
    bad = RadarScan(FRONT.name, 0, 1.0, np.r_[sc.range, np.nan], np.r_[sc.azimuth, 0.0], np.r_[sc.range_rate, -5.0])
    dets = process_radar(bad, FRONT, EGO, None, VEH)
    assert len(dets) == 1 and dets[0].meta['n_points'] == 3


def test_noise_free_sensor_keeps_positive_definite_r():
    spec = replace(FRONT, sigma_range=0.0, sigma_azimuth=0.0, sigma_range_rate=0.0)
    sc, pts = cars_scan(spec, EGO, [(40.0, LANE[1], 0.0)], [25.0], k=1)
    (d,) = process_radar(sc, spec, EGO, None, VEH)
    assert np.all(np.linalg.eigvalsh(d.R) > 0)


def test_speed():
    rng = np.random.default_rng(3)
    cars = [(10.0+12*i, LANE[i % 3], 0.0) for i in range(12)]
    sc, _ = cars_scan(FRONT, EGO, cars, rng.uniform(24, 36, len(cars)), rng)
    clutter = scan(FRONT, EGO, rng.uniform((5, -10), (150, 20), (6, 2)), rng.uniform(-5, 5, (6, 2)), rng)
    sc = RadarScan(FRONT.name, 0, 1.0, np.r_[sc.range, clutter.range], np.r_[sc.azimuth, clutter.azimuth],
                   np.r_[sc.range_rate, clutter.range_rate])
    mates = np.array([[-15.0, LANE[1], 0.0], [-30.0, LANE[1], 0.0]])
    process_radar(sc, FRONT, EGO, mates, VEH)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        dets = process_radar(sc, FRONT, EGO, mates, VEH)
    per = (time.perf_counter()-t0)/n
    assert len(dets) >= 12 and per < 2e-3, per
