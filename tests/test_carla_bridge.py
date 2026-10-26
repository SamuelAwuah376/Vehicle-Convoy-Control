"""CARLA bridge pieces that need no CARLA client: the OpenDRIVE road, frame conversions, the calibration and the
low-level longitudinal controller."""
from dataclasses import replace
import math
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from nohesi.carla.control import Calibration, Longitudinal
from nohesi.carla.opendrive import from_carla, to_carla, xodr
from nohesi.config import Road


def test_xodr_straight_road_lanes():
    road = Road(lanes=3, lane_width=3.6, length=10000.0)
    root = ET.fromstring(xodr(road, speed_limit=60.0))
    geometry = root.find('road/planView/geometry')
    assert float(geometry.get('y')) == pytest.approx(5.4) and float(geometry.get('hdg')) == 0.0
    assert float(geometry.get('length')) == pytest.approx(10000.0)
    lanes = root.findall('road/lanes/laneSection/right/lane')
    assert [int(l.get('id')) for l in lanes] == [-1, -2, -3]
    assert all(float(l.find('width').get('a')) == pytest.approx(3.6) for l in lanes)
    marks = [l.find('roadMark').get('type') for l in lanes]
    assert marks == ['broken', 'broken', 'solid']   # lane lines dashed, right edge solid


def test_frames_round_trip_and_handedness():
    x, y, yaw = to_carla(100.0, 3.6, 0.2)
    assert (float(x), float(y), float(yaw)) == pytest.approx((100.0, -3.6, -math.degrees(0.2)))
    bx, by, byaw = from_carla(x, y, yaw)
    assert (float(bx), float(by), float(byaw)) == pytest.approx((100.0, 3.6, 0.2))
    assert float(from_carla(0, 0, 190.0)[2]) == pytest.approx(math.radians(170.0))


def toy_calibration():
    speed = (0.0, 20.0, 40.0)
    return Calibration(blueprint='vehicle.tesla.model3', vehicle={'wheelbase': 3.0}, x_cg=0.1, x_velocity=0.1,
                       steer_speed=(0.0, 40.0), steer_gain=(1.2, 0.8), speed=speed,
                       throttle=(0.0, 0.5, 1.0), throttle_accel=((-0.2, -0.3, -0.6), (2.0, 1.5, 1.0), (8.0, 6.0, 4.0)),
                       brake=(0.0, 0.5, 1.0), brake_accel=((-0.2, -0.3, -0.6), (-5.0, -5.0, -5.0), (-10.0, -10.0, -10.0)))


def test_longitudinal_inverts_the_maps():
    cal = toy_calibration()
    ctl = Longitudinal(cal, 0.01, k_i=0.0)
    assert ctl(8.0, 0.0) == (pytest.approx(1.0), 0.0)
    assert ctl(1.5, 20.0) == (pytest.approx(0.5), 0.0)
    assert ctl(-5.0, 20.0) == (0.0, pytest.approx(0.5))
    assert ctl(-7.5, 20.0) == (0.0, pytest.approx(0.75))
    assert ctl(-0.3, 20.0) == (pytest.approx(0.0), 0.0)   # coasting
    assert cal.steer_gain_at(20.0) == pytest.approx(1.0)


def test_longitudinal_keeps_the_throttle_floor_while_braking():
    """Below the floor CARLA's drivetrain disengages: braking holds the throttle at the floor."""
    cal = replace(toy_calibration(), throttle=(0.3, 0.6, 1.0))
    ctl = Longitudinal(cal, 0.01, k_i=0.0)
    assert ctl(-5.0, 20.0) == (0.3, pytest.approx(0.5))
    assert ctl(1.5, 20.0) == (pytest.approx(0.6), 0.0)


def test_longitudinal_floor_follows_speed():
    """With brake_throttle the floor is the speed's zero-torque throttle: below its acceleration the car brakes with
    the throttle held there, never through the engine."""
    cal = replace(toy_calibration(), brake_throttle=(0.0, 0.5, 1.0))
    ctl = Longitudinal(cal, 0.01, k_i=0.0)
    assert ctl(1.0, 20.0) == (0.5, pytest.approx(0.0))            # below the floor's 1.5 m/s^2: brake from the floor
    assert ctl(2.0, 20.0)[0] > 0.5 and ctl(2.0, 20.0)[1] == 0.0
    assert ctl(-7.5, 20.0) == (0.5, pytest.approx(0.75))
    assert cal.floor_at(10.0) == pytest.approx(0.25)


def test_longitudinal_integral_corrects_a_weak_car():
    """The car delivers half the mapped acceleration: the integral raises the request until it tracks."""
    cal = toy_calibration()
    ctl = Longitudinal(cal, 0.01, k_i=4.0)
    v, a_cmd = 20.0, 1.0
    for _ in range(400):
        th, br = ctl(a_cmd, v)
        a_mapped = np.interp(th, cal.throttle, [np.interp(v, cal.speed, row) for row in cal.throttle_accel])
        v += 0.5*a_mapped*0.01
    assert ctl.bias > 0.5
    assert 0.5*a_mapped == pytest.approx(a_cmd, abs=0.05)
