"""Planner.escape_edge_grip: a control-law car near the road edge and moving toward it brakes no harder than the
friction circle leaves after its steering (planning.fallback.edge_grip_accel; off by default)."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import Planner, Scenario, bmw_m3
from nohesi.planning.fallback import edge_grip_accel
from nohesi.road import Road
from tests.test_nohesi import snapshot

ROAD, P, BRAKE = Road(Scenario().road), bmw_m3('dynamic'), 8.2404
ZONE = Planner().edge_grip_zone


def car(d, psi, v=18.0, vy=0.0):
    """Car 0 at lateral offset d, heading psi (rad, + toward +d), speed v, body lateral speed vy."""
    snap = snapshot(0.0, [(100.0, d, v), (-500.0, 0.0, v), (-1000.0, 0.0, v)], [])
    snap.team[0, 2] = psi
    return replace(snap, team_vy=np.array([vy, 0.0, 0.0]))


def test_braking_gives_way_to_steering_near_the_edge_moving_out():
    # Seed 1040: 3.96 m, 0.099 rad toward the left edge, 18 m/s, the law escape braking at the cap and steering -0.056.
    a = edge_grip_accel(car(3.96, 0.099), ROAD, P, 0, (-BRAKE, -0.056), BRAKE, ZONE)
    lateral = 18.0**2*np.tan(0.056)/P.wheelbase
    assert a == pytest.approx(-np.sqrt(BRAKE**2-lateral**2)) and -6.0 < a < -4.0
    # The right edge too, mirrored.
    assert edge_grip_accel(car(-3.96, -0.099), ROAD, P, 0, (-BRAKE, 0.056), BRAKE, ZONE) == pytest.approx(a)
    # A steering demand past the brake budget: no braking at all.
    assert edge_grip_accel(car(3.96, 0.099, v=30.0), ROAD, P, 0, (-BRAKE, -0.1), BRAKE, ZONE) == 0.0


@pytest.mark.parametrize('d, psi, vy, control', [
    (2.0, 0.099, 0.0, (-BRAKE, -0.056)),     # corner more than edge_grip_zone from the edge
    (3.96, -0.099, 0.0, (-BRAKE, -0.056)),   # near the edge but heading back in
    (3.96, 0.0, -1.0, (-BRAKE, -0.056)),     # straight, sliding inward
    (3.96, 0.099, 0.0, (2.0, -0.056)),       # accelerating
    (3.96, 0.099, 0.0, (-BRAKE, 0.0)),       # no steering demand: full braking stays
])
def test_unchanged_otherwise(d, psi, vy, control):
    assert edge_grip_accel(car(d, psi, vy=vy), ROAD, P, 0, control, BRAKE, ZONE) == pytest.approx(control[0])


def test_sliding_outward_counts_even_when_heading_straight():
    a = edge_grip_accel(car(3.96, 0.0, vy=1.0), ROAD, P, 0, (-BRAKE, -0.056), BRAKE, ZONE)
    assert a > -BRAKE
