"""Yard path planner (nohesi/yard/hybrid_astar.py): Dubins shots forward and reverse, and Hybrid A* on clearance maps
built here from boolean free masks (the same fields and `at` contract as sensors.ClearanceMap)."""
import math
from dataclasses import dataclass, replace
import numpy as np
from scipy.ndimage import distance_transform_edt
from nohesi.yard.config import Planning, Truck
from nohesi.yard.hybrid_astar import WORDS, HybridAStar, collision_free, dubins, dubins_sample, dubins_words
from nohesi.yard.truck import wrap

TRUCK, PLAN = Truck(), Planning()
R = TRUCK.min_radius


@dataclass
class Clearance:
    dist: np.ndarray
    x0: float
    y0: float
    res: float

    def at(self, xy):
        xy = np.asarray(xy, float)
        ix = np.floor((xy[..., 0]-self.x0)/self.res).astype(int)
        iy = np.floor((xy[..., 1]-self.y0)/self.res).astype(int)
        ny, nx = self.dist.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros(xy.shape[:-1])
        out[inside] = self.dist[iy[inside], ix[inside]]-self.res*math.sqrt(2)/2
        return np.maximum(out, 0.0)


def yard(boxes=(), length=100.0, width=70.0, res=0.5, border=1.0):
    """Clearance of a length x width yard walled on all sides (`border` thick), with axis-aligned obstacle boxes
    (x0, y0, x1, y1)."""
    nx, ny = int(round(length/res)), int(round(width/res))
    cx, cy = (np.arange(nx)+0.5)*res, (np.arange(ny)+0.5)*res
    X, Y = np.meshgrid(cx, cy)
    free = (X > border) & (X < length-border) & (Y > border) & (Y < width-border)
    for x0, y0, x1, y1 in boxes:
        free &= ~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
    return Clearance(distance_transform_edt(free)*res, 0.0, 0.0, res)


def check_path(clearance, path, start, goal):
    """The path starts at start, ends exactly at goal, is collision-free, samples are <= path_sample apart, and each
    step moves along the heading in its stated direction."""
    st = path.states
    assert np.allclose(st[0], start)
    assert st[-1, 0] == goal[0] and st[-1, 1] == goal[1] and abs(float(wrap(st[-1, 2]-goal[2]))) < 1e-9
    assert np.all(collision_free(TRUCK, PLAN, clearance, st))
    step = np.diff(st[:, :2], axis=0)
    assert np.all(np.hypot(*step.T) <= PLAN.path_sample+1e-9)
    along = (step[:, 0]*np.cos(st[:-1, 2])+step[:, 1]*np.sin(st[:-1, 2]))*path.direction[:-1]
    assert np.all(along >= -1e-9)
    assert np.all(np.abs(np.diff(st[:, 2])) <= np.hypot(*step.T)/R*1.05+1e-9)    # curvature within full lock


def reverse_length(path):
    return float(np.diff(path.s)[path.direction[:-1] < 0].sum())


def test_dubins_words_reach_the_goal():
    rng = np.random.default_rng(0)
    seen = dict.fromkeys(WORDS, 0)
    for _ in range(400):
        a = np.r_[rng.uniform(-15, 15, 2), rng.uniform(-np.pi, np.pi)]
        b = np.r_[rng.uniform(-15, 15, 2), rng.uniform(-np.pi, np.pi)]
        words = dubins_words(a, b, R)
        assert 'LSL' in words and 'RSR' in words     # always feasible
        for word, lengths in words.items():
            seen[word] += 1
            end = dubins_sample(a, word, lengths, R, 0.5)[-1]
            assert np.allclose(end[:2], b[:2], atol=1e-6) and abs(float(wrap(end[2]-b[2]))) < 1e-6, word
        path = dubins(a, b, R, 0.5)
        assert path is not None and abs(path.length-min(map(sum, words.values()))) < 1e-3*path.length+1e-9   # chords
        assert path.length >= np.hypot(*(b[:2]-a[:2]))-1e-9
        assert np.all(np.hypot(np.diff(path.x), np.diff(path.y)) <= 0.5+1e-12)
        assert np.all(path.direction == 1)
        back = dubins(a, b, R, 0.5, -1)
        assert np.all(back.direction == -1) and np.allclose(back.states[0], a)
        assert back.x[-1] == b[0] and back.y[-1] == b[1] and abs(float(wrap(back.yaw[-1]-b[2]))) < 1e-9
    assert all(seen.values()), seen


def test_straight_shots():
    fwd = dubins((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), R, 0.5)
    assert abs(fwd.length-10) < 1e-9 and len(fwd) == 21 and np.allclose(fwd.y, 0) and np.allclose(fwd.yaw, 0)
    assert np.all(fwd.direction == 1) and np.all(np.diff(fwd.x) > 0)
    rev = dubins((0.0, 0.0, 0.0), (-10.0, 0.0, 0.0), R, 0.5, -1)
    assert abs(rev.length-10) < 1e-9 and np.allclose(rev.y, 0, atol=1e-9) and np.allclose(rev.yaw, 0)
    assert np.all(rev.direction == -1) and np.all(np.diff(rev.x) < 0) and rev.cusps.size == 0
    # A diagonal reverse shot keeps the truck's heading opposite to its motion.
    rev = dubins((0.0, 0.0, 0.0), (-20.0, 8.0, 0.3), R, 0.5, -1)
    step = np.diff(rev.states[:, :2], axis=0)
    assert np.all(step[:, 0]*np.cos(rev.yaw[:-1])+step[:, 1]*np.sin(rev.yaw[:-1]) < 0)


def test_open_yard_plan():
    c = yard()
    start, goal = np.array([10.0, 35.0, 0.0]), np.array([75.0, 15.0, np.pi/2])
    res = HybridAStar(TRUCK, PLAN).plan(c, start, goal)
    assert res.reason == '' and res.path is not None
    check_path(c, res.path, start, goal)
    assert res.time_s < 1.0


def test_reverse_into_spot_against_a_wall():
    # The spot faces back toward the start with a wall right behind it: a forward arrival would need >= R of room
    # behind the spot, so the truck must reverse in.
    c = yard([(73.0, 0.0, 75.0, 70.0)])
    start, goal = np.array([10.0, 35.0, 0.0]), np.array([68.0, 30.0, np.pi])
    res = HybridAStar(TRUCK, PLAN).plan(c, start, goal)
    assert res.reason == '', res
    check_path(c, res.path, start, goal)
    assert np.any(res.path.direction == -1) and res.path.direction[-1] == -1
    # Default shots: drive past and back in a staged leg, not a reversing loop from afar.
    assert reverse_length(res.path) <= max(HybridAStar.staging)+1e-6 and len(res.path.cusps) == 1
    plain = HybridAStar(TRUCK, PLAN)
    plain.staging, plain.reverse_turn_max = (), math.inf      # the plain forward / reverse Dubins shots
    res = plain.plan(c, start, goal)
    assert res.reason == '' and reverse_length(res.path) > 40
    check_path(c, res.path, start, goal)


def test_gated_reverse_shot_is_the_fallback():
    # Every reverse shot is gated and the budget runs out before the primitives reach the goal: the cheapest
    # collision-free gated shot is returned instead of failing.
    c = yard([(73.0, 0.0, 75.0, 70.0)])
    planner = HybridAStar(TRUCK, replace(PLAN, max_expansions=40))
    planner.staging, planner.reverse_turn_max = (), 0.1
    start, goal = np.array([30.0, 35.0, 0.0]), np.array([68.0, 30.0, np.pi])
    res = planner.plan(c, start, goal)
    assert res.reason == '' and res.expansions == 40 and res.path.direction[-1] == -1
    check_path(c, res.path, start, goal)


def test_through_the_gap():
    c = yard([(48.0, 0.0, 52.0, 40.0), (48.0, 50.0, 52.0, 70.0)])
    start, goal = np.array([15.0, 15.0, 0.0]), np.array([85.0, 20.0, 0.0])
    res = HybridAStar(TRUCK, PLAN).plan(c, start, goal)
    assert res.reason == '', res
    check_path(c, res.path, start, goal)
    at_wall = (res.path.x >= 48) & (res.path.x <= 52)
    assert at_wall.any() and np.all((res.path.y[at_wall] > 40) & (res.path.y[at_wall] < 50))


def test_around_obstacles():
    c = yard([(30.0, 20.0, 36.0, 50.0), (50.0, 5.0, 56.0, 30.0), (62.0, 40.0, 68.0, 46.0)])
    start, goal = np.array([10.0, 35.0, 0.0]), np.array([80.0, 25.0, np.pi])
    res = HybridAStar(TRUCK, PLAN).plan(c, start, goal)
    assert res.reason == '', res
    check_path(c, res.path, start, goal)
    assert res.time_s < 3.0


def test_enclosed_goal_fails_fast():
    c = yard([(60.0, 25.0, 82.0, 27.0), (60.0, 43.0, 82.0, 45.0), (60.0, 25.0, 62.0, 45.0), (80.0, 25.0, 82.0, 45.0)])
    res = HybridAStar(TRUCK, PLAN).plan(c, (10.0, 35.0, 0.0), (68.0, 35.0, 0.0))
    assert res.path is None and res.reason == 'no_path' and res.expansions == 0 and res.time_s < 0.5
    # A 5 m gap in the enclosure passes the rear-axle grid but not the truck: the search itself fails.
    c = yard([(60.0, 25.0, 82.0, 27.0), (60.0, 43.0, 82.0, 45.0), (60.0, 25.0, 62.0, 32.5), (60.0, 37.5, 62.0, 45.0),
              (80.0, 25.0, 82.0, 45.0)])
    res = HybridAStar(TRUCK, PLAN).plan(c, (10.0, 35.0, 0.0), (68.0, 35.0, 0.0))
    assert res.path is None and res.reason in ('no_path', 'budget') and res.time_s < 3.0


def test_blocked_endpoints():
    c = yard([(40.0, 30.0, 50.0, 40.0)])
    planner = HybridAStar(TRUCK, PLAN)
    res = planner.plan(c, (44.0, 35.0, 0.0), (80.0, 35.0, 0.0))
    assert (res.path, res.reason, res.expansions) == (None, 'start_blocked', 0)
    res = planner.plan(c, (10.0, 35.0, 0.0), (44.0, 35.0, 0.0))
    assert (res.path, res.reason) == (None, 'goal_blocked')
    assert not collision_free(TRUCK, PLAN, c, [[44.0, 35.0, 0.0], [10.0, 35.0, 0.0]])[0]
    assert collision_free(TRUCK, PLAN, c, [[44.0, 35.0, 0.0], [10.0, 35.0, 0.0]])[1]


def test_budget():
    c = yard([(48.0, 0.0, 52.0, 40.0), (48.0, 50.0, 52.0, 70.0)])
    res = HybridAStar(TRUCK, replace(PLAN, max_expansions=5)).plan(c, (15.0, 15.0, 0.0), (85.0, 20.0, 0.0))
    assert (res.path, res.reason, res.expansions) == (None, 'budget', 5)
