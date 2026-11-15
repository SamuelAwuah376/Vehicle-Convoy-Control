import itertools
import time
import numpy as np
import pytest
from nohesi.geometry import clearances
from nohesi.yard.config import Planning, Truck
from nohesi.yard.coordination import Coordinator, Zone, conflict_matrix, conflict_zones
from nohesi.yard.paths import Path
from nohesi.yard.truck import corners, disk_centres, disk_offsets

T, P = Truck(), Planning()
_, R = disk_offsets(T)
REACH = 2*R+P.zone_margin
GATE = round(P.gate_buffer/P.path_sample)


def line(x0, y0, yaw, length, reverse=False):
    """Straight path from (x0, y0) at heading yaw, driven forward (or in reverse: moving against the heading)."""
    s = np.arange(0, length+1e-9, P.path_sample)*(-1 if reverse else 1)
    return Path(x0+s*np.cos(yaw), y0+s*np.sin(yaw), np.full(len(s), yaw), np.full(len(s), -1 if reverse else 1))


def towards(x0, y0, x1, y1):
    yaw = np.arctan2(y1-y0, x1-x0)
    return line(x0, y0, yaw, np.hypot(x1-x0, y1-y0))


def curve(rng, length=150.0):
    """Forward path of slowly varying curvature from a random start."""
    step, f, ph = P.path_sample, rng.uniform(0.02, 0.1), rng.uniform(0, 2*np.pi)
    s = np.arange(0, length+1e-9, step)
    yaw = rng.uniform(-np.pi, np.pi)+np.cumsum(np.r_[0, np.full(len(s)-1, step)]*0.12*np.sin(f*s+ph))
    x, y = rng.uniform(-20, 20, 2)
    return Path(x+np.r_[0, np.cumsum(step*np.cos(yaw[:-1]))], y+np.r_[0, np.cumsum(step*np.sin(yaw[:-1]))], yaw,
                np.ones(len(s), int))


def brute(a, b):
    """(Na, Nb) conflict matrix from every cover-disk pair."""
    da, db = disk_centres(T, a.states), disk_centres(T, b.states)
    return np.linalg.norm(da[:, None, :, None]-db[None, :, None, :], axis=-1).min(axis=(2, 3)) < REACH


def check_zones(a, b, zones):
    """Every conflicting sample pair lies in exactly one zone, zones are disjoint on both paths, and each bound is a
    conflicting sample."""
    hit = brute(a, b)
    inside = np.zeros_like(hit, int)
    for z in zones:
        inside[z.a_in:z.a_out+1, z.b_in:z.b_out+1] += 1
        assert hit[z.a_in].any() and hit[z.a_out].any() and hit[:, z.b_in].any() and hit[:, z.b_out].any()
    assert (inside[hit] == 1).all()
    for z, y in itertools.combinations(zones, 2):
        assert z.a_out < y.a_in or y.a_out < z.a_in
        assert z.b_out < y.b_in or y.b_out < z.b_in


def simulate(paths, order=None, ticks=3000, coord=None):
    """Kinematic run: each truck moves one sample per tick unless stopped. Asserts at every tick that no two trucks
    hold the same zone and that no two footprints come within zone_margin. Returns (coordinator, progress history
    (T, n), stop history)."""
    n = len(paths)
    coord = coord or Coordinator(T, P, n)
    if order is not None:
        coord.order = list(order)
    for i, path in enumerate(paths):
        coord.set_path(i, path, 0)
    p, ends = np.zeros(n, int), np.array([len(path)-1 for path in paths])
    hist, stops_hist = [p.copy()], []
    for _ in range(ticks):
        stops = coord.update(p, [True]*n)
        for z in coord.zones:
            assert not (coord._hold(z.a, z) and coord._hold(z.b, z)), z
        if stops_hist:   # a rolling truck learns of a new (or nearer) stop at least LOOKAHEAD - 1 sample ahead of it
            for i, (s, prev) in enumerate(zip(stops, stops_hist[-1])):
                if hist[-1][i] > hist[-2][i] and s is not None and (prev is None or s < prev):
                    assert s-p[i] >= coord.look-1, (i, int(p[i]), s)
        stops_hist.append(stops)
        limit = np.array([e if s is None else min(s, e) for s, e in zip(stops, ends)])
        p = p+(p < limit)
        hist.append(p.copy())
        if (p == ends).all() or len(hist) > 3 and (np.diff(hist[-3:], axis=0) == 0).all():
            break   # done, or nobody moved for two ticks: a fixed point
    hist = np.array(hist)
    states = [paths[i].states[hist[:, i]] for i in range(n)]
    disks, feet = [disk_centres(T, s) for s in states], [corners(T, s) for s in states]
    for i, j in itertools.combinations(range(n), 2):
        assert np.linalg.norm(disks[i][:, :, None]-disks[j][:, None], axis=-1).min() >= REACH
        assert clearances(feet[i], feet[j]).min() >= P.zone_margin
    return coord, hist, stops_hist


def first_reach(hist, i, k):
    return int(np.argmax(hist[:, i] >= k))


# --- conflict zones ---------------------------------------------------------------------------------------------------

def test_right_angle_crossing_gives_one_zone_with_the_analytic_bounds():
    a, b = towards(-40, 0, 40, 0), towards(0, -40, 0, 40)
    zones = conflict_zones(T, P, 0, a, 1, b)
    assert len(zones) == 1
    z = zones[0]
    assert (z.a, z.b) == (0, 1)
    # a's rear axle at x conflicts while some disk (offsets off) is within REACH of the line x = 0 (b's disks there)
    off, _ = disk_offsets(T)
    lo, hi = -REACH-off.max(), REACH-off.min()
    assert a.x[z.a_in]-P.path_sample < lo <= a.x[z.a_in] and a.x[z.a_out] < hi <= a.x[z.a_out]+P.path_sample
    assert b.y[z.b_in]-P.path_sample < lo <= b.y[z.b_in] and b.y[z.b_out] < hi <= b.y[z.b_out]+P.path_sample
    check_zones(a, b, zones)
    # the arguments' order does not matter: zones always carry a < b
    assert conflict_zones(T, P, 1, b, 0, a) == zones


def test_parallel_paths_10m_apart_do_not_conflict():
    assert conflict_zones(T, P, 0, towards(-40, 0, 40, 0), 1, towards(-40, 10, 40, 10)) == []
    assert conflict_zones(T, P, 0, towards(-40, 0, 40, 0), 2, towards(40, -10, -40, -10)) == []


def test_path_ending_on_the_other_path_gives_a_zone_with_its_end():
    a = line(-30, 0, np.pi, 32, reverse=True)      # reverses east, nose west, parks across b's lane
    b = towards(0, -40, 0, 40)
    zones = conflict_zones(T, P, 0, a, 1, b)
    assert len(zones) == 1 and zones[0].a_out == len(a)-1
    check_zones(a, b, zones)


def test_double_crossing_gives_two_zones_and_matches_brute_force():
    a = towards(-40, 0, 80, 0)
    s = np.arange(0, 120+1e-9, P.path_sample)
    yaw = np.arctan2(-30*np.pi/60*np.sin(np.pi*s/60), 1.0)    # b: y = 15 cos(pi x/60): crosses y = 0 twice
    b = Path(s-30, 15*np.cos(np.pi*s/60), yaw, np.ones(len(s), int))
    zones = conflict_zones(T, P, 0, a, 1, b)
    assert len(zones) == 2
    check_zones(a, b, zones)


def test_random_curved_paths_match_brute_force_and_merge_correctly():
    rng = np.random.default_rng(3)
    for _ in range(8):
        a, b = curve(rng), curve(rng)
        assert (conflict_matrix(T, P, a, b) == brute(a, b)).all()
        check_zones(a, b, conflict_zones(T, P, 0, a, 1, b))


def test_150m_paths_take_under_50ms():
    rng = np.random.default_rng(1)
    pairs = [(towards(-75, 0, 75, 0), towards(0, -75, 0, 75)), (towards(-75, 0, 75, 0), towards(-75, 4, 75, 4)),
             (curve(rng), curve(rng))]
    for a, b in pairs:
        conflict_zones(T, P, 0, a, 1, b)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            conflict_zones(T, P, 0, a, 1, b)
            times.append(time.perf_counter()-t0)
        assert min(times) < 0.05, times


# --- the shield -------------------------------------------------------------------------------------------------------

CROSS = [towards(-40, 0, 40, 0), towards(0, -40, 0, 40)]


@pytest.mark.parametrize('order', [[0, 1], [1, 0]])
def test_two_trucks_crossing_finish_and_order_picks_who_goes_first(order):
    coord, hist, stops = simulate(CROSS, order)
    assert (hist[-1] == [len(p)-1 for p in CROSS]).all()
    z = coord.zones[0]
    first, second = order
    assert first_reach(hist, first, z.side(first)[0]) < first_reach(hist, second, z.side(second)[0])
    assert all(s[first] is None for s in stops)
    held = [s[second] for s in stops if s[second] is not None]
    assert held and set(held) == {z.side(second)[0]-GATE-1}


STAR = [line(-40*np.cos(a), -40*np.sin(a), a, 80) for a in np.radians([0, 60, 120])]
TRIANGLE = [towards(-40, 0, 70, 0), towards(0, -40, 0, 70), towards(60, -20, -20, 60)]


@pytest.mark.parametrize('paths', [STAR, TRIANGLE], ids=['star', 'triangle'])
def test_three_trucks_never_share_a_zone_and_all_finish(paths):
    for order in itertools.permutations(range(3)):
        for trim in [(0, 0, 0), (0, 7, 15), (12, 0, 5), (20, 10, 0)]:
            cut = [p.sub(k, len(p)-1) for p, k in zip(paths, trim)]
            coord, hist, _ = simulate(cut, order)
            assert (hist[-1] == [len(p)-1 for p in cut]).all(), (order, trim)


def test_three_trucks_order_sets_the_crossing_sequence_at_a_shared_centre():
    for order in itertools.permutations(range(3)):
        coord, hist, _ = simulate(STAR, order)
        enter = [first_reach(hist, i, min(z.side(i)[0] for z in coord.zones if i in (z.a, z.b))) for i in range(3)]
        assert list(np.argsort(enter)) == list(order)


def abc(lead):
    """Review repro: A drives along y = 0, crossing B at x = 30 and C at x = 51, so A's zones with B and with C are
    separate clusters a little over a gate apart, and A is already within reach of requesting AC while still inside
    AB. C is trimmed to reach its own request window `lead` ticks after A reaches A's (both moving a sample a tick).
    Returns (paths, A's request sample for AC)."""
    paths = [towards(0, 0, 80, 0), towards(30, -45, 30, 40), towards(51, -70, 51, 40)]
    coord = Coordinator(T, P, 3)
    for i, path in enumerate(paths):
        coord.set_path(i, path, 0)
    ab, ac = (next(z for z in coord.zones if (z.a, z.b) == pair) for pair in ((0, 1), (0, 2)))
    ask = ac.a_in-GATE-coord.look
    assert ab.a_out < ac.a_in-GATE-1 and ab.a_in-GATE < ask+lead <= ab.a_out
    paths[2] = paths[2].sub(ac.b_in-(ask+lead+GATE+coord.look), len(paths[2])-1)
    return paths, ask


def zone(coord, a, b):
    return next(z for z in coord.zones if (z.a, z.b) == (a, b))


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_a_truck_inside_one_cluster_requests_the_next_in_time(order):
    # C wants AC while A, inside AB, is already within reach of it: A asked first and gets both, whatever the order
    paths, ask = abc(5)
    coord, hist, stops = simulate(paths, order)
    ac = zone(coord, 0, 2)
    assert (hist[-1] == [len(p)-1 for p in paths]).all()
    assert all(s[0] is None for s in stops)
    assert first_reach(hist, 0, ac.a_in) < first_reach(hist, 2, ac.b_in)
    assert any(s[2] == ac.b_in-GATE-1 for s in stops)


@pytest.mark.parametrize('order', [[2, 0, 1], [1, 2, 0], [0, 2, 1]])
def test_a_tie_for_the_next_cluster_is_denied_a_lookahead_ahead(order):
    # A (inside AB) and C reach their AC windows on the same tick: the earlier in order takes AC, and when that is C
    # A hears of its stop LOOKAHEAD - 1 sample before it (simulate() checks this for every stop)
    paths, ask = abc(0)
    coord, hist, stops = simulate(paths, order)
    ac = zone(coord, 0, 2)
    assert (hist[-1] == [len(p)-1 for p in paths]).all()
    if order.index(2) < order.index(0):
        t = next(t for t, s in enumerate(stops) if s[0] is not None)
        assert hist[t][0] == ask and stops[t][0] == ac.a_in-GATE-1 == ask+coord.look-1
        assert first_reach(hist, 2, ac.b_in) < first_reach(hist, 0, ac.a_in)
    else:
        assert all(s[0] is None for s in stops)


def test_parked_truck_in_the_zone_blocks_the_other_for_good():
    paths = [line(-30, 0, np.pi, 32, reverse=True), towards(0, -45, 0, 40)]
    coord, hist, stops = simulate(paths, [0, 1])
    z = coord.zones[0]
    assert hist[-1][0] == len(paths[0])-1 and z.a_out == len(paths[0])-1
    assert hist[-1][1] == z.b_in-GATE-1 and stops[-1][1] == z.b_in-GATE-1
    assert coord.blocking(1) == [0] and coord.blocking(0) == []
    near = coord.zones_of(1)[0]
    assert near[0] == pytest.approx(P.path_sample) and near[1] < 0 and near[2]
    assert coord.priority_wait(1, hist[-1]) is False      # truck 0 sits at its zone's out: only the shield stops 1
    # order only breaks ties: truck 0 asked first above. When truck 1 gets there first it crosses, then truck 0 parks
    paths[1] = towards(0, -30, 0, 40)
    coord, hist, _ = simulate(paths, [0, 1])
    assert (hist[-1] == [len(p)-1 for p in paths]).all()


def test_simultaneous_requests_go_to_the_earlier_truck_and_hold_releases():
    coord = Coordinator(T, P, 2)
    for i, path in enumerate(CROSS):
        coord.set_path(i, path, 0)
    z = coord.zones[0]
    start = [z.side(i)[0]-GATE-coord.look for i in (0, 1)]       # both just within reach of requesting
    assert coord.update(start, [True, True]) == [None, z.b_in-GATE-1]
    assert coord.owner == [0] and coord.blocking(1) == [0]
    # one sample farther out, nobody requests it
    coord = Coordinator(T, P, 2)
    for i, path in enumerate(CROSS):
        coord.set_path(i, path, 0)
    assert coord.update([k-1 for k in start], [True, True]) == [None, None] and coord.owner == [None]
    # truck 0 holds still (HOLD) before its gate: its token lapses and truck 1 takes the zone
    coord.update(start, [True, True])
    assert coord.owner == [0]
    assert coord.update(start, [False, True]) == [z.a_in-GATE-1, None] and coord.owner == [1]
    # a granted token is kept against a later request of a higher-priority truck
    assert coord.update(start, [True, True]) == [z.a_in-GATE-1, None]
    # a truck inside its gate holds the zone whatever it wants; the other stays at its gate
    inside = [z.a_in-GATE+1, z.b_in-GATE-1]
    coord = Coordinator(T, P, 2)
    for i, path in enumerate(CROSS):
        coord.set_path(i, path, 0)
    assert coord.update(inside, [False, True]) == [None, z.b_in-GATE-1] and coord.blocking(1) == [0]
    # past the gate start but short of the zone when the partner took it: the stop is the current progress
    late = [z.a_in-GATE, z.b_in-2]
    coord = Coordinator(T, P, 2)
    for i, path in enumerate(CROSS):
        coord.set_path(i, path, 0)
    coord.order = [1, 0]
    assert coord.update(late, [True, True]) == [z.a_in-GATE, None]


def test_set_path_recomputes_zones_on_the_current_path():
    coord = Coordinator(T, P, 3)
    coord.set_path(0, CROSS[0], 0)
    assert coord.zones == []
    coord.set_path(1, CROSS[1], 0)
    (z,) = coord.zones
    coord.set_path(1, CROSS[1].sub(20, len(CROSS[1])-1), 0)            # indices now count from sample 20
    assert coord.zones == [Zone(0, 1, z.a_in, z.a_out, z.b_in-20, z.b_out-20)]
    coord.set_path(2, towards(-40, 10, 40, 10), 0)                      # crosses truck 1 only
    assert sorted((y.a, y.b) for y in coord.zones) == [(0, 1), (1, 2)]
    coord.set_path(1, None, 0)
    assert coord.zones == [] and coord.update([0, 0, 0], [True]*3) == [None]*3
    assert coord.zones_of(0) == [None]*3 and coord.zones_of(1) == [None]*3


def test_a_partner_reservation_survives_a_replan():
    coord = Coordinator(T, P, 2)
    for i, path in enumerate(CROSS):
        coord.set_path(i, path, 0)
    z = coord.zones[0]
    start = [z.side(i)[0]-GATE-coord.look for i in (0, 1)]
    coord.order = [1, 0]
    coord.update(start, [True, True])
    assert coord.owner == [1]
    coord.set_path(0, CROSS[0].sub(10, len(CROSS[0])-1), start[0]-10)  # truck 0 replans: truck 1 keeps its token
    assert coord.owner == [1]
    coord.order = [0, 1]
    assert coord.update([start[0]-10, start[1]], [True, True]) == [coord.zones[0].a_in-GATE-1, None]


def test_priority_wait_follows_the_order():
    coord = Coordinator(T, P, 3)
    for i, path in enumerate([*CROSS, towards(-40, 60, 40, 60)]):     # truck 2 shares no zone
        coord.set_path(i, path, 0)
    z = coord.zones[0]
    p = [0, 0, 0]
    assert not coord.priority_wait(0, p) and coord.priority_wait(1, p) and not coord.priority_wait(2, p)
    coord.order = [2, 1, 0]
    assert coord.priority_wait(0, p) and not coord.priority_wait(1, p) and not coord.priority_wait(2, p)
    assert coord.priority_wait(0, [0, z.b_out-1, 0]) and not coord.priority_wait(0, [0, z.b_out, 0])
    assert not coord.priority_wait(0, [z.a_out+1, 0, 0])               # truck 0 is past the zone already


def test_zones_of_reports_the_nearest_zone_ahead():
    coord = Coordinator(T, P, 2)
    for i, path in enumerate(CROSS):
        coord.set_path(i, path, 0)
    z = coord.zones[0]
    coord.update([10, 30], [True, False])
    d0, d1, held = coord.zones_of(0)[1]
    assert coord.zones_of(0)[0] is None
    assert d0 == pytest.approx(CROSS[0].s[z.a_in-GATE]-CROSS[0].s[10])
    assert d1 == pytest.approx(CROSS[1].s[z.b_in-GATE]-CROSS[1].s[30]) and not held
    assert coord.zones_of(1)[0] == (pytest.approx(d1), pytest.approx(d0), False)
    coord.update([z.a_in, 30], [True, True])
    d0, d1, held = coord.zones_of(1)[0]
    assert d1 < 0 and held
    coord.update([z.a_out+1, 30], [True, True])
    assert coord.zones_of(1) == [None, None]
