"""Yard perception (nohesi/yard/sensors.py): the occupancy grid and its clearance map, the leader's lidar scan with
occlusion and missed obstacles, reveals from the followers' near field, and the near-field ranges."""
from dataclasses import replace
import time
import numpy as np
from nohesi.yard.config import YardScenario
from nohesi.yard.sensors import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, initial_map, lidar_origin, near_field, reveal
from nohesi.yard.truck import corners
from nohesi.yard.world import Obstacles, YardWorld, berms, box, obb

SC = YardScenario()
SITE, SENS, TRUCK = SC.site, SC.sensors, SC.truck


def centres(grid):
    """(ny, nx, 2) cell centres."""
    iy, ix = np.mgrid[0:grid.cells.shape[0], 0:grid.cells.shape[1]]
    return np.stack((grid.x0+(ix+0.5)*grid.res, grid.y0+(iy+0.5)*grid.res), -1)


def world_with(extra, missable, n_followers=3, sensors=SENS):
    """Seed-0 world whose obstacles are the berms plus `extra` (kind 'rock')."""
    w = YardWorld(replace(SC, sensors=sensors), 0, n_followers=n_followers)
    b = berms(SITE)
    w.obstacles = Obstacles(np.array(b+list(extra)), ['berm']*len(b)+['rock']*len(extra),
                            np.array([False]*len(b)+list(missable), bool))
    return w


def rect_distance(rect, pts):
    """(outside distance, depth inside) of points (P, 2) for a rectangle's corners (4, 2), both >= 0."""
    mid, a, b = rect.mean(0), (rect[1]-rect[0])/2, (rect[3]-rect[0])/2
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    u, v = np.abs((pts-mid)@a)/la, np.abs((pts-mid)@b)/lb
    return np.hypot(np.maximum(u-la, 0), np.maximum(v-lb, 0)), np.maximum(np.minimum(la-u, lb-v), 0)


def contains(rect, pts):
    """Points (..., 2) inside (or on) a rectangle (4, 2)."""
    pts = np.asarray(pts, float)
    return (rect_distance(rect, pts.reshape(-1, 2))[0] == 0).reshape(pts.shape[:-1])


def cell_value(grid, xy):
    iy, ix, inside = grid.cell_of(np.asarray(xy, float))
    assert np.all(inside)
    return grid.cells[iy, ix]


def test_grid_bounds_and_cell_of_round_trip():
    g = OccupancyGrid.for_site(SITE, SENS)
    assert (g.x0, g.y0, g.res) == (-64.0, -4.0, 0.5) and g.cells.shape == (156, 336) and g.cells.dtype == np.uint8
    assert np.all(g.cells == UNKNOWN)
    assert np.isclose(g.x0+g.cells.shape[1]*g.res, SITE.length+SITE.berm_width+2)
    assert np.isclose(g.y0+g.cells.shape[0]*g.res, SITE.width+SITE.berm_width+2)
    c = centres(g)
    iy, ix, inside = g.cell_of(c)
    rows, cols = np.mgrid[0:156, 0:336]
    assert inside.all() and np.array_equal(iy, rows) and np.array_equal(ix, cols)
    # Anywhere inside a cell maps to it; off the grid: clipped indices and inside False.
    jitter = np.random.default_rng(0).uniform(-0.49, 0.49, c.shape)*g.res
    assert np.array_equal(g.cell_of(c+jitter)[1], ix)
    iy, ix, inside = g.cell_of(np.array([[-65.0, 0.0], [0.0, 75.0], [200.0, -50.0], [10.0, 10.0]]))
    assert inside.tolist() == [False, False, False, True]
    assert ix.min() >= 0 and ix.max() < 336 and iy.min() >= 0 and iy.max() < 156
    assert g.cell_of(np.zeros(2))[0].shape == ()
    h = g.copy()
    h.cells[0, 0] = FREE
    assert g.cells[0, 0] == UNKNOWN


def test_mark_polygon_marks_exactly_the_centres_inside():
    g = OccupancyGrid.for_site(SITE, SENS)
    x0, x1, y0, y1 = 3.1, 7.3, 2.2, 4.9
    g.mark_polygon(box(x0, x1, y0, y1), OCCUPIED)
    c = centres(g)
    want = (c[..., 0] >= x0) & (c[..., 0] <= x1) & (c[..., 1] >= y0) & (c[..., 1] <= y1)
    assert want.sum() == 9*6 and np.array_equal(g.cells == OCCUPIED, want)
    assert np.all(g.cells[~want] == UNKNOWN)
    # Clockwise corners, a rotated rectangle and one hanging off the grid.
    for poly in (box(x0, x1, y0, y1)[::-1], obb(40.3, 20.7, 9.5, 3.8, 0.7), obb(-63.0, 30.0, 6.0, 4.0, 0.3)):
        g = OccupancyGrid.for_site(SITE, SENS)
        g.mark_polygon(poly, FREE)
        assert np.array_equal(g.cells == FREE, contains(poly, c))


def test_known_free():
    g = OccupancyGrid.for_site(SITE, SENS)
    g.mark_polygon(box(0, 40, 0, 30), FREE)
    poly = box(10.0, 19.5, 10.0, 13.8)
    assert g.known_free(poly) and g.known_free(poly, 2.0)
    g.cells[g.cell_of(np.array([21.25, 12.25]))[:2]] = OCCUPIED      # centre 1.75 m east of the polygon
    assert g.known_free(poly, 1.7) and not g.known_free(poly, 1.8)
    g.cells[g.cell_of(np.array([15.25, 12.25]))[:2]] = UNKNOWN       # inside
    assert not g.known_free(poly)
    assert not g.known_free(box(-70, -60, 30, 33))                    # off the grid


def test_clearance_distances():
    g = OccupancyGrid.for_site(SITE, SENS)
    g.mark_polygon(box(0, 40, 0, 30), FREE)
    block = np.array([20.25, 15.25])                                  # one OCCUPIED cell centre
    g.cells[g.cell_of(block)[:2]] = OCCUPIED
    cm = g.clearance()
    r, off = g.res, g.res*np.sqrt(2)/2
    for dx, dy in ((0, 0), (1, 0), (-4, 0), (0, 6), (3, 4)):
        p = block+r*np.array([dx, dy])
        assert np.isclose(cm.dist[g.cell_of(p)[:2]], r*np.hypot(dx, dy))
        assert np.isclose(cm.at(p), max(r*np.hypot(dx, dy)-off, 0))
    # The nearest blocking cell may be the UNKNOWN ground around the free box (5.5 m west of x = 5.25); UNKNOWN
    # cells and points off the grid read 0.
    assert np.isclose(cm.at(np.array([0.25, 15.25])), r-off)
    assert np.isclose(cm.at(np.array([5.25, 15.25])), 11*r-off)
    assert cm.at(np.array([-10.0, 15.0])) == 0
    assert cm.at(np.array([[-100.0, 0.0], [20.0, 500.0], [np.nan, 1.0]])).tolist() == [0.0, 0.0, 0.0]
    # Any leading shape; anywhere in a cell reads that cell's value.
    pts = np.random.default_rng(1).uniform((0, 0), (40, 30), (7, 5, 2))
    out = cm.at(pts)
    assert out.shape == (7, 5) and np.all(out >= 0)
    assert np.allclose(out, np.maximum(cm.dist[g.cell_of(pts)[:2]]-off, 0))
    assert np.allclose(out.ravel(), [cm.at(p) for p in pts.reshape(-1, 2)])
    # An all-FREE grid: the grid's outside blocks.
    h = g.copy()
    h.cells[:] = FREE
    assert np.isclose(h.clearance().dist[0, 5], r) and np.isclose(h.clearance().dist[10, 20], 11*r)
    # extra polygons block in a copy only.
    lead = corners(TRUCK, np.array([[10.0, 10.0, 0.0]]))
    cm2 = g.clearance(lead)
    assert cm2.at(np.array([12.0, 10.0])) == 0 and cm.at(np.array([12.0, 10.0])) > 0
    assert np.all(g.cells[g.cell_of(np.array([12.0, 10.0]))[:2]] == FREE)


def test_scan_occlusion():
    w = world_with([box(25, 28, 54, 58)], [False])
    o = lidar_origin(w, SENS)
    assert np.allclose(o, (SITE.leader_pose[0]+SENS.lidar_mount, SITE.leader_pose[1]))
    g, missed = initial_map(w, SENS, np.random.default_rng(0))
    assert not missed.any()
    assert cell_value(g, (25.001, 56.0)) == OCCUPIED and cell_value(g, (25.001, 55.9)) == OCCUPIED
    assert np.all(cell_value(g, np.column_stack((np.arange(19.0, 24.9, 0.25), np.full(24, 56.1)))) == FREE)
    # The box's inside past its near face and its shadow stay unknown.
    shadow = np.stack(np.meshgrid(np.arange(25.75, 45.0, 0.5), np.arange(55.25, 57.0, 0.5)), -1)
    assert np.all(cell_value(g, shadow) == UNKNOWN)
    # The same scan without the box sees that ground.
    g0, _ = initial_map(world_with([], []), SENS, np.random.default_rng(0))
    assert np.all(cell_value(g0, shadow) == FREE)


def test_missed_obstacle_reads_free_until_revealed():
    sens = replace(SENS, miss_prob=0.99)
    rock, far = box(40, 43, 40, 43), box(80, 83, 10, 13)
    w = world_with([rock, far], [True, True], sensors=sens)
    k, kf = len(w.obstacles)-2, len(w.obstacles)-1
    g, missed = initial_map(w, sens, np.random.default_rng(0))
    assert missed[k] and missed[kf] and not missed[:k].any()           # berms are never missable
    c = centres(g)
    inside = contains(rock, c)
    assert inside.sum() == 36 and np.all(g.cells[inside] == FREE)
    revealed = np.zeros(len(w.obstacles), bool)
    assert reveal(g, w, missed, revealed, sens) == [] and not revealed.any()
    # Follower 1 pulls up short of the rock: front bumper 10.5 m from it, then 9.5 m.
    front = TRUCK.length-TRUCK.rear_overhang
    w.states[1] = (40-10.5-front, 41.5, 0.0, 0.0)
    assert reveal(g, w, missed, revealed, sens) == [] and np.all(g.cells[inside] == FREE)
    w.states[1] = (40-9.5-front, 41.5, 0.0, 0.0)
    assert reveal(g, w, missed, revealed, sens) == [k]
    assert revealed[k] and not revealed[kf] and np.all(g.cells[inside] == OCCUPIED)
    assert reveal(g, w, missed, revealed, sens) == []                  # once only
    w.states[0] = (70.0, 11.5, 0.0, 0.0)                                # front bumper 2.6 m short of the far one
    assert reveal(g, w, missed, revealed, sens) == [kf] and revealed.sum() == 2


def brute_near_field(w, i, n, step=0.005):
    """near_field by ray marching (5 mm steps) against point-in-rectangle tests."""
    state = w.states[i]
    centre = state[:2]+(TRUCK.length/2-TRUCK.rear_overhang)*np.array([np.cos(state[2]), np.sin(state[2])])
    own = w.footprints()[i]
    others = (*w.obstacles.corners, corners(TRUCK, w.leader[None])[0], *np.delete(w.footprints(), i, 0))
    t = np.arange(0, 25, step)
    out = []
    for a in state[2]+np.arange(n)*2*np.pi/n:
        pts = centre+t[:, None]*np.array([np.cos(a), np.sin(a)])
        edge = t[np.flatnonzero(~contains(own, pts))[0]]
        hit = np.zeros(len(t), bool)
        for p in others:
            hit |= contains(p, pts)
        first = t[np.flatnonzero(hit)[0]] if hit.any() else np.inf
        out.append(np.clip(first-edge, 0, SENS.near_range))
    return np.array(out)


def test_near_field_box_ahead():
    front = TRUCK.length-TRUCK.rear_overhang
    w = world_with([box(50+front+4.0, 50+front+5.6, 33, 37)], [True])
    w.states[0] = (50.0, 35.0, 0.0, 0.0)
    assert np.allclose(near_field(w, 0, SENS), [4.0]+[SENS.near_range]*7)
    # Rotated: the truck faces +y with the box ahead of it; a missed obstacle still counts.
    w = world_with([box(33, 37, 50+front+4.0, 50+front+5.6)], [True])
    w.states[0] = (35.0, 50.0, np.pi/2, 0.0)
    assert np.allclose(near_field(w, 0, SENS), [4.0]+[SENS.near_range]*7)
    # A follower alongside (centres 5 m apart: 1.2 m gap) on the left, the box 2.5 m behind, 16 rays.
    w = world_with([box(50-TRUCK.rear_overhang-4.0, 50-TRUCK.rear_overhang-2.5, 33, 37)], [False])
    w.states[0] = (50.0, 35.0, 0.0, 0.0)
    w.states[1] = (50.0, 40.0, 0.0, 0.0)
    d = near_field(w, 0, SENS, n=16)
    assert d.shape == (16,) and np.isclose(d[4], 1.2) and np.isclose(d[8], 2.5) and np.isclose(d[0], SENS.near_range)
    assert np.allclose(near_field(w, 1, SENS)[6], 1.2)


def test_near_field_matches_ray_marching():
    rng = np.random.default_rng(3)
    for seed in (0, 1, 2):
        w = YardWorld(SC, seed, n_followers=3)
        for _ in range(3):
            w.states[:, :3] = rng.uniform((5, 5, -np.pi), (95, 65, np.pi), (3, 3))
            for i in range(3):
                assert np.allclose(near_field(w, i, SENS, n=12), brute_near_field(w, i, 12), atol=0.02)


def test_initial_map_seeds():
    road = box(-SITE.road_length, 0.0, SITE.width/2-SITE.road_width/2, SITE.width/2+SITE.road_width/2)
    times = []
    for seed in range(20):
        w = YardWorld(SC, seed)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        g, missed = initial_map(w, SENS, rng)
        times.append(time.perf_counter()-t0)
        obs = w.obstacles
        assert g.cells.shape == (156, 336) and missed.shape == (len(obs),) and missed.dtype == bool
        assert not np.any(missed & ~obs.missable)
        # Exactly len(obstacles) draws.
        ref = np.random.default_rng(seed)
        ref.random(len(obs))
        assert rng.random() == ref.random()
        c = centres(g)
        on_road = contains(road, c)
        assert on_road.sum() == 120*32 and np.all(g.cells[on_road] == FREE)
        for poly in (corners(TRUCK, w.leader[None])[0], *w.footprints()):
            assert g.known_free(poly)
        # OCCUPIED cells sit on seen obstacles (centre within half a cell diagonal); FREE cells never lie deeper
        # than that inside one.
        half_diag = g.res*np.sqrt(0.5)+1e-9
        occ, free = c[g.cells == OCCUPIED], c[g.cells == FREE]
        seen = obs.corners[~missed]
        assert len(occ) > 100
        assert np.all(np.min([rect_distance(p, occ)[0] for p in seen], 0) <= half_diag)
        assert np.all(np.max([rect_distance(p, free)[1] for p in seen], 0) <= half_diag)
        # Deterministic.
        g2, missed2 = initial_map(w, SENS, np.random.default_rng(seed))
        assert np.array_equal(g2.cells, g.cells) and np.array_equal(missed2, missed)
    assert max(times) < 0.5


def test_timing():
    w = YardWorld(SC, 0, n_followers=3)
    g, missed = initial_map(w, SENS, np.random.default_rng(0))
    missed = w.obstacles.missable.copy()          # worst case: every missable obstacle missed
    revealed = np.zeros(len(missed), bool)
    w.states[:, :3] = ((30, 30, 0), (50, 40, 1), (70, 20, 2))
    t0 = time.perf_counter()
    for _ in range(200):
        near_field(w, 1, SENS)
        reveal(g, w, missed, np.zeros_like(revealed), SENS)
    assert (time.perf_counter()-t0)/200 < 5e-3
