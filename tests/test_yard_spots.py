"""Yard spots (nohesi.yard.spots): the safety check's rejection reasons on hand-built grids, the candidate lattice, and
the scripted driver on generated yards with the leader's real scan."""
from collections import Counter
from dataclasses import dataclass
import numpy as np
import pytest
from nohesi.yard import spots
from nohesi.yard.config import YardScenario
from nohesi.yard.sensors import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, initial_map
from nohesi.yard.spots import Driver, candidates, grid_status, validate
from nohesi.yard.truck import corners
from nohesi.yard.world import YardWorld, berms, box, follower_starts

SC = YardScenario()
SITE, TRUCK, PL = SC.site, SC.truck, SC.planning
START = follower_starts(TRUCK, SITE, 1)[0, :3]


def yard_grid(inside=FREE):
    """The site grid with the yard's inside set to `inside`, the access road FREE and the berms OCCUPIED."""
    g = OccupancyGrid.for_site(SITE, SC.sensors)
    g.mark_polygon(box(0, SITE.length, 0, SITE.width), inside)
    g.mark_polygon(box(-SITE.road_length, 0, SITE.width/2-SITE.road_width/2, SITE.width/2+SITE.road_width/2), FREE)
    for b in berms(SITE):
        g.mark_polygon(b, OCCUPIED)
    return g


def foot(pose):
    return corners(TRUCK, np.asarray(pose, float)[None])[0]


def test_cell_states_and_reasons():
    assert (spots.UNKNOWN, spots.FREE, spots.OCCUPIED) == (UNKNOWN, FREE, OCCUPIED)
    assert set(spots.REASONS) == {'unscanned', 'occupied', 'outside', 'entrance', 'leader', 'too_close', 'unreachable'}


def test_open_spot_accepted():
    g = yard_grid()
    assert validate(SC, g, (50.0, 30.0, 0.0), [], ()) == ('', None)
    assert validate(SC, g, (80.0, 20.0, np.pi/2), [(50.0, 30.0, 0.0)], foot((12.0, 56.0, 0.0))[None]) == ('', None)


def test_unscanned():
    g = yard_grid()
    g.mark_polygon(box(40, 60, 20, 40), UNKNOWN)
    assert validate(SC, g, (50.0, 30.0, 0.0), [], ())[0] == 'unscanned'
    # Footprint on FREE ground, but UNKNOWN cells within spot_margin of it.
    g = yard_grid()
    pose = np.array([50.0, 30.0, 0.0])
    side = foot(pose)[:, 1].max()   # the footprint's north side
    g.mark_polygon(box(40, 60, side+0.6, side+3.0), UNKNOWN)
    assert validate(SC, g, pose, [], ())[0] == 'unscanned'
    g = yard_grid()
    g.mark_polygon(box(40, 60, side+1.3, side+3.0), UNKNOWN)   # beyond the margin
    assert validate(SC, g, pose, [], ())[0] == ''
    # The whole yard unscanned: nothing passes, nothing is a candidate.
    assert validate(SC, yard_grid(UNKNOWN), (50.0, 30.0, 0.0), [], ())[0] == 'unscanned'
    assert len(candidates(SC, yard_grid(UNKNOWN))) == 0


def test_occupied():
    g = yard_grid()
    g.mark_polygon(box(52, 55, 29, 32), OCCUPIED)   # a rock pile under the footprint
    assert validate(SC, g, (50.0, 30.0, 0.0), [], ())[0] == 'occupied'
    # As the scan sees it: the near face OCCUPIED, the pile's inside and its shadow UNKNOWN. A known obstacle is the
    # stronger reason.
    g = yard_grid()
    g.mark_polygon(box(52, 60, 29, 32), UNKNOWN)
    g.mark_polygon(box(52, 52.5, 29, 32), OCCUPIED)
    assert validate(SC, g, (50.0, 30.0, 0.0), [], ())[0] == 'occupied'


def test_outside():
    g = yard_grid()
    assert validate(SC, g, (50.0, 1.5, 0.0), [], ())[0] == 'outside'               # over the south berm
    assert validate(SC, g, (50.0, 2.5, 0.0), [], ())[0] == 'outside'               # 0.6 m from it: inside the margin
    assert validate(SC, g, (50.0, 1.9+PL.spot_margin+0.05, 0.0), [], ())[0] == ''  # just clear of the margin
    assert validate(SC, g, (SITE.length-5.0, 40.0, 0.0), [], ())[0] == 'outside'   # nose into the east berm
    # 'outside' is checked before the grid: over the berm and over UNKNOWN ground beyond it still reads 'outside'.
    assert validate(SC, yard_grid(UNKNOWN), (50.0, 1.5, 0.0), [], ())[0] == 'outside'


def test_entrance():
    g = yard_grid()
    assert validate(SC, g, (5.0, SITE.width/2, 0.0), [], ())[0] == 'entrance'
    assert validate(SC, g, (12.0, SITE.width/2-12.0, np.pi/2), [], ())[0] == 'entrance'
    assert validate(SC, g, (SITE.keep_clear+TRUCK.rear_overhang+0.5, SITE.width/2, 0.0), [], ())[0] == ''


def test_leader():
    g = yard_grid()
    lx, ly, _ = SITE.leader_pose
    assert validate(SC, g, (lx, ly, 0.0), [], ())[0] == 'leader'                            # on it (FREE in the map)
    assert validate(SC, g, (lx+TRUCK.length+0.5, ly, 0.0), [], ())[0] == 'leader'           # 0.5 m behind it
    assert validate(SC, g, (lx+TRUCK.length+PL.spot_spacing+0.1, ly, 0.0), [], ())[0] == ''


def test_too_close():
    g = yard_grid()
    spot = (50.0, 30.0, 0.0)
    assert validate(SC, g, (52.0, 31.0, 0.0), [spot], ())[0] == 'too_close'                    # overlapping
    assert validate(SC, g, (50.0, 30.0+TRUCK.width+1.0, 0.0), [spot], ())[0] == 'too_close'    # 1 m side gap
    assert validate(SC, g, (50.0, 30.0+TRUCK.width+PL.spot_spacing+0.1, 0.0), [spot], ())[0] == ''
    assert validate(SC, g, (52.0, 31.0, 0.0), [None, np.array(spot)], ())[0] == 'too_close'   # None entries ignored
    assert validate(SC, g, (52.0, 31.0, 0.0), [], ())[0] == ''


@dataclass
class _Result:
    path: object
    reason: str
    expansions: int = 0
    time_s: float = 0.0


class _Planner:
    """Records the clearance map and the endpoints it is asked to plan on; finds a path iff `ok`."""
    def __init__(self, ok):
        self.ok, self.calls = ok, []

    def plan(self, clearance, start, goal):
        self.calls.append((clearance, np.asarray(start), np.asarray(goal)))
        return _Result('path' if self.ok else None, '' if self.ok else 'no_path')


def test_planning_plumbing():
    g = yard_grid()
    spot, pose, other = (80.0, 50.0, 0.0), (50.0, 30.0, 0.0), foot((70.0, 15.0, np.pi/2))
    p = _Planner(False)
    reason, result = validate(SC, g, pose, [spot], other[None], p, START)
    assert reason == 'unreachable' and result.reason == 'no_path' and len(p.calls) == 1
    clr, start, goal = p.calls[0]
    assert np.allclose(start, START) and np.allclose(goal, pose)
    # The assigned spot's and the extra footprints are obstacles in the planner's map; the pick's own ground is not.
    assert clr.at(np.array([[82.0, 50.0], [70.0, 17.0]])).max() == 0.0 and clr.at(np.array([53.0, 30.0])) > 1.0
    p = _Planner(True)
    reason, result = validate(SC, g, pose, [spot], other[None], p, START)
    assert reason == '' and result.path == 'path'
    # A static rejection never plans; no planner or no start: static checks only.
    assert validate(SC, g, (5.0, SITE.width/2, 0.0), [], (), p, START) == ('entrance', None) and len(p.calls) == 1
    assert validate(SC, g, pose, [], (), p, None) == ('', None) and len(p.calls) == 1


def _hybrid_astar():
    mod = pytest.importorskip('nohesi.yard.hybrid_astar')
    return mod.HybridAStar(TRUCK, PL)


def test_unreachable_walled_off_pocket():
    planner = _hybrid_astar()
    g = yard_grid()
    for wall in (box(60, 90, 10, 11), box(60, 90, 31, 32), box(60, 61, 10, 32), box(89, 90, 10, 32)):
        g.mark_polygon(wall, OCCUPIED)
    pocket = (72.0, 21.0, 0.0)
    assert validate(SC, g, pocket, [], ())[0] == ''     # statically a fine spot
    reason, result = validate(SC, g, pocket, [], (), planner, START)
    assert reason == 'unreachable' and result.path is None and result.reason in ('no_path', 'budget', 'goal_blocked')
    # The same yard outside the pocket: reachable, and the path ends at the spot.
    reason, result = validate(SC, g, (40.0, 20.0, np.pi), [], (), planner, START)
    assert reason == '' and result.path is not None
    assert np.allclose(result.path.states[-1, :2], (40.0, 20.0), atol=PL.goal_tolerance)


def test_candidates():
    g = yard_grid()
    c = candidates(SC, g)
    assert c.ndim == 2 and c.shape[1] == 3 and len(c) > 500
    assert set(np.round(c[:, 2], 6)) <= {round(y, 6) for y in spots.YAWS}
    assert all(validate(SC, g, p, [], ())[0] == '' for p in c)
    lattice = spots.lattice(SC)
    rejected = lattice[~(lattice[:, None] == c[None]).all(-1).any(1)]
    assert len(rejected) == len(lattice)-len(c) and all(validate(SC, g, p, [], ())[0] for p in rejected[::7])
    # Against an assigned spot: none too close to it.
    spot = (50.0, 30.0, 0.0)
    c2 = candidates(SC, g, [spot])
    assert 0 < len(c2) < len(c) and all(validate(SC, g, p, [spot], ())[0] == '' for p in c2[::5])


def test_grid_status_is_known_free():
    w = YardWorld(SC, 3)
    g, _ = initial_map(w, SC.sensors, np.random.default_rng([3, 1]))
    rng = np.random.default_rng(0)
    poses = np.column_stack((rng.uniform(-5, 105, 600), rng.uniform(-5, 75, 600), rng.uniform(-np.pi, np.pi, 600)))
    occ, unk = grid_status(TRUCK, g, poses, PL.spot_margin)
    free = np.array([g.known_free(f, PL.spot_margin) for f in corners(TRUCK, poses)])
    assert np.array_equal(free, ~occ & ~unk) and occ.any() and unk.any() and free.any()


def test_driver_preference_and_free_clicks():
    d = Driver(SC, np.random.default_rng(0))
    poses = np.array([[90.0, 35.0, np.pi], [90.0, 35.0, 0.0], [50.0, 35.0, np.pi], [50.0, 62.0, np.pi]])
    assert d.preference(poses).tolist() == [3.0, 1.0, 1.0, 3.0]
    clicks = np.array([d.free_click() for _ in range(200)])
    assert np.all((clicks[:, 0] >= 0) & (clicks[:, 0] <= SITE.length) & (clicks[:, 1] >= 0) &
                  (clicks[:, 1] <= SITE.width) & (np.abs(clicks[:, 2]) <= np.pi))


def test_driver_without_room_gives_up():
    w = YardWorld(SC, 0, n_followers=1)
    pose, picks = Driver(SC, np.random.default_rng(1)).pick(0, w, yard_grid(UNKNOWN), [], None, None)
    assert pose is None and len(picks) == spots.ATTEMPTS and not any(p['accepted'] for p in picks)
    assert {p['reason'] for p in picks} <= {'unscanned', 'outside', 'entrance', 'leader'}


def _assign(seed, planner):
    w = YardWorld(SC, seed)
    g, _ = initial_map(w, SC.sensors, np.random.default_rng([seed, 1]))
    return w, g, Driver(SC, np.random.default_rng([seed, 2])).assign_all(w, g, planner)


def _check_assignment(seeds, planner):
    done, clicks, reasons = 0, 0, Counter()
    for seed in seeds:
        w, g, (spots_, picks) = _assign(seed, planner)
        assert len(spots_) == w.n and [p['follower'] for p in picks] == sorted(p['follower'] for p in picks)
        for i, s in enumerate(spots_):
            mine = [p for p in picks if p['follower'] == i]
            assert sum(p['accepted'] for p in mine) == (s is not None) and len(mine) <= spots.ATTEMPTS
            assert all(set(p) == {'follower', 'pose', 'accepted', 'reason', 'result'} for p in mine)
            assert all(p['reason'] in spots.REASONS for p in mine if not p['accepted'])
            if s is not None:
                assert mine[-1]['accepted'] and np.array_equal(mine[-1]['pose'], s)
                earlier = [t for t in spots_[:i] if t is not None]
                assert validate(SC, g, s, earlier, ())[0] == ''     # accepted spots keep spot_spacing to the earlier ones
                if planner is not None:
                    assert mine[-1]['result'].path is not None
        done += all(s is not None for s in spots_)
        for p in picks:
            if not p['accepted']:
                reasons[p['reason']] += 1
                p0 = p['pose']
                clicks += p0[2] not in spots.YAWS   # a free click (lattice candidates carry the four yaws)
    rate = done/len(seeds)
    print(f'\nassigned all followers: {done}/{len(seeds)} seeds ({rate:.0%}), rejected {sum(reasons.values())} '
          f'({clicks} free clicks): {dict(reasons)}')
    return rate, clicks, reasons


def test_driver_assigns_all_followers_static():
    rate, clicks, reasons = _check_assignment(range(30), None)
    assert rate >= 0.9 and clicks > 0 and reasons


def test_driver_assigns_all_followers_planned():
    rate, clicks, reasons = _check_assignment(range(30), _hybrid_astar())
    assert rate >= 0.8 and clicks > 0 and reasons


def test_driver_reproducible():
    a, b = _assign(4, None)[2], _assign(4, None)[2]
    assert all(np.array_equal(s, t) for s, t in zip(a[0], b[0]))
    assert [(p['reason'], tuple(p['pose'])) for p in a[1]] == [(p['reason'], tuple(p['pose'])) for p in b[1]]
