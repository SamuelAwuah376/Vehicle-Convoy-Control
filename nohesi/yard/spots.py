"""Parking spots: the cab display's safety check on the leader driver's picks, the lattice of candidate spots, and the
scripted driver model of simulation and RL training.

The leader's driver picks a spot for each follower in queue order; the check accepts the pick, or rejects it with the
first failing reason (cheapest check first):
    'outside'      the footprint is not inside the yard [0, length] x [0, width] shrunk by Planning.spot_margin
    'entrance'     the footprint is within Site.keep_clear of the entrance centre (0, width/2): it would block arrivals
    'leader'       within Planning.spot_spacing of the leader's footprint
    'too_close'    within spot_spacing of another assigned spot's footprint
    'occupied'     a cell within spot_margin of the footprint is OCCUPIED (a known obstacle: it takes precedence over
                   'unscanned', since the lidar maps only an obstacle's near face and its interior stays UNKNOWN)
    'unscanned'    a cell within spot_margin of the footprint is UNKNOWN (or off the grid)
    'unreachable'  Hybrid A* finds no path from the follower's pose around the mapped obstacles, the extra footprints
                   (the leader, waiting followers) and the other assigned spots
Poses are rear-axle [x, y, yaw] (docs/yard_interfaces.md)."""
import numpy as np
from ..geometry import clearances
from .truck import body_centre, corners
from .world import leader_state

UNKNOWN, FREE, OCCUPIED = 0, 1, 2   # sensors.py cell states
REASONS = ('outside', 'entrance', 'leader', 'too_close', 'occupied', 'unscanned', 'unreachable')   # check order
YAWS = (0.0, np.pi/2, np.pi, -np.pi/2)   # candidate headings
ATTEMPTS = 12        # driver picks per follower before giving up
BERM_NEAR = 15.0     # m: a spot whose footprint is this close to the yard edge is "along a berm"
PREFER = 3.0         # driver's weight on candidates along a berm and nose out (vs 1)
NOSE_OUT = -0.5      # cos(yaw) below this: the heading points roughly toward x = 0 (within 60 deg of -x)
CHUNK = 256          # poses per vectorised grid lookup


def _rect_distance(truck, poses, xy):
    """(M, P) distance from points xy (M, P, 2) to the footprints of rear-axle poses (M, >= 3); 0 inside."""
    c = body_centre(truck, poses)
    cos, sin = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx, dy = xy[..., 0]-c[:, None, 0], xy[..., 1]-c[:, None, 1]
    along = np.abs(dx*cos+dy*sin)-truck.length/2
    across = np.abs(dy*cos-dx*sin)-truck.width/2
    return np.hypot(np.maximum(along, 0.0), np.maximum(across, 0.0))


def grid_status(truck, grid, poses, margin):
    """(occupied (M,), unknown (M,)) bool: whether any grid cell whose centre lies within `margin` of the footprint of
    each rear-axle pose (M, >= 3) is OCCUPIED / UNKNOWN; cells off the grid count as UNKNOWN. Vectorised
    OccupancyGrid.known_free (known_free == ~occupied & ~unknown) over a fixed cell window around each footprint."""
    poses = np.atleast_2d(np.asarray(poses, float))
    res, (ny, nx) = grid.res, grid.cells.shape
    reach = float(np.hypot(truck.length/2+margin, truck.width/2+margin))
    n = int(np.ceil(2*reach/res))+2
    k = np.arange(n)
    centres = body_centre(truck, poses)
    occ, unk = np.zeros(len(poses), bool), np.zeros(len(poses), bool)
    for s in range(0, len(poses), CHUNK):
        p, c = poses[s:s+CHUNK], centres[s:s+CHUNK]
        m = len(p)
        ix = np.floor((c[:, 0]-reach-grid.x0)/res).astype(int)[:, None]+k      # (m, n)
        iy = np.floor((c[:, 1]-reach-grid.y0)/res).astype(int)[:, None]+k
        px = np.broadcast_to((grid.x0+(ix+0.5)*res)[:, None, :], (m, n, n))
        py = np.broadcast_to((grid.y0+(iy+0.5)*res)[:, :, None], (m, n, n))
        near = _rect_distance(truck, p, np.stack((px, py), -1).reshape(m, n*n, 2)) <= margin+1e-9
        on = ((iy >= 0) & (iy < ny))[:, :, None] & ((ix >= 0) & (ix < nx))[:, None, :]
        vals = grid.cells[np.clip(iy, 0, ny-1)[:, :, None], np.clip(ix, 0, nx-1)[:, None, :]]
        vals = np.where(on, vals, UNKNOWN).reshape(m, n*n)
        occ[s:s+m] = np.any(near & (vals == OCCUPIED), 1)
        unk[s:s+m] = np.any(near & (vals == UNKNOWN), 1)
    return occ, unk


def _poses(a):
    """(M, 3) rear-axle poses from a pose list / array, None entries dropped."""
    a = [np.asarray(p, float)[:3] for p in (a if a is not None else ()) if p is not None]
    return np.array(a, float).reshape(-1, 3)


def static_reasons(scenario, grid, poses, assigned=()):
    """(M,) object array: the first failing static check of each rear-axle pose (M, >= 3), '' where all pass. The
    checks run cheapest first, each on the poses still passing."""
    t, site, pl = scenario.truck, scenario.site, scenario.planning
    poses = np.atleast_2d(np.asarray(poses, float))[:, :3]
    assigned = _poses(assigned)
    reason = np.full(len(poses), '', object)
    feet = corners(t, poses)
    m, gap = pl.spot_margin, pl.spot_spacing
    lead = corners(t, leader_state(site)[None])[0]
    spots = corners(t, assigned) if len(assigned) else np.zeros((0, 4, 2))

    def check(name, fails):
        idx = np.flatnonzero(reason == '')
        if len(idx):
            reason[idx[fails(idx)]] = name

    check('outside', lambda i: (feet[i, :, 0].min(1) < m) | (feet[i, :, 0].max(1) > site.length-m) |
          (feet[i, :, 1].min(1) < m) | (feet[i, :, 1].max(1) > site.width-m))
    check('entrance', lambda i: _rect_distance(t, poses[i], np.broadcast_to((0.0, site.width/2), (len(i), 1, 2)))[:, 0]
          < site.keep_clear)
    check('leader', lambda i: clearances(feet[i], np.broadcast_to(lead, (len(i), 4, 2))) < gap)
    check('too_close', lambda i: np.any([clearances(feet[i], np.broadcast_to(s, (len(i), 4, 2))) < gap for s in spots],
                                        0) if len(spots) else np.zeros(len(i), bool))
    idx = np.flatnonzero(reason == '')
    if len(idx):
        occ, unk = grid_status(t, grid, poses[idx], m)
        reason[idx[occ]] = 'occupied'
        reason[idx[~occ & unk]] = 'unscanned'
    return reason


def validate(scenario, grid, pose, assigned=(), others=(), planner=None, start=None):
    """The safety check on one pick: (reason, plan_result). reason '' = accepted, else the first failing REASONS entry.
    assigned: poses (3,) already accepted for other followers; others: (K, 4, 2) extra footprints the path must avoid
    (the leader, waiting followers). When the static checks pass and planner (HybridAStar) and start (the follower's
    pose) are given, the spot must be reachable: plan_result is the PlanResult on grid.clearance(extra=others + the
    assigned spots' footprints) ('unreachable' when it has no path); otherwise plan_result is None."""
    pose = np.asarray(pose, float)[:3]
    assigned = _poses(assigned)
    reason = static_reasons(scenario, grid, pose[None], assigned)[0]
    if reason or planner is None or start is None:
        return reason, None
    extra = [np.asarray(others, float).reshape(-1, 4, 2), corners(scenario.truck, assigned).reshape(-1, 4, 2)]
    extra = np.concatenate(extra)
    result = planner.plan(grid.clearance(extra=extra if len(extra) else ()), np.asarray(start, float)[:3], pose)
    return ('' if result.path is not None else 'unreachable'), result


def lattice(scenario):
    """(M, 3) rear-axle poses of every lattice spot: body centres every spot_lattice m over the yard (offset half a
    step from the edges), each with the four YAWS."""
    site, t, step = scenario.site, scenario.truck, scenario.planning.spot_lattice
    cx, cy = np.meshgrid(np.arange(step/2, site.length, step), np.arange(step/2, site.width, step), indexing='ij')
    cx, cy = cx.ravel(), cy.ravel()
    mid = t.length/2-t.rear_overhang
    out = [np.column_stack((cx-mid*np.cos(yaw), cy-mid*np.sin(yaw), np.full(len(cx), yaw))) for yaw in YAWS]
    return np.concatenate(out)


def candidates(scenario, grid, assigned=()):
    """(M, 3) candidate spot poses: the lattice() poses that pass validate's static checks (no planning), also against
    the `assigned` spots when given."""
    poses = lattice(scenario)
    return poses[static_reasons(scenario, grid, poses, assigned) == '']


def default_others(world, follower):
    """The footprints follower's path must avoid besides the map and the assigned spots: the leader and the followers
    later in the queue where they stand (the earlier ones drive off first; conflict zones handle the transient)."""
    t = world.scenario.truck
    return np.concatenate((corners(t, world.leader[None]), world.footprints()[follower+1:]))


class Driver:
    """The scripted leader driver. Each attempt is, with probability scenario.driver_bad_pick, a free click (a uniformly
    random pose inside the yard, any heading, that the check usually rejects), otherwise a random candidate, weighted
    PREFER x for spots along a berm (footprint within BERM_NEAR of the yard edge) and nose out (heading roughly toward
    x = 0: reverse in, drive out forward). Up to ATTEMPTS attempts per follower. All randomness comes from `rng`."""

    def __init__(self, scenario, rng, attempts=ATTEMPTS):
        self.scenario, self.rng, self.attempts = scenario, rng, int(attempts)

    def preference(self, poses):
        """(M,) the driver's weight on candidate poses (M, 3)."""
        site = self.scenario.site
        feet = corners(self.scenario.truck, poses)
        edge = np.minimum.reduce((feet[..., 0], site.length-feet[..., 0], feet[..., 1], site.width-feet[..., 1])).min(1)
        return np.where((edge <= BERM_NEAR) & (np.cos(poses[:, 2]) < NOSE_OUT), PREFER, 1.0)

    def free_click(self):
        site = self.scenario.site
        return np.array([self.rng.uniform(0, site.length), self.rng.uniform(0, site.width),
                         self.rng.uniform(-np.pi, np.pi)])

    def pick(self, follower, world, grid, assigned=(), others=None, planner=None):
        """Attempts until the check accepts one: (pose (3,) or None, picks). picks: one {'pose', 'accepted', 'reason',
        'result'} per attempt, result the attempt's PlanResult (None if a static check rejected it, or no planner).
        assigned: spots (3,) of followers earlier in the queue; others: (K, 4, 2) footprints to avoid besides them (None:
        default_others: the leader and the followers later in the queue)."""
        assigned = _poses(assigned)
        others = default_others(world, follower) if others is None else others
        start = world.states[follower, :3]
        pool = candidates(self.scenario, grid, assigned)
        weight = self.preference(pool) if len(pool) else np.zeros(0)
        picks = []
        for _ in range(self.attempts):
            if self.rng.random() < self.scenario.driver_bad_pick or weight.sum() <= 0:
                pose = self.free_click()
            else:
                k = int(self.rng.choice(len(pool), p=weight/weight.sum()))
                weight[k] = 0.0   # never the same candidate twice
                pose = pool[k].copy()
            reason, result = validate(self.scenario, grid, pose, assigned, others, planner, start)
            picks.append({'pose': pose, 'accepted': not reason, 'reason': reason, 'result': result})
            if not reason:
                return pose, picks
        return None, picks

    def assign_all(self, world, grid, planner=None):
        """Serve the followers in queue order (F1 first), each accepted spot joining `assigned` for the next, with
        default_others: (spots: per follower (3,) or None, picks: every attempt with its 'follower')."""
        spots, log = [], []
        for i in range(world.n):
            pose, picks = self.pick(i, world, grid, spots, None, planner)
            spots.append(pose)
            log += [{'follower': i, **p} for p in picks]
        return spots, log
