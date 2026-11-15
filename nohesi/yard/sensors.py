"""Perception and the shared map: the leader's roof-lidar scan of the yard into an occupancy grid, the followers'
near-field sensing that reveals what the scan missed, and the clearance (distance) map the planner searches.

Cells are UNKNOWN, FREE or OCCUPIED (uint8). Cell (iy, ix) covers [x0 + ix res, x0 + (ix+1) res) along x (and the same
along y from y0), its centre at (x0 + (ix+.5) res, y0 + (iy+.5) res). Only ground the scan saw, or the convoy drove, is
FREE: the planner treats UNKNOWN as blocked, like OCCUPIED. Sensing is 2D ray casting against the obstacles' rectangles
(nohesi.yard.world.Obstacles). Everything in m and rad, world frame unless a docstring says otherwise."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt
from ..geometry import clearances
from .truck import body_centre, corners
from .world import box

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
TOL = 1e-9      # m: a cell centre this close outside a polygon edge still counts as inside


def _cells(x0, y0, res, shape, xy):
    """(iy, ix, inside) of points xy (..., 2) on a grid: indices clipped into the grid (safe to index with), and the
    mask of the points that really lie on it."""
    xy = np.asarray(xy, float)
    ny, nx = shape
    ix = np.floor(np.clip(np.nan_to_num((xy[..., 0]-x0)/res, nan=-1.0), -1, nx)).astype(np.intp)
    iy = np.floor(np.clip(np.nan_to_num((xy[..., 1]-y0)/res, nan=-1.0), -1, ny)).astype(np.intp)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.clip(iy, 0, ny-1), np.clip(ix, 0, nx-1), inside


def _orientation(p):
    """+1 for a counter-clockwise polygon (V, 2), -1 for a clockwise one."""
    return 1.0 if np.sum(p[:, 0]*np.roll(p[:, 1], -1)-np.roll(p[:, 0], -1)*p[:, 1]) >= 0 else -1.0


def _inside(p, x, y):
    """Points (x, y) (broadcast together) inside the convex polygon p (V, 2), either orientation, boundary included."""
    sign, inside = _orientation(p), np.ones(np.broadcast(x, y).shape, bool)
    for (ax, ay), (ex, ey) in zip(p, np.roll(p, -1, 0)-p):
        inside &= sign*(ex*(y-ay)-ey*(x-ax)) >= -TOL*np.hypot(ex, ey)
    return inside


def _distance(p, x, y):
    """Distance (P,) of points (x, y) (P,) to the convex polygon p (V, 2): 0 inside."""
    best = np.full(len(x), np.inf)
    for (ax, ay), (ex, ey) in zip(p, np.roll(p, -1, 0)-p):
        rx, ry, ll = x-ax, y-ay, ex*ex+ey*ey
        s = np.clip((rx*ex+ry*ey)/ll, 0, 1) if ll > 0 else 0.0
        best = np.minimum(best, np.hypot(rx-s*ex, ry-s*ey))
    return np.where(_inside(p, x, y), 0.0, best)


def _edges(polys):
    """Edge starts a and vectors d (S, 2) of polygons (K, V, 2), closing edges included."""
    polys = np.asarray(polys, float)
    return polys.reshape(-1, 2), (np.roll(polys, -1, 1)-polys).reshape(-1, 2)


def _first_hit(origin, u, a, d):
    """Distance along rays origin + t u (origin (2,), unit directions u (N, 2), t >= 0) to the first crossing with
    the segments a + s d, s in [0, 1] (S, 2): (N,), inf where a ray crosses none. Parallel segments never count."""
    if not len(a):
        return np.full(len(u), np.inf)
    w = a-origin
    den = u[:, None, 0]*d[None, :, 1]-u[:, None, 1]*d[None, :, 0]                  # (N, S) u x d
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[None, :, 0]*d[None, :, 1]-w[None, :, 1]*d[None, :, 0])/den          # (w x d)/(u x d)
        s = (w[None, :, 0]*u[:, None, 1]-w[None, :, 1]*u[:, None, 0])/den          # (w x u)/(u x d)
    ok = (np.abs(den) > 1e-12) & (t >= 0) & (s >= 0) & (s <= 1)
    return np.where(ok, t, np.inf).min(1)


@dataclass(eq=False)
class ClearanceMap:
    """Distance from each cell centre to the nearest non-FREE cell centre (m; 0 on UNKNOWN and OCCUPIED cells, and
    the grid's outside counts as non-FREE), as OccupancyGrid.clearance() builds it. Treat dist as read-only: at()
    caches a padded copy of it (rebuilt only if `dist` is reassigned)."""
    dist: np.ndarray        # (ny, nx) m
    x0: float
    y0: float
    res: float

    def at(self, xy):
        """(...) clearance (m) at points xy (..., 2): the value of the cell holding the point minus res*sqrt(2)/2, the
        farthest a point lies from its cell centre (conservative), clipped at 0; 0 outside the grid (and at NaN)."""
        if getattr(self, '_src', None) is not self.dist:
            # Zero border: points off the grid clip onto it, so the lookup needs no mask (the planner's hot path).
            self._src, self._val = self.dist, np.pad(np.maximum(self.dist-self.res*np.sqrt(0.5), 0.0), 1)
        xy = np.asarray(xy, float)
        (h, w), k = self._val.shape, 1.0/self.res
        ix = np.fmin(np.fmax(xy[..., 0]*k+(1.0-self.x0*k), 0.0), w-1).astype(np.intp)    # fmax: NaN -> 0
        iy = np.fmin(np.fmax(xy[..., 1]*k+(1.0-self.y0*k), 0.0), h-1).astype(np.intp)
        return self._val[iy, ix]


@dataclass(eq=False)
class OccupancyGrid:
    """The shared map: UNKNOWN / FREE / OCCUPIED cells (see the module docstring for the cell convention)."""
    x0: float               # m: west edge of column 0
    y0: float               # m: south edge of row 0
    res: float              # m per cell
    cells: np.ndarray       # (ny, nx) uint8

    @classmethod
    def for_site(cls, site, sensors):
        """All-UNKNOWN grid at sensors.grid_resolution over the yard, the access road and the berms, plus 2 m."""
        b, yc, half = site.berm_width, site.width/2, site.road_width/2+site.berm_width
        x0, x1 = -site.road_length-b-2.0, site.length+b+2.0
        y0, y1 = min(-b, yc-half)-2.0, max(site.width+b, yc+half)+2.0
        res = sensors.grid_resolution
        nx, ny = (int(np.ceil((hi-lo)/res-TOL)) for lo, hi in ((x0, x1), (y0, y1)))
        return cls(x0, y0, res, np.full((ny, nx), UNKNOWN, np.uint8))

    @property
    def shape(self):
        return self.cells.shape

    def copy(self):
        return OccupancyGrid(self.x0, self.y0, self.res, self.cells.copy())

    def cell_of(self, xy):
        """(iy, ix, inside) for points xy (..., 2): int index arrays (clipped into the grid, so always safe to index
        with) and the bool mask of the points that lie on the grid."""
        return _cells(self.x0, self.y0, self.res, self.cells.shape, xy)

    def _window(self, lo, hi):
        """The cells whose centres lie in the box lo..hi (2,), clipped to the grid: the (rows, columns) slices and the
        centres' x (1, w) and y (h, 1)."""
        (ny, nx), r = self.cells.shape, self.res
        ix0, iy0 = (max(int(np.ceil((v-o)/r-0.5-TOL)), 0) for v, o in zip(lo, (self.x0, self.y0)))
        ix1 = min(int(np.floor((hi[0]-self.x0)/r-0.5+TOL))+1, nx)
        iy1 = min(int(np.floor((hi[1]-self.y0)/r-0.5+TOL))+1, ny)
        ix1, iy1 = max(ix1, ix0), max(iy1, iy0)
        return (slice(iy0, iy1), slice(ix0, ix1)), self.x0+(np.arange(ix0, ix1)[None, :]+0.5)*r, \
            self.y0+(np.arange(iy0, iy1)[:, None]+0.5)*r

    def mark_polygon(self, corners, value):
        """Set every cell whose centre lies inside the convex polygon corners (V, 2) (boundary included) to value."""
        p = np.asarray(corners, float)
        win, x, y = self._window(p.min(0), p.max(0))
        self.cells[win][_inside(p, x, y)] = value

    def known_free(self, corners, margin=0.0):
        """True if every cell whose centre lies within `margin` (m) of the convex polygon corners (V, 2) is FREE (with
        margin 0: every cell mark_polygon would set). False if that ground reaches off the grid."""
        p = np.asarray(corners, float)
        lo, hi = p.min(0)-margin, p.max(0)+margin
        ny, nx = self.cells.shape
        if lo[0] < self.x0 or lo[1] < self.y0 or hi[0] > self.x0+nx*self.res or hi[1] > self.y0+ny*self.res:
            return False
        win, x, y = self._window(lo, hi)
        jy, jx = np.nonzero(self.cells[win] != FREE)
        return not len(jy) or bool(np.all(_distance(p, x[0, jx], y[jy, 0]) > margin+TOL))

    def clearance(self, extra=()):
        """ClearanceMap of this grid: distance (m) from each cell centre to the nearest non-FREE cell centre, UNKNOWN
        and OCCUPIED both blocking, the grid's outside too. `extra` polygons (K, 4, 2) are rasterised OCCUPIED in a copy
        first (the leader, parked or waiting trucks); this grid is left unchanged."""
        grid = self
        if len(extra):
            grid = self.copy()
            for poly in extra:
                grid.mark_polygon(poly, OCCUPIED)
        free = np.pad(grid.cells == FREE, 1, constant_values=False)
        return ClearanceMap(distance_transform_edt(free, sampling=self.res)[1:-1, 1:-1], self.x0, self.y0, self.res)


def lidar_origin(world, sensors):
    """(2,) the leader's roof lidar: lidar_mount ahead of its rear axle along its heading."""
    x, y, yaw = world.leader[:3]
    return np.array([x+sensors.lidar_mount*np.cos(yaw), y+sensors.lidar_mount*np.sin(yaw)])


def initial_map(world, sensors, rng):
    """The leader's scan: (OccupancyGrid, missed (K,) bool).

    One rng.random() per obstacle in index order, missable or not (exactly K draws): an obstacle is missed when it is
    missable and its draw is < miss_prob; missed obstacles neither block rays nor show. 360 deg of rays,
    lidar_resolution apart (ray 0 along the leader's heading), from lidar_origin() out to lidar_range: cells along a
    ray (sampled every res/2) are FREE up to its first hit on an obstacle edge, and the cell holding the point 1e-3 m
    past the hit (inside the obstacle) is OCCUPIED. The access road (x in [-road_length, 0], within road_width/2 of
    y = width/2, berm cells excluded) is FREE, the convoy drove it, and so are the leader's and the followers'
    footprints (known through V2V, not mapped). Hit cells are written last: an obstacle seen in a cell wins."""
    site, truck, obs = world.scenario.site, world.scenario.truck, world.obstacles
    grid = OccupancyGrid.for_site(site, sensors)
    missed = np.asarray(obs.missable, bool) & (rng.random(len(obs)) < sensors.miss_prob)
    origin = lidar_origin(world, sensors)
    n_rays = int(round(360.0/sensors.lidar_resolution))
    ang = world.leader[2]+np.deg2rad(np.arange(n_rays)*sensors.lidar_resolution)
    u = np.stack((np.cos(ang), np.sin(ang)), -1)
    hit = _first_hit(origin, u, *_edges(obs.corners[~missed]))
    # Free space along the rays.
    step = grid.res/2
    t = np.arange(int(np.ceil(sensors.lidar_range/step))+1)*step
    iy, ix, inside = grid.cell_of(origin+t[None, :, None]*u[:, None, :])
    keep = inside & (t[None, :] < np.minimum(hit, sensors.lidar_range)[:, None])
    grid.cells[iy[keep], ix[keep]] = FREE
    # The access road, without the berm cells along it.
    yc, half = site.width/2, site.road_width/2
    road = box(-site.road_length, 0.0, yc-half, yc+half)
    win, x, y = grid._window(road.min(0), road.max(0))
    on_road = _inside(road, x, y)
    for k in (k for k, kind in enumerate(obs.kind) if kind == 'berm'):
        on_road &= ~_inside(obs.corners[k], x, y)
    grid.cells[win][on_road] = FREE
    # The convoy's own footprints.
    for poly in (*corners(truck, world.leader[None]), *world.footprints()):
        grid.mark_polygon(poly, FREE)
    # The hits.
    seen = hit <= sensors.lidar_range
    iy, ix, inside = grid.cell_of(origin+(hit[seen, None]+1e-3)*u[seen])
    grid.cells[iy[inside], ix[inside]] = OCCUPIED
    return grid, missed


def reveal(grid, world, missed, revealed, sensors):
    """Missed obstacles not yet revealed whose footprint clearance to any follower's footprint is < near_range: each
    is rasterised OCCUPIED into `grid` and flagged in `revealed` (K,) bool, in place. Returns the newly revealed
    indices (ascending)."""
    todo = np.flatnonzero(np.asarray(missed, bool) & ~np.asarray(revealed, bool))
    if not len(todo) or not world.n:
        return []
    polys, feet = world.obstacles.corners[todo], world.footprints()
    # Bounding circles first: polygons whose circles are near_range apart are, too.
    cp, cf = polys.mean(1), feet.mean(1)
    rp = np.linalg.norm(polys-cp[:, None], axis=-1).max(1)
    rf = np.linalg.norm(feet-cf[:, None], axis=-1).max(1)
    near = np.linalg.norm(cp[:, None]-cf[None], axis=-1)-rp[:, None]-rf[None] < sensors.near_range     # (T, n)
    if not near.any():
        return []
    ko, kf = np.nonzero(near)
    close = np.zeros(len(todo), bool)
    close[ko[clearances(polys[ko], feet[kf]) < sensors.near_range]] = True
    new = todo[close]
    for k in new:
        grid.mark_polygon(world.obstacles.corners[k], OCCUPIED)
        revealed[k] = True
    return [int(k) for k in new]


def near_field(world, i, sensors, n=8):
    """(n,) free distance (m) from follower i's footprint edge along n rays from its body centre, in the truck frame
    (ray 0 straight ahead, then counter-clockwise, 2 pi/n apart), to the first true obstacle (missed ones too), the
    leader or another follower; clipped to [0, near_range] (0: touching or overlapping)."""
    truck, state = world.scenario.truck, world.states[i]
    centre = body_centre(truck, state[None])[0]
    ang = np.arange(n)*2*np.pi/n
    u = np.stack((np.cos(state[2]+ang), np.sin(state[2]+ang)), -1)
    others = np.concatenate((world.obstacles.corners.reshape(-1, 4, 2), corners(truck, world.leader[None]),
                             np.delete(world.footprints(), i, 0)))
    hit = _first_hit(centre, u, *_edges(others))
    with np.errstate(divide='ignore'):
        edge = np.minimum(truck.length/2/np.abs(np.cos(ang)), truck.width/2/np.abs(np.sin(ang)))
    return np.clip(hit-edge, 0.0, sensors.near_range)
