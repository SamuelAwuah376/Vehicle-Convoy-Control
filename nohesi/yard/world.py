"""The staging yard: berms, an entrance from the access road, static obstacles (some of which the leader's scan misses),
the leader parked at its scan pose and up to three followers queued on the access road. Obstacles are oriented
rectangles (4 corners); collisions use the separating-axis test of nohesi.geometry."""
from dataclasses import dataclass
import numpy as np
from ..geometry import clearances, overlaps
from .truck import corners

KINDS = ('berm', 'rock', 'equipment', 'windrow')


def box(x0, x1, y0, y1):
    """Axis-aligned rectangle corners (4, 2), counter-clockwise."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def obb(cx, cy, length, width, yaw):
    """Oriented rectangle corners (4, 2) centred at (cx, cy), `length` along yaw."""
    local = np.array([[-length, -width], [length, -width], [length, width], [-length, width]])/2
    c, s = np.cos(yaw), np.sin(yaw)
    return local@np.array([[c, s], [-s, c]])+(cx, cy)


@dataclass
class Obstacles:
    corners: np.ndarray     # (K, 4, 2)
    kind: list              # KINDS entry per obstacle
    missable: np.ndarray    # (K,) bool: low / small enough that the leader's lidar can miss it

    def __len__(self):
        return len(self.kind)


def berms(site):
    """Yard and access-road berms: the yard's four sides with the entrance gap in the west side, the road's two sides
    and its west end."""
    L, W, b, e, r, rw = site.length, site.width, site.berm_width, site.entrance_width, site.road_length, site.road_width
    yc = W/2
    out = [box(-b, L+b, W, W+b), box(-b, L+b, -b, 0), box(L, L+b, 0, W),
           box(-b, 0, 0, yc-e/2), box(-b, 0, yc+e/2, W),
           box(-r, 0, yc-rw/2-b, yc-rw/2), box(-r, 0, yc+rw/2, yc+rw/2+b), box(-r-b, -r, yc-rw/2-b, yc+rw/2+b)]
    return out


def leader_state(site):
    return np.array([*site.leader_pose, 0.0])


def follower_starts(truck, site, n):
    """Rear-axle states (n, 4) of the queued followers: on the road centre line, the first one's front bumper 2 m
    short of the entrance, follower_gap apart."""
    x0 = -2.0-truck.front_overhang-truck.wheelbase
    return np.array([[x0-k*site.follower_gap, site.width/2, 0.0, 0.0] for k in range(n)])


def generate(scenario, rng, n_followers):
    """Obstacles for one seed: berms plus rock piles, parked equipment and windrows inside the yard, kept off the
    entrance's keep-clear circle, off the leader (6 m) and 3 m apart."""
    site, truck = scenario.site, scenario.truck
    polys, kinds, miss = [], [], []
    for poly in berms(site):
        polys.append(poly); kinds.append('berm'); miss.append(False)
    lead = corners(truck, leader_state(site)[None])[0]
    placed = [lead]

    def try_place(make, kind, missable):
        for _ in range(200):
            poly = make()
            if np.any(poly[:, 0] < 1.0) or np.any(poly[:, 0] > site.length-1.0) or \
                    np.any(poly[:, 1] < 1.0) or np.any(poly[:, 1] > site.width-1.0):
                continue
            if np.min(np.hypot(poly[:, 0], poly[:, 1]-site.width/2)) < site.keep_clear:
                continue
            others = np.array(placed)
            gaps = clearances(np.repeat(poly[None], len(others), 0), others)
            need = np.full(len(others), 3.0)
            need[0] = 6.0   # the leader
            if np.any(gaps < need):
                continue
            placed.append(poly); polys.append(poly); kinds.append(kind); miss.append(missable)
            return

    uniform = lambda: (rng.uniform(0, site.length), rng.uniform(0, site.width), rng.uniform(-np.pi, np.pi))
    for _ in range(int(rng.integers(site.rock_piles[0], site.rock_piles[1]+1))):
        side = rng.uniform(2.5, 6.0)
        try_place(lambda: obb(*uniform()[:2], side, side*rng.uniform(0.7, 1.0), rng.uniform(-np.pi, np.pi)), 'rock', True)
    for _ in range(int(rng.integers(site.parked[0], site.parked[1]+1))):
        try_place(lambda: obb(*uniform()[:2], 8.0, 3.5, rng.choice((0.0, np.pi/2))+rng.normal(0, 0.1)), 'equipment', False)
    for _ in range(int(rng.integers(site.windrows[0], site.windrows[1]+1))):
        try_place(lambda: obb(*uniform()[:2], rng.uniform(12, 20), 1.5, rng.uniform(-np.pi, np.pi)), 'windrow', True)
    return Obstacles(np.array(polys), kinds, np.array(miss, bool)), follower_starts(truck, site, n_followers)


class YardWorld:
    """Ground truth: obstacles, the leader (parked at its scan pose) and the followers' states."""
    def __init__(self, scenario, seed, n_followers=None):
        self.scenario, self.seed = scenario, seed
        self.rng = np.random.default_rng(seed)
        lo, hi = scenario.followers
        self.n = int(n_followers if n_followers is not None else self.rng.integers(lo, hi+1))
        self.obstacles, starts = generate(scenario, self.rng, self.n)
        self.leader = leader_state(scenario.site)
        self.states = starts.copy()          # (n, 4) rear axle [x, y, yaw, v]
        self.steer = np.zeros(self.n)
        self.accel = np.zeros(self.n)

    def footprints(self, states=None):
        return corners(self.scenario.truck, self.states if states is None else states)

    def collisions(self):
        """[(follower, kind, other)]: kind 'obstacle' (other = obstacle index), 'leader' or 'truck' (other follower)."""
        t, found = self.scenario.truck, []
        feet = self.footprints()
        obs = self.obstacles.corners
        lead = corners(t, self.leader[None])[0]
        for i in range(self.n):
            hit = overlaps(np.repeat(feet[i][None], len(obs), 0), obs)
            found += [(i, 'obstacle', int(k)) for k in np.flatnonzero(hit)]
            if overlaps(feet[i][None], lead[None])[0]:
                found.append((i, 'leader', -1))
            for j in range(i+1, self.n):
                if overlaps(feet[i][None], feet[j][None])[0]:
                    found.append((i, 'truck', j))
        return found
