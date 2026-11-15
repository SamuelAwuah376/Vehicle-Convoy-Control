"""Hybrid A* for a rigid truck at the rear axle, forward and reverse, closed by an analytic Dubins shot that ends the
path exactly at the goal.

Motion: the kinematic bicycle at the rear axle moves on arcs of curvature k = tan(steer)/wheelbase; over a signed arc
length sigma (< 0 reversing) a pose (x, y, yaw) goes exactly to
    yaw + k sigma,   (x, y) + chord (cos, sin)(yaw + k sigma/2),   chord = 2 sin(k sigma/2)/k = sigma sinc(k sigma/2pi).
A reverse path is the same geometric curve driven backward: the reverse Dubins path from A to B is the forward Dubins
path from (A, yaw+pi) to (B, yaw+pi), with the truck's heading the Dubins heading - pi.

Paths keep the heading continuous (unwrapped from the start's yaw): the last sample is the goal exactly in x and y, and
in yaw modulo 2 pi (exactly, unless the path winds).
"""
import heapq
import math
import time
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .paths import Path
from .truck import disk_centres, disk_offsets, wrap

TWO_PI = 2*math.pi
WORDS = ('LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL')
_TURN = {'L': 1.0, 'S': 0.0, 'R': -1.0}     # curvature sign per letter (L: counter-clockwise)
_ENDPOINT_TOL = 1e-6                         # m and rad: a Dubins candidate's sampled endpoint must be this close


@dataclass
class PlanResult:
    path: Path | None
    reason: str        # '' (found), 'start_blocked', 'goal_blocked', 'no_path' (search exhausted), 'budget'
    expansions: int
    time_s: float


def collision_free(truck, planning, clearance, states):
    """(M,) bool for rear-axle states (M, >= 3) or (3,): every cover disk's centre has a clearance (clearance.at) of at
    least the disk radius + planning.clearance."""
    _, radius = disk_offsets(truck)
    return np.all(clearance.at(disk_centres(truck, states)) >= radius+planning.clearance, axis=-1)


def _arc(x, y, yaw, kappa, sigma):
    """Exact constant-curvature motion of the rear axle: the poses after signed arc lengths sigma (broadcast)."""
    half = 0.5*kappa*sigma
    chord = sigma*np.sinc(half/np.pi)
    return x+chord*np.cos(yaw+half), y+chord*np.sin(yaw+half), yaw+2*half


# ---------------------------------------------------------------------------------------------------------- Dubins

def _mod2pi(a):
    a %= TWO_PI
    return 0.0 if a > TWO_PI-1e-9 else a     # a rounding-level full turn is no turn (else a spurious loop)


def dubins_words(start, goal, radius):
    """{word: (3,) segment lengths (m)} of every Dubins word of WORDS whose formula applies between the poses
    start -> goal ([x, y, heading], forward motion) at turning radius `radius` (Shkel & Lumelsky's normalised
    formulas). Unverified: dubins() keeps only a candidate whose sampled endpoint is the goal."""
    dx, dy = goal[0]-start[0], goal[1]-start[1]
    d = math.hypot(dx, dy)/radius
    theta = math.atan2(dy, dx) if d > 0 else 0.0
    a, b = _mod2pi(start[2]-theta), _mod2pi(goal[2]-theta)
    sa, ca, sb, cb, cab = math.sin(a), math.cos(a), math.sin(b), math.cos(b), math.cos(a-b)
    out = {}
    p2 = 2+d*d-2*cab+2*d*(sa-sb)
    if p2 > -1e-10:
        tmp = math.atan2(cb-ca, d+sa-sb)
        out['LSL'] = (_mod2pi(tmp-a), math.sqrt(max(p2, 0.0)), _mod2pi(b-tmp))
    p2 = 2+d*d-2*cab+2*d*(sb-sa)
    if p2 > -1e-10:
        tmp = math.atan2(ca-cb, d-sa+sb)
        out['RSR'] = (_mod2pi(a-tmp), math.sqrt(max(p2, 0.0)), _mod2pi(tmp-b))
    p2 = -2+d*d+2*cab+2*d*(sa+sb)
    if p2 > -1e-10:
        p = math.sqrt(max(p2, 0.0))
        tmp = math.atan2(-ca-cb, d+sa+sb)-math.atan2(-2.0, p)
        out['LSR'] = (_mod2pi(tmp-a), p, _mod2pi(tmp-b))
    p2 = -2+d*d+2*cab-2*d*(sa+sb)
    if p2 > -1e-10:
        p = math.sqrt(max(p2, 0.0))
        tmp = math.atan2(ca+cb, d-sa-sb)-math.atan2(2.0, p)
        out['RSL'] = (_mod2pi(a-tmp), p, _mod2pi(b-tmp))
    c = (6-d*d+2*cab+2*d*(sa-sb))/8
    if abs(c) <= 1+1e-12:
        p = _mod2pi(TWO_PI-math.acos(min(max(c, -1.0), 1.0)))
        t = _mod2pi(a-math.atan2(ca-cb, d-sa+sb)+p/2)
        out['RLR'] = (t, p, _mod2pi(a-b-t+p))
    c = (6-d*d+2*cab+2*d*(sb-sa))/8
    if abs(c) <= 1+1e-12:
        p = _mod2pi(TWO_PI-math.acos(min(max(c, -1.0), 1.0)))
        t = _mod2pi(-a-math.atan2(ca-cb, d+sa-sb)+p/2)
        out['LRL'] = (t, p, _mod2pi(b-a-t+p))
    return {w: (radius*t, radius*p, radius*q) for w, (t, p, q) in out.items()}


def dubins_sample(start, word, lengths, radius, sample):
    """(N, 3) poses along a Dubins word from start ([x, y, heading]), by exact arc integration: each segment is cut
    into equal pieces at most `sample` long, so segment ends are samples. N >= 2 (a zero-length path repeats start)."""
    x, y, yaw = (float(v) for v in start[:3])
    heads, counts, u = [], [], []
    for letter, ell in zip(word, lengths):
        if ell <= 1e-9:            # a rounding-level segment: no sample (the endpoint check bounds the error)
            continue
        m = max(1, int(math.ceil(ell/sample-1e-9)))
        k = _TURN[letter]/radius
        heads.append((x, y, yaw, k))
        counts.append(m)
        s = np.arange(1, m+1)*(ell/m)
        s[-1] = ell
        u.append(s)
        half = 0.5*k*ell
        chord = ell*math.sin(half)/half if half else ell
        x, y, yaw = x+chord*math.cos(yaw+half), y+chord*math.sin(yaw+half), yaw+2*half
    first = np.array([start[:3]], float)
    if not heads:
        return np.repeat(first, 2, 0)
    h = np.repeat(np.array(heads), counts, 0)
    px, py, pyaw = _arc(h[:, 0], h[:, 1], h[:, 2], h[:, 3], np.concatenate(u))
    return np.vstack((first, np.column_stack((px, py, pyaw))))


def _ranked(start, goal, radius, direction):
    """[(word, lengths)] of the Dubins words for the motion start -> goal in `direction` (+1 / -1), shortest first."""
    flip = math.pi if direction < 0 else 0.0
    words = dubins_words((start[0], start[1], start[2]+flip), (goal[0], goal[1], goal[2]+flip), radius)
    return sorted(words.items(), key=lambda kv: sum(kv[1]))


def _dubins_states(start, goal, radius, sample, direction, ranked):
    """(N, 3) truck poses along the first word of `ranked` (from _ranked) whose sampled endpoint is the goal, with
    the first sample exactly start and the last exactly goal (its yaw modulo 2 pi, kept continuous); None if no word
    verifies."""
    flip = math.pi if direction < 0 else 0.0
    a = (float(start[0]), float(start[1]), float(start[2])+flip)
    gx, gy, gyaw = float(goal[0]), float(goal[1]), float(goal[2])
    for word, lengths in ranked:
        st = dubins_sample(a, word, lengths, radius, sample)
        ex, ey, eyaw = st[-1]
        if abs(ex-gx) <= _ENDPOINT_TOL and abs(ey-gy) <= _ENDPOINT_TOL and \
                abs(float(wrap(eyaw-flip-gyaw))) <= _ENDPOINT_TOL:
            st[:, 2] -= flip
            st[0] = start[:3]
            st[-1] = gx, gy, gyaw+TWO_PI*round((st[-1, 2]-gyaw)/TWO_PI)
            return st
    return None


def dubins(start, goal, radius, sample, direction=1):
    """Shortest Dubins path from start to goal (rear-axle [x, y, yaw]) at turning radius `radius` as a Path, samples at
    most `sample` apart, or None if no candidate's sampled endpoint is the goal (never, for correct formulas).
    direction -1: the reverse path (the forward curve between the poses turned by pi, driven backward)."""
    st = _dubins_states(start, goal, radius, sample, direction, _ranked(start, goal, radius, direction))
    return None if st is None else Path(st[:, 0], st[:, 1], st[:, 2], np.full(len(st), -1 if direction < 0 else 1))


# -------------------------------------------------------------------------------------------------------- Hybrid A*

class HybridAStar:
    """Hybrid A* over rear-axle states with forward and reverse arc primitives.

    Nodes are keyed by (ix, iy, iyaw, direction) at Planning.xy_resolution and yaw_bins (the start's direction is 0:
    its first primitive pays no switch or steer change); each key keeps its cheapest continuous state. A node is
    expanded with every (direction, steer fraction) arc, Planning.step long, all collision-checked in one call at
    samples at most Planning.path_sample apart. Cost: step (x reverse_cost in reverse) + switch_cost on a change of
    direction + steer_cost |fraction| + steer_change_cost |fraction change|. Priority f = g + heuristic_weight h, with
    h = max(Euclidean distance, grid distance) to the goal; the grid distance runs over ~1 m cells where the rear axle
    has clearance >= width/2 (8-connected, scipy dijkstra from the goal), and nodes on cells it cannot reach are pruned.
    A node within goal_tolerance (and 0.1 goal_tolerance rad) of the goal also ends the search, with the goal appended.

    Every analytic_every-th expansion within analytic_range of the goal tries analytic shots (turning radius
    Truck.min_radius) in order of cost, and the cheapest collision-free one ends the search: the forward and the
    reverse Dubins shot to the goal, and, for each leg L in `staging`, the staged shot (a forward Dubins path to the
    pose L ahead of the spot, then L straight in reverse: the drive-past-and-back-in manoeuvre). Shot costs: length
    (x reverse_cost in reverse) + switch_cost per change of direction (from the node's, and at a staged cusp). A
    reverse shot that turns by more than `reverse_turn_max` (a reversing loop: slow, hard to track) does not end the
    search: the cheapest collision-free one is kept and returned only if the search fails ('budget' or 'no_path').
    staging = () and reverse_turn_max = inf give plain Hybrid A* with forward and reverse Dubins shots."""
    staging = (8.0, 16.0)           # m: straight reverse legs of the staged shot (about one and two truck lengths)
    reverse_turn_max = math.pi/2    # rad of total heading change for a reverse shot that may end the search

    def __init__(self, truck, planning):
        self.truck, self.planning = truck, planning
        p = planning
        n_sub = max(1, int(math.ceil(p.step/p.path_sample-1e-9)))
        fracs = np.asarray(p.steer_fractions, float)
        self.dirs = np.repeat([1, -1], len(fracs))
        self.fracs = np.tile(fracs, 2)
        kappa = np.tan(self.fracs*truck.steer_max)/truck.wheelbase
        sigma = self.dirs[:, None]*p.step*np.arange(1, n_sub+1)/n_sub
        self.lx, self.ly, self.lyaw = _arc(0.0, 0.0, 0.0, kappa[:, None], sigma)    # (P, n_sub) in the node's frame
        self.cost = p.step*np.where(self.dirs > 0, 1.0, p.reverse_cost)+p.steer_cost*np.abs(self.fracs)
        self.radius = truck.min_radius
        self.offsets, r = disk_offsets(truck)
        self.need = r+p.clearance

    def _free(self, clearance, states):
        """collision_free with the disk geometry cached: (M,) bool for states (M, 3)."""
        c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
        centres = np.stack((states[:, 0, None]+self.offsets*c[:, None], states[:, 1, None]+self.offsets*s[:, None]), -1)
        return np.all(clearance.at(centres) >= self.need, axis=-1)

    def plan(self, clearance, start, goal):
        """PlanResult for rear-axle poses start -> goal ((>= 3,) [x, y, yaw, ...]) on a ClearanceMap."""
        t0 = time.perf_counter()
        p = self.planning
        start, goal = np.asarray(start, float)[:3], np.asarray(goal, float)[:3]
        expansions = 0
        fallback = [math.inf, None]     # the cheapest collision-free gated shot: [total cost, (node, states, dirs)]

        def result(path, reason):
            return PlanResult(path, reason, expansions, time.perf_counter()-t0)

        if not self._free(clearance, start[None])[0]:
            return result(None, 'start_blocked')
        if not self._free(clearance, goal[None])[0]:
            return result(None, 'goal_blocked')
        hgrid, hx0, hy0, hres = self._grid_distance(clearance, goal)
        nyh, nxh = hgrid.shape
        gx, gy, gyaw = (float(v) for v in goal)
        i, j = int((start[0]-hx0)//hres), int((start[1]-hy0)//hres)
        h0 = max(hgrid[j, i], math.hypot(gx-start[0], gy-start[1])) if 0 <= i < nxh and 0 <= j < nyh else math.inf
        if not math.isfinite(h0):
            return result(None, 'no_path')
        legs = self._legs(clearance, goal)
        w, tol, bin_w = p.heuristic_weight, p.goal_tolerance, TWO_PI/p.yaw_bins
        n_prim, dirs, fracs = len(self.dirs), self.dirs, self.fracs
        dirs_l, fracs_l = dirs.tolist(), fracs.tolist()
        # Node lists (index = node id): state, cost, direction, steer fraction, parent, primitive samples, key.
        X, Y, YAW, G, D, F = [float(start[0])], [float(start[1])], [float(start[2])], [0.0], [0], [0.0]
        parent, samples, keys = [-1], [None], [self._key(start, 0)]
        best, closed = {keys[0]: 0}, set()
        heap = [(w*h0, 0)]
        shots = 0

        def finish(n, tail, tail_dirs):
            parts = [Path(tail[:, 0], tail[:, 1], tail[:, 2], tail_dirs)]
            while parent[n] >= 0:
                m = parent[n]
                st = np.vstack(((X[m], Y[m], YAW[m]), samples[n]))
                parts.append(Path(st[:, 0], st[:, 1], st[:, 2], np.full(len(st), D[n])))
                n = m
            return result(Path.concat(parts[::-1]), '')

        def give_up(reason):
            return result(None, reason) if fallback[1] is None else finish(*fallback[1])

        while heap:
            n = heapq.heappop(heap)[1]
            key = keys[n]
            if key in closed or best[key] != n:
                continue
            if expansions >= p.max_expansions:
                return give_up('budget')
            closed.add(key)
            expansions += 1
            x, y, yaw, g, d = X[n], Y[n], YAW[n], G[n], D[n]
            dist = math.hypot(gx-x, gy-y)
            if dist <= tol and abs(float(wrap(yaw-gyaw))) <= 0.1*tol:
                end = gyaw+TWO_PI*round((yaw-gyaw)/TWO_PI)
                return finish(n, np.array([[x, y, yaw], [gx, gy, end]]), np.full(2, d or 1))
            if dist <= p.analytic_range:
                if shots % p.analytic_every == 0:
                    shot = self._shot(clearance, (x, y, yaw), goal, d, g, legs, fallback, n)
                    if shot is not None:
                        return finish(n, *shot)
                shots += 1
            # Every primitive at once: (P, n_sub) samples, one collision call; the end states scored and keyed.
            c, s = math.cos(yaw), math.sin(yaw)
            roll = np.stack((x+c*self.lx-s*self.ly, y+s*self.lx+c*self.ly, yaw+self.lyaw), -1)
            free = self._free(clearance, roll.reshape(-1, 3)).reshape(n_prim, -1).all(1)
            ex, ey, eyaw = roll[:, -1, 0], roll[:, -1, 1], roll[:, -1, 2]
            hi, hj = np.floor((ex-hx0)/hres).astype(int), np.floor((ey-hy0)/hres).astype(int)
            inside = (hi >= 0) & (hi < nxh) & (hj >= 0) & (hj < nyh)
            h = np.where(inside, hgrid[np.clip(hj, 0, nyh-1), np.clip(hi, 0, nxh-1)], np.inf)
            ok = np.flatnonzero(free & (h < np.inf))
            if not len(ok):
                continue
            h = np.maximum(h, np.hypot(gx-ex, gy-ey))
            gn = g+self.cost
            if d:
                gn = gn+p.switch_cost*(dirs != d)+p.steer_change_cost*np.abs(fracs-F[n])
            fn = (gn+w*h).tolist()
            gn = gn.tolist()
            kx = np.floor(ex/p.xy_resolution).astype(int).tolist()
            ky = np.floor(ey/p.xy_resolution).astype(int).tolist()
            kyaw = (np.floor((eyaw % TWO_PI)/bin_w).astype(int) % p.yaw_bins).tolist()
            for i in ok.tolist():
                k = (kx[i], ky[i], kyaw[i], dirs_l[i])
                if k in closed:
                    continue
                m = best.get(k)
                if m is not None and G[m] <= gn[i]:
                    continue
                m = len(G)
                best[k] = m
                X.append(float(ex[i])); Y.append(float(ey[i])); YAW.append(float(eyaw[i]))
                G.append(gn[i]); D.append(dirs_l[i]); F.append(fracs_l[i])
                parent.append(n); samples.append(roll[i]); keys.append(k)
                heapq.heappush(heap, (fn[i], m))
        return give_up('no_path')

    def _key(self, state, d):
        p = self.planning
        return (int(math.floor(state[0]/p.xy_resolution)), int(math.floor(state[1]/p.xy_resolution)),
                int(math.floor((state[2] % TWO_PI)/(TWO_PI/p.yaw_bins))) % p.yaw_bins, int(d))

    def _legs(self, clearance, goal):
        """[(stage pose, (m, 3) straight reverse samples stage -> goal)] of the collision-free staging legs."""
        legs, r, sample = [], self.radius, self.planning.path_sample
        for L in self.staging:
            stage = (goal[0]+L*math.cos(goal[2]), goal[1]+L*math.sin(goal[2]), float(goal[2]))
            st = _dubins_states(stage, goal, r, sample, -1, _ranked(stage, goal, r, -1))
            if st is not None and self._free(clearance, st).all():
                legs.append((stage, st))
        return legs

    def _shot(self, clearance, state, goal, d, g, legs, fallback, n):
        """(states, dirs) of the cheapest collision-free shot from `state` (node n: direction d, cost g) to the goal
        that may end the search, or None; a cheaper collision-free gated shot updates `fallback` (see the class)."""
        p, r = self.planning, self.radius
        cands = []          # (cost, tie-break, direction (+1 / -1) or leg index (>= 2), ranked words, gated)
        for direction in (1, -1):
            ranked = _ranked(state, goal, r, direction)
            word, lengths = ranked[0]
            cost = sum(lengths)*(1.0 if direction > 0 else p.reverse_cost)+p.switch_cost*(d == -direction)
            turn = sum(ell for letter, ell in zip(word, lengths) if letter != 'S')/r
            cands.append((cost, direction, direction, ranked, direction < 0 and turn > self.reverse_turn_max))
        for i, (stage, back) in enumerate(legs):
            ranked = _ranked(state, stage, r, 1)
            back_len = math.hypot(back[-1, 0]-back[0, 0], back[-1, 1]-back[0, 1])
            cost = sum(ranked[0][1])+p.switch_cost*(1+(d == -1))+p.reverse_cost*back_len
            cands.append((cost, 2+i, 2+i, ranked, False))
        for cost, _, which, ranked, gated in sorted(cands, key=lambda c: c[:2]):
            if gated and g+cost >= fallback[0]:
                continue
            if which < 2:
                st = _dubins_states(state, goal, r, p.path_sample, which, ranked)
                dirs = None if st is None else np.full(len(st), which)
            else:
                stage, back = legs[which-2]
                st = _dubins_states(state, stage, r, p.path_sample, 1, ranked)
                if st is not None:
                    dirs = np.r_[np.ones(len(st)-1, int), -np.ones(len(back), int)]
                    turns = TWO_PI*round((st[-1, 2]-back[0, 2])/TWO_PI)     # keep the heading continuous
                    st = np.vstack((st[:-1], back+(0.0, 0.0, turns)))
            if st is None or not self._free(clearance, st[1:]).all():
                continue
            if not gated:
                return st, dirs
            fallback[:] = g+cost, (n, st, dirs)
        return None

    def _grid_distance(self, clearance, goal):
        """Shortest 8-connected distance (m) from the goal over coarse cells (~1 m: blocks of k x k map cells) that hold
        a map cell where the rear axle has clearance >= width/2 (inf: unreachable), with the coarse grid's origin and
        cell size."""
        res = float(clearance.res)
        ny, nx = clearance.dist.shape
        k = max(1, int(round(1.0/res)))
        cx = clearance.x0+(np.arange(nx)+0.5)*res
        cy = clearance.y0+(np.arange(ny)+0.5)*res
        c = clearance.at(np.stack(np.broadcast_arrays(cx[None, :], cy[:, None]), -1))
        nyc, nxc = -(-ny//k), -(-nx//k)
        pad = np.full((nyc*k, nxc*k), -np.inf)
        pad[:ny, :nx] = c
        ok = pad.reshape(nyc, k, nxc, k).max((1, 3)) >= self.truck.width/2
        cres = k*res
        gi, gj = int((goal[0]-clearance.x0)//cres), int((goal[1]-clearance.y0)//cres)
        ok[gj, gi] = True          # the goal is collision-free (checked), so its own cell counts
        idx = np.arange(nyc*nxc).reshape(nyc, nxc)
        rows, cols, wts = [], [], []
        for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = (slice(0, nyc-dj), slice(max(0, -di), nxc-max(0, di)))
            b = (slice(dj, nyc), slice(max(0, di), nxc-max(0, -di)))
            m = ok[a] & ok[b]
            rows.append(idx[a][m]); cols.append(idx[b][m]); wts.append(np.full(int(m.sum()), cres*math.hypot(dj, di)))
        graph = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(nyc*nxc, nyc*nxc)).tocsr()
        dist = dijkstra(graph, directed=False, indices=gj*nxc+gi)
        return dist.reshape(nyc, nxc), float(clearance.x0), float(clearance.y0), cres
