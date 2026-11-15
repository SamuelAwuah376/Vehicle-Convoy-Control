"""Hybrid A* for a rigid truck at the rear axle, forward and reverse, closed by an analytic Dubins shot that ends the
path exactly at the goal.

Motion: the kinematic bicycle at the rear axle moves on arcs of curvature tan(steer)/wheelbase; with the signed arc
length sigma (< 0 reversing) a pose (x, y, yaw) goes exactly to
    yaw + k sigma,   (x, y) + chord (cos, sin)(yaw + k sigma/2),   chord = 2 sin(k sigma/2)/k = sigma sinc(k sigma/2pi).
A reverse path is the same geometric curve driven backward: the reverse Dubins path from A to B is the forward Dubins
path from (A, yaw+pi) to (B, yaw+pi), with the truck's heading the Dubins heading - pi.

Paths keep the heading continuous (unwrapped from the start's yaw), so the last sample equals the goal in x, y and in
yaw modulo 2 pi (exactly, when the path does not wind).
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
_ENDPOINT_TOL = 1e-6                         # m and rad: a Dubins candidate must hit the goal this closely


@dataclass
class PlanResult:
    path: Path | None
    reason: str        # '' (found), 'start_blocked', 'goal_blocked', 'no_path' (search exhausted), 'budget'
    expansions: int
    time_s: float


def collision_free(truck, planning, clearance, states):
    """(M,) bool for rear-axle states (M, >= 3) or (3,): every cover disk's centre has clearance (clearance.at) of at
    least its radius + planning.clearance."""
    _, radius = disk_offsets(truck)
    return np.all(clearance.at(disk_centres(truck, states)) >= radius+planning.clearance, axis=-1)


def _arc(x, y, yaw, kappa, sigma):
    """Exact constant-curvature motion of the rear axle: poses after signed arc lengths sigma (broadcast)."""
    half = 0.5*kappa*sigma
    chord = sigma*np.sinc(half/np.pi)
    return x+chord*np.cos(yaw+half), y+chord*np.sin(yaw+half), yaw+2*half


# ---------------------------------------------------------------------------------------------------------- Dubins

def _mod2pi(a):
    a %= TWO_PI
    return 0.0 if a > TWO_PI-1e-9 else a     # a rounding-level full turn is no turn (else a spurious loop)


def dubins_words(start, goal, radius):
    """{word: (t, p, q) segment lengths (m)} of every Dubins word of WORDS whose formula applies between the poses
    start -> goal ([x, y, heading], forward motion) at turning radius `radius` (Shkel & Lumelsky's normalised
    formulas). Unverified: dubins() keeps only candidates whose sampled endpoint is the goal."""
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
    return {w: tuple(radius*v for v in tpq) for w, tpq in out.items()}


def dubins_sample(start, word, lengths, radius, sample):
    """(N, 3) poses along a Dubins word from start ([x, y, heading]), by exact arc integration: each segment is cut
    into equal pieces at most `sample` long, so segment ends are samples. N >= 2 (a zero-length path repeats start)."""
    x, y, yaw = (float(v) for v in start[:3])
    parts = [np.array([[x, y, yaw]])]
    for letter, ell in zip(word, lengths):
        if ell <= 1e-9:            # a rounding-level segment: no sample (the endpoint check bounds the error)
            continue
        u = np.linspace(0.0, ell, max(1, int(math.ceil(ell/sample-1e-9)))+1)[1:]
        px, py, pyaw = _arc(x, y, yaw, _TURN[letter]/radius, u)
        parts.append(np.column_stack((px, py, pyaw)))
        x, y, yaw = px[-1], py[-1], pyaw[-1]
    states = np.concatenate(parts)
    return states if len(states) > 1 else np.repeat(states, 2, 0)


def dubins(start, goal, radius, sample, direction=1):
    """Shortest Dubins path from start to goal (rear-axle [x, y, yaw]) at turning radius `radius`, as a Path with
    samples at most `sample` apart, or None if no candidate's sampled endpoint is the goal (never, for correct
    formulas). direction -1: the reverse path (the forward curve between the poses turned by pi, driven backward)."""
    flip = math.pi if direction < 0 else 0.0
    a = (float(start[0]), float(start[1]), float(start[2])+flip)
    b = (float(goal[0]), float(goal[1]), float(goal[2])+flip)
    for word, lengths in sorted(dubins_words(a, b, radius).items(), key=lambda kv: sum(kv[1])):
        st = dubins_sample(a, word, lengths, radius, sample)
        end = st[-1]
        if abs(end[0]-b[0]) <= _ENDPOINT_TOL and abs(end[1]-b[1]) <= _ENDPOINT_TOL and \
                abs(float(wrap(end[2]-b[2]))) <= _ENDPOINT_TOL:
            yaw = st[:, 2]-flip
            yaw[0] = float(start[2])
            st[-1, :2] = b[:2]
            yaw[-1] = float(goal[2])+TWO_PI*round((yaw[-1]-float(goal[2]))/TWO_PI)
            return Path(st[:, 0], st[:, 1], yaw, np.full(len(st), -1 if direction < 0 else 1))
    return None


# -------------------------------------------------------------------------------------------------------- Hybrid A*

class HybridAStar:
    """Hybrid A* over rear-axle states with forward and reverse arc primitives.

    Nodes are keyed by (ix, iy, iyaw, direction) at Planning.xy_resolution and yaw_bins (the start's direction is 0:
    its first primitive pays no switch); each key keeps its cheapest continuous state. A node is expanded with every
    (direction, steer fraction) arc, Planning.step long, all collision-checked in one call at samples at most
    Planning.path_sample apart. Cost: step (x reverse_cost in reverse) + switch_cost on a change of direction +
    steer_cost |fraction| + steer_change_cost |fraction change|. Priority f = g + heuristic_weight h, with
    h = max(Euclidean distance, grid distance) to the goal; the grid distance runs over ~1 m cells where the rear axle
    has clearance >= width/2 (8-connected, scipy dijkstra from the goal) and nodes on cells it cannot reach are
    pruned. Every analytic_every-th expansion within analytic_range of the goal tries the forward and the reverse
    Dubins shot to the goal (turning radius Truck.min_radius); the cheaper collision-free one ends the search."""
    anytime = False
    staged = ()

    def __init__(self, truck, planning):
        self.truck, self.planning = truck, planning
        p = planning
        self.n_sub = max(1, int(math.ceil(p.step/p.path_sample-1e-9)))
        fracs = np.asarray(p.steer_fractions, float)
        self.dirs = np.repeat([1, -1], len(fracs))
        self.fracs = np.tile(fracs, 2)
        kappa = np.tan(self.fracs*truck.steer_max)/truck.wheelbase
        sigma = self.dirs[:, None]*p.step*np.arange(1, self.n_sub+1)/self.n_sub
        self.lx, self.ly, self.lyaw = _arc(0.0, 0.0, 0.0, kappa[:, None], sigma)    # (P, n_sub) in the node's frame
        self.cost = p.step*np.where(self.dirs > 0, 1.0, p.reverse_cost)+p.steer_cost*np.abs(self.fracs)
        self.radius = truck.min_radius

    def plan(self, clearance, start, goal):
        """PlanResult for rear-axle poses start -> goal ((>= 3,) [x, y, yaw, ...]) on a ClearanceMap."""
        t0 = time.perf_counter()
        truck, p = self.truck, self.planning
        start, goal = np.asarray(start, float)[:3], np.asarray(goal, float)[:3]

        def result(path, reason, n):
            return PlanResult(path, reason, n, time.perf_counter()-t0)

        if not collision_free(truck, p, clearance, start)[0]:
            return result(None, 'start_blocked', 0)
        if not collision_free(truck, p, clearance, goal)[0]:
            return result(None, 'goal_blocked', 0)
        hgrid, hx0, hy0, hres = self._grid_distance(clearance, goal)
        nyh, nxh = hgrid.shape
        gx, gy, gyaw = (float(v) for v in goal)
        w, tol, bin_w = p.heuristic_weight, p.goal_tolerance, TWO_PI/p.yaw_bins
        n_prim, dirs, fracs = len(self.dirs), self.dirs, self.fracs
        dirs_list = dirs.tolist()

        def h_start():
            i, j = int((start[0]-hx0)//hres), int((start[1]-hy0)//hres)
            inside = 0 <= i < nxh and 0 <= j < nyh
            return max(hgrid[j, i], math.hypot(gx-start[0], gy-start[1])) if inside else math.inf

        h0 = h_start()
        if not math.isfinite(h0):
            return result(None, 'no_path', 0)
        # Node arrays (index = node id): state, cost, direction, steer fraction, parent, primitive samples.
        X, Y, YAW, G, D, F = [float(start[0])], [float(start[1])], [float(start[2])], [0.0], [0], [0.0]
        parent, samples, keys = [-1], [None], [self._key(start[0], start[1], start[2], 0)]
        best = {keys[0]: 0}
        closed = set()
        heap = [(w*h0, 0)]
        expansions = shots = 0
        sol_cost, sol = math.inf, None
        while heap:
            fpop, n = heapq.heappop(heap)
            if fpop >= sol_cost:
                break
            key = keys[n]
            if key in closed or best[key] != n:
                continue
            if expansions >= p.max_expansions:
                if sol is not None:
                    break
                return result(None, 'budget', expansions)
            closed.add(key)
            expansions += 1
            x, y, yaw, g, d = X[n], Y[n], YAW[n], G[n], D[n]
            dist = math.hypot(gx-x, gy-y)
            if dist <= tol and abs(float(wrap(yaw-gyaw))) <= 0.1*tol:
                end = gyaw+TWO_PI*round((yaw-gyaw)/TWO_PI)
                tail = Path([x, gx], [y, gy], [yaw, end], [d or 1]*2)
                return result(self._path(n, tail, X, Y, YAW, D, parent, samples), '', expansions)
            if dist <= p.analytic_range:
                if shots % p.analytic_every == 0:
                    tail = self._shot(clearance, (x, y, yaw), goal, d)
                    if tail is not None:
                        if not self.anytime:
                            return result(self._path(n, tail[1], X, Y, YAW, D, parent, samples), '', expansions)
                        if g+tail[0] < sol_cost:
                            sol_cost, sol = g+tail[0], (n, tail[1])
                shots += 1
            # All primitives at once: (P, n_sub) samples, one collision call, the end states keyed and scored.
            c, s = math.cos(yaw), math.sin(yaw)
            px, py, pyaw = x+c*self.lx-s*self.ly, y+s*self.lx+c*self.ly, yaw+self.lyaw
            roll = np.stack((px, py, pyaw), -1)
            free = collision_free(truck, p, clearance, roll.reshape(-1, 3)).reshape(n_prim, -1).all(1)
            ex, ey, eyaw = px[:, -1], py[:, -1], pyaw[:, -1]
            hi, hj = np.floor((ex-hx0)/hres).astype(int), np.floor((ey-hy0)/hres).astype(int)
            inside = (hi >= 0) & (hi < nxh) & (hj >= 0) & (hj < nyh)
            hgr = np.where(inside, hgrid[np.clip(hj, 0, nyh-1), np.clip(hi, 0, nxh-1)], np.inf)
            h = np.maximum(hgr, np.hypot(gx-ex, gy-ey))
            ok = np.flatnonzero(free & np.isfinite(h) & (g+self.cost+h < sol_cost))
            if not len(ok):
                continue
            gn = g+self.cost
            if d:
                gn = gn+p.switch_cost*(dirs != d)+p.steer_change_cost*np.abs(fracs-F[n])
            fn = gn+w*h
            kx = np.floor(ex/p.xy_resolution).astype(int).tolist()
            ky = np.floor(ey/p.xy_resolution).astype(int).tolist()
            kyaw = (np.floor((eyaw % TWO_PI)/bin_w).astype(int) % p.yaw_bins).tolist()
            gn_l, fn_l = gn.tolist(), fn.tolist()
            for i in ok.tolist():
                k = (kx[i], ky[i], kyaw[i], dirs_list[i])
                if k in closed:
                    continue
                m = best.get(k)
                if m is not None and G[m] <= gn_l[i]:
                    continue
                m = len(G)
                best[k] = m
                X.append(float(ex[i])); Y.append(float(ey[i])); YAW.append(float(eyaw[i]))
                G.append(gn_l[i]); D.append(dirs_list[i]); F.append(float(fracs[i]))
                parent.append(n); samples.append(roll[i]); keys.append(k)
                heapq.heappush(heap, (fn_l[i], m))
        if sol is not None:
            return result(self._path(sol[0], sol[1], X, Y, YAW, D, parent, samples), '', expansions)
        return result(None, 'no_path', expansions)

    def _key(self, x, y, yaw, d):
        p = self.planning
        return (int(math.floor(x/p.xy_resolution)), int(math.floor(y/p.xy_resolution)),
                int(math.floor((yaw % TWO_PI)/(TWO_PI/p.yaw_bins))) % p.yaw_bins, int(d))

    def _shot(self, clearance, state, goal, d):
        """The cheaper collision-free of the forward and the reverse Dubins shot from state to goal, or None. Cost:
        length (x reverse_cost in reverse), + switch_cost when the shot's direction differs from the node's d."""
        p = self.planning
        cands = []
        for direction in (1, -1):
            path = dubins(state, goal, self.radius, p.path_sample, direction)
            if path is not None:
                cost = path.length*(1.0 if direction > 0 else p.reverse_cost)+p.switch_cost*(d != 0 and d != direction)
                cands.append((cost, direction, path))
        for L in self.staged:
            gyaw = float(goal[2])
            stage = (goal[0]+L*math.cos(gyaw), goal[1]+L*math.sin(gyaw), gyaw)
            fwd = dubins(state, stage, self.radius, p.path_sample, 1)
            back = dubins(stage, goal, self.radius, p.path_sample, -1)
            if fwd is not None and back is not None:
                cost = fwd.length+p.switch_cost*(1+(d == -1))+p.reverse_cost*back.length
                cands.append((cost, 2, Path.concat([fwd, back])))
        for cost, _, path in sorted(cands, key=lambda c: c[:2]):
            if collision_free(self.truck, p, clearance, path.states[1:]).all():
                return cost, path
        return None

    def _path(self, n, tail, X, Y, YAW, D, parent, samples):
        """Concatenate the primitives from the start to node n, then `tail` (which starts at node n)."""
        parts = [tail]
        while parent[n] >= 0:
            m = parent[n]
            st = np.vstack(([X[m], Y[m], YAW[m]], samples[n]))
            parts.append(Path(st[:, 0], st[:, 1], st[:, 2], np.full(len(st), D[n])))
            n = m
        return Path.concat(parts[::-1])

    def _grid_distance(self, clearance, goal):
        """Shortest 8-connected distance (m) from the goal over coarse cells (~1 m: a block of k x k map cells) that
        hold a map cell where the rear axle has clearance >= width/2 (inf: unreachable), and the coarse grid's
        origin and cell size."""
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
