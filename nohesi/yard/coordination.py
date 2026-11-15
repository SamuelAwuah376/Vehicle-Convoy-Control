"""Multi-truck coordination by path-velocity decomposition: each follower keeps its planned path, and pairwise conflict
zones on those paths are shared under a token shield (mutual exclusion), so only the order in which trucks pass them is
left to decide (the policy's job; the scripted baseline uses a fixed priority list).

A zone is a pair of sample ranges, one on each truck's path, outside of which the two footprints cannot come within
Planning.zone_margin of each other. A truck holds a zone while its progress lies in [in - gate, out] (gate =
gate_buffer/path_sample samples), and the shield stops every other truck at its own gate, so two trucks never occupy
the same zone. Progress is a sample index and only grows along a path; a replan or a back-off is a new path."""
from dataclasses import dataclass
import numpy as np
from .truck import disk_centres, disk_offsets

COARSE = 1.0        # m of arc per cell of the coarse pairwise test in conflict_zones
LOOKAHEAD = 6.0     # m before a zone's gate within which a truck that wants to move requests the zone


@dataclass
class Zone:
    a: int; b: int                 # follower indices, a < b
    a_in: int; a_out: int          # sample range on a's path whose footprint conflicts with b's path (zone_margin)
    b_in: int; b_out: int          # ... on b's path

    def side(self, x):
        """(in, out) on truck x's path."""
        return (self.a_in, self.a_out) if x == self.a else (self.b_in, self.b_out)

    def other(self, x):
        return self.b if x == self.a else self.a


def _cells(path, disks):
    """Coarse cells of a path: contiguous sample blocks COARSE m of arc long. Returns (sizes, rep, rad): the block
    sizes, the block's middle sample, and per block and disk the farthest that disk's centre gets at any sample of the
    block from where it is at the middle sample (M, n)."""
    cell = np.floor(path.s/COARSE).astype(int)
    starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    sizes = np.diff(np.r_[starts, len(cell)])
    rep = starts+(sizes-1)//2
    dev = np.linalg.norm(disks-disks[np.repeat(rep, sizes)], axis=-1)
    return sizes, rep, np.maximum.reduceat(dev, starts, axis=0)


def conflict_matrix(truck, planning, path_a, path_b):
    """(Na, Nb) bool: samples ka, kb conflict when a cover disk of a at ka and one of b at kb are closer than
    2 radius + zone_margin. Exact at the full resolution, computed on ~1 m cells first: a cell pair whose middle
    samples' disks are closer than the reach less both cells' disk excursions conflicts throughout, one farther than
    the reach plus the excursions nowhere (triangle inequality), and only the cell pairs in between are tested sample
    by sample."""
    _, r = disk_offsets(truck)
    reach = 2*r+planning.zone_margin
    da, db = disk_centres(truck, path_a.states), disk_centres(truck, path_b.states)
    na, ra, wa = _cells(path_a, da)
    nb, rb, wb = _cells(path_b, db)
    ca, cb = da[ra], db[rb]                                                   # (Ma, n, 2), (Mb, n, 2)
    d = np.hypot(ca[:, :, None, None, 0]-cb[None, None, :, :, 0], ca[:, :, None, None, 1]-cb[None, None, :, :, 1])
    slack = wa[:, :, None, None]+wb[None, None]                               # (Ma, n, Mb, n)
    sure = np.any(d < reach-slack, axis=(1, 3))
    maybe = np.any(d < reach+slack, axis=(1, 3)) & ~sure
    expand = lambda m: np.repeat(np.repeat(m, na, 0), nb, 1)
    hit = expand(sure)
    ka, kb = np.nonzero(expand(maybe))
    if len(ka):
        gap = da[ka][:, :, None]-db[kb][:, None]
        close = np.min(gap[..., 0]**2+gap[..., 1]**2, axis=(1, 2)) < reach**2
        hit[ka[close], kb[close]] = True
    return hit


def _runs(idx):
    """[(lo, hi)]: inclusive runs of consecutive integers in sorted idx."""
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    return [(int(lo), int(hi)) for lo, hi in zip(idx[np.r_[0, cut+1]], idx[np.r_[cut, len(idx)-1]])]


def conflict_zones(truck, planning, a, path_a, b, path_b):
    """[Zone] between trucks a and b (any order; the zones carry a < b), sorted along a's path. The conflicting
    samples of a (conflict_matrix) form connected runs; the samples of b conflicting with one run form runs of their
    own; each pair of runs is a zone, and zones whose ranges overlap on either path are merged (to a fixed point), so
    the zones' ranges are disjoint on both paths and every conflicting sample pair lies inside one zone."""
    if a > b:
        a, path_a, b, path_b = b, path_b, a, path_a
    hit = conflict_matrix(truck, planning, path_a, path_b)
    boxes = [[lo, hi, *cols] for lo, hi in _runs(np.flatnonzero(hit.any(1)))
             for cols in _runs(np.flatnonzero(hit[lo:hi+1].any(0)))]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                p, q = boxes[i], boxes[j]
                if p[0] <= q[1] and q[0] <= p[1] or p[2] <= q[3] and q[2] <= p[3]:
                    boxes[i] = [min(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), max(p[3], q[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [Zone(a, b, *box) for box in sorted(boxes)]


class Coordinator:
    """Token shield over the conflict zones of up to n trucks' current paths.

    Each zone has a token (self.owner, parallel to self.zones: a truck index or None). update() gives it, in order:
    1. to a truck whose progress lies in the zone's hold range [in - gate, out] (both inside, which the shield never
       lets happen but a replan can: the current owner, else the one earlier in self.order). An owner that is not
       inside loses it once it has passed out or no longer wants to move;
    2. to a truck that wants to move and is within gate + LOOKAHEAD of entering: trucks request in self.order, so on
       simultaneous requests the earlier one wins. Zones that lie closer than a gate apart on a truck's path (with any
       partners) form a cluster, which is requested whole: all its tokens at once or none, and a denied truck drops
       the reservations it has there. The shield therefore stops trucks only outside every hold range, and nobody
       waits while holding a zone: no hold-and-wait deadlock, pairwise or cyclic, except through a truck that stands
       inside a zone (HOLD there, or parked at the end of its path: it holds the zone for good, which blocking()
       exposes) or a replan that puts a zone under a truck;
    3. a truck is stopped at the entry gate (first in - gate - 1, clipped to >= its progress) of the first cluster on
       its path in which another truck holds a zone; the stop moves on to the gate of the first zone it does not hold
       when it is already inside the cluster.
    Zones the truck or its partner has passed are ignored. Callers should report a truck that is still rolling as
    wanting to move: releasing its token while it brakes could let the partner in."""
    def __init__(self, truck, planning, n):
        self.truck, self.planning, self.n = truck, planning, n
        self.order = list(range(n))
        self.gate = round(planning.gate_buffer/planning.path_sample)
        self.look = round(LOOKAHEAD/planning.path_sample)
        self.paths = [None]*n
        self.progress = np.zeros(n, int)
        self.zones, self.owner = [], []
        self._blockers = [[] for _ in range(n)]

    def set_path(self, i, path, progress=0):
        """Give truck i a new path (None: none, it conflicts with nothing) at sample `progress`, and recompute its
        zones against every other truck's current path. A partner's reservation (not a hold of i) carries over to
        the new zones that overlap, on the partner's path, a zone it had reserved."""
        self.paths[i], self.progress[i] = path, int(progress)
        old = [(z, o) for z, o in zip(self.zones, self.owner) if i in (z.a, z.b)]
        kept = [(z, o) for z, o in zip(self.zones, self.owner) if i not in (z.a, z.b)]
        for j in range(self.n):
            if path is None or j == i or self.paths[j] is None:
                continue
            for z in conflict_zones(self.truck, self.planning, i, path, j, self.paths[j]):
                lo, hi = z.side(j)
                carried = any(o == j and z0.other(j) == i and z0.side(j)[0] <= hi and lo <= z0.side(j)[1]
                              for z0, o in old)
                kept.append((z, j if carried else None))
        self.zones, self.owner = [z for z, _ in kept], [o for _, o in kept]

    def _hold(self, x, z):
        lo, hi = z.side(x)
        return lo-self.gate <= self.progress[x] <= hi

    def _clusters(self, x):
        """Truck x's live zones (neither x nor its partner past out) as clusters of indices into self.zones, in order
        along x's path: a zone joins the previous cluster when its gate stop (in - gate - 1) is not past the cluster's
        out, i.e. a truck stopped at it would still be holding the cluster."""
        p, out, end = self.progress, [], -np.inf
        live = [k for k, z in enumerate(self.zones) if x in (z.a, z.b) and p[x] <= z.side(x)[1]
                and p[z.other(x)] <= z.side(z.other(x))[1]]
        for k in sorted(live, key=lambda k: self.zones[k].side(x)[0]):
            lo, hi = self.zones[k].side(x)
            if out and lo-self.gate-1 <= end:
                out[-1].append(k)
                end = max(end, hi)
            else:
                out.append([k])
                end = hi
        return out

    def update(self, progress, wants):
        """progress (n,): current sample per truck; wants (n,) bool: the truck wants to move along its path now.
        Returns per truck the last sample it may reach (None: no limit), and records the tokens (self.owner) and who
        blocks whom (blocking())."""
        p = np.array(progress, int)
        for i, path in enumerate(self.paths):
            if path is not None:
                p[i] = min(max(p[i], 0), len(path)-1)
        self.progress, wants, g = p, np.asarray(wants, bool), self.gate
        rank = {t: r for r, t in enumerate(self.order)}
        for k, z in enumerate(self.zones):
            inside, o = [x for x in (z.a, z.b) if self._hold(x, z)], self.owner[k]
            if inside:
                o = o if o in inside else min(inside, key=lambda x: rank.get(x, self.n+x))
            elif o is not None and (p[o] > z.side(o)[1] or not wants[o]):
                o = None
            self.owner[k] = o
        clusters = [self._clusters(x) for x in range(self.n)]
        for x in self.order:
            if not wants[x] or not clusters[x]:
                continue
            c = clusters[x][0]
            if p[x] < min(self.zones[k].side(x)[0] for k in c)-g-self.look:
                continue
            if all(self.owner[k] in (None, x) for k in c):
                for k in c:
                    self.owner[k] = x
            else:
                for k in c:
                    if self.owner[k] == x and not self._hold(x, self.zones[k]):
                        self.owner[k] = None
        stops = [None]*self.n
        for x in range(self.n):
            self._blockers[x] = []
            for c in clusters[x]:
                others = sorted({self.owner[k] for k in c}-{None, x})
                if others:
                    gate = min(self.zones[k].side(x)[0] for k in c if self.owner[k] != x)-g-1
                    stops[x], self._blockers[x] = max(int(gate), int(p[x])), others
                    break
        return stops

    def blocking(self, i):
        """Trucks holding a zone that stops truck i, as of the last update."""
        return list(self._blockers[i])

    def priority_wait(self, i, progress):
        """The scripted baseline's rule: truck i must wait while a truck earlier in self.order shares a zone with it
        that i has not passed and the earlier truck has not yet passed (progress < out)."""
        p = np.asarray(progress, int)
        ahead = set(self.order[:self.order.index(i)])
        return any(z.other(i) in ahead and p[i] <= z.side(i)[1] and p[z.other(i)] < z.side(z.other(i))[1]
                   for z in self.zones if i in (z.a, z.b))

    def zones_of(self, i):
        """Observation of truck i's coordination, as of the last update (or set_path): per truck j (n,), None for
        j == i or when no live zone with j lies ahead, else the nearest one along i's path not yet passed by either as
        (metres along i's path to its in-gate, metres along j's path to j's in-gate, j holds the zone's token); a
        distance is negative once that truck is past its gate."""
        out, p, g = [None]*self.n, self.progress, self.gate
        if self.paths[i] is None:
            return out
        for k, z in enumerate(self.zones):
            if i not in (z.a, z.b):
                continue
            j = z.other(i)
            (i_in, i_out), (j_in, j_out) = z.side(i), z.side(j)
            if p[i] > i_out or p[j] > j_out:
                continue
            si, sj = self.paths[i].s, self.paths[j].s
            di = float(si[max(i_in-g, 0)]-si[p[i]])
            if out[j] is None or di < out[j][0]:
                out[j] = (di, float(sj[max(j_in-g, 0)]-sj[p[j]]), self.owner[k] == j)
        return out
