"""Leader-follower haul convoy on a quarry site: the simulation behind the promo video and the memo's numbers.

One human-driven lead haul truck and three driverless followers carry rock from a pit floor up an
8-9 % ramp and along a haul road to a run-of-mine (ROM) pad, where each truck spots and paddock-dumps.

* Site: a terraced pit (three 10 m benches), a ramp cut along the pit wall, an ex-pit haul road with a
  light-vehicle crossing, and a ROM pad beside a crusher plant (terrain exported as a heightfield).
* Trucks: loaded 60 t-class rigid haul trucks (100 t gross, 520 kW), kinematic bicycle with steering lag and
  rate limits, first-order drive/brake lag, power- and grade-limited acceleration, 0.3 g service braking.
* Leader: a human driver model (speed profile from limits and curves, a natural lateral wander, a 1 s reaction).
* Followers: track the leader's front-axle path from its V2V breadcrumbs (20 Hz, 40-60 ms latency, 2 % loss)
  with a Stanley controller, and keep spacing with CACC (predecessor acceleration fed forward over V2V).
  They replicate the operator's line, not a map, so no site survey is needed.
* Safety: each follower checks its own swept corridor ahead with its perception; an object predicted to be in
  it within 2.5 s triggers a convoy-wide hold over V2V. Resume needs a clear corridor and an operator tap.
* A second, empty convoy runs the return lane.

Outputs (in --out): scene.json (static geometry), terrain.bin (heights + colours), traj.json (30 fps states and
HUD data) and metrics.json (tracking error, gaps, stop margins).

Run:  python venture/sim/convoy_sim.py --out venture/video/data
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

FPS = 30
DT = 0.01            # plant step (100 Hz)
CTRL_EVERY = 5       # controllers and perception at 20 Hz
G = 9.81

# Pit: elliptical, rim semi-axes A x B, three 10 m benches to a floor at -30 m
PIT_A, PIT_B, PIT_DEPTH = 170.0, 115.0, 30.0
ROAD_HALF = 11.0     # 22 m haul road
LANE_OFF = 5.5       # lane centre from road centre (keep right)
BERM_IN, BERM_OUT, BERM_H = 11.5, 15.5, 1.8
PAD = (545.0, 700.0, 172.0, 292.0)        # ROM pad x0, x1, y0, y1
PLANT = (585.0, 735.0, 292.0, 365.0)       # crusher plant footprint
ACCESS_X = 380.0                           # light-vehicle access road (north-south)

# Truck (60 t class rigid haul truck, generic)
WHEELBASE = 4.2
LENGTH = 10.2
WIDTH = 5.1
REAR_OVERHANG = 3.3  # rear axle to tail
MASS_LOADED, MASS_EMPTY = 100_000.0, 40_000.0
POWER = 520_000.0
ROLL_RES = 0.025
A_BRAKE_MAX = 3.0
TAU_DRIVE = 0.5
TAU_STEER = 0.15
STEER_MAX = math.radians(36.0)
STEER_RATE = math.radians(22.0)

# Convoy spacing (front axle to front axle minus truck length = bumper gap, along the path)
GAP_STANDSTILL = 20.0
TIME_GAP = 3.0


# --------------------------------------------------------------------------------------------- noise and helpers

def _hash(i, j, seed):
    n = (i.astype(np.int64) * 374761393 + j.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFFFF) / 4294967296.0


def value_noise(x, y, seed=0):
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    u, v = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    xi, yi = xi.astype(np.int64), yi.astype(np.int64)
    a, b = _hash(xi, yi, seed), _hash(xi + 1, yi, seed)
    c, d = _hash(xi, yi + 1, seed), _hash(xi + 1, yi + 1, seed)
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v


def fbm(x, y, octaves=4, seed=0):
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        total = total + amp * value_noise(x * freq, y * freq, seed + 17 * o)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def catmull_rom(pts, n_per_seg=24):
    pts = np.asarray(pts, float)
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        # centripetal parameterisation avoids cusps
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        for t in np.linspace(t1, t2, n_per_seg, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(pts[-1])
    return np.array(out)


def resample(xy, ds):
    d = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s_new = np.arange(0.0, s[-1], ds)
    return np.column_stack([np.interp(s_new, s, xy[:, 0]), np.interp(s_new, s, xy[:, 1])]), s_new


def moving_average(a, n):
    if n <= 1:
        return a.copy()
    pad = n // 2
    ap = np.concatenate([np.full(pad, a[0]), a, np.full(pad, a[-1])])
    k = np.ones(n) / n
    return np.convolve(ap, k, mode='valid')[: len(a)]


# --------------------------------------------------------------------------------------------- polyline

class Polyline:
    """A growable 2D polyline with arc length, per-point heading and speed, and windowed projection."""

    def __init__(self, capacity=20000):
        self.xy = np.zeros((capacity, 2))
        self.s = np.zeros(capacity)
        self.head = np.zeros(capacity)
        self.v = np.zeros(capacity)
        self.n = 0

    @classmethod
    def from_points(cls, xy, head=None, v=None):
        xy = np.asarray(xy, float)
        if head is None:
            d = np.gradient(xy, axis=0)
            head = np.arctan2(d[:, 1], d[:, 0])
        pl = cls(len(xy) + 8)
        for i, p in enumerate(xy):
            pl.append(p, head[i], 0.0 if v is None else v[i])
        return pl

    def append(self, p, head=None, v=0.0):
        if self.n and np.hypot(*(p - self.xy[self.n - 1])) < 0.05:
            return
        if self.n == len(self.s):
            for name in ('xy', 's', 'head', 'v'):
                arr = getattr(self, name)
                setattr(self, name, np.concatenate([arr, np.zeros_like(arr)]))
        self.xy[self.n] = p
        self.s[self.n] = 0.0 if self.n == 0 else self.s[self.n - 1] + np.hypot(*(p - self.xy[self.n - 1]))
        if head is None:
            head = 0.0 if self.n == 0 else math.atan2(*(p - self.xy[self.n - 1])[::-1])
        self.head[self.n] = head
        self.v[self.n] = v
        self.n += 1

    def project(self, p, hint=0, back=40, fwd=400):
        """Nearest point to p: (arc length, signed lateral offset (+left), heading, index)."""
        i0 = max(0, hint - back)
        i1 = min(self.n - 1, hint + fwd)
        if i1 <= i0:
            return self.s[i0], 0.0, self.head[i0], i0
        a = self.xy[i0:i1]
        ab = self.xy[i0 + 1:i1 + 1] - a
        ap = p - a
        l2 = np.maximum((ab * ab).sum(1), 1e-9)
        t = np.clip((ap * ab).sum(1) / l2, 0.0, 1.0)
        proj = a + ab * t[:, None]
        d2 = ((p - proj) ** 2).sum(1)
        k = int(np.argmin(d2))
        i = i0 + k
        seg = math.sqrt(l2[k])
        s = self.s[i] + t[k] * seg
        e = (ab[k, 0] * ap[k, 1] - ab[k, 1] * ap[k, 0]) / seg
        h0, h1 = self.head[i], self.head[i + 1]
        head = h0 + wrap(h1 - h0) * t[k]
        return s, e, head, i

    def at(self, s):
        n = self.n
        s = min(max(s, 0.0), self.s[n - 1])
        i = int(np.searchsorted(self.s[:n], s) - 1)
        i = min(max(i, 0), n - 2)
        seg = self.s[i + 1] - self.s[i]
        t = 0.0 if seg <= 0 else (s - self.s[i]) / seg
        p = self.xy[i] + (self.xy[i + 1] - self.xy[i]) * t
        head = self.head[i] + wrap(self.head[i + 1] - self.head[i]) * t
        v = self.v[i] + (self.v[i + 1] - self.v[i]) * t
        return p, head, v, i

    @property
    def length(self):
        return self.s[self.n - 1] if self.n else 0.0


# --------------------------------------------------------------------------------------------- site

@dataclass
class Site:
    road_xy: np.ndarray        # road centreline, 0.5 m spacing (loaded direction: pit -> pad)
    road_s: np.ndarray
    road_z: np.ndarray
    road_head: np.ndarray
    road_kappa: np.ndarray
    s_ramp: tuple
    s_cross: float             # arc length of the access-road crossing
    xs: np.ndarray = None
    ys: np.ndarray = None
    height: np.ndarray = None
    colour: np.ndarray = None
    interp: RegularGridInterpolator = None

    def lane(self, side, wander=None):
        """Lane centreline: side +1 = loaded lane (right of the pit->pad direction), -1 = return lane."""
        nx, ny = -np.sin(self.road_head), np.cos(self.road_head)      # left normal
        off = -side * LANE_OFF + (0.0 if wander is None else wander)
        return self.road_xy + np.column_stack([nx * off, ny * off])

    def z_at(self, x, y):
        return self.interp(np.column_stack([np.atleast_1d(y), np.atleast_1d(x)]))


def bench_profile(m, nz):
    """Elevation inside the pit from the inward distance m (metres) from the rim: 3 benches of 10 m."""
    z = np.zeros_like(m)
    face = 2.7 + 0.9 * nz            # horizontal run of a 10 m face (about 73-77 degrees)
    bench = 14.0
    x = m.copy()
    level = 0.0
    for k in range(3):
        on_face = (x > 0) & (x <= face)
        z = np.where(on_face, level - 10.0 * x / face, z)
        past = x > face
        z = np.where(past, level - 10.0, z)
        level -= 10.0
        x = x - face - (bench if k < 2 else 0.0)
        z = np.where((x > 0) & (k < 2), level, z)
    return z


def natural_ground(x, y):
    rolling = (1.6 * np.sin(x / 95.0 + 0.4) * np.cos(y / 120.0) + 1.1 * np.sin((x + 0.6 * y) / 70.0)
               + 1.4 * (fbm(x / 140.0, y / 140.0, 4, 3) - 0.5))
    r = np.hypot(x - 260.0, (y - 80.0) * 1.2)
    ang = np.arctan2(y - 80.0, x - 260.0)
    ridge = 0.55 + 0.45 * fbm(np.cos(ang) * 3 + 10, np.sin(ang) * 3 + 10, 4, 7)
    hills = 95.0 * smoothstep(600.0, 2600.0, r) * ridge + 18.0 * smoothstep(450.0, 900.0, r) * fbm(x / 260, y / 260, 3, 9)
    return rolling + hills


def pit_ground(x, y):
    rho = np.sqrt((x / PIT_A) ** 2 + (y / PIT_B) ** 2)
    m = (1.0 - rho) * PIT_B + 5.0 * (fbm(x / 55.0, y / 55.0, 3, 11) - 0.5)
    nz = fbm(x / 12.0, y / 12.0, 3, 5) - 0.5
    zb = bench_profile(np.maximum(m, 0.0), nz) + 0.35 * nz
    zo = natural_ground(x, y)
    inside = m > 0
    return np.where(inside, zb + zo * (1.0 - smoothstep(0.0, 8.0, m)), zo), m


def build_route():
    """Road centreline: pit floor (loading area) -> wall ramp (counter-clockwise) -> ex-pit road -> ROM pad."""
    # one continuous counter-clockwise spiral: along the pit floor from the loading area (north), then up the
    # wall ramp, which cuts through the benches as it widens out to the rim
    floor = []
    for u in np.linspace(0.0, 1.0, 9)[:-1]:
        phi = math.radians(95.0 + (215.0 - 95.0) * u)
        rho = 0.45 + (0.64 - 0.45) * u
        floor.append((PIT_A * rho * math.cos(phi), PIT_B * rho * math.sin(phi)))
    ramp = []
    phi0, phi1 = math.radians(215.0), math.radians(375.0)
    for u in np.linspace(0.0, 1.0, 17):
        phi = phi0 + (phi1 - phi0) * u
        rho = 0.64 + (1.075 - 0.64) * u
        ramp.append((PIT_A * rho * math.cos(phi), PIT_B * rho * math.sin(phi)))
    # out of the pit: a 60 m right-hand arc onto the haul road east to the ROM pad
    x0, y0 = ramp[-1]
    h0 = math.radians(112.0)
    cx, cy = x0 + 60.0 * math.cos(h0 - math.pi / 2), y0 + 60.0 * math.sin(h0 - math.pi / 2)
    a0 = math.atan2(y0 - cy, x0 - cx) % (2 * math.pi)   # right-hand arc: the angle decreases to pi/2
    arc = [(cx + 60.0 * math.cos(a), cy + 60.0 * math.sin(a)) for a in np.linspace(a0, math.pi / 2, 7)[1:]]
    exit_road = arc + [(300.0, 120.0), (ACCESS_X, 128.0), (450.0, 140.0), (505.0, 165.0), (548.0, 192.0), (575.0, 205.0)]
    ctrl = floor + ramp + exit_road
    dense = catmull_rom(ctrl, 30)
    xy, s = resample(dense, 0.5)
    # heading and curvature (smoothed)
    d = np.gradient(xy, axis=0)
    head = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    head = moving_average(head, 9)
    kappa = np.gradient(head) / 0.5
    kappa = moving_average(kappa, 15)
    # ramp extent along s (nearest points to the first and last ramp control points)
    i_r0 = int(np.argmin(np.hypot(*(xy - np.array(ramp[0])).T)))
    i_r1 = int(np.argmin(np.hypot(*(xy - np.array(ramp[-1])).T)))
    i_c = int(np.argmin(np.abs(xy[:, 0] - ACCESS_X) + 10 * (xy[:, 1] < 100)))
    s_r0, s_r1, s_c = s[i_r0], s[i_r1], s[i_c]
    # elevation: floor -30, a linear ramp, then a smoothed natural profile outside
    z_out = moving_average(natural_ground(xy[:, 0], xy[:, 1]), 241)
    z_ramp_end = z_out[i_r1]
    z = np.where(s <= s_r0, -PIT_DEPTH, np.where(s >= s_r1, z_out, -PIT_DEPTH + (z_ramp_end + PIT_DEPTH) * (s - s_r0) / (s_r1 - s_r0)))
    z = moving_average(z, 61)
    z[: max(1, i_r0 - 40)] = -PIT_DEPTH
    return xy, s, z, head, kappa, (s_r0, s_r1), s_c


def build_site():
    xy, s, z, head, kappa, s_ramp, s_c = build_route()
    site = Site(xy, s, z, head, kappa, s_ramp, s_c)

    # non-uniform rectilinear grid: 3 m in the action area, growing outwards to the horizon
    def axis(lo, hi, step, far, n_out):
        inner = np.arange(lo, hi + 1e-6, step)
        grow = np.geomspace(step, (far - (hi - lo) / 2) / n_out * 3.0, n_out)
        grow = grow / grow.sum() * (far - (hi - lo) / 2)
        left = lo - np.cumsum(grow)[::-1]
        right = hi + np.cumsum(grow)
        return np.concatenate([left, inner, right])

    def refine(a, lo, hi, step):
        keep = a[(a < lo) | (a > hi)]
        return np.sort(np.concatenate([keep, np.arange(lo, hi + 1e-6, step)]))

    xs = refine(axis(-330.0, 830.0, 3.0, 4200.0, 26), -186.0, 192.0, 1.5)
    ys = refine(axis(-270.0, 470.0, 3.0, 4200.0, 26), -128.0, 130.0, 1.5)
    X, Y = np.meshgrid(xs, ys)
    H, m_in = pit_ground(X, Y)

    # ROM pad and plant flattened, with 1:1.5 transitions
    z_pad = float(z[-1])
    for (x0, x1, y0, y1) in (PAD, PLANT):
        dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
        dd = np.hypot(dx, dy)
        H = z_pad + np.clip(H - z_pad, -0.7 * dd, 0.9 * dd)

    # haul road: nearest centreline sample; berms except at the crossing, on the floor and at the pad
    tree = cKDTree(xy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    dist, idx = tree.query(pts, distance_upper_bound=120.0)
    near = np.isfinite(dist)
    dist = np.where(near, dist, 1e9)
    idx = np.where(near, idx, 0)
    zr = z[idx]
    sr = s[idx]
    berm_on = (sr > s_ramp[0] + 15) & (np.abs(sr - s_c) > 18.0) & (sr < s[-1] - 30.0)
    berm = np.where(berm_on, BERM_H * np.clip(1.0 - np.abs(dist - (BERM_IN + BERM_OUT) / 2) / ((BERM_OUT - BERM_IN) / 2), 0, 1), 0.0)
    berm = berm * (0.85 + 0.3 * fbm(X.ravel() / 4.0, Y.ravel() / 4.0, 2, 21))
    w = np.where(berm_on, BERM_OUT, ROAD_HALF + 1.0)
    dd = np.maximum(dist - w, 0.0)
    Hf = H.ravel()
    Hn = zr + np.clip(Hf - zr, -0.75 * dd, 1.45 * dd)
    Hn = np.where(dist <= w, zr + berm, Hn)
    Hf = np.where(near, Hn, Hf)
    H = Hf.reshape(X.shape)

    # access road (light vehicles), 10 m wide, straight north-south at ACCESS_X
    ay = np.clip(Y, -10.0, 330.0)
    za = moving_average(natural_ground(np.full(200, ACCESS_X), np.linspace(-10, 330, 200)), 41)
    zacc = np.interp(ay, np.linspace(-10, 330, 200), za)
    d_acc = np.hypot(X - ACCESS_X, Y - ay)
    i_cross = int(np.argmin(np.abs(s - s_c)))
    zacc = np.where(np.abs(Y - xy[i_cross, 1]) < 40.0, z[i_cross] + (zacc - z[i_cross]) * smoothstep(12.0, 40.0, np.abs(Y - xy[i_cross, 1])), zacc)
    dd = np.maximum(d_acc - 5.5, 0.0)
    road_mask = (dist.reshape(X.shape) <= ROAD_HALF + 1.0)
    Ha = zacc + np.clip(H - zacc, -0.8 * dd, 1.2 * dd)
    H = np.where(road_mask, H, np.where(d_acc < 60.0, Ha, H))

    # ---- colours
    gy, gx = np.gradient(H, ys, xs)
    slope = np.hypot(gx, gy)
    n1 = fbm(X / 9.0, Y / 9.0, 3, 31)
    n2 = fbm(X / 70.0, Y / 70.0, 3, 37)
    dirt = np.stack([168 + 22 * (n1 - .5), 141 + 18 * (n1 - .5), 104 + 14 * (n1 - .5)], -1)
    scrub_mask = smoothstep(0.52, 0.72, n2)[..., None] * (m_in < -6)[..., None]
    scrub = np.stack([118 + 20 * n1, 118 + 16 * n1, 74 + 10 * n1], -1)
    col = dirt * (1 - 0.75 * scrub_mask) + scrub * 0.75 * scrub_mask
    strata = 0.5 + 0.5 * np.sin(H * 2.9 + 2.0 * fbm(X / 25.0, Y / 25.0, 2, 41))
    rock = np.stack([138 + 30 * strata + 14 * (n1 - .5), 128 + 27 * strata + 12 * (n1 - .5), 114 + 22 * strata + 10 * (n1 - .5)], -1)
    steep = smoothstep(0.55, 1.2, slope)[..., None]
    inpit = (m_in > -2)[..., None]
    floor_col = np.stack([196 + 16 * (n1 - .5), 188 + 14 * (n1 - .5), 172 + 12 * (n1 - .5)], -1)
    col = np.where(inpit, floor_col, col)
    col = col * (1 - steep) + rock * steep
    # road surface with wheel tracks and darker edges
    D = dist.reshape(X.shape)
    lat = D  # unsigned
    track = np.zeros_like(D)
    for c in (LANE_OFF - 1.85, LANE_OFF + 1.85):
        track += np.exp(-((lat - c) / 0.55) ** 2)
    road_c = np.stack([172 + 10 * (n1 - .5), 152 + 9 * (n1 - .5), 120 + 8 * (n1 - .5)], -1) - 20 * track[..., None]
    road_c = road_c - 16 * smoothstep(8.5, 11.0, lat)[..., None]
    on_road = (D <= ROAD_HALF)[..., None]
    col = np.where(on_road, road_c, col)
    berm_c = np.stack([150 + 18 * (n1 - .5), 128 + 14 * (n1 - .5), 98 + 12 * (n1 - .5)], -1)
    on_berm = ((D > ROAD_HALF) & (D < BERM_OUT + 0.5))[..., None] & (berm.reshape(X.shape) > 0.15)[..., None]
    col = np.where(on_berm, berm_c, col)
    acc_c = np.stack([186 + 8 * (n1 - .5), 170 + 8 * (n1 - .5), 140 + 6 * (n1 - .5)], -1)
    col = np.where(((d_acc < 5.0) & ~road_mask)[..., None], acc_c, col)
    for (x0, x1, y0, y1) in (PAD, PLANT):
        inside = ((X > x0) & (X < x1) & (Y > y0) & (Y < y1))[..., None]
        col = np.where(inside, np.stack([190 + 10 * (n1 - .5), 178 + 9 * (n1 - .5), 154 + 8 * (n1 - .5)], -1), col)
    # cheap ambient occlusion from concavity
    lap = np.zeros_like(H)
    lap[1:-1, 1:-1] = (H[1:-1, :-2] + H[1:-1, 2:] + H[:-2, 1:-1] + H[2:, 1:-1] - 4 * H[1:-1, 1:-1])
    ao = np.clip(1.0 - 0.06 * np.clip(lap, 0, 8), 0.7, 1.0)
    col = col * ao[..., None]
    # far terrain fades toward a hazy olive-brown
    far = smoothstep(500.0, 2200.0, np.hypot(X - 260.0, Y - 80.0))[..., None]
    col = col * (1 - far * 0.5) + np.array([150, 140, 112]) * far * 0.5
    col = np.clip(col, 0, 255).astype(np.uint8)

    site.xs, site.ys, site.height, site.colour = xs, ys, H.astype(np.float32), col
    site.interp = RegularGridInterpolator((ys, xs), H, bounds_error=False, fill_value=None)
    return site


# --------------------------------------------------------------------------------------------- vehicles

@dataclass
class Truck:
    name: str
    role: str                  # 'lead' (human) or 'follow'
    x: float                   # rear axle
    y: float
    yaw: float
    v: float = 0.0
    a: float = 0.0
    steer: float = 0.0
    steer_cmd: float = 0.0
    u: float = 0.0             # commanded acceleration
    mass: float = MASS_LOADED
    odo: float = 0.0
    hint: int = 0
    road_hint: int = 0
    grade: float = 0.0
    mode: str = 'FOLLOW'       # FOLLOW, HOLD, SPOT, DUMP, DONE
    tip: float = 0.0
    payload: float = 1.0
    dump_t: float = -1.0
    lat_err: float = 0.0
    gap: float = float('nan')
    brake: float = 0.0
    spot_path: Polyline = None
    spot_hint: int = 0
    spot_end: float = 0.0
    bc: Polyline = None        # breadcrumbs (leader front-axle path) as received
    last_pred: dict = None
    det_state: str = ''
    det_range: float = float('nan')
    log_err: list = field(default_factory=list)
    stopped_at: float = -1.0
    s_lane: float = 0.0
    det_stop_dist: float = None

    @property
    def front(self):
        return np.array([self.x + WHEELBASE * math.cos(self.yaw), self.y + WHEELBASE * math.sin(self.yaw)])

    def a_limits(self):
        v = max(self.v, 0.6)
        trac = min(0.12 * G, POWER / (self.mass * v))
        resist = G * (self.grade + ROLL_RES)
        return -A_BRAKE_MAX, trac - resist

    def step(self, dt):
        # drive/brake lag, then physical limits
        self.a += (self.u - self.a) * dt / TAU_DRIVE
        lo, hi = self.a_limits()
        net = min(max(self.a, lo), hi)
        if self.v <= 0.0 and net < 0.0:
            net = 0.0
        self.brake = float(np.clip(-self.u / 1.5, 0.0, 1.0)) if self.u < -0.15 else 0.0
        # steering lag and rate limit
        target = min(max(self.steer_cmd, -STEER_MAX), STEER_MAX)
        rate = (target - self.steer) / TAU_STEER
        rate = min(max(rate, -STEER_RATE), STEER_RATE)
        self.steer += rate * dt
        self.v = max(0.0, self.v + net * dt)
        self.yaw += self.v * math.tan(self.steer) / WHEELBASE * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.odo += self.v * dt


def stanley(truck, path, hint, k=1.6):
    """Steering for the front axle to track a front-axle path; returns (cmd, lateral error, s, index)."""
    s, e, head, i = path.project(truck.front, hint)
    he = wrap(head - truck.yaw)
    cmd = he + math.atan2(-k * e, truck.v + 1.2)
    return cmd, e, s, i


# --------------------------------------------------------------------------------------------- V2V channel

class V2V:
    """Broadcast channel: per-receiver latency 40-60 ms and 2 % independent loss."""

    def __init__(self, rng, latency=0.04, jitter=0.02, loss=0.02):
        self.rng, self.latency, self.jitter, self.loss = rng, latency, jitter, loss
        self.queue = []   # (t_deliver, receiver, msg)

    def send(self, t, msg, receivers):
        for r in receivers:
            if self.rng.random() < self.loss:
                continue
            self.queue.append((t + self.latency + self.jitter * self.rng.random(), r, msg))

    def deliver(self, t):
        due = [q for q in self.queue if q[0] <= t]
        self.queue = [q for q in self.queue if q[0] > t]
        due.sort(key=lambda q: q[0])
        return due


# --------------------------------------------------------------------------------------------- light vehicle

class Pickup:
    """Site light vehicle that rolls through the crossing in front of follower 1, stops in the lane, then leaves."""
    L, W = 5.6, 2.1

    def __init__(self, y_road):
        self.x, self.y, self.yaw, self.v = ACCESS_X - 2.0, 290.0, -math.pi / 2, 0.0
        self.y_road = y_road
        self.state = 'parked'
        self.t_state = 0.0
        self.odo = 0.0
        self.y_stop = y_road - LANE_OFF + 0.6

    def start(self, t, y_start):
        self.y = y_start
        self.v = 8.0
        self.state = 'approach'
        self.t_state = t

    def step(self, t, dt):
        if self.state == 'approach':
            d = self.y - self.y_stop
            if d < 9.0:
                self.v = max(0.0, self.v - 4.5 * dt)
                if self.v <= 0.0:
                    self.state, self.t_state = 'stopped', t
        elif self.state == 'stopped':
            if t - self.t_state > 2.6:
                self.state, self.t_state = 'leave', t
        elif self.state == 'leave':
            if self.y > 70.0:
                self.v = min(9.0, self.v + 2.2 * dt)
            else:                                   # pulls up at the south end of the access road
                self.v = max(0.0, self.v - 3.0 * dt)
                if self.v <= 0.0:
                    self.state = 'parked_south'
        self.y -= self.v * dt
        self.odo += self.v * dt

    def corners(self, t_ahead=0.0):
        y = self.y - self.v * t_ahead
        hl, hw = self.L / 2, self.W / 2
        return np.array([[self.x - hw, y - hl], [self.x + hw, y - hl], [self.x + hw, y + hl], [self.x - hw, y + hl]])


def corridor_hit(truck, poly, d_look=80.0, half_w=WIDTH / 2 + 1.6, obstacle=None):
    """Distance along the truck's path ahead to the first sample within half_w of the obstacle polygon."""
    s0, _, _, i = poly.project(truck.front, truck.hint if truck.role == 'follow' else 0)
    j1 = int(np.searchsorted(poly.s[:poly.n], s0 + d_look))
    pts = poly.xy[i:max(i + 2, j1)]
    if len(pts) == 0:
        return None
    c = obstacle
    lo, hi = c.min(0), c.max(0)
    dx = np.maximum(np.maximum(lo[0] - pts[:, 0], pts[:, 0] - hi[0]), 0)
    dy = np.maximum(np.maximum(lo[1] - pts[:, 1], pts[:, 1] - hi[1]), 0)
    hit = np.nonzero(np.hypot(dx, dy) < half_w)[0]
    if len(hit) == 0:
        return None
    return poly.s[i + hit[0]] - s0


# --------------------------------------------------------------------------------------------- convoy

class Convoy:
    def __init__(self, name, site, side, rng, n_follow=3, loaded=True, start_s=None, wander_seed=0, t_start=0.0):
        self.name, self.site, self.side, self.rng = name, site, side, rng
        self.t_start = t_start
        road_xy = site.road_xy if side > 0 else site.road_xy[::-1]
        road_head = site.road_head if side > 0 else site.road_head[::-1] + math.pi
        s_road = site.road_s if side > 0 else (site.road_s[-1] - site.road_s[::-1])
        kappa = site.road_kappa if side > 0 else -site.road_kappa[::-1]
        z = site.road_z if side > 0 else site.road_z[::-1]
        nx, ny = -np.sin(road_head), np.cos(road_head)
        rs = np.random.default_rng(100 + wander_seed)
        ph1, ph2 = rs.uniform(0, 6.28, 2)
        wander = 0.32 * np.sin(2 * np.pi * s_road / 88.0 + ph1) + 0.16 * np.sin(2 * np.pi * s_road / 33.0 + ph2)
        off = -LANE_OFF + wander
        lane = road_xy + np.column_stack([nx * off, ny * off])
        self.lane = Polyline.from_points(lane)
        self.lane_s_road = s_road
        self.grade = np.gradient(z) / np.maximum(np.gradient(s_road), 1e-6)
        # leader speed profile: limits, curves, the pad approach; then a backward pass for gentle braking
        v_lim = np.where(s_road < (site.s_ramp[1] if side > 0 else 0), 6.0, 9.7)
        if side < 0:
            s_ramp_rev = (site.road_s[-1] - site.s_ramp[1], site.road_s[-1] - site.s_ramp[0])
            v_lim = np.where(s_road > s_ramp_rev[0], 6.0, 9.7)
        v_curve = np.sqrt(1.3 / np.maximum(np.abs(kappa), 1e-4))
        v_t = np.minimum(v_lim, v_curve)
        if side > 0:
            v_t = np.minimum(v_t, 4.0 + 5.7 * np.clip((s_road[-1] - s_road) / 60.0, 0, 1))
        for i in range(len(v_t) - 2, -1, -1):
            ds = s_road[i + 1] - s_road[i]
            v_t[i] = min(v_t[i], math.sqrt(v_t[i + 1] ** 2 + 2 * 0.55 * ds))
        self.v_prof = v_t
        self.lane.v[: self.lane.n] = v_t[: self.lane.n]
        self.trucks: list[Truck] = []
        start_s = start_s if start_s is not None else 0.0
        spacing = LENGTH + GAP_STANDSTILL + 0.6
        names = ['LEAD'] + [f'AUTO {k}' for k in range(1, n_follow + 1)]
        for k, nm in enumerate(names):
            s_front = start_s - k * spacing
            p, head, _, i = self.lane.at(s_front)
            yaw = head
            rear = p - WHEELBASE * np.array([math.cos(yaw), math.sin(yaw)])
            t = Truck(nm, 'lead' if k == 0 else 'follow', rear[0], rear[1], yaw,
                      mass=MASS_LOADED if loaded else MASS_EMPTY, payload=1.0 if loaded else 0.0)
            t.hint = i
            t.road_hint = i
            self.trucks.append(t)
        # breadcrumbs the leader drove before t=0 (pre-filled, as if received)
        for f in self.trucks[1:]:
            f.bc = Polyline(30000)
            for s in np.arange(max(0.0, start_s - (n_follow + 1) * spacing - 5.0), start_s, 0.5):
                p, head, v, _ = self.lane.at(s)
                f.bc.append(p, head, 0.0)
            f.hint = 0
        self.lead_true = Polyline(40000)   # ground truth of the leader's front-axle path
        for s in np.arange(max(0.0, start_s - (n_follow + 1) * spacing - 5.0), start_s, 0.5):
            p, head, _, _ = self.lane.at(s)
            self.lead_true.append(p, head)
        self.state = 'RUN'
        self.hold_t = -1.0
        self.clear_t = -1.0
        self.resume_t = -1.0
        self.alert_t = -1.0
        self.events = []
        self.lead_hint = self.trucks[0].hint
        self.pad_entry = None
        self.spots = None
        self.active = t_start <= 0.0

    # ----- leader (human driver model)
    def drive_leader(self, t):
        L = self.trucks[0]
        if L.mode in ('SPOT', 'DUMP', 'DONE'):
            return
        cmd, e, s, i = stanley(L, self.lane, self.lead_hint, k=1.4)
        self.lead_hint = i
        L.steer_cmd = cmd
        L.grade = float(self.grade[min(int(s / 0.5), len(self.grade) - 1)])
        v_ref = float(np.interp(s + 3.0, self.lane.s[: self.lane.n], self.v_prof[: self.lane.n]))
        # convoy pacing: the lead tablet shows the convoy stretched (e.g. after the ramp) and the operator eases off
        excess = max([f.gap - (GAP_STANDSTILL + TIME_GAP * f.v) for f in self.trucks[1:]
                      if f.mode == 'FOLLOW' and math.isfinite(f.gap)] or [0.0])
        if self.state == 'RUN':
            v_ref *= min(1.0, max(0.7, 1.0 - 0.012 * (excess - 6.0)))
        u = 0.55 * (v_ref - L.v)
        u = min(max(u, -1.2), 0.7)
        if self.alert_t >= 0 and t >= self.alert_t + 1.0 and self.state in ('HOLD', 'CLEAR'):
            u = -1.3 if L.v > 0.05 else -0.3
        if self.state == 'RESUMED' and t < self.resume_t + 1.0:   # the operator pulls away a moment after the tap
            u = min(u, 0.0)
        L.u = u
        self._lead_s = s

    # ----- followers
    def drive_follower(self, t, k):
        f = self.trucks[k]
        if f.mode in ('SPOT', 'DUMP', 'DONE'):
            return
        bc = f.bc
        cmd, e, s_self, i = stanley(f, bc, f.hint)
        f.hint = i
        f.steer_cmd = cmd
        # longitudinal: CACC on path distance to the predecessor (its broadcast state)
        msg = f.last_pred
        u = 0.0
        if msg is not None:
            age = t - msg['t']
            p_pred = np.array(msg['front']) + age * msg['v'] * np.array([math.cos(msg['head']), math.sin(msg['head'])])
            s_pred, _, _, _ = bc.project(p_pred, max(0, i), back=5, fwd=1200)
            gap = s_pred - s_self - LENGTH
            f.gap = gap
            g_des = GAP_STANDSTILL + TIME_GAP * f.v
            u = 0.62 * msg['a'] + 0.55 * (msg['v'] - f.v) + 0.14 * (gap - g_des)
            if gap < 6.0:
                u = min(u, -2.5)
        # site speed profile (posted limits and curve speeds along the route) as the follower's speed cap
        f.grade = self._grade_at(f)
        v_cap = float(np.interp(f.s_lane + 4.0, self.lane.s[: self.lane.n], self.v_prof[: self.lane.n]))
        u = min(u, 0.6 * (v_cap + 0.4 - f.v))
        # convoy hold (obstacle anywhere in the convoy): stop smoothly, the detecting truck stops hard if needed
        if self.state in ('HOLD', 'CLEAR') or (self.state == 'RESUMED' and t < self.resume_t + 1.0):
            u = min(u, -1.4 if f.v > 0.05 else -0.3)
        if f.det_stop_dist is not None:
            d = max(f.det_stop_dist, 0.3)
            u = min(u, -min(A_BRAKE_MAX, max(1.0, f.v ** 2 / (2 * d))) if f.v > 0.05 else -0.3)
        f.u = min(max(u, -A_BRAKE_MAX), 0.9)
        f.lat_err = self._true_err(f)

    def _grade_at(self, tr):
        s, e, _, i = self.lane.project(tr.front, tr.road_hint, back=30, fwd=300)
        tr.road_hint = i
        tr.s_lane = s
        return float(self.grade[min(int(s / 0.5), len(self.grade) - 1)])

    def _true_err(self, f):
        s, e, _, i = self.lead_true.project(f.front, getattr(f, 'true_hint', 0), back=20, fwd=400)
        f.true_hint = i
        return e

    # ----- pad: spotting and dumping
    def setup_spots(self, spots):
        self.spots = spots

    def maybe_enter_pad(self, t, k):
        tr = self.trucks[k]
        if self.spots is None or tr.mode not in ('FOLLOW', 'HOLD'):
            return
        if tr.front[0] < 556.0:
            return
        sx, sy = self.spots[k]
        p0 = tr.front
        h0 = tr.yaw + tr.steer
        p3 = np.array([sx + WHEELBASE, sy])
        h3 = 0.0
        c1 = p0 + 30.0 * np.array([math.cos(h0), math.sin(h0)])
        c2 = p3 - 34.0 * np.array([math.cos(h3), math.sin(h3)])
        tt = np.linspace(0, 1, 160)[:, None]
        bez = (1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * c1 + 3 * (1 - tt) * tt ** 2 * c2 + tt ** 3 * p3
        ext = p3 + np.outer(np.linspace(0.5, 20, 40), [1.0, 0.0])
        tr.spot_path = Polyline.from_points(np.vstack([bez, ext]))
        tr.spot_hint = 0
        tr.spot_end = tr.spot_path.project(p3, 120, back=200, fwd=200)[0]
        tr.mode = 'SPOT'
        self.events.append((t, f'{tr.name}: dump spot assigned'))

    def drive_pad(self, t, k):
        tr = self.trucks[k]
        if tr.mode == 'SPOT':
            cmd, e, s, i = stanley(tr, tr.spot_path, tr.spot_hint)
            tr.spot_hint = i
            tr.steer_cmd = cmd
            d = tr.spot_end - s
            v_ref = min(4.0, math.sqrt(max(0.0, 2 * 0.6 * max(d - 0.3, 0))))
            u = 0.7 * (v_ref - tr.v)
            # keep clear of the truck ahead (paths diverge, but entries are close)
            if k > 0:
                ahead = self.trucks[k - 1]
                dist = np.hypot(*(ahead.front - tr.front))
                if dist < LENGTH + 8.0 and ahead.mode == 'SPOT':
                    u = min(u, -1.0)
            tr.u = min(max(u, -2.0), 0.6)
            tr.grade = 0.0
            if d < 0.4 and tr.v < 0.05:
                tr.mode, tr.dump_t = 'DUMP', t
                self.events.append((t, f'{tr.name}: dumping'))
        elif tr.mode == 'DUMP':
            dt_ = t - tr.dump_t
            tr.steer_cmd = 0.0
            if dt_ < 1.0:
                tr.tip = 0.0
                tr.u = -0.3
            elif dt_ < 8.0:
                x = (dt_ - 1.0) / 7.0
                tr.tip = math.radians(52.0) * (x * x * (3 - 2 * x))
                tr.u = -0.3
            elif dt_ < 10.0:
                tr.tip = math.radians(52.0)
                tr.u = -0.3
            elif dt_ < 16.0:
                x = (dt_ - 10.0) / 6.0
                tr.tip = math.radians(52.0) * (1 - x * x * (3 - 2 * x))
                tr.u = 0.25 if tr.v < 1.0 and dt_ < 14.0 else -0.4
            else:
                tr.tip = 0.0
                tr.u = -0.5
                if tr.v < 0.05:
                    tr.mode = 'DONE'
            if tr.tip > math.radians(22.0):
                tr.payload = max(0.0, tr.payload - DT / 3.2)
            tr.mass = MASS_EMPTY + (MASS_LOADED - MASS_EMPTY) * tr.payload
        elif tr.mode == 'DONE':
            tr.u = -0.5
            tr.steer_cmd = 0.0


# --------------------------------------------------------------------------------------------- scenario

def run(out_dir: Path, seed=7, t_end=None, quiet=False):
    rng = np.random.default_rng(seed)
    site = build_site()
    if not quiet:
        L = site.road_s[-1]
        r0, r1 = site.s_ramp
        dz = site.road_z[int(r1 / 0.5)] - site.road_z[int(r0 / 0.5)]
        print(f'route {L:.0f} m, ramp {r1 - r0:.0f} m at {100 * dz / (r1 - r0):.1f} %, crossing at s={site.s_cross:.0f} m, '
              f'min radius {1 / np.abs(site.road_kappa).max():.0f} m')

    loaded = Convoy('loaded', site, +1, rng, start_s=100.0, wander_seed=1)
    loaded.setup_spots([(646.0, 252.0), (646.0, 238.0), (646.0, 224.0), (646.0, 210.0)])
    # return convoy: timed so it passes the loaded convoy on the ex-pit road, clear of the crossing event
    empty = Convoy('empty', site, -1, rng, start_s=100.0, loaded=False, wander_seed=2, t_start=80.0)
    convoys = [loaded, empty]
    v2v = V2V(rng)
    i_cross = int(np.argmin(np.abs(site.road_s - site.s_cross)))
    pickup = Pickup(site.road_xy[i_cross, 1])
    pickup_triggered = False

    t = 0.0
    step = 0
    frames = []
    hud_events = []
    det_log = {}
    t_end = t_end or 260.0
    for c in convoys:
        for tr in c.trucks:
            tr.det_stop_dist = None
    while t < t_end:
        ctrl = step % CTRL_EVERY == 0
        for c in convoys:
            if not c.active:
                if t >= c.t_start:
                    c.active = True
                else:
                    continue
            L = c.trucks[0]
            if ctrl:
                # V2V: leader breadcrumbs + every truck's state to the truck behind it
                for k, tr in enumerate(c.trucks):
                    msg = {'t': t, 'front': tr.front.tolist(), 'head': tr.yaw + tr.steer, 'v': tr.v, 'a': tr.a, 'k': k, 'c': c.name}
                    rec = [(c.name, j) for j in range(k + 1, len(c.trucks))] if k == 0 else [(c.name, k + 1)] if k + 1 < len(c.trucks) else []
                    v2v.send(t, msg, rec)
        if ctrl:
            for td, (cname, j), msg in v2v.deliver(t):
                c = loaded if cname == 'loaded' else empty
                f = c.trucks[j]
                if msg.get('estop'):
                    continue
                if msg['k'] == 0 and f.mode in ('FOLLOW', 'HOLD'):
                    f.bc.append(np.array(msg['front']), msg['head'], msg['v'])
                if msg['k'] == j - 1:
                    f.last_pred = msg
        for c in convoys:
            if not c.active:
                continue
            L = c.trucks[0]
            if ctrl:
                c.lead_true.append(L.front, L.yaw + L.steer)
                c.drive_leader(t)
                for k in range(1, len(c.trucks)):
                    c.drive_follower(t, k)
                for k in range(len(c.trucks)):
                    if c.name == 'loaded':
                        c.maybe_enter_pad(t, k)
                    c.drive_pad(t, k)
        # light vehicle: starts so it enters follower 1's corridor when follower 1 is ~32 m from the crossing
        f1 = loaded.trucks[1]
        if not pickup_triggered and f1.bc is not None:
            # distance from F1's front to the crossing
            p_cross = site.road_xy[i_cross]
            d_cross = np.hypot(*(p_cross - f1.front))
            if d_cross < 105.0 and f1.front[0] < ACCESS_X:
                # the pickup's front reaches the convoy lane's corridor when follower 1 is ~45 m from the crossing
                y_enter = site.road_xy[i_cross, 1] - LANE_OFF + (WIDTH / 2 + 1.6) + Pickup.L / 2
                y_start = y_enter + 8.0 * (d_cross - 45.0) / max(f1.v, 1.0)
                pickup.start(t, y_start)
                pickup_triggered = True
                hud_events.append((t, 'LV-07 approaching crossing'))
        if pickup.state not in ('parked', 'parked_south'):
            pickup.step(t, DT)
        # perception (20 Hz): every follower checks its own corridor for the light vehicle
        if ctrl and pickup.state != 'parked':
            for c in convoys:
                if not c.active:
                    continue
                for k, tr in enumerate(c.trucks):
                    tr.det_stop_dist = None
                    if tr.role != 'follow' or tr.mode not in ('FOLLOW', 'HOLD'):
                        continue
                    rel = np.array([pickup.x, pickup.y]) - tr.front
                    rng_ = float(np.hypot(*rel))
                    bearing = abs(wrap(math.atan2(rel[1], rel[0]) - tr.yaw))
                    if rng_ > 85.0 or bearing > math.radians(60):
                        tr.det_state = ''
                        continue
                    tr.det_range = rng_
                    hit_now = corridor_hit(tr, tr.bc, obstacle=pickup.corners(0.0))
                    hit_pred = None
                    for ta in (0.5, 1.0, 1.5, 2.0, 2.5):
                        hp = corridor_hit(tr, tr.bc, obstacle=pickup.corners(ta))
                        if hp is not None:
                            hit_pred = hp
                            break
                    hit = hit_now if hit_now is not None else hit_pred
                    tr.det_state = 'TRACK' if hit is None else 'IN PATH'
                    if hit is not None:
                        tr.det_stop_dist = hit - 15.0 - WHEELBASE
                        if c.state == 'RUN':
                            c.state, c.hold_t = 'HOLD', t
                            c.alert_t = t + 0.05
                            hud_events.append((t, f'{tr.name}: light vehicle in path {hit:.0f} m ahead - CONVOY HOLD'))
                            det_log.update({'t_detect': t, 'range_at_detect': rng_, 'path_dist_at_detect': hit,
                                            'v_at_detect': tr.v, 'truck': tr.name})
                    if tr.name == 'AUTO 1' and c.name == 'loaded':
                        det_log.setdefault('first_track_t', t if tr.det_state else None)
            for c in convoys:
                if c.state == 'HOLD':
                    blocked = any(tr.det_state == 'IN PATH' for tr in c.trucks)
                    if not blocked and pickup.state == 'leave' and all(tr.v < 0.05 for tr in c.trucks[1:]):
                        c.state, c.clear_t = 'CLEAR', t
                        hud_events.append((t, 'Corridor clear - waiting for operator'))
                if c.state == 'CLEAR' and t > c.clear_t + 1.6:
                    c.state, c.resume_t = 'RESUMED', t
                    hud_events.append((t, 'Operator: RESUME'))
        for c in convoys:
            if not c.active:
                continue
            for tr in c.trucks:
                tr.step(DT)
                if tr.v < 0.03 and c.state in ('HOLD', 'CLEAR') and tr.stopped_at < 0 and tr.role == 'follow':
                    tr.stopped_at = t
        # log at 30 fps
        if len(frames) <= t * FPS:
            fr = {'t': round(t, 3), 'c': []}
            for c in convoys:
                rows = []
                for k, tr in enumerate(c.trucks):
                    rows.append([tr.x, tr.y, tr.yaw, tr.steer, tr.odo, tr.v, tr.a, tr.brake, tr.tip, tr.payload,
                                 tr.lat_err if tr.role == 'follow' else 0.0, tr.gap if tr.role == 'follow' else float('nan'),
                                 {'FOLLOW': 0, 'HOLD': 1, 'SPOT': 2, 'DUMP': 3, 'DONE': 4}.get(tr.mode, 0),
                                 {'': 0, 'TRACK': 1, 'IN PATH': 2}.get(tr.det_state, 0), tr.det_range if tr.det_state else 0.0])
                fr['c'].append({'state': c.state, 'active': c.active, 'trucks': rows})
            fr['lv'] = [pickup.x, pickup.y, pickup.yaw, pickup.odo, pickup.v]
            frames.append(fr)
        t += DT
        step += 1

    # ---------------------------------------------------------------- metrics
    def stats(convoy):
        out = {}
        for k, tr in enumerate(convoy.trucks[1:], start=1):
            errs = np.array([fr['c'][convoys.index(convoy)]['trucks'][k][10] for fr in frames
                             if fr['c'][convoys.index(convoy)]['active'] and fr['c'][convoys.index(convoy)]['trucks'][k][12] == 0
                             and fr['c'][convoys.index(convoy)]['trucks'][k][5] > 1.0])
            gaps = np.array([fr['c'][convoys.index(convoy)]['trucks'][k][11] for fr in frames
                             if fr['c'][convoys.index(convoy)]['active'] and fr['c'][convoys.index(convoy)]['trucks'][k][12] == 0])
            gaps = gaps[np.isfinite(gaps)]
            out[tr.name] = {'lat_rms_cm': float(100 * np.sqrt(np.mean(errs ** 2))) if len(errs) else None,
                            'lat_max_cm': float(100 * np.abs(errs).max()) if len(errs) else None,
                            'gap_min_m': float(gaps.min()) if len(gaps) else None}
        return out

    # stop margin: closest approach of any loaded follower's front bumper to the light vehicle while it is in the lane
    margin = 1e9
    for fr in frames:
        lvx, lvy = fr['lv'][0], fr['lv'][1]
        for row in fr['c'][0]['trucks'][1:]:
            fx = row[0] + (WHEELBASE + 2.4) * math.cos(row[2])
            fy = row[1] + (WHEELBASE + 2.4) * math.sin(row[2])
            margin = min(margin, math.hypot(fx - lvx, fy - lvy) - Pickup.L / 2)
    metrics = {
        'route_m': float(site.road_s[-1]),
        'ramp_grade_pct': float(100 * (site.road_z[int(site.s_ramp[1] / 0.5)] - site.road_z[int(site.s_ramp[0] / 0.5)]) / (site.s_ramp[1] - site.s_ramp[0])),
        'loaded': stats(loaded), 'empty': stats(empty), 'detection': det_log,
        'closest_approach_to_light_vehicle_m': float(margin),
        'v2v': {'rate_hz': 20, 'latency_ms': '40-60', 'loss_pct': 2},
        'hud_events': [(round(a, 2), b) for a, b in hud_events] + [(round(a, 2), b) for a, b in loaded.events],
        'duration_s': t_end,
    }
    if not quiet:
        print(json.dumps({k: metrics[k] for k in ('loaded', 'empty', 'detection', 'closest_approach_to_light_vehicle_m')}, indent=1))

    if out_dir is not None:
        export(out_dir, site, frames, metrics, loaded)
    return site, frames, metrics


def export(out_dir: Path, site: Site, frames, metrics, loaded):
    out_dir.mkdir(parents=True, exist_ok=True)
    H = site.height
    (out_dir / 'terrain_height.bin').write_bytes(H.astype('<f4').tobytes())
    (out_dir / 'terrain_colour.bin').write_bytes(site.colour.astype(np.uint8).tobytes())
    # truck ground contact (z, pitch, roll) from the heightfield, per frame
    zs = []
    for fr in frames:
        rows = []
        for ci, c in enumerate(fr['c']):
            for row in c['trucks']:
                x, y, yaw = row[0], row[1], row[2]
                cx, sy_ = math.cos(yaw), math.sin(yaw)
                pts = np.array([[x - 1.0 * cx, y - 1.0 * sy_], [x + WHEELBASE * cx, y + WHEELBASE * sy_],
                                [x - 2.2 * sy_, y + 2.2 * cx], [x + 2.2 * sy_, y - 2.2 * cx]])
                rows.append(pts)
        lv = fr['lv']
        rows.append(np.array([[lv[0], lv[1] + 1.6], [lv[0], lv[1] - 1.6], [lv[0] - 0.9, lv[1]], [lv[0] + 0.9, lv[1]]]))
        zs.append(np.vstack(rows))
    allp = np.vstack(zs)
    zz = site.z_at(allp[:, 0], allp[:, 1]).reshape(len(frames), -1, 4)
    for fi, fr in enumerate(frames):
        k = 0
        for c in fr['c']:
            for row in c['trucks']:
                zr, zf, zl, zrt = zz[fi, k]
                pitch = math.atan2(zf - zr, WHEELBASE + 1.0)
                roll = math.atan2(zl - zrt, 4.4)
                z_mid = zr + (zf - zr) * (1.0 / (WHEELBASE + 1.0))
                row[:] = [round(float(v), 4) if isinstance(v, (float, np.floating)) and math.isfinite(v) else (None if isinstance(v, float) else v) for v in row]
                row.extend([round(float(z_mid), 3), round(float(pitch), 4), round(float(roll), 4)])
                k += 1
        zl = zz[fi, k]
        fr['lv'] = [round(float(v), 3) for v in fr['lv']] + [round(float(zl.mean()), 3), round(float(math.atan2(zl[0] - zl[1], 3.2)), 4)]
    scene = {
        'grid': {'xs': [round(float(v), 3) for v in site.xs], 'ys': [round(float(v), 3) for v in site.ys]},
        'road': {'xy': np.round(site.road_xy[::4], 3).tolist(), 'z': np.round(site.road_z[::4], 3).tolist(),
                 'head': np.round(site.road_head[::4], 4).tolist(), 's': np.round(site.road_s[::4], 2).tolist(),
                 'ramp': list(site.s_ramp), 's_cross': site.s_cross, 'half_width': ROAD_HALF, 'lane_off': LANE_OFF},
        'access_x': ACCESS_X, 'pad': PAD, 'plant': PLANT,
        'pit': {'a': PIT_A, 'b': PIT_B, 'depth': PIT_DEPTH},
        'truck': {'wheelbase': WHEELBASE, 'length': LENGTH, 'width': WIDTH, 'rear_overhang': REAR_OVERHANG},
        'spots': loaded.spots, 'fps': FPS,
        'z_pad': float(site.road_z[-1]),
    }
    (out_dir / 'scene.json').write_text(json.dumps(scene))
    (out_dir / 'traj.json').write_text(json.dumps({'fps': FPS, 'frames': frames}, separators=(',', ':')))
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(f'wrote {out_dir} ({len(frames)} frames)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'video' / 'data')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--t-end', type=float, default=None)
    ap.add_argument('--no-export', action='store_true')
    a = ap.parse_args()
    run(None if a.no_export else a.out, seed=a.seed, t_end=a.t_end)
