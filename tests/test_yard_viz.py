"""nohesi.yard.viz on a synthetic YardEpisode-like record: hand-made forward / reverse paths with cusps, a ray-cast scan
grid, a missed rock revealed mid-run that forces a back-off and a replan, rejected driver picks and the event log."""
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image
from nohesi.geometry import clearances
from nohesi.yard import viz
from nohesi.yard.config import YardScenario
from nohesi.yard.paths import Path
from nohesi.yard.truck import corners
from nohesi.yard.world import YardWorld

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
HOLD, CREEP, GO, BACK_OFF, REPLAN = range(5)


def _drive(start, moves, step=0.25):
    """A Path from rear-axle pose `start` through (direction, length, curvature) moves, samples `step` apart."""
    x, y, yaw = (float(v) for v in start)
    xs, ys, yaws, ds = [x], [y], [yaw], []
    for d, length, kappa in moves:
        m = max(1, int(np.ceil(length/step)))
        h = d*length/m
        for _ in range(m):
            if kappa:
                new = yaw+h*kappa
                x += (np.sin(new)-np.sin(yaw))/kappa
                y -= (np.cos(new)-np.cos(yaw))/kappa
                yaw = new
            else:
                x += h*np.cos(yaw); y += h*np.sin(yaw)
            xs.append(x); ys.append(y); yaws.append(yaw); ds.append(d)
    return Path(xs, ys, yaws, ds+[ds[-1]])


def _at(path, s):
    return [np.interp(s, path.s, path.x), np.interp(s, path.s, path.y), np.interp(s, path.s, np.unwrap(path.yaw))]


class _Track:
    """One follower's per-dt rows (state, action, safety stop)."""
    def __init__(self, start, dt):
        self.dt, self.rows = dt, [(np.r_[start[:3], 0.0], HOLD, False)]

    @property
    def t(self):
        return (len(self.rows)-1)*self.dt

    def hold(self, until, action=HOLD, safety=False):
        last = self.rows[-1][0].copy(); last[3] = 0.0
        while self.t < until-1e-9:
            self.rows.append((last, action, safety))

    def run(self, path, i0=0, i1=None, action=GO, v_fwd=2.8, v_rev=1.4, creep=0.0, dwell=1.0):
        """Drive samples i0 .. i1 of `path`: constant speed per direction, `dwell` s stopped at each cusp, the last
        `creep` m at CREEP speed."""
        i1 = len(path)-1 if i1 is None else int(i1)
        legs = [i0, *(int(c) for c in path.cusps if i0 < c < i1), i1]
        for a, b in zip(legs[:-1], legs[1:]):
            d, pos, end = int(path.direction[a]), path.s[a], path.s[b]
            while pos < end-1e-9:
                slow = b == i1 and end-pos <= creep
                v = 0.7 if slow else (v_fwd if d > 0 else v_rev)
                pos = min(pos+v*self.dt, end)
                self.rows.append((np.r_[_at(path, pos), d*v], CREEP if slow else action, False))
            if b != i1:
                self.hold(self.t+dwell, action)
        self.rows[-1][0][3] = 0.0


def _scan_grid(world, missed, res=0.5):
    """A stand-in for sensors.initial_map: the leader's lidar ray-cast against the obstacles it did not miss."""
    site, sensors, truck = world.scenario.site, world.scenario.sensors, world.scenario.truck
    x0, y0 = -site.road_length-site.berm_width-2.0, -site.berm_width-2.0
    nx = int(round((site.length+site.road_length+2*site.berm_width+4.0)/res))
    ny = int(round((site.width+2*site.berm_width+4.0)/res))
    X, Y = np.meshgrid(x0+(np.arange(nx)+0.5)*res, y0+(np.arange(ny)+0.5)*res)
    pts = np.column_stack((X.ravel(), Y.ravel()))
    occ = np.zeros(len(pts), bool)
    for c, m in zip(world.obstacles.corners, missed):
        if not m:
            occ |= MplPath(c).contains_points(pts)
    occ = occ.reshape(ny, nx)
    lead = world.leader
    origin = lead[:2]+sensors.lidar_mount*np.array([np.cos(lead[2]), np.sin(lead[2])])
    ang = np.radians(np.arange(0.0, 360.0, sensors.lidar_resolution))[:, None]
    r = np.arange(0.0, sensors.lidar_range, res/2)[None]
    ix = np.floor((origin[0]+r*np.cos(ang)-x0)/res).astype(int)
    iy = np.floor((origin[1]+r*np.sin(ang)-y0)/res).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = np.clip(ix, 0, nx-1), np.clip(iy, 0, ny-1)
    hit = occ[iy, ix] & inside
    first = np.where(hit.any(1), hit.argmax(1), r.shape[1])
    cells = np.zeros((ny, nx), np.uint8)
    free = (np.arange(r.shape[1])[None] < first[:, None]) & inside
    cells[iy[free], ix[free]] = FREE
    rays = np.flatnonzero(first < r.shape[1])
    cells[iy[rays, first[rays]], ix[rays, first[rays]]] = OCCUPIED
    road = (X >= -site.road_length) & (X <= 0) & (np.abs(Y-site.width/2) <= site.road_width/2)
    cells[road] = FREE
    for foot in corners(truck, np.vstack((lead[None], world.states))):
        cells[MplPath(foot).contains_points(pts).reshape(ny, nx)] = FREE
    return SimpleNamespace(cells=cells, x0=x0, y0=y0, res=res)


def _nearest(world, kind, xy):
    obs = world.obstacles
    ks = [k for k in range(len(obs)) if obs.kind[k] == kind]
    return min(ks, key=lambda k: np.hypot(*(obs.corners[k].mean(0)-xy)))


def synthetic_episode(duration=12.0, dt=0.1, seed=0):
    """A YardEpisode-like record (SimpleNamespace with the documented fields) on YardWorld(YardScenario(), seed) with
    three followers: F1 and F2 drive in and back into spots on the south side (one cusp each), F3 parks nose-in; a
    rock the scan missed is revealed in F2's reverse leg (safety stop, back-off, replan); F3 holds for the crossing."""
    scenario = YardScenario()
    world = YardWorld(scenario, seed, n_followers=3)
    obs = world.obstacles
    missed = np.zeros(len(obs), bool)
    rock_a, rock_b, rock_c = (_nearest(world, 'rock', xy) for xy in ((73, 38), (89, 16), (94, 26)))
    missed[[rock_a, rock_b, rock_c]] = True
    grid0 = _scan_grid(world, missed)
    theta = np.arccos(1-11.0/30)     # S-curve from the road centre line (y = 35) to y = 46
    entry = lambda x_start, x_turn: [(1, 4.0-x_start, 0.0), (1, 15*theta, 1/15), (1, 15*theta, -1/15),
                                     (1, x_turn-(4.0+30*np.sin(theta)), 0.0)]
    starts = world.states
    p1 = _drive(starts[0, :3], entry(starts[0, 0], 73.0)+[(1, 9*np.pi/2, 1/9), (-1, 48.0, 0.0)])
    p2 = _drive(starts[1, :3], entry(starts[1, 0], 65.0)+[(1, 9*np.pi/2, 1/9), (-1, 48.0, 0.0)])
    p3 = _drive(starts[2, :3], entry(starts[2, 0], 90.0))
    tracks = [_Track(s, dt) for s in starts]
    events = []
    # F1: straight in, cusp, back into its spot.
    tracks[0].run(p1, creep=4.0)
    events.append({'time': tracks[0].t, 'follower': 0, 'kind': 'parked', 'detail': '0.04 m'})
    # F2: starts 5 s later; 3.5 m into its reverse leg the missed rock is revealed.
    f2 = tracks[1]
    f2.hold(5.0)
    cusp = int(p2.cusps[0])
    k_rev = int(np.searchsorted(p2.s, p2.s[cusp]+3.5))
    f2.run(p2, 0, k_rev)
    t_rev = f2.t
    events.append({'time': t_rev, 'follower': 1, 'kind': 'safety_stop', 'detail': 'obstacle 6.1 m'})
    f2.hold(t_rev+1.5, GO, safety=True)
    events.append({'time': f2.t, 'follower': 1, 'kind': 'back_off', 'detail': '3.0 m'})
    f2.run(p2.retrace(k_rev, k_rev-12), action=BACK_OFF, v_fwd=0.7, v_rev=0.7)
    t_replan = f2.t
    events.append({'time': t_replan, 'follower': 1, 'kind': 'replan', 'detail': 'rock revealed on path'})
    here = f2.rows[-1][0]
    p2b = _drive(here[:3], [(-1, 15*0.5236, -1/15), (-1, 15*0.5236, 1/15), (-1, 9.5, 0.0), (-1, 15*0.5236, 1/15),
                            (-1, 15*0.5236, -1/15), (-1, max(here[1]-15-9.5-15-7.0, 1.0), 0.0)])
    f2.hold(t_replan+1.0, REPLAN)
    f2.run(p2b, creep=4.0)
    events.append({'time': f2.t, 'follower': 1, 'kind': 'parked', 'detail': '0.11 m'})
    # F3: in behind them, holds west of the two reverse columns until they are clear.
    f3 = tracks[2]
    f3.hold(10.0)
    k_gate = int(np.searchsorted(p3.x, 60.0))
    f3.run(p3, 0, k_gate)
    f3.hold(max(f3.t, t_replan+15.0))
    f3.run(p3, k_gate, creep=4.0)
    events.append({'time': f3.t, 'follower': 2, 'kind': 'parked', 'detail': '0.07 m'})

    T = int(round(duration/dt))+1
    for tr in tracks:
        tr.hold((T-1)*dt)
    rows = [tr.rows[:T] for tr in tracks]
    states = np.array([[r[0] for r in rs] for rs in rows]).transpose(1, 0, 2)
    time = np.arange(T)*dt
    revealed_at = np.full(len(obs), np.inf)
    revealed_at[rock_a] = t_rev
    # rock_b: revealed when F1's footprint first comes within near_range of it; rock_c is never revealed.
    feet = corners(scenario.truck, states[:, 0])
    near = clearances(feet, np.repeat(obs.corners[rock_b][None], T, 0)) < scenario.sensors.near_range
    if near.any():
        revealed_at[rock_b] = time[int(np.argmax(near))]
    spots = [p1.states[-1], p2.states[-1], p3.states[-1]]
    picks = [{'time': 0.0, 'follower': 0, 'pose': spots[0], 'accepted': True, 'reason': ''},
             {'time': 0.0, 'follower': 1, 'pose': np.array([40.0, 54.0, 0.0]), 'accepted': False,
              'reason': 'overlaps_obstacle'},
             {'time': 0.0, 'follower': 1, 'pose': spots[1], 'accepted': True, 'reason': ''},
             {'time': 0.0, 'follower': 2, 'pose': np.array([31.0, 18.0, np.pi/2]), 'accepted': False,
              'reason': 'unknown_ground'},
             {'time': 0.0, 'follower': 2, 'pose': spots[2], 'accepted': True, 'reason': ''}]
    keep = [e for e in events if e['time'] <= time[-1]+1e-9]
    parked = sum(e['kind'] == 'parked' for e in keep)
    return SimpleNamespace(
        scenario=scenario, seed=seed, n=3, time=time, states=states, steer=np.zeros((T, 3)),
        actions=np.array([[r[1] for r in rs] for rs in rows]).T, safety_stop=np.array([[r[2] for r in rs]
                                                                                        for rs in rows]).T,
        obstacles=obs, missed=missed, revealed_at=revealed_at, leader=world.leader, grid0=grid0, spots=spots,
        picks=picks, paths=[(0.0, 0, p1), (0.0, 1, p2), (0.0, 2, p3)]+([(t_replan, 1, p2b)] if t_replan <= time[-1]
                                                                       else []),
        events=keep, end_reason='parked' if parked == 3 else 'timeout')


def _expected_frames(T, stride):
    return len(range(0, T, stride))+((T-1) % stride != 0)


def test_animate_writes_a_gif_with_one_frame_per_half_second(tmp_path):
    ep = synthetic_episode(12.0)
    out = tmp_path/'yard.gif'
    viz.animate(ep, str(out), fps=10)
    T = len(ep.time)
    with Image.open(out) as im:
        assert im.format == 'GIF'
        assert im.n_frames == _expected_frames(T, 5)   # dt 0.1 s: stride 5
        im.seek(im.n_frames-1)
        assert im.info['duration'] == 100*(1+round(viz.END_HOLD*10))   # the end pause merges into the last frame


def test_animate_with_a_stride_keeps_the_last_sample(tmp_path):
    ep = synthetic_episode(6.0)
    out = tmp_path/'yard.gif'
    viz.animate(ep, str(out), fps=8, stride=7)
    with Image.open(out) as im:
        assert im.n_frames == _expected_frames(len(ep.time), 7)


def test_plot_writes_a_png(tmp_path):
    ep = synthetic_episode(100.0)
    assert ep.end_reason == 'parked' and len(ep.paths) == 4 and np.isfinite(ep.revealed_at).sum() == 2
    out = tmp_path/'yard.png'
    viz.plot(ep, str(out))
    with Image.open(out) as im:
        assert im.format == 'PNG' and im.size[0] > 1000 and im.size[1] > 500


def test_viz_imports_without_the_planning_modules():
    code = ('import sys, nohesi.yard.viz; '
            'bad = [m for m in ("sensors", "episode", "env", "hybrid_astar", "mpc", "coordination", "spots") '
            'if "nohesi.yard."+m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], check=True)
