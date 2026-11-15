"""Top-down rendering of a yard-parking episode (the YardEpisode record of nohesi.yard.episode): animate() writes a GIF,
plot() one PNG. The record is read by attribute name only, so this module imports none of the planning modules.

Layers, bottom to top: the leader's lidar scan (unknown light grey, free near-white, occupied dark); berms and
obstacles by kind (rock piles brown, parked equipment ochre, windrows tan), an obstacle the scan missed drawn as a
dashed outline until a follower reveals it (then filled and flashed red for REVEAL_FLASH s); rejected driver picks (red
X, reason); assigned spots (outlined footprints in the follower's colour, tagged F1/F2/F3); the planned paths (rear
axle, forward segments solid, reverse dashed, cusps as dots); the leader and the followers (a dark cab box with a white
heading arrow at the front end, a motion arrow at the bumper the truck drives toward, a red outline while the safety
supervisor brakes it)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import ListedColormap, to_hex, to_rgb, to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
from PIL import Image

ACTIONS = ('HOLD', 'CREEP', 'GO', 'BACK_OFF', 'REPLAN')   # mirrors nohesi.yard.episode.ACTIONS
COLORS = ('#2a78d6', '#d55181', '#008300')   # followers F1-F3: blue, magenta, green (colour-blind safe as a set)
LEADER = '#3d4250'
SCAN = ('#d4d4d1', '#fbfbf8', '#3a3a3a')     # UNKNOWN, FREE, OCCUPIED
KINDS = {'berm': ('#6e6e6e', '#474747'), 'rock': ('#8b5a2b', '#4a2c10'),   # kind: (fill, edge)
         'equipment': ('#c39a14', '#6b5200'), 'windrow': ('#d2b48c', '#8a6d45')}
KIND_LABELS = {'berm': 'berm', 'rock': 'rock pile', 'equipment': 'parked equipment', 'windrow': 'windrow'}
ALERT = '#e0161b'                            # safety stop, rejected pick, reveal flash, collision
ALERT_INK = '#9b1116'                        # ... as text
INK, INK_2 = '#1b1b1b', '#52514e'
END_COLORS = {'parked': '#1a7f37', 'collision': ALERT, 'timeout': '#a35f00', 'stalled': '#a35f00'}
REVEAL_FLASH = 4.0      # s an obstacle stays highlighted after a follower reveals it
EVENT_WINDOW = 4.0      # s an event stays in the animation's event line
FRAME_DT = 0.5          # s of simulation per GIF frame (default stride)
END_HOLD = 2.0          # s the GIF pauses on its last frame
GIF_DPI, PNG_DPI = 110, 150
PATH_OFFSET = 0.55     # m: followers' drawn paths are shifted sideways by this much each, so shared routes stay visible
REVERSE = (0, (4.0, 2.5))   # dash pattern of reverse segments


# ---------------------------------------------------------------------------------------------------------- geometry
def _local(state, pts):
    """Truck-frame points (m, 2) (rear axle origin, +x forward) placed at a rear-axle state [x, y, yaw, ...]."""
    x, y, yaw = (float(v) for v in state[:3])
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.asarray(pts, float)
    return np.column_stack((x+pts[:, 0]*c-pts[:, 1]*s, y+pts[:, 0]*s+pts[:, 1]*c))


def _shapes(truck):
    """Truck-frame outlines: body (= truck.corners), cab box at the front, heading arrow in the cab, motion arrows
    ahead of the front and behind the rear bumper, and a halo 0.5 m outside the body."""
    r, f, h = -truck.rear_overhang, truck.length-truck.rear_overhang, truck.width/2
    cab = min(2.6, 0.3*truck.length)
    return {'body': [[r, -h], [f, -h], [f, h], [r, h]],
            'cab': [[f-cab, -h], [f, -h], [f, h], [f-cab, h]],
            'heading': [[f-0.35, 0], [f-cab+0.45, -0.62*h], [f-cab+0.45, 0.62*h]],
            'ahead': [[f+0.5, -0.65*h], [f+2.4, 0], [f+0.5, 0.65*h]],
            'behind': [[r-0.5, -0.65*h], [r-2.4, 0], [r-0.5, 0.65*h]],
            'halo': [[r-0.5, -h-0.5], [f+0.5, -h-0.5], [f+0.5, h+0.5], [r-0.5, h+0.5]],
            'mid': [[(r+f)/2, 0]]}


def _mix(color, other, w):
    a, b = np.array(to_rgb(color)), np.array(to_rgb(other))
    return to_hex((1-w)*a+w*b)


def _f(i):
    return f'F{i+1}'


def _color(i):
    return COLORS[i % len(COLORS)]


def _view(ep):
    """(x0, x1, y0, y1): the scan grid, the obstacles (berms) and every truck position, plus a 1 m margin."""
    pts = [np.asarray(ep.obstacles.corners, float).reshape(-1, 2), np.asarray(ep.leader, float)[None, :2],
           np.asarray(ep.states, float)[..., :2].reshape(-1, 2)]
    lo, hi = np.nanmin(np.vstack(pts), 0)-3.0, np.nanmax(np.vstack(pts), 0)+3.0
    grid = getattr(ep, 'grid0', None)
    if grid is not None:
        ny, nx = grid.cells.shape
        lo = np.minimum(lo, (grid.x0, grid.y0))
        hi = np.maximum(hi, (grid.x0+nx*grid.res, grid.y0+ny*grid.res))
    return lo[0]-1.0, hi[0]+1.0, lo[1]-1.0, hi[1]+1.0


def _offset(i, n):
    """Sideways shift (m, to the truck's left) of follower i's drawn path: overlapping routes become parallel lines."""
    return (i-(n-1)/2)*PATH_OFFSET


def _segments(path, offset=0.0):
    """Forward and reverse polylines of a Path ((m, 2) each, NaN rows between segments) and its cusp points, shifted
    `offset` m to the truck's left."""
    yaw = np.asarray(path.yaw, float)
    x, y = path.x-offset*np.sin(yaw), path.y+offset*np.cos(yaw)
    fwd, rev = [], []
    for a, b in path.segments():
        xy = np.column_stack((x[a:b+1], y[a:b+1]))
        (fwd if path.direction[a] > 0 else rev).append(np.vstack((xy, [[np.nan, np.nan]])))
    empty = np.zeros((0, 2))
    cusps = np.asarray(path.cusps, int)
    return (np.vstack(fwd) if fwd else empty, np.vstack(rev) if rev else empty,
            np.column_stack((x[cusps], y[cusps])) if len(cusps) else empty)


def _grow(c, m):
    """Rectangle corners (4, 2), counter-clockwise, pushed `m` outward on every side."""
    u = (c[1]-c[0])/max(np.linalg.norm(c[1]-c[0]), 1e-9)
    v = (c[3]-c[0])/max(np.linalg.norm(c[3]-c[0]), 1e-9)
    return c+m*np.array([-u-v, u-v, u+v, v-u])


def _top_label_at(foot):
    """Anchor (x, y) just above a footprint (4, 2)."""
    return float(foot[:, 0].mean()), float(foot[:, 1].max())+0.5


# --------------------------------------------------------------------------------------------------------- the figure
def _figure(view, width, top, bottom, dpi, left=0.42, right=0.14):
    """Figure sized so the equal-aspect map fills its axes exactly; `top` / `bottom` inches are left for the HUD and
    the legend."""
    x0, x1, y0, y1 = view
    aw = width-left-right
    ah = aw*(y1-y0)/(x1-x0)
    height = ah+top+bottom
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor='white')
    ax = fig.add_axes((left/width, bottom/height, aw/width, ah/height))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor(SCAN[0])
    ax.tick_params(labelsize=7, length=2, pad=1.5, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_color('#9a9a96'); spine.set_linewidth(0.6)
    return fig, ax


def _draw_scan(ax, grid):
    """The scan under everything, and its OCCUPIED cells (the lidar hits) again above the obstacle fills: the faces the
    leader saw get a dark rim, an obstacle it missed has none."""
    if grid is None:
        return
    cells = np.asarray(grid.cells)
    ny, nx = cells.shape
    extent = (grid.x0, grid.x0+nx*grid.res, grid.y0, grid.y0+ny*grid.res)
    ax.imshow(cells, origin='lower', cmap=ListedColormap(SCAN), vmin=0, vmax=2, interpolation='nearest',
              extent=extent, zorder=0)
    hits = np.zeros((ny, nx, 4))
    hits[cells == 2] = to_rgba(SCAN[2], 0.9)
    ax.imshow(hits, origin='lower', interpolation='nearest', extent=extent, zorder=1.6)


def _obstacle_style(poly, kind, hidden):
    """A missed obstacle not yet revealed: dashed outline only; otherwise filled."""
    fill, edge = KINDS.get(kind, ('#8a8a8a', '#4a4a4a'))
    poly.set_facecolor('none' if hidden else fill)
    poly.set_edgecolor(edge)
    poly.set_linestyle((0, (3.0, 2.0)) if hidden else '-')
    poly.set_linewidth(1.3 if hidden else 0.7)


def _draw_obstacles(ax, ep, t=None):
    """One polygon per obstacle; styled for time t (None: the episode's end)."""
    obs = ep.obstacles
    missed, revealed_at = _missed(ep)
    polys = []
    for k, (c, kind) in enumerate(zip(np.asarray(obs.corners, float), obs.kind)):
        poly = ax.add_patch(Polygon(c, closed=True, zorder=1.2 if kind == 'berm' else 1.4))
        hidden = missed[k] and (t is None and not np.isfinite(revealed_at[k]) or t is not None and t < revealed_at[k])
        _obstacle_style(poly, kind, hidden)
        polys.append(poly)
    return polys


def _missed(ep):
    k = len(ep.obstacles.kind)
    missed = getattr(ep, 'missed', None)
    revealed_at = getattr(ep, 'revealed_at', None)
    missed = np.zeros(k, bool) if missed is None else np.asarray(missed, bool)
    revealed_at = np.full(k, np.inf) if revealed_at is None else np.asarray(revealed_at, float)
    return missed, revealed_at


def _truck_artists(ax, color, label, z, label_size=7.0):
    cab = _mix(color, '#000', 0.45)
    art = {'body': ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, facecolor=color, edgecolor=cab, lw=0.8,
                                        zorder=z)),
           'cab': ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, facecolor=cab, edgecolor='none', zorder=z+0.1)),
           'heading': ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, facecolor='white', edgecolor='none',
                                           zorder=z+0.2)),
           'motion': ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, facecolor=color, edgecolor=cab, lw=0.6,
                                          zorder=z+0.2, visible=False)),
           'stop': ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor=ALERT, lw=2.2,
                                        zorder=z+0.3, visible=False)),
           'label': ax.text(0, 0, label, color='white', fontsize=label_size, fontweight='bold', ha='center',
                            va='center', zorder=z+0.4)}
    return art


def _place_truck(art, shapes, state, stop=False, rotate_label=False):
    for key in ('body', 'cab', 'heading'):
        art[key].set_xy(_local(state, shapes[key]))
    v = float(state[3]) if len(state) > 3 else 0.0
    moving = abs(v) > 0.05
    art['motion'].set_visible(moving)
    if moving:
        art['motion'].set_xy(_local(state, shapes['ahead' if v > 0 else 'behind']))
    art['stop'].set_visible(bool(stop))
    if stop:
        art['stop'].set_xy(_local(state, shapes['halo']))
    art['label'].set_position(_local(state, shapes['mid'])[0])
    if rotate_label:
        deg = np.degrees(float(state[2]))
        art['label'].set_rotation((deg+90) % 180-90)
        art['label'].set_rotation_mode('anchor')


def _draw_leader(ax, ep, shapes):
    art = _truck_artists(ax, LEADER, 'Leader', 5.0, label_size=6.5)
    _place_truck(art, shapes, np.r_[np.asarray(ep.leader, float)[:3], 0.0], rotate_label=True)
    return art


def _draw_rejected(ax, truck, pick, alpha=1.0):
    """A rejected pick: red X at the would-be body centre, its footprint dotted, the reason beside it."""
    from .truck import body_centre, corners
    pose = np.asarray(pick['pose'], float)[:3]
    foot = corners(truck, pose[None])[0]
    cx, cy = body_centre(truck, pose[None])[0]
    outline = ax.add_patch(Polygon(foot, closed=True, fill=False, edgecolor=ALERT, lw=0.8, ls=(0, (1.5, 1.5)),
                                   alpha=0.7*alpha, zorder=1.8))
    mark, = ax.plot([cx], [cy], 'x', color=ALERT, ms=7, mew=2.0, alpha=alpha, zorder=6.5)
    reason = str(pick.get('reason', '') or 'rejected').replace('_', ' ')
    text = ax.text(cx+1.3, cy, f"{_f(int(pick['follower']))} rejected: {reason}", fontsize=6.5, color=ALERT_INK,
                   ha='left', va='center', alpha=alpha, zorder=6.6,
                   bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.75*alpha))
    return [outline, mark, text]


def _spot_artists(ax, i):
    color = _color(i)
    outline = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, facecolor=to_rgba(color, 0.10), edgecolor=color,
                                   lw=1.4, zorder=2.0, visible=False))
    tag = ax.text(0, 0, _f(i), fontsize=6.5, fontweight='bold', color='white', ha='center', va='bottom', zorder=6.0,
                  visible=False, bbox=dict(boxstyle='round,pad=0.15', facecolor=color, edgecolor='none'))
    return outline, tag


def _place_spot(truck, outline, tag, pose):
    from .truck import corners
    visible = pose is not None
    outline.set_visible(visible); tag.set_visible(visible)
    if visible:
        foot = corners(truck, np.asarray(pose, float)[None, :3])[0]
        outline.set_xy(foot)
        tag.set_position(_top_label_at(foot))


def _legend(fig, extra=(), ncol=None):
    handles = [(Patch(facecolor=SCAN[1], edgecolor='#a0a09c', lw=0.5), Patch(facecolor=SCAN[0], edgecolor='#a0a09c',
                                                                             lw=0.5), Patch(facecolor=SCAN[2])),
               *(Patch(facecolor=KINDS[k][0], edgecolor=KINDS[k][1], lw=0.7) for k in ('berm', 'rock', 'equipment',
                                                                                         'windrow')),
               Patch(fill=False, edgecolor=KINDS['rock'][1], lw=1.2, ls=(0, (3.0, 2.0))),
               Line2D([], [], color=INK_2, lw=1.6), Line2D([], [], color=INK_2, lw=1.6, ls=REVERSE),
               Patch(facecolor=to_rgba(INK_2, 0.1), edgecolor=INK_2, lw=1.2),
               Line2D([], [], ls='none', marker='x', color=ALERT, ms=6, mew=2),
               Patch(fill=False, edgecolor=ALERT, lw=2.0)]
    labels = ['scan: free / unknown / occupied', *(KIND_LABELS[k] for k in ('berm', 'rock', 'equipment', 'windrow')),
              'missed by the scan', 'forward path', 'reverse path', 'assigned spot', 'rejected pick', 'safety stop']
    for h, lab in extra:
        handles.append(h); labels.append(lab)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=ncol or (len(labels)+1)//2,
               fontsize=7, frameon=False, handlelength=2.0, handleheight=0.9, columnspacing=1.3, borderaxespad=0.25,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})


def _follower_hud(fig, n, y, size):
    """Per follower: a coloured square (identity) and a monospaced status text in ink."""
    out = []
    for i in range(n):
        x = 0.012+i*0.33
        fig.text(x, y, '■', color=_color(i), fontsize=size+2, va='top', ha='left')
        out.append(fig.text(x+0.017, y-0.002, '', color=INK, fontsize=size, va='top', ha='left', family='monospace'))
    return out


def _episode_title(ep):
    n = int(ep.n)
    return f"Yard parking · {getattr(ep.scenario, 'version', '')} · seed {ep.seed} · " \
           f"{n} follower{'s' if n != 1 else ''}"


def _parked_at(ep):
    """Per follower: time of its 'parked' event (inf: never)."""
    out = np.full(int(ep.n), np.inf)
    for e in ep.events:
        if e.get('kind') == 'parked' and 0 <= int(e['follower']) < len(out):
            out[int(e['follower'])] = min(out[int(e['follower'])], float(e['time']))
    return out


def _event_text(e):
    detail = str(e.get('detail', '') or '')
    who = _f(int(e['follower'])) if e.get('follower') is not None and int(e['follower']) >= 0 else ''
    return f"{float(e['time']):.1f} s {who} {str(e.get('kind', '')).replace('_', ' ')}" + \
        (f': {detail[:48]}' if detail else '')


def _collision_points(ep, truck):
    """Body centre of each collision event's follower at the event time: [(time, x, y)]."""
    from .truck import body_centre
    out = []
    time = np.asarray(ep.time, float)
    for e in ep.events:
        if e.get('kind') == 'collision':
            k = int(np.clip(np.searchsorted(time, float(e['time'])), 0, len(time)-1))
            i = int(e['follower'])
            out.append((float(e['time']), *body_centre(truck, ep.states[k, i][None])[0]))
    return out


# ------------------------------------------------------------------------------------------------------------ the GIF
class _GifWriter(PillowWriter):
    """PillowWriter that maps every frame onto one shared palette (built from the first frame plus the colours that
    appear later) and saves with optimize=True, so the pixels a frame does not change are written transparent: the
    static background is stored once and a 120 s episode stays at a few MB. Identical frames merge into one longer
    frame (Pillow), which is how the end pause is made."""
    def __init__(self, fps, extra_colors=()):
        super().__init__(fps=fps)
        self._extra = [tuple(int(round(255*c)) for c in to_rgb(col)) for col in extra_colors]
        self._palette = None

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        im = self._frames[-1].convert('RGB')
        if self._palette is None:
            # <= 254 entries: a free index is left for the transparency of unchanged pixels.
            base = im.quantize(colors=254-len(self._extra), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            pal = base.getpalette()
            colors = [tuple(pal[3*j:3*j+3]) for _, j in sorted(base.getcolors(256), key=lambda c: c[1])]+self._extra
            self._palette = Image.new('P', (1, 1))
            self._palette.putpalette([v for c in colors for v in c])
        self._frames[-1] = im.quantize(palette=self._palette, dither=Image.Dither.NONE)

    def finish(self):
        self._frames[0].save(self.outfile, save_all=True, append_images=self._frames[1:],
                             duration=int(round(1000/self.fps)), loop=0, optimize=True)


def _frame_indices(ep, stride=None):
    """Record indices animate() draws: every `stride`-th sample (default FRAME_DT of simulation) and the last one."""
    time = np.asarray(ep.time, float)
    if stride is None:
        dt = float(np.median(np.diff(time))) if len(time) > 1 else float(ep.scenario.planning.dt)
        stride = max(1, int(round(FRAME_DT/max(dt, 1e-9))))
    frames = list(range(0, len(time), int(stride)))
    if frames[-1] != len(time)-1:
        frames.append(len(time)-1)
    return frames


def animate(ep, path, fps=10, stride=None):
    """Top-down GIF of a YardEpisode, one frame per `stride` record samples (default: about FRAME_DT s of
    simulation) plus the last sample, which carries the end reason and is held for END_HOLD s."""
    truck = ep.scenario.truck
    shapes = _shapes(truck)
    n, time = int(ep.n), np.asarray(ep.time, float)
    states = np.asarray(ep.states, float)
    view = _view(ep)
    fig, ax = _figure(view, width=12.0, top=0.78, bottom=0.5, dpi=GIF_DPI)
    _draw_scan(ax, ep.grid0)
    polys = _draw_obstacles(ax, ep, t=time[0])
    missed, revealed_at = _missed(ep)
    flash = {}
    for k in np.flatnonzero(missed & np.isfinite(revealed_at)):
        c = np.asarray(ep.obstacles.corners[k], float)
        centre, grown = c.mean(0), _grow(c, 1.0)
        flash[k] = (ax.add_patch(Polygon(grown, closed=True, fill=False, edgecolor=ALERT, lw=2.6, zorder=4.5,
                                         visible=False)),
                    ax.text(centre[0], grown[:, 1].max()+0.4, 'revealed', fontsize=7, fontweight='bold',
                            color=ALERT_INK, ha='center', va='bottom', zorder=6.8, visible=False,
                            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.8)))
    _draw_leader(ax, ep, shapes)

    # Rejected picks, shown from their pick time on.
    picks = [p for p in ep.picks if p.get('pose') is not None]
    rejected = [(float(p['time']), _draw_rejected(ax, truck, p)) for p in picks if not p.get('accepted', False)]
    # Spots: the latest accepted pick at or before t; ep.spots[i] when the record has no accepted pick for i.
    accepted = [[(float(p['time']), np.asarray(p['pose'], float)) for p in picks
                 if p.get('accepted', False) and int(p['follower']) == i] for i in range(n)]
    spot_art = [_spot_artists(ax, i) for i in range(n)]
    # Paths: per follower the latest plan at or before t.
    plans = [[(float(tp), _segments(p, _offset(i, n))) for tp, j, p in ep.paths if int(j) == i] for i in range(n)]
    plan_times = [np.array([tp for tp, _ in plans[i]]) for i in range(n)]
    lines = [(ax.plot([], [], color=_color(i), lw=1.7, zorder=3.0, solid_capstyle='round')[0],
              ax.plot([], [], color=_color(i), lw=1.7, ls=REVERSE, zorder=3.0)[0],
              ax.plot([], [], ls='none', marker='o', ms=3.6, mfc='white', mec=_color(i), mew=1.2, zorder=3.1)[0])
             for i in range(n)]
    trucks = [_truck_artists(ax, _color(i), _f(i), 5.0) for i in range(n)]
    crashes = [(tc, ax.plot([x], [y], marker='*', ms=15, color=ALERT, mec='white', mew=0.8, zorder=7.0,
                            visible=False)[0]) for tc, x, y in _collision_points(ep, truck)]

    # HUD: title and end reason, one status per follower, the latest events.
    top = fig.get_figheight()
    y1, y2, y3 = 1-0.09/top, 1-0.36/top, 1-0.60/top
    head = fig.text(0.012, y1, '', fontsize=10.5, fontweight='bold', color=INK, va='top')
    end = fig.text(0.988, y1, '', fontsize=10.5, fontweight='bold', ha='right', va='top')
    status = _follower_hud(fig, n, y2, 9.0)
    ticker = fig.text(0.012, y3, '', fontsize=8, color=INK_2, va='top')
    _legend(fig, ncol=11)
    title = _episode_title(ep)
    parked_at = _parked_at(ep)
    events = sorted(ep.events, key=lambda e: float(e['time']))
    actions = np.asarray(ep.actions)
    safety = np.asarray(ep.safety_stop, bool)
    last = len(time)-1

    def update(k):
        t = time[k]
        for j, poly in enumerate(polys):
            if missed[j]:
                _obstacle_style(poly, ep.obstacles.kind[j], t < revealed_at[j])
        for j, (halo, tag) in flash.items():
            age = t-revealed_at[j]
            on = 0 <= age <= REVEAL_FLASH
            halo.set_visible(on); tag.set_visible(on)
            if on:
                halo.set_alpha(1.0-0.7*age/REVEAL_FLASH)
        for tp, arts in rejected:
            for a in arts:
                a.set_visible(t >= tp-1e-9)
        for i in range(n):
            pose = None
            if accepted[i]:
                done = [p for tp, p in accepted[i] if tp <= t+1e-9]
                pose = done[-1] if done else None
            elif i < len(ep.spots) and ep.spots[i] is not None:
                pose = ep.spots[i]
            _place_spot(truck, *spot_art[i], pose)
            m = int(np.searchsorted(plan_times[i], t+1e-9))-1
            fwd, rev, cusp = plans[i][m][1] if m >= 0 else (np.zeros((0, 2)),)*3
            for line, xy in zip(lines[i], (fwd, rev, cusp)):
                line.set_data(xy[:, 0], xy[:, 1])
            _place_truck(trucks[i], shapes, states[k, i], stop=safety[k, i])
            a = int(actions[k, i])
            name = ACTIONS[a] if 0 <= a < len(ACTIONS) else str(a)
            note = '  SAFETY STOP' if safety[k, i] else ('  parked' if parked_at[i] <= t+1e-9 else '')
            status[i].set_text(f'{_f(i)} {name:<8} {states[k, i, 3]:+4.1f} m/s{note}')
            status[i].set_color(ALERT_INK if safety[k, i] else INK)
        for tc, mark in crashes:
            mark.set_visible(t >= tc-1e-9)
        head.set_text(f'{title} · t = {t:5.1f} s')
        if k == last:
            reason = str(ep.end_reason)
            end.set_text(f'END: {reason}'); end.set_color(END_COLORS.get(reason, INK))
        else:
            end.set_text('')
        recent = [e for e in events if t-EVENT_WINDOW < float(e['time']) <= t+1e-9]
        ticker.set_text('   ·   '.join(_event_text(e) for e in recent[-3:]))

    frames = _frame_indices(ep, stride)
    frames += [frames[-1]]*int(round(END_HOLD*fps))
    extra = [*COLORS, *(_mix(c, '#000', 0.45) for c in COLORS), *(_mix(c, '#fff', 0.9) for c in COLORS), LEADER,
             _mix(LEADER, '#000', 0.45), ALERT, ALERT_INK, INK, INK_2, *END_COLORS.values(),
             *(c for pair in KINDS.values() for c in pair), *SCAN, '#ffffff']
    writer = _GifWriter(fps, extra)
    with writer.saving(fig, path, GIF_DPI):
        for k in frames:
            update(k)
            writer.grab_frame()
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ the PNG
def plot(ep, path):
    """One PNG of the whole episode: the scan, the obstacles (missed ones dashed, tagged with their reveal time), the
    spots and rejected picks, every planned path (superseded ones faint), the driven rear-axle tracks, the trucks at
    their final states (start positions outlined)."""
    from .truck import corners
    truck = ep.scenario.truck
    shapes = _shapes(truck)
    n, time = int(ep.n), np.asarray(ep.time, float)
    states = np.asarray(ep.states, float)
    fig, ax = _figure(_view(ep), width=13.0, top=0.85, bottom=0.78, dpi=PNG_DPI, left=0.6)
    ax.set_xlabel('x (m)', fontsize=7.5, color=INK_2, labelpad=1)
    ax.set_ylabel('y (m)', fontsize=7.5, color=INK_2, labelpad=1)
    _draw_scan(ax, ep.grid0)
    _draw_obstacles(ax, ep)   # styled for the episode's end: missed and never revealed stay dashed
    missed, revealed_at = _missed(ep)
    for k in np.flatnonzero(missed):
        c = np.asarray(ep.obstacles.corners[k], float)
        note = f'missed · revealed {revealed_at[k]:.0f} s' if np.isfinite(revealed_at[k]) else \
            'missed · never revealed'
        ax.text(c[:, 0].mean(), c[:, 1].max()+0.4, note, fontsize=6, color=INK, ha='center', va='bottom', zorder=6.6,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.75))
    _draw_leader(ax, ep, shapes)
    picks = [p for p in ep.picks if p.get('pose') is not None]
    for p in picks:
        if not p.get('accepted', False):
            _draw_rejected(ax, truck, p)
    for i in range(n):
        color = _color(i)
        final = ep.spots[i] if i < len(ep.spots) else None
        for p in picks:   # accepted picks superseded by a reassignment: faint dotted outlines
            if p.get('accepted', False) and int(p['follower']) == i and \
                    (final is None or not np.allclose(np.asarray(p['pose'], float)[:3], np.asarray(final, float)[:3])):
                ax.add_patch(Polygon(corners(truck, np.asarray(p['pose'], float)[None, :3])[0], closed=True,
                                     fill=False, edgecolor=color, lw=0.9, ls=(0, (1.5, 1.5)), alpha=0.6, zorder=1.9))
        if final is not None:
            _place_spot(truck, *_spot_artists(ax, i), final)
        mine = [p for _, j, p in ep.paths if int(j) == i]
        for m, p in enumerate(mine):
            fwd, rev, cusp = _segments(p, _offset(i, n))
            current = m == len(mine)-1
            kw = dict(color=color, lw=1.5 if current else 0.9, alpha=1.0 if current else 0.35, zorder=3.0 if current
                      else 2.6)
            ax.plot(fwd[:, 0], fwd[:, 1], solid_capstyle='round', **kw)
            ax.plot(rev[:, 0], rev[:, 1], ls=REVERSE, **kw)
            if current and len(cusp):
                ax.plot(cusp[:, 0], cusp[:, 1], ls='none', marker='o', ms=3.6, mfc='white', mec=color, mew=1.2,
                        zorder=3.1)
        ax.plot(states[:, i, 0], states[:, i, 1], color=color, lw=4.5, alpha=0.22, solid_capstyle='round', zorder=2.4)
        ax.add_patch(Polygon(corners(truck, states[0, i][None])[0], closed=True, fill=False, edgecolor=color, lw=0.9,
                             ls=(0, (2.0, 1.5)), zorder=4.0))
        art = _truck_artists(ax, color, _f(i), 5.0)
        _place_truck(art, shapes, np.r_[states[-1, i, :3], 0.0], stop=bool(np.asarray(ep.safety_stop)[-1, i]))
    for _, x, y in _collision_points(ep, truck):
        ax.plot([x], [y], marker='*', ms=15, color=ALERT, mec='white', mew=0.8, zorder=7.0)

    # Header: title, outcome and counts, one summary per follower.
    top = fig.get_figheight()
    fig.text(0.012, 1-0.08/top, _episode_title(ep), fontsize=11, fontweight='bold', color=INK, va='top')
    reason = str(ep.end_reason)
    kinds = [str(e.get('kind')) for e in ep.events]
    fig.text(0.988, 1-0.08/top, f'END: {reason} at {time[-1]:.1f} s', fontsize=11, fontweight='bold', ha='right',
             va='top', color=END_COLORS.get(reason, INK))
    counts = f"picks rejected {sum(not p.get('accepted', False) for p in ep.picks)} · plans {len(ep.paths)} " \
             f"· replans {kinds.count('replan')} · safety stops {kinds.count('safety_stop')} " \
             f"· obstacles missed {int(missed.sum())} (revealed {int(np.isfinite(revealed_at[missed]).sum())})"
    fig.text(0.012, 1-0.62/top, counts, fontsize=8, color=INK_2, va='top')
    parked_at = _parked_at(ep)
    for i, text in enumerate(_follower_hud(fig, n, 1-0.36/top, 8.5)):
        driven = float(np.sum(np.hypot(*np.diff(states[:, i, :2], axis=0).T)))
        fate = f'parked {parked_at[i]:.1f} s' if np.isfinite(parked_at[i]) else 'not parked'
        text.set_text(f'{_f(i)} {fate} · driven {driven:.0f} m')
    track = (Line2D([], [], color=INK_2, lw=4.5, alpha=0.22, solid_capstyle='round'), 'driven track')
    faint = (Line2D([], [], color=INK_2, lw=0.9, alpha=0.35), 'superseded path')
    start = (Patch(fill=False, edgecolor=INK_2, lw=0.9, ls=(0, (2.0, 1.5))), 'start')
    _legend(fig, extra=(track, faint, start), ncol=7)
    fig.savefig(path, dpi=PNG_DPI, facecolor='white')
    plt.close(fig)
