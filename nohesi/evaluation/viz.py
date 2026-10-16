import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from convoy.visualization import COLORS
from ..geometry import body_centre, footprint
from ..road import Road
from ..scoring.score import TERMS


def _cumulative(ep):
    return np.vstack((np.zeros((1, 3)), np.cumsum(sum(ep.score[t] for t in TERMS), axis=0)))


def plot(ep, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    cumulative = _cumulative(ep)
    for i, color in enumerate(COLORS):
        axes[0].plot(ep.time, ep.team[:, i, 3], color=color, label=f'Car {i}')
        axes[1].plot(ep.time, ep.team_d[:, i], color=color)
        axes[2].plot(ep.time, cumulative[:, i], color=color)
    axes[2].plot(ep.time, cumulative.sum(axis=1), color='black', label='Team')
    axes[0].set(ylabel='Speed (m/s)'); axes[0].legend(ncol=3)
    axes[1].set(ylabel='Lateral d (m)')
    axes[2].set(ylabel='Score', xlabel='Time (s)'); axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f'No Hesi {ep.scenario.version} | seed {ep.seed} | {ep.policy} | score {ep.scenario.score.version}')
    fig.savefig(path, dpi=140); plt.close(fig)


def _panel(ax, ep, behind, ahead, label=''):
    """Draw one episode on `ax`; returns update(k) -> artists. k past the episode end shows its last frame."""
    road, cfg = Road(ep.scenario.road), ep.scenario.vehicle
    half = road.width/2
    lines = [ax.axhline(y, color='gray', lw=1 if abs(y) == half else 0.6, ls='-' if abs(y) == half else '--')
             for y in np.linspace(-half, half, road.cfg.lanes+1)]
    team = [ax.add_patch(plt.Polygon(np.zeros((4, 2)), color=c)) for c in COLORS]
    traffic = [ax.add_patch(plt.Polygon(np.zeros((4, 2)), color='#888888')) for _ in range(ep.traffic_s.shape[1])]
    title = ax.set_title('', fontsize=9, loc='left')
    ax.set_ylim(-half-1, half+1); ax.set_aspect('equal'); ax.set_yticks([])
    cumulative = _cumulative(ep)
    # Latest executed plan per vehicle (faint line ahead of the car; dashed when it is a fallback).
    plans = ep.planner_log.plans if ep.planner_log else []
    plan_times = [np.array([p[0] for p in plans if p[1] == i]) for i in range(3)]
    plan_list = [[p for p in plans if p[1] == i] for i in range(3)]
    paths = [ax.plot([], [], color=c, lw=1.2, alpha=0.45, zorder=1)[0] for c in COLORS]
    ticks = [ax.plot([], [], 'o', color=c, ms=2, alpha=0.45, zorder=1)[0] for c in COLORS]
    prefix = f'{label} | ' if label else ''

    def update(k):
        ended = k >= len(ep.time)
        k = min(k, len(ep.time)-1)
        centre = ep.team_s[k].mean()
        ax.set_xlim(centre-behind, centre+ahead)
        for i, poly in enumerate(team):
            poly.set_xy(footprint(ep.team[k, i], cfg))
        # Traffic heading from consecutive frames (nonzero while changing lanes; 0 across a respawn).
        n = min(k+1, len(ep.time)-1); p = n-1 if n > 0 else 0
        same = ep.traffic_id[n] == ep.traffic_id[p]
        yaw = np.where(same, np.arctan2(ep.traffic_d[n]-ep.traffic_d[p], np.maximum(ep.traffic_s[n]-ep.traffic_s[p], 1e-6)), 0)
        for j, poly in enumerate(traffic):
            poly.set_xy(footprint(np.array([ep.traffic_s[k, j], ep.traffic_d[k, j], yaw[j], 0]), cfg))
        for i in range(3):
            n = np.searchsorted(plan_times[i], ep.time[k]+1e-9)-1 if plans else -1
            if n < 0:
                continue
            _, _, source, z = plan_list[i][n]
            # Draw the body-centre path (CG + (l_f - l_r)/2 along the heading).
            x, y = body_centre(z, cfg).T
            paths[i].set_data(x, y); paths[i].set_linestyle('-' if source == 'qp' else '--')
            ticks[i].set_data(x[10::10], y[10::10])   # one dot per planned second
        scores = ' '.join(f'{v:.0f}' for v in cumulative[k])
        combo = f' | combo x{ep.combo_multiplier[k-1]:g}' if ep.combo_multiplier is not None and k else ''
        status = f' | ENDED ({ep.end_reason})' if ended or (k == len(ep.time)-1 and ep.end_reason != 'time') else ''
        title.set_text(f'{prefix}t={ep.time[k]:.1f} s | team {cumulative[k].sum():.0f} | cars {scores}{combo}{status}')
        return team+traffic+[title]+lines+paths+ticks
    return update


def animate(ep, path, behind=60, ahead=140):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    update = _panel(ax, ep, behind, ahead)
    stride = max(1, int(round(0.2/ep.scenario.timing.dt)))
    animation = FuncAnimation(fig, update, frames=range(0, len(ep.time), stride), blit=False)
    animation.save(path, writer=PillowWriter(fps=15)); plt.close(fig)


def animate_compare(episodes, labels, path, behind=60, ahead=140):
    """Stacked, time-synchronised animation of several episodes (same dt); a shorter episode holds its last frame."""
    fig, axes = plt.subplots(len(episodes), 1, figsize=(12, 2.4*len(episodes)), squeeze=False)
    updates = [_panel(ax, ep, behind, ahead, label) for ax, ep, label in zip(axes[:, 0], episodes, labels)]
    fig.tight_layout()
    stride = max(1, int(round(0.2/episodes[0].scenario.timing.dt)))
    frames = range(0, max(len(ep.time) for ep in episodes), stride)
    animation = FuncAnimation(fig, lambda k: sum((u(k) for u in updates), []), frames=frames, blit=False)
    animation.save(path, writer=PillowWriter(fps=15)); plt.close(fig)


def plot_compare(episodes, labels, path):
    """Team score, speeds and lane positions of several episodes, overlaid (one line style per episode)."""
    styles = ('-', '--', ':', '-.')
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for n, (ep, label) in enumerate(zip(episodes, labels)):
        ls = styles[n % len(styles)]
        axes[0].plot(ep.time, _cumulative(ep).sum(axis=1), color='black', ls=ls, label=label)
        for i, color in enumerate(COLORS):
            axes[1].plot(ep.time, ep.team[:, i, 3], color=color, ls=ls, lw=1)
            axes[2].plot(ep.time, ep.team_d[:, i], color=color, ls=ls, lw=1)
    axes[0].set(ylabel='Team score'); axes[0].legend()
    axes[1].set(ylabel='Speed (m/s)')
    axes[2].set(ylabel='Lateral d (m)', xlabel='Time (s)')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle('Comparison: ' + ' vs '.join(labels) + ' (colours = cars 0/1/2)')
    fig.savefig(path, dpi=140); plt.close(fig)
