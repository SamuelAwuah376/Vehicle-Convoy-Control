import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from convoy.visualization import COLORS
from ..geometry import footprint
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


def animate(ep, path, behind=60, ahead=140):
    road, cfg = Road(ep.scenario.road), ep.scenario.vehicle
    fig, ax = plt.subplots(figsize=(12, 2.8))
    half = road.width/2
    lines = [ax.axhline(y, color='gray', lw=1 if abs(y) == half else 0.6, ls='-' if abs(y) == half else '--')
             for y in np.linspace(-half, half, road.cfg.lanes+1)]
    team = [ax.add_patch(plt.Polygon(np.zeros((4, 2)), color=c)) for c in COLORS]
    traffic = [ax.add_patch(plt.Polygon(np.zeros((4, 2)), color='#888888')) for _ in range(ep.traffic_s.shape[1])]
    title = ax.set_title('')
    ax.set_ylim(-half-1, half+1); ax.set_aspect('equal'); ax.set_yticks([])
    cumulative = _cumulative(ep)
    stride = max(1, int(round(0.2/ep.scenario.timing.dt)))
    # Latest executed plan per vehicle (faint line ahead of the car; dashed when it is a fallback).
    plans = ep.planner_log.plans if ep.planner_log else []
    plan_times = [np.array([p[0] for p in plans if p[1] == i]) for i in range(3)]
    plan_list = [[p for p in plans if p[1] == i] for i in range(3)]
    paths = [ax.plot([], [], color=c, lw=1.2, alpha=0.45, zorder=1)[0] for c in COLORS]
    ticks = [ax.plot([], [], 'o', color=c, ms=2, alpha=0.45, zorder=1)[0] for c in COLORS]

    def update(k):
        centre = ep.team_s[k].mean()
        ax.set_xlim(centre-behind, centre+ahead)
        for i, poly in enumerate(team):
            poly.set_xy(footprint(ep.team[k, i], cfg))
        for j, poly in enumerate(traffic):
            poly.set_xy(footprint(np.array([ep.traffic_s[k, j], ep.traffic_d[k, j], 0, 0]), cfg))
        for i in range(3):
            n = np.searchsorted(plan_times[i], ep.time[k]+1e-9)-1 if plans else -1
            if n < 0:
                continue
            _, _, source, z = plan_list[i][n]
            # Draw the body-centre path (rear axle + half wheelbase along the heading).
            x = z[:, 0]+0.5*cfg.wheelbase*np.cos(z[:, 2]); y = z[:, 1]+0.5*cfg.wheelbase*np.sin(z[:, 2])
            paths[i].set_data(x, y); paths[i].set_linestyle('-' if source == 'qp' else '--')
            ticks[i].set_data(x[10::10], y[10::10])   # one dot per planned second
        scores = ' '.join(f'{v:.0f}' for v in cumulative[k])
        combo = f' | combo x{ep.combo_multiplier[k-1]:g}' if ep.combo_multiplier is not None and k else ''
        title.set_text(f't={ep.time[k]:.1f} s | team {cumulative[k].sum():.0f} | cars {scores}{combo}')
        return team+traffic+[title]+lines+paths+ticks

    animation = FuncAnimation(fig, update, frames=range(0, len(ep.time), stride), blit=False)
    animation.save(path, writer=PillowWriter(fps=15)); plt.close(fig)
