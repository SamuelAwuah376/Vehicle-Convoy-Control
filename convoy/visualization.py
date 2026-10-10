import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from .route import Route
from .geometry import footprint


COLORS = ['#087e8b', '#e08e45', '#9255a7']


def draw_route(ax, route, cfg):
    tangent = np.gradient(route.xy, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    for sign in (-1, 1):
        boundary = route.xy+sign*cfg.lane_width/2*normal
        ax.plot(*boundary.T, color='gray', linewidth=0.8)
    ax.plot(*route.xy.T, '--', color='silver', linewidth=0.7)
    ax.set(xlabel='x (m)', ylabel='y (m)'); ax.set_aspect('equal')


def plot(result, path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    draw_route(axes[0], Route(), result.cfg)
    for i, color in enumerate(COLORS):
        label = ['Leader', 'Follower 1', 'Follower 2'][i]
        axes[0].plot(*result.states[:, i, :2].T, color=color, label=label)
        axes[1].plot(result.time, result.states[:, i, 3], color=color, label=label)
        axes[3].plot(result.time, result.lateral[:, i], color=color)
    for i in range(2):
        axes[2].plot(result.time, result.gaps[:, i], color=COLORS[i+1], label=f'Gap {i+1}')
        desired = result.cfg.standstill_gap+result.cfg.headway*result.states[:, i+1, 3]
        axes[2].plot(result.time, desired, '--', color=COLORS[i+1], alpha=0.6)
    axes[0].legend(loc='upper right', ncol=3)
    axes[1].set(ylabel='Speed (m/s)')
    axes[2].set(ylabel='Bumper gap (m)', title='Solid: actual; dashed: desired time-headway gap')
    axes[2].legend()
    axes[3].set(ylabel='Cross-track error (m)', xlabel='Time (s)')
    for ax in axes[1:]:
        ax.grid(alpha=0.2)
    fig.suptitle(f'Three-vehicle convoy | {result.scenario} | {result.controller}')
    fig.savefig(path, dpi=150); plt.close(fig)


def animate(result, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    draw_route(ax, Route(), result.cfg)
    polygons = [plt.Polygon(footprint(result.states[0, i], result.cfg), color=color)
                for i, color in enumerate(COLORS)]
    for poly in polygons:
        ax.add_patch(poly)
    title = ax.set_title('')
    stride = max(1, int(0.3/result.cfg.dt))
    def update(k):
        for i, poly in enumerate(polygons):
            poly.set_xy(footprint(result.states[k, i], result.cfg))
        title.set_text(f'{result.scenario} | t={result.time[k]:.1f} s')
        return polygons+[title]
    animation = FuncAnimation(fig, update, frames=range(0, len(result.time), stride), blit=False)
    animation.save(path, writer=PillowWriter(fps=20)); plt.close(fig)
