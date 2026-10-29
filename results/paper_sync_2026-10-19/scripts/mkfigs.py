import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nohesi.geometry import body_centre, footprint
from nohesi.road import Road
from nohesi.evaluation.viz import _cumulative, COLORS
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times'], 'pdf.use14corefonts': True, 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 3, 'pdf.compression': 9})
o, l = pickle.load(open('/tmp/ep0.pkl', 'rb'))
import sys
T = float(sys.argv[1]) if len(sys.argv) > 1 else 36.0
k = int(round(T/o.scenario.timing.dt)); cfg = o.scenario.vehicle; road = Road(o.scenario.road); half = road.width/2
print('s', o.team_s[k], 'd', o.team_d[k])
lo, hi = float(sys.argv[2]), float(sys.argv[3])
fig, ax = plt.subplots(figsize=(7.16, 1.55), constrained_layout=True)
for y in np.linspace(-half, half, road.cfg.lanes+1):
    ax.axhline(y, color='0.4', lw=0.9 if abs(y) == half else 0.5, ls='-' if abs(y) == half else (0, (6, 4)), zorder=0)
for j in range(o.traffic_s.shape[1]):
    s, d, v = o.traffic_s[k, j], o.traffic_d[k, j], o.traffic_v[k, j]
    if lo-5 < s < hi+5:
        ax.add_patch(plt.Polygon(footprint(np.array([s, d, 0, 0]), cfg), color='0.65', zorder=2))
        ax.text(s, d, f'{v:.0f}', ha='center', va='center', fontsize=6, zorder=3)
plans = o.planner_log.plans
for i, c in enumerate(COLORS):
    ax.add_patch(plt.Polygon(footprint(o.team[k, i], cfg), color=c, zorder=4))
    bc = body_centre(o.team[k, i], cfg)
    ax.text(bc[0], bc[1], str(i), ha='center', va='center', fontsize=6, color='white', zorder=5, fontweight='bold')
    mine = [p for p in plans if p[1] == i and p[0] <= o.time[k]+1e-9]
    if mine:
        z = mine[-1][3]; x, y = body_centre(z, cfg).T
        ax.plot(x, y, color=c, lw=1.0, zorder=3, ls='-' if mine[-1][2] == 'qp' else '--')
        ax.plot(x[5::5], y[5::5], 'o', color=c, ms=2, zorder=3)
ax.set_xlim(lo, hi); ax.set_ylim(-half-0.6, half+0.6)
ax.set_xlabel('s (m)', style='italic'); ax.set_ylabel('d (m)'); ax.set_yticks([-3.6, 0, 3.6]); ax.set_yticklabels(['-3.6', '0', '3.6'])
fig.savefig('/tmp/figs/snapshot.pdf'); fig.savefig('/tmp/figs/snapshot.png', dpi=200); plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.6), sharex=True, constrained_layout=True,
                         gridspec_kw={'height_ratios': [1.4, 0.8, 1.1]})
for y in np.linspace(-half, half, road.cfg.lanes+1)[1:-1]:
    axes[0].axhline(y, color='0.5', lw=0.5, ls='--')
for i, c in enumerate(COLORS):
    axes[0].plot(o.time[::4], o.team_d[::4, i], color=c, lw=0.8, label=f'Car {i}')
axes[0].set_ylim(-half, half); axes[0].set_yticks([-3.6, 0, 3.6]); axes[0].set_yticklabels(['-3.6', '0', '3.6']); axes[0].set_ylabel('d (m)')
axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.32), frameon=False, handlelength=1.5)
cm=o.combo_multiplier; tt=o.time[1:]; import numpy as _n; ch=_n.r_[0, _n.flatnonzero(_n.diff(cm))+1, len(cm)-1]; axes[1].step(tt[ch], cm[ch], where='post', color='k', lw=0.8); axes[1].set_ylabel('Combo'); axes[1].set_ylim(0.8, 5.3)
axes[2].plot(o.time[::4], _cumulative(o).sum(1)[::4], color='k', lw=0.9, label='Overtaker + MPC')
axes[2].plot(l.time[::20], _cumulative(l).sum(1)[::20], color='0.55', lw=0.9, ls='--', label='Lane keeping')
axes[2].set_ylabel('Team score'); axes[2].set_xlabel('Time (s)'); axes[2].legend(frameon=False, loc='upper left')
axes[2].set_xlim(0, 120)
for a in axes: a.grid(alpha=0.2, lw=0.4)
fig.savefig('/tmp/figs/episode_seed0.pdf'); fig.savefig('/tmp/figs/episode_seed0.png', dpi=200); plt.close(fig)
