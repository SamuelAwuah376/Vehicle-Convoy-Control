"""Haul-road obstacle test (the user's call, 2026-11-15): the current stack and policy with the traffic cars replaced by
static obstacles, as a first read on whether it is a basis for autonomous heavy-truck convoys on mining / logging roads.

Stack as run_sim.py runs it: the milestone 8 PPO policy (models/ppo_v10_e5_dyn_b11c_2.75m, final), its env config
with run_sim.py's planner overrides, planner m4.4, the nonlinear MPC (acados) on the dynamic plant. Baseline: the
scripted overtaker on the same planner. Scenarios (nohesi.config):
    h1  v9's road and cars at the policy's trained speeds (70 mph start, up to 134 mph), 12 static obstacles in place
        of the 45 traffic cars
    h2  h1 in a mining envelope: 45 km/h start, 5-60 km/h commanded, gravel (mu 0.6)
Policies: 'ppo', 'overtaker', and 'ppo_edge' = ppo with the planner's corner-aware road-edge bound and edge grip
(Planner.nmpc_edge_yaw, escape_edge_grip; both off by default), which hold the body corners, not only the CG, inside
the edge at the plan's heading (the low-speed lane changes obstacles force run at 0.2-0.3 rad).
Writes results/haul_obstacles_<date>/: episodes.jsonl (one row per episode, appended as each finishes; a rerun skips
the jobs already in it), episodes.csv, summary.json (per scenario and policy), and with --store numbered
results/run_NNN_... folders (GIF, plots, metrics) for the episodes of those seeds.

    .venv/bin/python experiments/haul_road_obstacles.py [--scenarios h1,h2] [--seeds 1000-1019] [--workers 4]
        [--policies ppo,overtaker,ppo_edge] [--store 1000] [--duration 120]
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import numpy as np   # noqa: E402

PPO_RUN = ROOT/'models'/'ppo_v10_e5_dyn_b11c_2.75m'
PPO_MODEL = 'final'
PLANNER_OVERRIDES = (('retry_cooldown', 1.0), ('w_lateral', 10.0))   # run_sim.py's default, over the run's own
SCORE = 'v10'
NEAR_MISS = 0.5   # m: a pass of an obstacle with less footprint clearance than this counts as a near miss
EDGE_FIX = (('nmpc_edge_yaw', True), ('escape_edge_grip', True))   # policy 'ppo_edge'


def env_cfg_for(scenario, extra=()):
    from nohesi.rl.evaluate import load_env_cfg
    cfg = load_env_cfg(PPO_RUN)
    return replace(cfg, scenario=scenario, score=SCORE, vehicle_model='dynamic', mpc='nonlinear',
                   planner_overrides=(*cfg.planner_overrides, *PLANNER_OVERRIDES, *extra))


def haul_metrics(ep, m):
    """Flat per-episode numbers for a haul road: safety, progress, comfort and convoy cohesion."""
    from nohesi.geometry import clearances, footprints
    from nohesi.world import traffic_states
    from nohesi.road import Road
    dt, sc = ep.scenario.timing.dt, ep.scenario
    v = ep.team[:, :, 3]
    accel = ep.controls[:, :, 0] if len(ep.controls) else np.zeros((0, 3))
    lat = ep.team_lat if ep.team_lat is not None else np.zeros(ep.team.shape[:2]+(2,))
    ay = np.abs(v*lat[:, :, 1])                       # lateral acceleration ~ v r
    # Footprint clearance to the nearest obstacle (any direction, within 15 m along the road), every 0.25 s.
    road, stride, nearest = Road(sc.road), max(1, int(round(0.25/dt))), np.inf
    for k in range(0, len(ep.time), stride):
        near = np.flatnonzero(np.abs(ep.traffic_s[k][None, :]-ep.team_s[k][:, None]).min(axis=0) < 15.0)
        if not near.size:
            continue
        snap = type('S', (), {'traffic_s': ep.traffic_s[k, near], 'traffic_d': ep.traffic_d[k, near],
                              'traffic_v': ep.traffic_v[k, near], 'traffic_vd': None})
        obs = footprints(traffic_states(snap, road), sc.vehicle)
        team = footprints(ep.team[k], sc.vehicle)
        nearest = min(nearest, float(clearances(np.repeat(team, len(near), 0), np.tile(obs, (3, 1, 1))).min()))
    passes = [o.min_clearance for o in ep.overtakes]
    fallbacks = (m.get('planner') or {}).get('fallbacks', {})
    spread = ep.team_s.max(axis=1)-ep.team_s.min(axis=1)
    distance = ep.team_s[-1]-ep.team_s[0]
    collision = ep.collisions[0] if ep.collisions else None
    return {
        'seed': int(ep.seed), 'scenario': sc.version, 'policy': ep.policy,
        'collision': bool(ep.collisions), 'collision_kind': collision.kind if collision else '',
        'collision_vehicle': collision.vehicle if collision else '',
        'duration_s': round(float(ep.time[-1]), 2),
        'distance_m': round(float(distance.mean()), 1),
        'mean_speed_mps': round(float(v.mean()), 2),
        'progress_mps': round(float(distance.mean()/max(ep.time[-1], dt)), 2),
        'stopped_s': round(float((v[1:] < 0.5).sum(axis=0).mean()*dt), 2),          # per car
        'stuck_s': round(float(np.mean(m['stuck_time_s'])), 2),                      # per car, behind an obstacle
        'obstacles_passed': len(passes),
        'near_misses': int(sum(c < NEAR_MISS for c in passes)),
        'min_pass_clearance_m': round(float(min(passes)), 3) if passes else '',
        'min_obstacle_clearance_m': round(nearest, 3) if np.isfinite(nearest) else '',
        'min_teammate_clearance_m': round(float(m['min_clearance_teammates_m']), 3)
        if m.get('min_clearance_teammates_m') is not None else '',
        'peak_brake_mps2': round(float(-accel.min()), 2) if accel.size else 0.0,
        'hard_brake_s': round(float((accel < -0.4*9.81*sc.vehicle.friction).sum(axis=0).mean()*dt), 2),
        'peak_lat_accel_mps2': round(float(ay.max()), 2),
        'emergencies': int(fallbacks.get('emergency', 0)+fallbacks.get('escape', 0)),
        'fallbacks': int(sum(fallbacks.values())),
        'lane_changes': int((m.get('planner') or {}).get('lane_changes_completed', 0)),
        'lane_changes_cancelled': int((m.get('planner') or {}).get('lane_changes_cancelled', 0)),
        'mean_spread_m': round(float(spread.mean()), 1), 'max_spread_m': round(float(spread.max()), 1),
        'solve_ms_mean': round(float(((m.get('planner') or {}).get('solve_ms') or {}).get('mean', np.nan)), 2),
        'team_score': round(float(m['team_score']), 0),
    }


def _store(ep, label):
    from nohesi.evaluation.io import append_index, next_run_dir, save
    from nohesi.evaluation.viz import animate, plot
    out = next_run_dir(ROOT/'results', label)
    m = save(ep, out)
    plot(ep, out/'summary.png')
    animate(ep, out/'team.gif')
    append_index(out, m)
    return str(out.relative_to(ROOT))


def episode(job):
    """One episode -> haul_metrics row (in a worker process)."""
    scenario, policy, seed, duration, store = job
    import multiprocessing
    if multiprocessing.current_process().name != 'MainProcess' and not getattr(episode, 'quiet', False):
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)   # acados prints a line per failed QP (C stdout): not in the log
        episode.quiet = True
    from nohesi.evaluation.metrics import metrics
    cfg = env_cfg_for(scenario, EDGE_FIX if policy == 'ppo_edge' else ())
    started = time.perf_counter()
    if policy in ('ppo', 'ppo_edge'):
        import torch
        torch.set_num_threads(1)
        from nohesi.rl.env import NoHesiEnv
        from nohesi.rl.evaluate import load_policy, resolve
        model, normalize = load_policy(*resolve(PPO_RUN, PPO_MODEL))
        env = NoHesiEnv(cfg)
        if duration:
            env.scenario = replace(env.scenario, timing=replace(env.scenario.timing, duration=float(duration)))
        obs, _ = env.reset(seed=int(seed))
        done = False
        while not done:
            action, _ = model.predict(normalize(obs[None]), action_masks=env.action_masks()[None], deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated
        ep = env.episode()
        ep.policy = f'ppo-{PPO_RUN.name}'
    else:
        from nohesi.evaluation.runner import run
        from nohesi.rl.evaluate import scenario_for
        sc, planner, score = scenario_for(cfg)
        if duration:
            sc = replace(sc, timing=replace(sc.timing, duration=float(duration)))
        ep = run(sc, int(seed), policy, planner, score=score)
    row = haul_metrics(ep, metrics(ep))
    row['policy'] = policy
    row['wall_s'] = round(time.perf_counter()-started, 1)
    if store:
        row['stored'] = _store(ep, f'{scenario}_{policy}'+(f'-{PPO_RUN.name}' if policy != 'overtaker' else '') +
                               f'_score{SCORE}_seed{seed}_dynamic_nmpc_haul_obstacles')
    return row


def summarize(rows):
    def mean(key):
        vals = [r[key] for r in rows if r[key] != '' and r[key] is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    keys = ('distance_m', 'mean_speed_mps', 'progress_mps', 'stopped_s', 'stuck_s', 'obstacles_passed', 'near_misses',
            'min_pass_clearance_m', 'min_obstacle_clearance_m', 'min_teammate_clearance_m', 'peak_brake_mps2',
            'hard_brake_s', 'peak_lat_accel_mps2', 'emergencies', 'fallbacks', 'lane_changes', 'lane_changes_cancelled',
            'mean_spread_m', 'max_spread_m', 'solve_ms_mean', 'duration_s')
    crashes = [r for r in rows if r['collision']]
    return {'episodes': len(rows), 'crashes': len(crashes), 'crash_rate': round(len(crashes)/len(rows), 3),
            'crash_kinds': {k: sum(r['collision_kind'] == k for r in crashes) for k in ('traffic', 'edge', 'team')},
            'crash_seeds': [r['seed'] for r in crashes],
            'crash_times_s': [r['duration_s'] for r in crashes],
            **{k: mean(k) for k in keys},
            'stopped_share': round(float(np.mean([r['stopped_s']/max(r['duration_s'], 1e-9) for r in rows])), 3),
            'clean_episodes': sum(not r['collision'] and r['stopped_s'] < 0.25*r['duration_s'] for r in rows),
            'worst_min_obstacle_clearance_m': min((r['min_obstacle_clearance_m'] for r in rows
                                                   if r['min_obstacle_clearance_m'] != ''), default=None)}


POLICY_STYLE = {'ppo': ('PPO policy (milestone 8)', '#2a78d6'),
                'ppo_edge': ('PPO + corner-aware edge bound', '#eb6834'),
                'overtaker': ('Scripted overtaker (same planner)', '#1baf7a')}
SCENARIO_NAME = {'h1': 'h1: trained speeds\n(up to 60 m/s, tarmac)', 'h2': 'h2: mining envelope\n(5-16.7 m/s, gravel)'}


def plot_summary(summary, path):
    """Small multiples: crash rate, distance covered and time stopped per scenario and policy (summary.json)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = summary['results']
    scenarios = list(res)
    policies = [p for p in POLICY_STYLE if all(p in res[s] for s in scenarios)]
    panels = (('crash_rate', 'Episodes ending in a crash', lambda v: f'{100*v:.0f}%', 100.0),
              ('distance_m', 'Distance covered per car (m)', lambda v: f'{v:,.0f}', 1.0),
              ('stopped_share', 'Share of driving time stopped', lambda v: f'{100*v:.0f}%', 100.0))
    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=surface)
    width = 0.8/len(policies)
    for ax, (key, title, fmt, scale) in zip(axes, panels):
        ax.set_facecolor(surface)
        for j, pol in enumerate(policies):
            vals = [res[s][pol][key] or 0.0 for s in scenarios]
            x = np.arange(len(scenarios))+(j-(len(policies)-1)/2)*width
            bars = ax.bar(x, [v*scale for v in vals], width*0.92, color=POLICY_STYLE[pol][1], label=POLICY_STYLE[pol][0],
                          zorder=2)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height(), fmt(v), ha='center', va='bottom', fontsize=8,
                        color=muted)
        ax.set_title(title, loc='left', fontsize=11, color=ink)
        ax.set_xticks(np.arange(len(scenarios)), [SCENARIO_NAME.get(s, s) for s in scenarios], fontsize=9, color=ink)
        ax.tick_params(axis='y', labelsize=8, colors=muted, length=0)
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color=grid, lw=0.8, zorder=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(grid)
        if key in ('crash_rate', 'stopped_share'):
            ax.set_ylim(0, 100)
    n = summary['results'][scenarios[0]][policies[0]]['episodes']
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=len(policies), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), labelcolor=ink)
    fig.text(0.01, 0.01, f'{n} seeds per bar (1000-{999+n}), 120 s episodes, 3-vehicle team, 12 static obstacles in view; '
             'dynamic tyre model + nonlinear MPC', fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def _seeds(text):
    if '-' in text:
        lo, hi = map(int, text.split('-'))
        return list(range(lo, hi+1))
    return [int(s) for s in text.split(',')]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenarios', default='h1,h2')
    p.add_argument('--policies', default='ppo,overtaker,ppo_edge')
    p.add_argument('--seeds', default='1000-1019')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--store', default='', help='seeds whose episodes are saved to results/ with a GIF')
    p.add_argument('--duration', type=float, default=None, help='episode length (s); default the scenario (120 s)')
    p.add_argument('--out', default=None)
    p.add_argument('--plot-only', action='store_true', help='redraw summary.png from the output folder\'s summary.json')
    a = p.parse_args(argv)
    scenarios, policies, seeds = a.scenarios.split(','), a.policies.split(','), _seeds(a.seeds)
    store = set(_seeds(a.store)) if a.store else set()
    out = Path(a.out) if a.out else ROOT/'results'/f'haul_obstacles_{date.today().isoformat()}'
    out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        plot_summary(json.loads((out/'summary.json').read_text(encoding='utf-8')), out/'summary.png')
        return
    # Build the acados solvers once per scenario before the pool starts (workers then only load them).
    from nohesi.rl.evaluate import scenario_for
    from nohesi.world import World
    from nohesi.coordination.centralized import CentralizedTeam
    from nohesi.rl.scripted import LaneKeeping
    for name in scenarios:
        for extra in ((), EDGE_FIX) if 'ppo_edge' in policies else ((),):
            sc, planner, _ = scenario_for(env_cfg_for(name, extra))
            CentralizedTeam(sc, World(sc, 0), LaneKeeping(sc, World(sc, 0).road, World(sc, 0).plant), planner)
    jsonl = out/'episodes.jsonl'
    done = [json.loads(line) for line in jsonl.read_text(encoding='utf-8').splitlines()] if jsonl.exists() else []
    have = {(r['scenario'], r['policy'], r['seed']) for r in done}
    jobs = [(s, pol, seed, a.duration, seed in store) for s in scenarios for pol in policies
            for seed in seeds if (s, pol, seed) not in have]
    jobs.sort(key=lambda j: j[1] == 'overtaker')   # PPO episodes (the long ones) first
    rows, started = [r for r in done if r['scenario'] in scenarios and r['policy'] in policies and r['seed'] in seeds], \
        time.perf_counter()
    print(f'{len(rows)} episodes already in {jsonl}, {len(jobs)} to run', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(episode, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            rows.append(r)
            with jsonl.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(r)+'\n')
            print(f"[{n}/{len(jobs)} {time.perf_counter()-started:.0f}s] {r['scenario']} {r['policy']:9} seed {r['seed']}: "
                  f"{'CRASH '+r['collision_kind']+' at '+str(r['duration_s'])+' s' if r['collision'] else 'ok'} | "
                  f"{r['distance_m']:.0f} m, {r['mean_speed_mps']:.1f} m/s, stopped {r['stopped_s']:.1f} s, "
                  f"passed {r['obstacles_passed']}, min clearance {r['min_obstacle_clearance_m']} m, "
                  f"emergencies {r['emergencies']} ({r['wall_s']:.0f} s)", flush=True)
    rows.sort(key=lambda r: (r['scenario'], r['policy'], r['seed']))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out/'episodes.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {'ppo_run': str(PPO_RUN.relative_to(ROOT)), 'ppo_model': PPO_MODEL, 'score': SCORE,
               'planner_overrides': PLANNER_OVERRIDES, 'seeds': seeds, 'near_miss_m': NEAR_MISS,
               'results': {s: {pol: summarize([r for r in rows if r['scenario'] == s and r['policy'] == pol])
                               for pol in policies} for s in scenarios}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    plot_summary(summary, out/'summary.png')
    print(json.dumps(summary['results'], indent=2), flush=True)
    print(f'results in {out}')


if __name__ == '__main__':
    main()
