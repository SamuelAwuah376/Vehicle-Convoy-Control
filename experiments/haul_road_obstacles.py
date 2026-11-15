"""Haul-road obstacle test (the user's call, 2026-11-15): the current stack and policy with the traffic cars replaced by
static obstacles, as a first read on whether it is a basis for autonomous heavy-truck convoys on mining / logging roads.

Stack as run_sim.py runs it: the milestone 8 PPO policy (models/ppo_v10_e5_dyn_b11c_2.75m, final), its env config
with run_sim.py's planner overrides, planner m4.4, the nonlinear MPC (acados) on the dynamic plant. Baseline: the
scripted overtaker on the same planner. Scenarios (nohesi.config):
    h1  v9's road and cars at the policy's trained speeds (70 mph start, up to 134 mph), 12 static obstacles in place
        of the 45 traffic cars
    h2  h1 in a mining envelope: 45 km/h start, 5-60 km/h commanded, gravel (mu 0.6)
Writes results/haul_obstacles_<date>/: episodes.csv (one row per episode), summary.json (per scenario and policy),
and with --store numbered results/run_NNN_... folders (GIF, plots, metrics) for the PPO episodes of those seeds.

    .venv/bin/python experiments/haul_road_obstacles.py [--scenarios h1,h2] [--seeds 1000-1019] [--workers 4]
        [--policies ppo,overtaker] [--store 1000] [--duration 120]
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


def env_cfg_for(scenario):
    from nohesi.rl.evaluate import load_env_cfg
    cfg = load_env_cfg(PPO_RUN)
    return replace(cfg, scenario=scenario, score=SCORE, vehicle_model='dynamic', mpc='nonlinear',
                   planner_overrides=(*cfg.planner_overrides, *PLANNER_OVERRIDES))


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
    from nohesi.evaluation.metrics import metrics
    cfg = env_cfg_for(scenario)
    started = time.perf_counter()
    if policy == 'ppo':
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
        row['stored'] = _store(ep, f'{scenario}_{policy}'+(f'-{PPO_RUN.name}' if policy == 'ppo' else '') +
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
            'worst_min_obstacle_clearance_m': min((r['min_obstacle_clearance_m'] for r in rows
                                                   if r['min_obstacle_clearance_m'] != ''), default=None)}


def _seeds(text):
    if '-' in text:
        lo, hi = map(int, text.split('-'))
        return list(range(lo, hi+1))
    return [int(s) for s in text.split(',')]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenarios', default='h1,h2')
    p.add_argument('--policies', default='ppo,overtaker')
    p.add_argument('--seeds', default='1000-1019')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--store', default='', help='seeds whose PPO episodes are saved to results/ with a GIF')
    p.add_argument('--duration', type=float, default=None, help='episode length (s); default the scenario (120 s)')
    p.add_argument('--out', default=None)
    a = p.parse_args(argv)
    scenarios, policies, seeds = a.scenarios.split(','), a.policies.split(','), _seeds(a.seeds)
    store = set(_seeds(a.store)) if a.store else set()
    out = Path(a.out) if a.out else ROOT/'results'/f'haul_obstacles_{date.today().isoformat()}'
    out.mkdir(parents=True, exist_ok=True)
    # Build the acados solvers once per scenario before the pool starts (workers then only load them).
    from nohesi.rl.evaluate import scenario_for
    from nohesi.world import World
    from nohesi.coordination.centralized import CentralizedTeam
    from nohesi.rl.scripted import LaneKeeping
    for name in scenarios:
        sc, planner, _ = scenario_for(env_cfg_for(name))
        CentralizedTeam(sc, World(sc, 0), LaneKeeping(sc, World(sc, 0).road, World(sc, 0).plant), planner)
    jobs = [(s, pol, seed, a.duration, pol == 'ppo' and seed in store) for s in scenarios for pol in policies for seed in seeds]
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(episode, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            rows.append(r)
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
    print(json.dumps(summary['results'], indent=2))
    print(f'results in {out}')


if __name__ == '__main__':
    main()
