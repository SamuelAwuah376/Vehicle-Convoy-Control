"""Perception evaluation on scenario v9 (design section 12, phase P2 acceptance).

The scripted overtaker drives with the default nonlinear Planner and run_sim.py's overrides (dynamic plant, score
v10). Mode 'shadow' (default): the perception pipeline runs every step and its belief is evaluated, but the planner
drives on the true snapshot, so the driving is exactly the reference's. Mode 'ideal' is experiment E0: the planner
drives on the belief from perfect detections, and --verify runs the same seeds without perception for the comparison.

Per episode (nohesi.perception.metrics): GOSPA (p 2, c 5 m, alpha 2) of the confirmed tracks in the observation
window with its localization / missed / false parts, NEES of the matched tracks, the tracker's NIS, identity switches,
per-sensor detection probability against range, and the pipeline's time per stage.

Writes results/perception_<mode>_v9_<n>seeds_<duration>s/: seeds/seed_<seed>.json (one episode's metrics),
episodes.csv (one row per episode), summary.json and summary.md (all episodes pooled).

    .venv/Scripts/python experiments/perception_shadow.py [--seeds 1000-1049] [--duration 60] [--workers 16]
        [--mode shadow|ideal] [--verify K]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import numpy as np   # noqa: E402
from nohesi.perception import metrics as M   # noqa: E402
from nohesi.perception.pipeline import STAGES, PerceptionSystem   # noqa: E402

SCENARIO, POLICY, PLANT, MPC, SCORE = 'v9', 'overtaker', 'dynamic', 'nonlinear', 'v10'
PLANNER_OVERRIDES = (('retry_cooldown', 1.0), ('w_lateral', 10.0))   # run_sim.py's
DIGEST = ('time', 'team', 'team_s', 'team_d', 'traffic_s', 'traffic_d', 'traffic_v', 'traffic_id', 'controls')
time_now = time.perf_counter


class RecordingSystem(PerceptionSystem):
    """The pipeline with record=True that also counts every scan's detections against the truth at the scan time
    (metrics.DetectionCounter), for episode times >= 0; eval_time sums the seconds that took."""

    def reset_state(self):
        super().reset_state()
        self.counter = M.DetectionCounter(self.sensors, self.params, self.cfg.car_height)
        self.eval_time = 0.0

    def _queue_scan(self, spec, car, time, dets, raw, true):
        super()._queue_scan(spec, car, time, dets, raw, true)
        if time >= -1e-9:
            tic = time_now()
            self.counter.add(spec.name, car, time, dets, self.truth.at(time))
            self.eval_time += time_now()-tic


def scenario(duration, mode):
    from nohesi.config import SCENARIOS
    from nohesi.perception.config import Perception
    sc = SCENARIOS[SCENARIO]()
    sc = replace(sc, vehicle=replace(sc.vehicle, model=PLANT), timing=replace(sc.timing, duration=float(duration)))
    return replace(sc, perception=Perception(enabled=mode != 'off', mode='shadow' if mode == 'off' else mode))


def digest(ep):
    m = hashlib.sha256()
    for k in DIGEST:
        m.update(np.ascontiguousarray(getattr(ep, k)).tobytes())
    for t in sorted(ep.score):
        m.update(np.ascontiguousarray(ep.score[t]).tobytes())
    return m.hexdigest()


def peak_memory_mb():
    """This process's peak resident memory (MB), None where it cannot be read."""
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                            ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                            ('PeakPagefileUsage', ctypes.c_size_t)]
            c = Counters()
            c.cb = ctypes.sizeof(c)
            get = ctypes.windll.psapi.GetProcessMemoryInfo
            get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if get(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.PeakWorkingSetSize/2**20
            return None
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except Exception:
        return None


def drive(seed, duration, mode):
    """One episode: (EpisodeRun, the recording system or None, wall seconds)."""
    from nohesi.config import SCORES, Planner
    from nohesi.evaluation.runner import EpisodeRun
    held = []

    def factory(sc, world, s):
        if not sc.perception.enabled:
            return None
        held.append(RecordingSystem(sc.perception, sc, world.road, world.plant.params, s, record=True))
        return held[-1]
    start = time_now()
    r = EpisodeRun(scenario(duration, mode), seed, POLICY, replace(Planner(), mpc=MPC, **dict(PLANNER_OVERRIDES)),
                   score=SCORES[SCORE](), perception=factory)
    while not r.done:
        r.step()
    return r, (held[0] if held else None), time_now()-start


def driving(r):
    from nohesi.evaluation.metrics import metrics
    ep = r.episode()
    m = metrics(ep)
    return {'team_score': m['team_score'], 'collisions': len(m['collisions']), 'end_reason': m['end_reason'],
            'duration_s': m['duration_s'], 'overtakes': m['overtakes'], 'digest': digest(ep)}


def episode(job):
    """One seed -> its evaluation (runs in a worker process)."""
    seed, duration, mode, verify = job
    r, system, wall = drive(seed, duration, mode)
    tic = time_now()
    ev = M.evaluate_episode(system.beliefs, r.snaps, system.tracker.nis_records(), system.counter.result(),
                            gate=system.cfg.tracker.gate)
    eval_s = time_now()-tic+system.eval_time
    steps = len(r.controls)
    timing = {'wall_s': wall, 'steps': steps, 'calls': system.calls, 'stage_ms': system.timing(),
              'planner_ms_per_step': 1e3*float(np.mean(r.compute)) if r.compute else 0.0,
              'wall_ms_per_step': 1e3*wall/max(steps, 1), 'eval_s': eval_s, 'peak_memory_mb': peak_memory_mb()}
    out = {'seed': seed, 'mode': mode, 'duration': duration, 'run': driving(r), 'timing': timing,
           'tracker_stats': dict(system.tracker.stats), 'eval': ev, 'reference': None}
    if verify:
        ref, _, ref_wall = drive(seed, duration, 'off')
        out['reference'] = {**driving(ref), 'wall_s': ref_wall}
    return out


# ---------------------------------------------------------------------------------------------------------- output
def _json(x):
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def seed_record(res):
    return {k: res[k] for k in ('seed', 'mode', 'duration', 'run', 'reference', 'timing', 'tracker_stats')} | \
        {'metrics': M.summarize([res['eval']])}


def row(res, summary):
    g, n, ids = summary['gospa'], summary['nees'], summary['id_switches']
    nis = summary['nis'].get('all', {})
    t = res['timing']
    ref = res['reference']
    return {'seed': res['seed'], 'team_score': round(res['run']['team_score'], 1),
            'collisions': res['run']['collisions'], 'end_reason': res['run']['end_reason'],
            'gospa_m': _r(g.get('mean')), 'gospa_loc_m2': _r(g.get('localization')),
            'gospa_missed_m2': _r(g.get('missed')), 'gospa_false_m2': _r(g.get('false')),
            'rms_pos_m': _r(g.get('rms_position_error')), 'coverage': _r(g.get('coverage'), 4),
            'missed_per_frame': _r(g.get('missed_per_frame')), 'false_per_frame': _r(g.get('false_per_frame')),
            'nees4_mean': _r(n.get('state', {}).get('mean')), 'nees4_inside': _r(n.get('state', {}).get('inside'), 4),
            'nees2_mean': _r(n.get('position', {}).get('mean')),
            'nis_ratio': _r(nis.get('ratio'), 4), 'nis_inside': _r(nis.get('inside'), 4),
            'false_duplicate': _r(summary['false_tracks'].get('duplicate')),
            'false_ghost': _r(summary['false_tracks'].get('ghost')),
            'id_switches': ids['switches'], 'id_takeovers': ids['takeovers'], 'id_rebirths': ids['rebirths'],
            'track_swaps': ids['track_swaps'],
            'perception_ms_per_call': _r(t['stage_ms']['total']), 'planner_ms_per_step': _r(t['planner_ms_per_step']),
            'wall_s': _r(t['wall_s'], 1),
            'ref_team_score': None if ref is None else round(ref['team_score'], 1),
            'ref_collisions': None if ref is None else ref['collisions'],
            'ref_identical': None if ref is None else ref['digest'] == res['run']['digest']}


def _r(x, k=3):
    return None if x is None else round(float(x), k)


def timing_summary(results):
    stages = {s: float(np.mean([r['timing']['stage_ms'][s] for r in results])) for s in (*STAGES, 'total')}
    return {'stage_ms_per_call': stages,
            'planner_ms_per_step': float(np.mean([r['timing']['planner_ms_per_step'] for r in results])),
            'wall_ms_per_step': float(np.mean([r['timing']['wall_ms_per_step'] for r in results])),
            'episode_wall_s_mean': float(np.mean([r['timing']['wall_s'] for r in results])),
            'episode_wall_s_max': float(np.max([r['timing']['wall_s'] for r in results])),
            'eval_s_mean': float(np.mean([r['timing']['eval_s'] for r in results])),
            'peak_memory_mb_max': max((r['timing']['peak_memory_mb'] or 0.0 for r in results), default=0.0)}


def markdown(meta, summary, results, timing):
    g, nees, ids = summary['gospa'], summary['nees'], summary['id_switches']
    L = [f"# Perception {meta['mode']} evaluation: {meta['scenario']}, {len(results)} seeds x {meta['duration']:g} s",
         '', f"{meta['created']}. Policy {POLICY} (Planner mpc '{MPC}', overrides {dict(PLANNER_OVERRIDES)}), plant "
         f"{PLANT}, score {SCORE}; seeds {meta['seeds']}. Mode '{meta['mode']}': " +
         ('the planner drives on the truth, the belief is evaluated.' if meta['mode'] == 'shadow' else
          'the planner drives on the belief (experiment E0).'),
         f"Window {summary['window'][0]:g} to +{summary['window'][1]:g} m of the team centre, confirmed tracks, every "
         f"belief time t >= 0 ({summary['frames']} frames).", '',
         '## Driving', '',
         f"Team score mean {np.mean([r['run']['team_score'] for r in results]):,.0f}, crashed episodes "
         f"{sum(r['run']['collisions'] > 0 for r in results)}/{len(results)}."]
    refs = [r for r in results if r['reference'] is not None]
    if refs:
        same = sum(r['reference']['digest'] == r['run']['digest'] for r in refs)
        d = [r['run']['team_score']-r['reference']['team_score'] for r in refs]
        crashed = sum(r['run']['collisions'] > 0 for r in refs), sum(r['reference']['collisions'] > 0 for r in refs)
        L += [f"Against perception off on {len(refs)} seeds: {same}/{len(refs)} byte-identical; score difference "
              f"mean {np.mean(d):,.0f} (min {np.min(d):,.0f}, max {np.max(d):,.0f}); crashed episodes {crashed[0]} vs "
              f"{crashed[1]} off.", '', '| seed | score | off | difference | collisions | off |',
              '|---|---|---|---|---|---|']
        L += [f"| {r['seed']} | {r['run']['team_score']:,.0f} | {r['reference']['team_score']:,.0f} | "
              f"{r['run']['team_score']-r['reference']['team_score']:,.0f} | {r['run']['collisions']} | "
              f"{r['reference']['collisions']} |" for r in refs]
    L += ['', f"## GOSPA (p {g['p']:g}, c {g['c']:g} m, alpha 2), per frame", '',
          '| GOSPA (m) | localization (m^2) | missed (m^2) | false (m^2) | RMS position error of matches (m) | cars | '
          'tracks | missed | false | coverage |', '|---|---|---|---|---|---|---|---|---|---|',
          f"| {g['mean']:.2f} | {g['localization']:.2f} | {g['missed']:.2f} | {g['false']:.2f} | "
          f"{g['rms_position_error']:.3f} | {g['truth_per_frame']:.1f} | {g['tracks_per_frame']:.1f} | "
          f"{g['missed_per_frame']:.2f} | {g['false_per_frame']:.2f} | {g['coverage']:.4f} |", '',
          'By offset from the team centre (m):', '', '| from | to | cars | coverage | tracks | false fraction |',
          '|---|---|---|---|---|---|']
    L += [f"| {b['from']:g} | {b['to']:g} | {b['truth']} | {_f(b['coverage'], 4)} | {b['tracks']} | "
          f"{_f(b['false_fraction'], 4)} |" for b in summary['by_offset']]
    ft = summary.get('false_tracks', {})
    if ft.get('track_frames'):
        L += ['', f"False confirmed tracks ({ft['track_frames']} track-frames): duplicates (a car within c, matched to "
              f"another track) {ft['duplicate']:.3f}, misplaced (c to 3c from a car) {ft['misplaced']:.3f}, ghosts "
              f"(beyond 3c) {ft['ghost']:.3f}; {ft['never_matched']:.3f} from ids never matched in their episode "
              f"({ft['never_matched_ids']} ids, false for a median {_f(ft.get('never_matched_life_median_s'), 2)} s, "
              f"90 % {_f(ft.get('never_matched_life_p90_s'), 2)} s, max "
              f"{_f(ft.get('never_matched_life_max_s'), 1)} s); "
              f"mean existence {ft['p_exist_mean']:.3f} (matched tracks {_f(ft.get('p_exist_matched_mean'))})."]
    L += ['', '## NEES of the matched tracks', '',
          '| | samples | mean | 95 % bounds of the mean | median (chi2 median) | samples inside their 95 % interval | '
          'time steps inside (Bar-Shalom) | > 99.9 % quantile |', '|---|---|---|---|---|---|---|---|']
    for name in ('state', 'position'):
        n = nees.get(name)
        if n:
            L.append(f"| {name} ({n['dof']} dof) | {n['n']} | {n['mean']:.2f} | "
                     f"{n['bounds'][0]:.3f}-{n['bounds'][1]:.3f} | {n['median']:.2f} ({n['median_expected']:.2f}) | "
                     f"{n['inside']:.3f} | "
                     f"{n['per_time']['inside']:.3f} ({n['per_time']['pairs_per_step']:.0f} pairs per step, below "
                     f"{n['per_time']['below']:.2f}, above {n['per_time']['above']:.2f}) | {n['above_999']:.4f} |")
    if nees:
        L += ['', 'State NEES by offset (m), by time since last seen (s) and by lane change (true |v_d| > '
              f'{M.LANE_CHANGE_VD:g} m/s):', '', '| bin | samples | mean | median | inside |', '|---|---|---|---|---|']
        L += [f"| offset {b['from']:g} to {b['to']:g} | {b['n']} | {_f(b.get('mean'))} | {_f(b.get('median'))} | "
              f"{_f(b.get('inside'))} |" for b in nees['state']['by_offset']]
        L += [f"| age {b['from']:g} to {_f(b['to'], 1) if b['to'] is not None else 'inf'} | {b['n']} | "
              f"{_f(b.get('mean'))} | {_f(b.get('median'))} | {_f(b.get('inside'))} |" for b in nees['state']['by_age']]
        L += [f"| {'changing lane' if b['changing'] else 'keeping lane'} | {b['n']} | {_f(b.get('mean'))} | "
              f"{_f(b.get('median'))} | {_f(b.get('inside'))} |" for b in nees['state'].get('by_lane_change', ())]
        L += ['', 'Per state component: normalized error mean and spread (1 when consistent), error mean and RMS:', '',
              '| | z mean | z std | error mean | error RMS |', '|---|---|---|---|---|']
        L += [f"| {k} | {c['z_mean']:.3f} | {c['z_std']:.3f} | {c['err_mean']:.3f} | {c['err_rms']:.3f} |"
              for k, c in nees['components'].items()]
    L += ['', '## NIS (tracker records, gate-truncated chi-square)', '',
          '| group | records | mean | mean dof | expected | 95 % bounds | ratio | inside (0.95) | PIT (10 bins, flat '
          '= 0.1) |', '|---|---|---|---|---|---|---|---|---|']
    for name, s in summary['nis'].items():
        L.append(f"| {name} | {s['n']} | {s['mean']:.3f} | {s['mean_dof']:.2f} | {s['expected']:.3f} | "
                 f"{s['bounds'][0]:.3f}-{s['bounds'][1]:.3f} | {s['ratio']:.3f} | {s['inside']:.3f} | "
                 f"{' '.join(f'{p:.2f}' for p in s['pit'])} |")
    split = '' if ids.get('takeovers') is None else \
        f" ({ids['takeovers']} takeovers, the old track still confirmed; {ids['rebirths']} rebirths, the old one gone)"
    L += ['', '## Identity', '', f"{ids['switches']} identity switches{split} and {ids['track_swaps']} track swaps "
          f"over {ids['tracked_car_minutes']:.0f} tracked car-minutes ({_f(ids['switches_per_car_minute'], 4)} "
          f"switches per car-minute); {_f(ids['tracks_per_car'])} track ids per matched car.", '',
          '## Detection probability against range (after processing)', '',
          'Opportunities: traffic cars with their body centre in the field of view and nominal range; visible: at '
          f'least {M.VISIBLE:g} of the facing outline in sight.', '']
    det = summary['detection']
    if det:
        edges = next(iter(det.values()))['edges']
        head = ' | '.join(f'{edges[i]:g}-{edges[i+1]:g}' for i in range(len(edges)-1))
        L += [f'| sensor | all | visible | {head} | false per scan | teammate per scan |',
              '|---|---|---|' + '---|'*(len(edges)-1) + '---|---|']
        for name, d in det.items():
            cells = ' | '.join(f"{_f(p, 2)}/{_f(q, 2)}" if o else '' for p, q, o in zip(d['pd'], d['pd_visible'],
                                                                                    d['opp']))
            L.append(f"| {name} | {_f(d['pd_all'])} | {_f(d['pd_visible_all'])} | {cells} | "
                     f"{_f(d['false_per_scan'])} | {_f(d['mate_per_scan'], 4)} |")
        L += ['', 'Range cells: all / visible.']
    st = timing['stage_ms_per_call']
    L += ['', '## Time', '', '| stage | ms per call |', '|---|---|']
    L += [f'| {k} | {v:.2f} |' for k, v in st.items()]
    L += ['', f"Planner {timing['planner_ms_per_step']:.1f} ms per step; whole step {timing['wall_ms_per_step']:.1f} "
          f"ms (world, planner, perception); episode wall time mean {timing['episode_wall_s_mean']:.0f} s, max "
          f"{timing['episode_wall_s_max']:.0f} s with {meta['workers']} workers; evaluation "
          f"{timing['eval_s_mean']:.1f} s per episode; peak worker memory {timing['peak_memory_mb_max']:.0f} MB.", '']
    return '\n'.join(L)


def _f(x, k=3):
    return '' if x is None else f'{x:.{k}f}'


def parse_seeds(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b)+1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='1000-1049')
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--mode', choices=('shadow', 'ideal'), default='shadow')
    ap.add_argument('--verify', type=int, default=0, help='also run the first K seeds without perception')
    ap.add_argument('--out', default=None, help='output folder (default results/perception_<mode>_v9_...)')
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    out = Path(args.out) if args.out else \
        ROOT/'results'/f'perception_{args.mode}_{SCENARIO}_{len(seeds)}seeds_{args.duration:g}s'
    (out/'seeds').mkdir(parents=True, exist_ok=True)
    jobs = [(s, args.duration, args.mode, k < args.verify) for k, s in enumerate(seeds)]
    meta = {'scenario': SCENARIO, 'mode': args.mode, 'duration': args.duration, 'seeds': args.seeds,
            'workers': args.workers, 'policy': POLICY, 'plant': PLANT, 'mpc': MPC, 'score': SCORE,
            'planner_overrides': dict(PLANNER_OVERRIDES), 'created': datetime.now().isoformat(timespec='seconds'),
            'perception': asdict(scenario(args.duration, args.mode).perception)}
    results, start = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(episode, j): j[0] for j in jobs}
        for f in as_completed(futures):
            res = f.result()
            rec = seed_record(res)
            (out/'seeds'/f"seed_{res['seed']}.json").write_text(json.dumps(_json(rec), indent=1), encoding='utf-8')
            results.append(res)
            g = rec['metrics']['gospa']
            print(f"seed {res['seed']}: score {res['run']['team_score']:,.0f}, collisions {res['run']['collisions']}, "
                  f"GOSPA {g.get('mean', float('nan')):.2f} m, coverage {g.get('coverage') or 0:.3f}, "
                  f"wall {res['timing']['wall_s']:.0f} s ({len(results)}/{len(jobs)}, "
                  f"{time.perf_counter()-start:.0f} s)", flush=True)
    results.sort(key=lambda r: r['seed'])
    summary = M.summarize([r['eval'] for r in results])
    timing = timing_summary(results)
    rows = [row(r, M.summarize([r['eval']])) for r in results]
    with open(out/'episodes.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    refs = [r for r in results if r['reference'] is not None]
    drive_summary = {'team_score_mean': float(np.mean([r['run']['team_score'] for r in results])),
                     'crashed_episodes': int(sum(r['run']['collisions'] > 0 for r in results)),
                     'reference_seeds': len(refs),
                     'reference_identical': int(sum(r['reference']['digest'] == r['run']['digest'] for r in refs)),
                     'reference_team_score_mean': float(np.mean([r['reference']['team_score'] for r in refs]))
                     if refs else None,
                     'reference_crashed_episodes': int(sum(r['reference']['collisions'] > 0 for r in refs))}
    stats = {k: int(sum(r['tracker_stats'][k] for r in results)) for k in results[0]['tracker_stats']}
    doc = {'meta': meta, 'driving': drive_summary, 'timing': timing, 'tracker_stats': stats, 'summary': summary,
           'wall_s': time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(_json(doc), indent=1), encoding='utf-8')
    (out/'summary.md').write_text(markdown(meta, summary, results, timing), encoding='utf-8')
    print(f'outputs in {out} ({time.perf_counter()-start:.0f} s)')


if __name__ == '__main__':
    main()
