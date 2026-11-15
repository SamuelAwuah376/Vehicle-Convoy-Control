"""Evaluate a trained yard (convoy-parking) MaskablePPO policy against the scripted prioritized baseline.

Edit the settings below or pass the flags. The policy (deterministic, with the env's action masks, observations
normalised with the VecNormalize statistics saved next to the model) and the scripted baseline
(nohesi.yard.env.scripted_actions) play the same seeds with the env config the run was trained on, each in --workers
parallel processes (nohesi.yard.train.evaluate and .baseline). Seeds 1000-1049 are held out: training episodes draw
seeds >= 2000. The trainer picks best_model.zip on its eval_seeds (1000-1019 by default), so 'best' is optimistic on
those; the summary warns about any overlap.

    .venv/bin/python experiments/yard_eval.py                                  # final model of runs_rl/yard_ppo_001
    .venv/bin/python experiments/yard_eval.py --model best --workers 4
    .venv/bin/python experiments/yard_eval.py --model latest --seeds 1000-1009 --no-gif
    .venv/bin/python experiments/yard_eval.py --model runs_rl/yard_ppo_001/checkpoints/model_200000.zip

--model: 'final', 'best', 'latest' (newest checkpoint), a checkpoint timestep (200000: checkpoints/model_200000.zip) or
a model .zip path. Its VecNormalize is the .pkl the trainer saves alongside (final_/best_vecnormalize.pkl,
vecnormalize_<t>.pkl; --vecnorm overrides); if it is missing the policy sees raw observations, with a warning. The env
config is read from the config.json in the model's folder, its parent or the run folder (nohesi.yard.train.write_config:
config.json['train']['env'], env module config.json['train']['env_module']); without one, today's defaults (warning).
The model file is read once, so a trainer overwriting best_model.zip mid-evaluation cannot swap it.

Writes results/yard_eval_<run>_<model>_<date>/ (suffix _2, _3, ... if that exists):
    per_seed.csv    one row per seed: followers, then ppo_* and scripted_* success, parked, duration_s, safety_stops,
                    replans, back_off_m, collision, end_reason; delta_duration_s = ppo - scripted on seeds both solved
    summary.json    per policy nohesi.yard.train.aggregate() (success_rate, collision_rate, duration_s = mean over all
                    episodes, success_duration_s = over successes, safety_stops, replans, ... per episode) plus
                    by_followers (1/2/3); 'paired': ppo - scripted duration on the seeds both solved (mean, std, 95 % CI,
                    median, seeds faster/slower/equal, seeds solved by one only); the rendered cases; provenance; warnings
    summary.md      the same as tables (also printed)
    <case>_seed<s>_<policy>.png / .gif   nohesi.yard.viz plot / animate of that seed for both policies (ppo, scripted;
                    the figure titles name the policy). Cases: ppo_wins_most (largest duration gain over the baseline),
                    ppo_loses_most (largest loss), ppo_fails (a policy failure: a collision first, else one the baseline
                    solved). The episodes are replayed for this (and checked against the evaluation); --no-gif: PNGs only.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')   # env workers single-threaded (inherited by the spawned processes)

import numpy as np   # noqa: E402

# ---------------------------------------------------------------------------------------------- settings
RUN = 'runs_rl/yard_ppo_001'   # run folder (relative: to the current directory, else to the repo)
MODEL = 'final'                # 'final', 'best', 'latest', a checkpoint timestep, or a model .zip path
SEEDS = '1000-1049'            # held out (training draws seeds >= 2000); '1000-1009,1020' style
WORKERS = min(8, os.cpu_count() or 1)   # parallel processes per stage (policy envs, baseline episodes, renders)
GIF = True                     # False (--no-gif): the cases as PNGs only
OUT_ROOT = 'results'

POLICIES = ('ppo', 'scripted')
CSV_FIELDS = ('success', 'parked', 'duration_s', 'safety_stops', 'replans', 'back_off_m', 'collision', 'end_reason')
FOLLOWER_COUNTS = (1, 2, 3)
TRAIN_SEEDS_FROM = 2000        # nohesi.yard.env: training episodes draw seeds >= 2000
TIE_S = 1e-6                   # |ppo - scripted| duration below this: equal
YARD_CODE = ('nohesi/yard',)   # checked against the run's training commit

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print(f'WARNING: {msg}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------- inputs
def parse_seeds(text):
    """'1000-1049', '1000,1003' or a mix ('1000-1009,1020'); duplicates dropped, order kept."""
    out = []
    for part in str(text).replace(' ', '').split(','):
        if part:
            lo, sep, hi = part.partition('-')
            out.extend(range(int(lo), int(hi)+1) if sep else [int(part)])
    if not out:
        raise ValueError(f'no seeds in {text!r}')
    return list(dict.fromkeys(out))


def seed_text(seeds):
    """Compact seed list: 1000-1019,1030."""
    seeds, parts, k = sorted(int(s) for s in seeds), [], 0
    while k < len(seeds):
        j = k
        while j+1 < len(seeds) and seeds[j+1] == seeds[j]+1:
            j += 1
        parts.append(str(seeds[k]) if j == k else f'{seeds[k]}-{seeds[j]}')
        k = j+1
    return ','.join(parts)


def find(path):
    """`path` as given (relative: to the current directory) if it exists, else relative to the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() or p.exists() or not (ROOT/p).exists() else ROOT/p


def _rel(path):
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def checkpoints(run_dir):
    """{timestep: checkpoints/model_<t>.zip}, sorted."""
    out = {}
    for p in (Path(run_dir)/'checkpoints').glob('model_*.zip'):
        t = p.stem[len('model_'):]
        if t.isdigit():
            out[int(t)] = p
    return dict(sorted(out.items()))


def available_models(run_dir):
    return [w for w in ('final', 'best') if (Path(run_dir)/f'{w}_model.zip').exists()] + \
        [f'checkpoints/model_{t}.zip' for t in checkpoints(run_dir)]


def resolve_model(run_dir, which, vecnorm=None):
    """(model .zip, its VecNormalize .pkl or None if there is none) for --model (see the module docstring)."""
    from nohesi.rl.evaluate import resolve
    run_dir, which = Path(run_dir), str(which)
    if which in ('final', 'best'):
        model, norm = resolve(run_dir, which)
    else:
        if which == 'latest' or which.isdigit():
            ckpts = checkpoints(run_dir)
            if which == 'latest' and not ckpts:
                raise FileNotFoundError(f'no checkpoints in {run_dir/"checkpoints"}; models in {run_dir}: '
                                        f'{available_models(run_dir) or "none"}')
            model = ckpts[max(ckpts)] if which == 'latest' else run_dir/'checkpoints'/f'model_{which}.zip'
        else:
            given = Path(which).expanduser()
            options = [given, ROOT/given, run_dir/given]
            options += [p.with_name(p.name+'.zip') for p in options if p.suffix != '.zip']
            model = next((p for p in options if p.is_file()), given)
        model, norm = resolve(model)
    if not model.is_file():
        where = run_dir if run_dir.is_dir() else f'{run_dir} (no such folder)'
        raise FileNotFoundError(f'model {model} not found; models in {where}: '
                                f'{(available_models(run_dir) if run_dir.is_dir() else None) or "none"}')
    if vecnorm is not None:
        norm = find(vecnorm)
        if not norm.is_file():
            raise FileNotFoundError(f'--vecnorm {norm} not found')
    return model, (norm if norm.is_file() else None)


def model_label(model_path):
    """Output-folder tag: final, best, ckpt<t>, or the file's stem."""
    stem = Path(model_path).stem
    if stem in ('final_model', 'best_model'):
        return stem[:-len('_model')]
    if stem.startswith('model_') and stem[len('model_'):].isdigit():
        return f'ckpt{stem[len("model_"):]}'
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in stem)


def config_dir(model_path, run_dir):
    """The folder whose config.json describes the model: the model's own, its parent (a checkpoint), or the run's."""
    for d in (Path(model_path).parent, Path(model_path).parent.parent, Path(run_dir)):
        if (d/'config.json').is_file():
            return d
    return None


def load_run_config(cfg_dir):
    """(env config, env module, run info) from a run's config.json. The trainer's layout (nohesi.yard.train.write_config)
    has the train config under 'train', the env config inlined as train.env and its module as train.env_module; a flat
    {'env': ...} or a bare env-config dict are read too. Unknown env keys are dropped and missing ones take today's
    defaults (both warned about); no config.json, or none with an env config: the YardEnvConfig() defaults (warned)."""
    from nohesi.yard.train import ENV_MODULE, _tupled, env_function
    if cfg_dir is None:
        warn(f'no config.json for this model: the env runs with {ENV_MODULE}.YardEnvConfig() defaults, which may not '
             'be what it was trained on')
        return env_function(ENV_MODULE, 'YardEnvConfig')(), ENV_MODULE, {}
    path = Path(cfg_dir)/'config.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    train = d['train'] if isinstance(d.get('train'), dict) else d
    module = train.get('env_module')
    if not module and isinstance(d.get('env_config_type'), str) and '.' in d['env_config_type']:
        module = d['env_config_type'].rsplit('.', 1)[0]
    module = module or ENV_MODULE
    env_cls = env_function(module, 'YardEnvConfig')
    names = [f.name for f in fields(env_cls)]
    saved = next((x for x in (train.get('env'), d.get('env')) if isinstance(x, dict)), None)
    if saved is None:
        saved = {k: v for k, v in d.items() if k in names}   # a bare env config
    if not saved:
        warn(f'{_rel(path)} holds no env config: the env runs with {env_cls.__name__}() defaults')
    unknown, missing = sorted(set(saved)-set(names)), [k for k in names if k not in saved]
    if unknown:
        warn(f'{_rel(path)}: env keys unknown to today\'s {env_cls.__name__} dropped: {", ".join(unknown)}')
    if saved and missing:
        warn(f'{_rel(path)}: env keys missing, today\'s defaults used: {", ".join(missing)}')
    env_cfg = env_cls(**{k: _tupled(v) for k, v in saved.items() if k in names})
    current = getattr(env_cls(), 'version', None)
    if getattr(env_cfg, 'version', current) != current:
        warn(f'the run was trained on env version {env_cfg.version}, the code is at {current}: the observation, masks '
             'or reward may have changed since')
    info = {'config': _rel(path), 'run_id': train.get('run_id'), 'eval_seeds': train.get('eval_seeds'),
            'norm_obs': train.get('norm_obs', True), 'git': d.get('git'), 'observation_dim': d.get('observation_dim'),
            'action_nvec': d.get('action_nvec')}
    return env_cfg, module, info


def code_changed_since(commit, paths=YARD_CODE):
    """True if `paths` differ (committed or not) from the training commit, None if git cannot tell. Read-only."""
    if not commit:
        return None
    try:
        r = subprocess.run(['git', 'diff', '--quiet', commit, '--', *paths], cwd=ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: False, 1: True}.get(r.returncode)


# ---------------------------------------------------------------------------------------------- model
def read_model(model_path, tries=3):
    """(the .zip's bytes, MaskablePPO loaded from them). Retries a file caught mid-write by the trainer."""
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    for k in range(tries):
        data = Path(model_path).read_bytes()
        try:
            return data, MaskablePPO.load(io.BytesIO(data), device='cpu')
        except (zipfile.BadZipFile, EOFError) as e:
            if k == tries-1:
                raise
            print(f'{model_path} unreadable ({e!r}), probably being written: retrying in 5 s', flush=True)
            time.sleep(5)


def load_normalizer(vecnorm_path, expected, norm_obs=True):
    """ObsNormalizer from the VecNormalize .pkl; None: identity (warned about unless the run trained without it)."""
    from nohesi.rl.evaluate import ObsNormalizer
    if vecnorm_path is None:
        if norm_obs:
            warn(f'no VecNormalize at {_rel(expected)}: the policy sees raw, unnormalised observations (it trained on '
                 'normalised ones, so expect it to act poorly)')
        return ObsNormalizer()
    return ObsNormalizer.load(vecnorm_path)


def check_spaces(model, normalize, env_cfg, env_module):
    """The model and the normaliser fit the env's observation and action spaces (ValueError otherwise)."""
    from nohesi.yard.train import env_function
    env = env_function(env_module, 'make_env')(env_cfg, 0, 0)()
    try:
        obs, nvec = tuple(env.observation_space.shape), list(getattr(env.action_space, 'nvec', []))
    finally:
        env.close()
    if tuple(model.observation_space.shape) != obs:
        raise ValueError(f'the model observes {tuple(model.observation_space.shape)}, the env {obs}: trained on another '
                         'env version?')
    if list(getattr(model.action_space, 'nvec', [])) != nvec:
        raise ValueError(f'the model acts in {model.action_space}, the env in {nvec}')
    if normalize.obs_rms is not None and tuple(np.shape(normalize.obs_rms.mean)) != obs:
        raise ValueError(f'the VecNormalize statistics are {np.shape(normalize.obs_rms.mean)}, the env observes {obs}')


# ---------------------------------------------------------------------------------------------- results
def merge(ppo_rows, scripted_rows):
    """One per_seed.csv row per seed (in the policy's seed order) from the two nohesi.yard.train.summarize() lists."""
    scripted = {int(r['seed']): r for r in scripted_rows}
    out = []
    for p in ppo_rows:
        b = scripted[int(p['seed'])]
        if p['followers'] != b['followers']:
            warn(f"seed {p['seed']}: {p['followers']} followers for ppo, {b['followers']} for scripted")
        row = {'seed': int(p['seed']), 'followers': int(p['followers'])}
        for name, r in (('ppo', p), ('scripted', b)):
            row.update({f'{name}_{k}': r[k] for k in CSV_FIELDS})
        row['delta_duration_s'] = float(p['duration_s']-b['duration_s']) if p['success'] and b['success'] else None
        out.append(row)
    return out


def policy_summary(rows):
    """nohesi.yard.train.aggregate() plus the same by follower count."""
    from nohesi.yard.train import aggregate
    out = aggregate(rows)
    out['by_followers'] = {}
    for n in sorted({*FOLLOWER_COUNTS, *(int(r['followers']) for r in rows)}):
        sub = [r for r in rows if int(r['followers']) == n]
        a = aggregate(sub) if sub else {}
        out['by_followers'][str(n)] = {'episodes': len(sub), 'success_rate': a.get('success_rate'),
                                       'duration_s': a.get('duration_s'),
                                       'success_duration_s': a.get('success_duration_s')}
    return out


def _deltas(rows):
    d = np.array([r['delta_duration_s'] for r in rows if r['delta_duration_s'] is not None], float)
    n = len(d)
    std = float(d.std(ddof=1)) if n > 1 else (0.0 if n else None)
    return d, {'seeds_both_solved': n, 'delta_duration_mean_s': float(d.mean()) if n else None,
               'delta_duration_std_s': std,
               'delta_duration_ci95_s': 1.96*std/np.sqrt(n) if n > 1 else None,
               'delta_duration_median_s': float(np.median(d)) if n else None,
               'ppo_faster': int(np.sum(d < -TIE_S)), 'ppo_slower': int(np.sum(d > TIE_S)),
               'equal': int(np.sum(np.abs(d) <= TIE_S))}


def paired(rows):
    """ppo - scripted duration on the seeds both solved (negative: the policy is faster), overall and by follower
    count, and the seeds only one of them solved."""
    _, out = _deltas(rows)
    out['solved_by_ppo_only'] = [r['seed'] for r in rows if r['ppo_success'] and not r['scripted_success']]
    out['solved_by_scripted_only'] = [r['seed'] for r in rows if r['scripted_success'] and not r['ppo_success']]
    out['solved_by_neither'] = [r['seed'] for r in rows if not (r['scripted_success'] or r['ppo_success'])]
    out['by_followers'] = {}
    for n in sorted({*FOLLOWER_COUNTS, *(r['followers'] for r in rows)}):
        _, sub = _deltas([r for r in rows if r['followers'] == n])
        out['by_followers'][str(n)] = {k: sub[k] for k in ('seeds_both_solved', 'delta_duration_mean_s',
                                                           'ppo_faster', 'ppo_slower')}
    return out


def select_cases(rows):
    """[(case, seed)]: ppo_wins_most (the largest duration gain on a seed both solved, if the policy is faster
    anywhere), ppo_loses_most (the largest loss, if slower anywhere), ppo_fails (a policy failure: a collision first,
    then one the baseline solved, then the lowest seed)."""
    both, out = [r for r in rows if r['delta_duration_s'] is not None], []
    if both:
        win = min(both, key=lambda r: (r['delta_duration_s'], r['seed']))
        if win['delta_duration_s'] < -TIE_S:
            out.append(('ppo_wins_most', win['seed']))
        loss = max(both, key=lambda r: (r['delta_duration_s'], -r['seed']))
        if loss['delta_duration_s'] > TIE_S:
            out.append(('ppo_loses_most', loss['seed']))
    fails = [r for r in rows if not r['ppo_success']]
    if fails:
        out.append(('ppo_fails', min(fails, key=lambda r: (not r['ppo_collision'], not r['scripted_success'],
                                                           r['seed']))['seed']))
    return out


def case_entry(case, seed, rows, gif):
    r = next(r for r in rows if r['seed'] == seed)
    files = {p: [f'{case}_seed{seed}_{p}.png']+([f'{case}_seed{seed}_{p}.gif'] if gif else []) for p in POLICIES}
    return {'case': case, 'seed': seed, 'followers': r['followers'],
            **{p: {k: r[f'{p}_{k}'] for k in ('end_reason', 'parked', 'duration_s')} for p in POLICIES},
            'delta_duration_s': r['delta_duration_s'], 'files': files}


def _clean(x):
    """JSON-safe: numpy scalars to Python, NaN/inf to null."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def _csv_value(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ''
        return int(v) if float(v).is_integer() else round(float(v), 3)
    return '' if v is None else v


def write_csv(path, rows):
    columns = ['seed', 'followers', *(f'{p}_{k}' for p in POLICIES for k in CSV_FIELDS), 'delta_duration_s']
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _csv_value(r.get(k)) for k in columns})


def _f(v, spec='.2f'):
    return '-' if v is None else format(v, spec)


def _signed(v, spec='.1f'):
    return '-' if v is None else format(v, '+'+spec)


def markdown(s):
    """summary.md from the summary.json dict."""
    P, S, d, m = s['policies']['ppo'], s['policies']['scripted'], s['paired'], s['model']
    env = s['env_config']
    lines = [f"# Yard policy evaluation: {s['run']} / {m['label']}", '',
             f"- model: `{m['path']}` ({m['num_timesteps'] if m['num_timesteps'] is not None else '?'} timesteps, "
             f"sha256 {m['sha256'][:12]})",
             f"- VecNormalize: `{m['vecnormalize']}`" if m['vecnormalize'] else
             '- VecNormalize: **none** (raw observations)',
             f"- env: `{s['env_module']}` scenario {env.get('scenario')}, version {env.get('version')}, followers "
             f"{tuple(env.get('followers') or ()) or 'scenario default'} (config `{s['run_config'].get('config')}`)",
             f"- seeds: {seed_text(s['seeds'])} ({len(s['seeds'])}); ppo deterministic with action masks; scripted: "
             'the prioritized baseline (scripted_actions)',
             f"- {s['date']}, commit {(s['git'] or {}).get('commit', '?')}"
             f"{' (dirty)' if (s['git'] or {}).get('dirty') else ''}, {s['workers']} workers", '',
             '| | ppo | scripted |', '|---|---:|---:|']

    def row(label, key, spec='.2f'):
        lines.append(f'| {label} | {_f(P.get(key), spec)} | {_f(S.get(key), spec)} |')
    row('episodes', 'episodes', 'd')
    row('success rate', 'success_rate')
    row('collision rate', 'collision_rate')
    row('timeout rate', 'timeout_rate')
    row('stalled rate', 'stalled_rate')
    row('parked fraction', 'parked_frac')
    row('mean duration, all episodes (s)', 'duration_s', '.1f')
    row('mean duration, successes (s)', 'success_duration_s', '.1f')
    for n in P['by_followers']:
        cells = []
        for x in (P, S):
            b = x['by_followers'].get(n, {})
            cells.append(f"{_f(b.get('duration_s'), '.1f')} / {_f(b.get('success_duration_s'), '.1f')} "
                         f"(n {b.get('episodes', 0)})")
        lines.append(f"| mean duration, {n} follower{'s' if n != '1' else ''}: all / successes (s) | "
                     f'{cells[0]} | {cells[1]} |')
    row('safety stops / episode', 'safety_stops')
    row('replans / episode', 'replans')
    row('back-off / episode (m)', 'back_off_m', '.1f')

    seeds = lambda xs: ', '.join(map(str, xs)) or '-'
    ci = f" (95 % CI of the mean ±{_f(d['delta_duration_ci95_s'], '.1f')})" if d['delta_duration_ci95_s'] else ''
    lines += ['', '## Paired: ppo - scripted duration on the seeds both solved (negative: ppo faster)', '',
              '| | |', '|---|---|',
              f"| seeds both solved | {d['seeds_both_solved']} |",
              f"| mean ± std (s) | {_signed(d['delta_duration_mean_s'])} ± {_f(d['delta_duration_std_s'], '.1f')}{ci} |",
              f"| median (s) | {_signed(d['delta_duration_median_s'])} |",
              f"| ppo faster / slower / equal | {d['ppo_faster']} / {d['ppo_slower']} / {d['equal']} |",
              f"| solved by ppo only | {seeds(d['solved_by_ppo_only'])} |",
              f"| solved by scripted only | {seeds(d['solved_by_scripted_only'])} |",
              f"| solved by neither | {seeds(d['solved_by_neither'])} |", '',
              '## By follower count', '',
              '| followers | seeds | ppo success | scripted success | both solved | mean Δ (s) | ppo faster / slower |',
              '|---:|---:|---:|---:|---:|---:|---|']
    for n, b in d['by_followers'].items():
        p, q = P['by_followers'].get(n, {}), S['by_followers'].get(n, {})
        lines.append(f"| {n} | {p.get('episodes', 0)} | {_f(p.get('success_rate'))} | {_f(q.get('success_rate'))} | "
                     f"{b['seeds_both_solved']} | {_signed(b['delta_duration_mean_s'])} | "
                     f"{b['ppo_faster']} / {b['ppo_slower']} |")
    lines += ['', '## Rendered cases', '']
    if s['cases']:
        lines += ['| case | seed | followers | ppo | scripted | Δ (s) | files |', '|---|---:|---:|---|---|---:|---|']
        for c in s['cases']:
            out = lambda p: f"{c[p]['end_reason']}, {c[p]['parked']}/{c['followers']} parked, {c[p]['duration_s']:.1f} s"
            files = ' '.join(f'[{f}]({f})' for p in POLICIES for f in c['files'][p])
            lines.append(f"| {c['case']} | {c['seed']} | {c['followers']} | {out('ppo')} | {out('scripted')} | "
                         f"{_signed(c['delta_duration_s'])} | {files} |")
        lines.append('')
    shown = {c['case'] for c in s['cases']}
    absent = {'ppo_wins_most': 'ppo was faster on no seed both solved', 'ppo_loses_most': 'ppo was slower on no seed '
              'both solved', 'ppo_fails': 'ppo solved every seed'}
    lines += [f'- no {k}: {why}' for k, why in absent.items() if k not in shown]
    if s['warnings']:
        lines += ['', '## Warnings', '', *(f'- {w}' for w in s['warnings'])]
    return '\n'.join(lines)+'\n'


def write_outputs(out_dir, rows, summary):
    write_csv(out_dir/'per_seed.csv', rows)
    summary['warnings'] = list(WARNINGS)
    (out_dir/'summary.json').write_text(json.dumps(_clean(summary), indent=2)+'\n', encoding='utf-8')
    md = markdown(summary)
    (out_dir/'summary.md').write_text(md, encoding='utf-8')
    return md


def unique_dir(path):
    path, k = Path(path), 2
    out = path
    while out.exists():
        out, k = path.with_name(f'{path.name}_{k}'), k+1
    return out


# ---------------------------------------------------------------------------------------------- rendering
def render(job):
    """Replay job['seed'] with job['policy'] in this process, then viz.plot (and viz.animate) it. Returns the replay's
    nohesi.yard.train.summarize() row."""
    from nohesi.yard.train import env_function, summarize
    from nohesi.yard.viz import animate, plot
    env = env_function(job['env_module'], 'make_env')(job['env_cfg'], 0, 0)()
    if job['policy'] == 'ppo':
        import torch
        from sb3_contrib import MaskablePPO
        torch.set_num_threads(1)
        model, normalize = MaskablePPO.load(io.BytesIO(job['model_bytes']), device='cpu'), job['normalize']
        act = lambda obs: model.predict(normalize(obs[None]), action_masks=env.action_masks()[None],
                                        deterministic=True)[0][0]
    else:
        scripted = env_function(job['env_module'], 'scripted_actions')
        act = lambda obs: scripted(env)
    try:
        obs, _ = env.reset(seed=int(job['seed']))
        done, decisions, info = False, 0, {}
        while not done:
            obs, _, terminated, truncated, info = env.step(np.asarray(act(obs), dtype=np.int64))
            decisions, done = decisions+1, terminated or truncated
        ep = env.episode()
    finally:
        env.close()
    if is_dataclass(ep.scenario) and hasattr(ep.scenario, 'version'):   # viz titles show it: name the policy there
        ep.scenario = replace(ep.scenario, version=f"{ep.scenario.version} · {job['title']}")
    out = Path(job['out_dir'])
    for name in job['files']:
        (plot if name.endswith('.png') else animate)(ep, out/name)
    return summarize(info['episode_metrics'], decisions)


def render_cases(cases, rows, env_cfg, env_module, model_bytes, normalize, ppo_title, out_dir, workers):
    """Both policies' PNG/GIF for every case; a replay that differs from its evaluation row is warned about."""
    jobs = []
    for c in cases:
        for p in POLICIES:
            jobs.append({'case': c['case'], 'seed': c['seed'], 'policy': p, 'files': c['files'][p],
                         'title': ppo_title if p == 'ppo' else 'scripted baseline', 'env_cfg': env_cfg,
                         'env_module': env_module, 'out_dir': str(out_dir),
                         'model_bytes': model_bytes if p == 'ppo' else None,
                         'normalize': normalize if p == 'ppo' else None})
    names = ', '.join(f"{c['case']} seed {c['seed']}" for c in cases)
    print(f'[render] {len(jobs)} episodes ({names}) on {min(workers, len(jobs))} workers', flush=True)
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(render, job) for job in jobs]
            results = []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as e:   # noqa: BLE001 - one failed render must not cost the evaluation
                    results.append(e)
    else:
        results = []
        for job in jobs:
            try:
                results.append(render(job))
            except Exception as e:   # noqa: BLE001
                results.append(e)
    by_seed = {r['seed']: r for r in rows}
    for job, res in zip(jobs, results):
        tag = f"{job['case']} seed {job['seed']} {job['policy']}"
        if isinstance(res, Exception):
            warn(f'{tag}: rendering failed: {res!r}')
            continue
        r = by_seed[job['seed']]
        want = (r[f"{job['policy']}_end_reason"], r[f"{job['policy']}_parked"], r[f"{job['policy']}_duration_s"])
        if (res['end_reason'], res['parked']) != want[:2] or abs(res['duration_s']-want[2]) > TIE_S:
            warn(f'{tag}: the replay ({res["end_reason"]}, {res["parked"]} parked, {res["duration_s"]:.1f} s) differs '
                 f'from the evaluation ({want[0]}, {want[1]} parked, {want[2]:.1f} s): the figures show the replay')


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default=RUN, help=f'run folder (default {RUN})')
    p.add_argument('--model', default=MODEL, help="'final', 'best', 'latest', a checkpoint timestep or a model .zip "
                                                 f'(default {MODEL})')
    p.add_argument('--vecnorm', help='VecNormalize .pkl (default: the one saved next to the model)')
    p.add_argument('--seeds', default=SEEDS, help=f"'1000-1049', '1000,1003' or '1000-1009,1020' (default {SEEDS})")
    p.add_argument('--workers', type=int, default=WORKERS, help=f'parallel processes per stage (default {WORKERS})')
    p.add_argument('--no-gif', dest='gif', action='store_false', default=GIF, help='render the cases as PNGs only')
    p.add_argument('--out-root', default=OUT_ROOT, help=f'parent of the output folder (default {OUT_ROOT}/)')
    a = p.parse_args(argv)
    WARNINGS.clear()
    from nohesi.rl.evaluate import resolve
    from nohesi.rl.train import git_commit
    from nohesi.yard import train as yt

    start, wall = time.time(), {}
    seeds, workers = parse_seeds(a.seeds), max(1, a.workers)
    run_dir = find(a.run)
    model_path, vecnorm_path = resolve_model(run_dir, a.model, a.vecnorm)
    label = model_label(model_path)
    cfg_dir = config_dir(model_path, run_dir)
    env_cfg, env_module, info = load_run_config(cfg_dir)
    if cfg_dir is not None and cfg_dir.resolve() not in model_path.resolve().parents:
        warn(f'{_rel(model_path)} has no config.json of its own: the env config is the run folder\'s, '
             f'{_rel(cfg_dir/"config.json")}')
    run_name = (cfg_dir or model_path.parent).name

    trained_on = [s for s in seeds if s >= TRAIN_SEEDS_FROM]
    if trained_on:
        warn(f'{len(trained_on)} seeds >= {TRAIN_SEEDS_FROM} ({seed_text(trained_on)}) are in the training seed range')
    overlap = sorted(set(seeds) & set(info.get('eval_seeds') or ()))
    if label == 'best' and overlap:
        warn(f'best_model.zip was selected on the run\'s eval_seeds; {len(overlap)} of the seeds here '
             f'({seed_text(overlap)}) are among them, so its results there are optimistic')
    commit = (info.get('git') or {}).get('commit')
    if code_changed_since(commit):
        warn(f"{', '.join(YARD_CODE)} changed since the run's training commit {commit[:10]}: the policy plays today's env")

    model_bytes, model = read_model(model_path)
    normalize = load_normalizer(vecnorm_path, resolve(model_path)[1], info.get('norm_obs', True))
    check_spaces(model, normalize, env_cfg, env_module)
    timesteps = int(getattr(model, 'num_timesteps', 0)) or None
    ppo_title = f"PPO {label}{f' (t {timesteps})' if timesteps and label in ('final', 'best') else ''}"

    out_dir = unique_dir(find(a.out_root)/f'yard_eval_{run_name}_{label}_{date.today().isoformat()}')
    out_dir.mkdir(parents=True)
    print(f'model {_rel(model_path)} ({timesteps} timesteps) | VecNormalize {_rel(vecnorm_path) or "none"} | env '
          f'{env_module} {env_cfg} | seeds {seed_text(seeds)} ({len(seeds)}) | {workers} workers\noutputs in {out_dir}',
          flush=True)

    t = time.time()
    print(f'[ppo] {len(seeds)} episodes ...', flush=True)
    policy = yt.evaluate(model, normalize, env_cfg, seeds, n_envs=min(workers, len(seeds)), subproc=workers > 1,
                         env_module=env_module)
    wall['ppo_s'] = round(time.time()-t, 1)
    print(f"[ppo] {yt._line(policy['summary'])} | {wall['ppo_s']:.0f} s", flush=True)
    t = time.time()
    print(f'[scripted] {len(seeds)} episodes ...', flush=True)
    base = yt.baseline(env_cfg, seeds, workers=workers, env_module=env_module)
    wall['scripted_s'] = round(time.time()-t, 1)
    print(f"[scripted] {yt._line(base['summary'])} | {wall['scripted_s']:.0f} s", flush=True)

    rows = merge(policy['per_seed'], base['per_seed'])
    cases = [case_entry(c, s, rows, a.gif) for c, s in select_cases(rows)]
    summary = {'run': run_name, 'run_dir': _rel(run_dir),
               'model': {'label': label, 'path': _rel(model_path), 'sha256': hashlib.sha256(model_bytes).hexdigest(),
                         'num_timesteps': timesteps, 'vecnormalize': _rel(vecnorm_path),
                         'normalized': normalize.obs_rms is not None},
               'env_module': env_module, 'env_config': asdict(env_cfg), 'run_config': info, 'seeds': seeds,
               'workers': workers, 'deterministic': True, 'date': date.today().isoformat(), 'git': git_commit(),
               'policies': {'ppo': policy_summary(policy['per_seed']), 'scripted': policy_summary(base['per_seed'])},
               'paired': paired(rows), 'cases': cases, 'wall_s': wall}
    md = write_outputs(out_dir, rows, summary)   # the numbers are safe before the (slower) rendering

    if cases:
        t = time.time()
        render_cases(cases, rows, env_cfg, env_module, model_bytes, normalize, ppo_title, out_dir, workers)
        wall['render_s'] = round(time.time()-t, 1)
    wall['total_s'] = round(time.time()-start, 1)
    md = write_outputs(out_dir, rows, summary)
    print('\n'+md)
    print(f'outputs in {out_dir}')
    return summary


if __name__ == '__main__':
    main()
