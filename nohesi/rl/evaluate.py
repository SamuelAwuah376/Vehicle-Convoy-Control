"""Evaluate a trained MaskablePPO policy on held-out seeds and compare it with the scripted overtaker.

    python -m nohesi.rl.evaluate runs_rl/<run_id>                      # best model, seeds 1000-1019, + overtaker
    python -m nohesi.rl.evaluate runs_rl/<run_id> --mpc nonlinear --vehicle dynamic   # transfer evaluation
    python -m nohesi.rl.evaluate runs_rl/<run_id> --store              # also save the first seed to results/ with a GIF

Policy actions are deterministic with the env's action masks applied; observations are normalised with the
VecNormalize statistics saved next to the model (training=False, norm_reward=False).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_SEEDS = tuple(range(1000, 1020))
SUMMARY_FIELDS = ('team_score', 'collision', 'close_passes', 'draft_passes', 'train_passes', 'draft_time_s',
                  'train_time_s', 'mean_speed_mps', 'top_speed_time_s', 'maneuvers', 'fallbacks', 'rejections', 'decisions',
                  'duration_s')


# ---------------------------------------------------------------------------------------------- metrics
def _total(x):
    """Sum of a count dict / list / number (None -> 0)."""
    if x is None:
        return 0.0
    if isinstance(x, dict):
        return float(sum(_total(v) for v in x.values()))
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(np.sum(x))
    return float(x)


def _draft_time(m):
    """Team drafting time (car-seconds summed over the three cars) from metrics()['draft_time_s'], None if the
    score has no draft term."""
    value = m.get('draft_time_s')
    return None if value is None else _total(value)


def summarize(m, decisions=None):
    """Flat per-episode numbers from a nohesi metrics() dict."""
    planner = m.get('planner') or {}
    return {'seed': m.get('seed'), 'team_score': float(m['team_score']),
            'collision': bool(m.get('collisions')) or m.get('end_reason') == 'collision',
            'close_passes': int(m.get('close_passes_scored', 0)),
            'draft_passes': int(m.get('draft_passes_scored', 0)),
            'train_passes': int(m.get('train_passes_scored', 0)),
            'train_time_s': float(m.get('train_time_s', 0.0)),
            'mean_speed_mps': m.get('mean_speed_mps'),
            'top_speed_time_s': _total(m.get('top_speed_time_s')),
            'draft_time_s': _draft_time(m),
            'maneuvers': _total((m.get('maneuvers') or {}).get('count')),
            'fallbacks': _total(planner.get('fallbacks')), 'rejections': _total(planner.get('rejections')),
            'decisions': decisions, 'duration_s': float(m.get('duration_s', np.nan))}


def aggregate(rows):
    """Mean (and std of team score) over per-episode summaries; collision -> collision_rate."""
    out = {'episodes': len(rows)}
    for f in SUMMARY_FIELDS:
        vals = [r[f] for r in rows if r.get(f) is not None]
        key = 'collision_rate' if f == 'collision' else f
        out[key] = float(np.mean(vals)) if vals else None
    scores = [r['team_score'] for r in rows]
    out['team_score_std'] = float(np.std(scores)) if scores else None
    return out


# ---------------------------------------------------------------------------------------------- policy rollouts
class ObsNormalizer:
    """Applies saved VecNormalize observation statistics without a VecNormalize wrapper."""

    def __init__(self, obs_rms=None, clip_obs=10.0, epsilon=1e-8):
        self.obs_rms, self.clip_obs, self.epsilon = obs_rms, clip_obs, epsilon

    @classmethod
    def from_vecnormalize(cls, vecnorm):
        return cls(vecnorm.obs_rms if vecnorm.norm_obs else None, vecnorm.clip_obs, vecnorm.epsilon)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as stream:
            vecnorm = pickle.load(stream)   # VecNormalize pickles without its venv
        return cls.from_vecnormalize(vecnorm)

    def __call__(self, obs):
        if self.obs_rms is None:
            return obs
        return np.clip((obs-self.obs_rms.mean)/np.sqrt(self.obs_rms.var+self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)


def make_eval_vec_env(env_cfg, n_envs, subproc=True):
    """Raw (unnormalised) eval envs; seeds are set explicitly per episode by run_policy."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from .env import make_env
    thunks = [make_env(env_cfg, rank, 0) for rank in range(n_envs)]
    return SubprocVecEnv(thunks) if subproc and n_envs > 1 else DummyVecEnv(thunks)


def run_policy(model, normalize, vec_env, seeds, deterministic=True):
    """Run one episode per seed on vec_env (seeds handed to free envs as they finish).

    Returns [(seed, episode_metrics, decisions, summed aux, summed score terms)] in seed order."""
    from sb3_contrib.common.maskable.utils import get_action_masks
    n, queue = vec_env.num_envs, list(seeds)
    active, steps, results = [None]*n, np.zeros(n, int), {}
    obs = np.zeros((n,)+vec_env.observation_space.shape, np.float32)

    def start(i):
        seed = queue.pop(0)
        o, _ = vec_env.env_method('reset', seed=int(seed), indices=[i])[0]
        obs[i], active[i], steps[i] = o, seed, 0

    for i in range(min(n, len(queue))):
        start(i)
    while any(a is not None for a in active):
        masks = get_action_masks(vec_env)
        actions, _ = model.predict(normalize(obs), action_masks=masks, deterministic=deterministic)
        next_obs, _, dones, infos = vec_env.step(actions)
        obs[:] = next_obs
        for i in range(n):
            if active[i] is None:
                continue
            steps[i] += 1
            if dones[i]:
                info = infos[i]
                if 'episode_metrics' not in info:
                    raise RuntimeError("env did not report 'episode_metrics' at episode end")
                results[active[i]] = (active[i], info['episode_metrics'], int(steps[i]))
                active[i] = None
                if queue:
                    start(i)
    return [results[s] for s in seeds]


def evaluate_model(model, normalize, env_cfg, seeds=EVAL_SEEDS, n_envs=None, vec_env=None, subproc=True):
    """Per-seed summaries and aggregate for an in-memory model; creates (and closes) eval envs unless given."""
    own = vec_env is None
    if own:
        vec_env = make_eval_vec_env(env_cfg, n_envs or min(len(seeds), 10), subproc)
    try:
        rows = [summarize(m, d) for _, m, d in run_policy(model, normalize, vec_env, list(seeds))]
    finally:
        if own:
            vec_env.close()
    return {'per_seed': rows, 'summary': aggregate(rows)}


# ---------------------------------------------------------------------------------------------- scripted baseline
def scenario_for(env_cfg):
    """(scenario, planner, score) matching the RL env's configuration, for scripted runs."""
    from ..config import SCENARIOS, SCORES, Planner
    try:   # the env's own builder, if it exposes one, is authoritative
        from . import env as rl_env
        if hasattr(rl_env, 'build_scenario'):
            scenario, planner = rl_env.build_scenario(env_cfg)
            return scenario, planner, scenario.score
    except ImportError:
        pass
    scenario = SCENARIOS[env_cfg.scenario]()
    scenario = replace(scenario, vehicle=replace(scenario.vehicle, model=env_cfg.vehicle_model))
    planner = replace(Planner(), mpc=env_cfg.mpc)
    return scenario, planner, SCORES[env_cfg.score]()


def _overtaker_one(args):
    env_cfg, seed = args
    from ..evaluation.metrics import metrics
    from ..evaluation.runner import run
    scenario, planner, score = scenario_for(env_cfg)
    ep = run(scenario, seed, 'overtaker', planner, score=score)
    return summarize(metrics(ep), decisions=int(round(ep.time[-1]/planner.decision_interval)))


def overtaker_baseline(env_cfg, seeds=EVAL_SEEDS, workers=None, cache_dir=ROOT/'runs_rl'/'baselines'):
    """Scripted overtaker on the same seeds/scenario/score; cached as JSON under runs_rl/baselines/."""
    seeds = [int(s) for s in seeds]
    tag = '_'.join(f'{k}-{v}' for k, v in sorted(asdict(env_cfg).items())
                   if k in ('scenario', 'score', 'vehicle_model', 'mpc', 'version'))
    path = Path(cache_dir)/f'overtaker_{tag}_seeds{seeds[0]}-{seeds[-1]}_{len(seeds)}.json'
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached.get('seeds') == seeds:
            return cached
    with ProcessPoolExecutor(max_workers=workers or min(len(seeds), 16)) as pool:
        rows = list(pool.map(_overtaker_one, [(env_cfg, s) for s in seeds]))
    out = {'seeds': seeds, 'env_cfg': asdict(env_cfg), 'per_seed': rows, 'summary': aggregate(rows)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    return out


# ---------------------------------------------------------------------------------------------- loading
def load_env_cfg(run_dir):
    """EnvConfig from runs_rl/<run>/config.json (unknown keys dropped)."""
    from .config import EnvConfig
    saved = json.loads((Path(run_dir)/'config.json').read_text(encoding='utf-8'))['train']['env']
    names = {f.name for f in fields(EnvConfig)}
    return EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in names})


def resolve(run_or_model, which='best'):
    """(model_path, vecnorm_path) for a run directory (best/final) or an explicit model .zip."""
    p = Path(run_or_model)
    if p.is_dir():
        model = p/('best_model.zip' if which == 'best' else 'final_model.zip')
        return model, p/('best_vecnormalize.pkl' if which == 'best' else 'final_vecnormalize.pkl')
    return p, p.with_name(p.stem.replace('model', 'vecnormalize')+'.pkl')


def load_policy(model_path, vecnorm_path):
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_path), device='cpu')
    return model, ObsNormalizer.load(vecnorm_path)


# ---------------------------------------------------------------------------------------------- store one episode
def store_episode(model, normalize, env_cfg, seed, label, root=ROOT/'results', policy='ppo'):
    """Run one deterministic episode in-process and save it like run_sim.py (numbered results/ run + GIF)."""
    from ..evaluation.io import append_index, next_run_dir, save
    from ..evaluation.viz import animate, plot
    from .env import NoHesiEnv
    env = NoHesiEnv(env_cfg)
    obs, _ = env.reset(seed=int(seed))
    done = False
    while not done:
        action, _ = model.predict(normalize(obs[None]), action_masks=env.action_masks()[None], deterministic=True)
        obs, _, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated
    episode = finished_episode(env)
    episode.policy = policy
    out = next_run_dir(root, label)
    m = save(episode, out)
    plot(episode, out/'summary.png')
    animate(episode, out/'team.gif')
    append_index(out, m)
    env.close()
    return out, m


def finished_episode(env):
    """The nohesi Episode of the env's last finished episode."""
    for name in ('episode', 'last_episode'):
        ep = getattr(env, name, None)
        ep = ep() if callable(ep) else ep
        if ep is not None:
            return ep
    for name in ('build_episode', 'get_episode', 'to_episode'):
        if hasattr(env, name):
            return getattr(env, name)()
    raise AttributeError('NoHesiEnv exposes no finished Episode (expected env.episode)')


# ---------------------------------------------------------------------------------------------- API + CLI
def evaluate(model_path, vecnorm_path, env_cfg, seeds=EVAL_SEEDS, n_envs=None, baseline=True, store=False,
             label=None, subproc=True):
    """Per-seed metrics + summary for a saved policy; with baseline, the scripted overtaker on the same seeds
    and the per-seed score difference; with store, the first seed is saved to results/ with a GIF."""
    import torch
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model, normalize = load_policy(model_path, vecnorm_path)
    out = {'model': str(model_path), 'vecnormalize': str(vecnorm_path), 'env_cfg': asdict(env_cfg),
           'seeds': [int(s) for s in seeds], **evaluate_model(model, normalize, env_cfg, seeds, n_envs, subproc=subproc)}
    if baseline:
        base = overtaker_baseline(env_cfg, seeds)
        out['overtaker'] = {'per_seed': base['per_seed'], 'summary': base['summary']}
        out['ppo_minus_overtaker'] = {'team_score_mean': out['summary']['team_score']-base['summary']['team_score'],
                                      'seeds_won': int(sum(p['team_score'] > b['team_score']
                                                           for p, b in zip(out['per_seed'], base['per_seed'])))}
    if store:
        seed = int(seeds[0])
        label = label or f'{env_cfg.scenario}_ppo-{Path(model_path).parent.name}_score{env_cfg.score}_seed{seed}' + \
            (f'_{env_cfg.vehicle_model}' if env_cfg.vehicle_model != 'kinematic' else '') + \
            ('_nmpc' if env_cfg.mpc == 'nonlinear' else '')
        path, _ = store_episode(model, normalize, env_cfg, seed, label, policy=f'ppo-{Path(model_path).parent.name}')
        out['stored'] = str(path)
    return out


def _seed_list(text):
    if '-' in text:
        lo, hi = map(int, text.split('-'))
        return list(range(lo, hi+1))
    return [int(s) for s in text.split(',')]


def print_report(out):
    keys = ('team_score', 'team_score_std', 'collision_rate', 'close_passes', 'draft_passes', 'train_passes',
            'draft_time_s', 'train_time_s', 'mean_speed_mps', 'top_speed_time_s', 'maneuvers',
            'fallbacks', 'rejections', 'decisions')
    cols = [('ppo', out['summary'])]+([('overtaker', out['overtaker']['summary'])] if 'overtaker' in out else [])
    print(f"{'':16}"+''.join(f'{name:>12}' for name, _ in cols))
    for k in keys:
        print(f'{k:16}'+''.join(f"{s[k]:12.2f}" if s.get(k) is not None else f"{'-':>12}" for _, s in cols))
    if 'ppo_minus_overtaker' in out:
        d = out['ppo_minus_overtaker']
        print(f"PPO - overtaker: {d['team_score_mean']:+.1f} mean team score, PPO higher on {d['seeds_won']}/{len(out['seeds'])} seeds")
    if 'stored' in out:
        print(f"stored episode: {out['stored']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', help='runs_rl/<run_id> directory or a model .zip')
    p.add_argument('--which', choices=('best', 'final'), default='best')
    p.add_argument('--vecnorm', help='VecNormalize .pkl (default: next to the model)')
    p.add_argument('--seeds', default='1000-1019', help="'1000-1019' or '1000,1003'")
    p.add_argument('--envs', type=int, default=None, help='parallel eval envs (default min(#seeds, 10))')
    p.add_argument('--mpc', choices=('linear', 'nonlinear'))
    p.add_argument('--vehicle', choices=('kinematic', 'dynamic'))
    p.add_argument('--scenario')
    p.add_argument('--score')
    p.add_argument('--no-baseline', action='store_true')
    p.add_argument('--store', action='store_true', help='save the first seed to results/ with GIF')
    p.add_argument('--out', help='write the full result JSON here (default <run>/eval_<tag>.json)')
    a = p.parse_args(argv)
    model_path, vecnorm_path = resolve(a.run, a.which)
    vecnorm_path = Path(a.vecnorm) if a.vecnorm else vecnorm_path
    run_dir = Path(a.run) if Path(a.run).is_dir() else model_path.parent
    if (run_dir/'config.json').exists():
        env_cfg = load_env_cfg(run_dir)
    else:
        from .config import EnvConfig
        env_cfg = EnvConfig()
    overrides = {k: v for k, v in (('mpc', a.mpc), ('vehicle_model', a.vehicle), ('scenario', a.scenario),
                                   ('score', a.score)) if v}
    env_cfg = replace(env_cfg, **overrides)
    out = evaluate(model_path, vecnorm_path, env_cfg, _seed_list(a.seeds), a.envs, not a.no_baseline, a.store)
    print_report(out)
    tag = '_'.join(f'{k}-{v}' for k, v in overrides.items()) or 'default'
    path = Path(a.out) if a.out else run_dir/f'eval_{a.which}_{tag}.json'
    path.write_text(json.dumps(out, indent=2, default=str)+'\n', encoding='utf-8')
    print(f'results in {path}')


if __name__ == '__main__':
    main()
