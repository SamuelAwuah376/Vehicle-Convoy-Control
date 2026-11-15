"""MaskablePPO training for the yard parking env (nohesi.yard.env.YardEnv).

    python -m nohesi.yard.train --run-id yard_ppo_001 --timesteps 2000000 --envs 8
    python -m nohesi.yard.train --run-id yard_ppo_001 --resume runs_rl/yard_ppo_001/checkpoints/model_1000000.zip
    python -m nohesi.yard.train --run-id yard_dbg --timesteps 20000 --envs 2 --no-subproc

--resume continues the same run from a checkpoint (timestep count, LR schedule and reward normalisation carry on); its
settings are reloaded from runs_rl/<run_id>/config.json.

Everything for a run goes to runs_rl/<run_id>/:
    config.json                 YardTrainConfig + YardEnvConfig, the env functions used, observation and action sizes,
                                library versions, git commit (a resume adds resume_<time>.json instead)
    tb/                         TensorBoard: SB3 train/* and rollout/*, plus episode/* (training episodes, mean per
                                rollout) and eval/* (the held-out evaluations)
    checkpoints/model_<t>.zip + vecnormalize_<t>.pkl   every checkpoint_every timesteps
    best_model.zip + best_vecnormalize.pkl             best held-out evaluation (rank(): success rate first)
    final_model.zip + final_vecnormalize.pkl
    eval_history.csv            one row per evaluation of the policy on eval_seeds (policy 'ppo'), and one row for the
                                scripted baseline on the same seeds (policy 'scripted', recorded once per run)
    baseline.json               the scripted baseline's per-seed results and summary
    hangs/                      step-watchdog dumps from the workers (nohesi.rl.watchdog), if any

Timesteps are env decisions, summed over all envs. Policy evaluations are deterministic with the env's action masks
applied, observations normalised with the training VecNormalize statistics.

Env injection. This module never imports the env at import time; it talks to an "env module", any module providing
    YardEnvConfig                                        the env config (frozen dataclass): YardTrainConfig.env's default
    make_env(cfg, rank=0, base_seed=0, seed_pool=None)   a thunk building one env (pickled into SubprocVecEnv workers)
    scripted_actions(env) -> (3,) int                    the scripted baseline's actions for env's current decision
named by YardTrainConfig.env_module (default 'nohesi.yard.env', imported when first needed). The envs follow the YardEnv
contract (docs/yard_interfaces.md): reset(seed=s) plays seed s, action_masks() -> flat bool, info['episode_metrics'] at
each episode's end. train(), evaluate() and baseline() also take make_env= and scripted_fn= arguments, which override
the module's functions (tests/test_yard_train.py runs everything on a fake env that way); config.json records the
functions actually used.

Threads: as in nohesi.rl.train, env workers are single-threaded (OMP/MKL/OpenBLAS = 1) and the learner runs torch with
1 thread while collecting rollouts and torch_threads during the PPO update.
"""
import argparse
import csv
import functools
import importlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import numpy as np

# Generic pieces of the road team's trainer (all torch- and env-free at import, so spawned workers stay lean).
from ..rl.evaluate import ObsNormalizer, _total, run_policy
from ..rl.train import (LinearSchedule, _jsonable, _vecnorm_path, close_vec_env, git_commit, single_thread_workers,
                        versions)
from ..rl.vec_worker import Thunk, subproc_vec_env

ROOT = Path(__file__).resolve().parents[2]
ENV_MODULE = 'nohesi.yard.env'
EVAL_SEEDS = tuple(range(1000, 1020))
END_REASONS = ('parked', 'collision', 'timeout', 'stalled')
COUNT_FIELDS = ('safety_stops', 'replans', 'replans_failed', 'back_off_m', 'picks_rejected', 'reassigned',
                'path_length_m', 'driven_m')
EPISODE_TB_FIELDS = ('success', 'parked_frac', 'collision', 'timeout', 'stalled', 'duration_s', *COUNT_FIELDS)
EVAL_FIELDS = ('timesteps', 'wall_s', 'policy', 'episodes', 'success_rate', 'parked_frac', 'collision_rate',
               'timeout_rate', 'stalled_rate', 'duration_s', 'duration_s_std', 'success_duration_s', *COUNT_FIELDS,
               'decisions', 'best')


def env_function(module, name, given=None):
    """`given` if set, else `name` from the env module (imported now: the lazy import of nohesi.yard.env)."""
    return given if given is not None else getattr(importlib.import_module(module), name)


def _qualname(obj):
    return f'{getattr(obj, "__module__", "?")}.{getattr(obj, "__qualname__", repr(obj))}'


@dataclass
class YardTrainConfig:
    run_id: str = 'yard_ppo'
    env: object = None               # env_module's YardEnvConfig (None: its defaults)
    env_module: str = ENV_MODULE     # provides YardEnvConfig, make_env, scripted_actions (module docstring)
    n_envs: int = 8
    subproc: bool = True             # SubprocVecEnv workers (False: DummyVecEnv in-process, for tests/debugging)
    total_timesteps: int = 2_000_000
    # PPO
    n_steps: int = 256               # per env: 8 x 256 = 2048 decisions per update
    batch_size: int = 512
    n_epochs: int = 10
    learning_rate: float = 3e-4      # linearly decayed to lr_final over the run
    lr_final: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    net_pi: tuple = (256, 256)
    net_vf: tuple = (256, 256)
    activation: str = 'tanh'         # 'tanh' or 'relu'
    # VecNormalize
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    # evaluation / checkpoints
    eval_every: int = 50_000         # timesteps between held-out evaluations
    eval_seeds: tuple = EVAL_SEEDS   # held-out seeds (training episodes draw seeds >= 2000)
    eval_envs: int = 10              # parallel eval envs (and scripted-baseline worker processes)
    eval_at_start: bool = True       # evaluate the untrained policy at t=0
    baseline: bool = True            # run the scripted baseline once on eval_seeds (eval_history.csv, baseline.json)
    checkpoint_every: int = 100_000
    seed: int = 0                    # PPO/torch seed; the env RNGs are seeded 2000 + 1000*seed + rank
    torch_threads: int = 4
    device: str = 'cpu'
    out_root: str = 'runs_rl'
    resume: str = None               # checkpoint .zip to continue from (its vecnormalize_<t>.pkl alongside)
    verbose: int = 1

    def __post_init__(self):
        if self.env is None:
            self.env = env_function(self.env_module, 'YardEnvConfig')()
        self.eval_seeds = tuple(int(s) for s in self.eval_seeds)
        self.net_pi, self.net_vf = tuple(self.net_pi), tuple(self.net_vf)
        if self.n_envs*self.n_steps % self.batch_size:
            raise ValueError('n_envs*n_steps must be a multiple of batch_size')
        if self.activation not in ('tanh', 'relu'):
            raise ValueError("activation must be 'tanh' or 'relu'")

    @property
    def run_dir(self):
        root = Path(self.out_root)
        return (root if root.is_absolute() else ROOT/root)/self.run_id


# ---------------------------------------------------------------------------------------------- metrics
def _flag(x):
    return bool(np.any(x)) if isinstance(x, (list, tuple, np.ndarray)) else bool(x)


def summarize(m, decisions=None):
    """Flat per-episode numbers from a YardEnv info['episode_metrics'] dict (counts given per follower are summed)."""
    parked, followers = int(_total(m.get('parked'))), int(_total(m.get('followers')))
    end_reason = str(m.get('end_reason', ''))
    out = {'seed': None if m.get('seed') is None else int(m['seed']), 'success': _flag(m.get('success')),
           'parked': parked, 'followers': followers, 'parked_frac': parked/followers if followers else None,
           'collision': _flag(m.get('collision')) or end_reason == 'collision', 'end_reason': end_reason,
           'duration_s': np.nan if m.get('duration_s') is None else float(m['duration_s']), 'decisions': decisions}
    out.update({k: _total(m.get(k)) for k in COUNT_FIELDS})
    return out


def aggregate(rows):
    """Rates and means over per-episode summaries. parked_frac: parked trucks / followers over all episodes;
    success_duration_s: mean duration of the successful episodes (None if none)."""
    n = len(rows)

    def mean(key, subset=rows):
        vals = [float(r[key]) for r in subset if r.get(key) is not None and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None
    followers = sum(r['followers'] for r in rows)
    reasons = Counter(r['end_reason'] for r in rows)
    durations = [r['duration_s'] for r in rows if np.isfinite(r['duration_s'])]
    out = {'episodes': n, 'success_rate': mean('success'),
           'parked_frac': sum(r['parked'] for r in rows)/followers if followers else None,
           'collision_rate': mean('collision'),
           'timeout_rate': reasons['timeout']/n if n else None, 'stalled_rate': reasons['stalled']/n if n else None,
           'duration_s': mean('duration_s'), 'duration_s_std': float(np.std(durations)) if durations else None,
           'success_duration_s': mean('duration_s', [r for r in rows if r['success']])}
    out.update({k: mean(k) for k in (*COUNT_FIELDS, 'decisions')})
    out['end_reasons'] = dict(reasons)
    return out


def rank(summary):
    """Best-model key (higher is better): the success rate, then a shorter mean duration of the successful episodes;
    while no episode succeeds, the parked fraction. Rounded as eval_history.csv stores it, so a resumed run compares
    alike."""
    def get(key):
        value = summary.get(key)
        return 0.0 if value in (None, '') else round(float(value), 4)
    success = get('success_rate')
    if success > 0:
        return (success, 0.0, -get('success_duration_s'))
    return (0.0, get('parked_frac'), 0.0)


def _line(s):
    def f(key, fmt):
        return '-' if s.get(key) is None else format(s[key], fmt)
    return (f"success {f('success_rate', '.2f')} | parked {f('parked_frac', '.2f')} | "
            f"collisions {f('collision_rate', '.2f')} | duration {f('duration_s', '.0f')} s "
            f"(successes {f('success_duration_s', '.0f')} s) | safety stops {f('safety_stops', '.1f')} | "
            f"replans {f('replans', '.1f')} | len {f('decisions', '.0f')}")


# ---------------------------------------------------------------------------------------------- envs, rollouts
def make_vec_env(make_env, env_cfg, n_envs, base_seed, subproc=True):
    """n_envs envs from make_env(env_cfg, rank, base_seed): nohesi.rl.vec_worker's torch-free SubprocVecEnv (it
    replaces dead workers), or a DummyVecEnv in-process."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    thunks = [make_env(env_cfg, rank, base_seed) for rank in range(n_envs)]
    return subproc_vec_env(thunks) if subproc and n_envs > 1 else DummyVecEnv(thunks)


def evaluate(model, normalize, env_cfg, seeds=EVAL_SEEDS, *, vec_env=None, n_envs=None, subproc=True,
             make_env=None, env_module=ENV_MODULE, deterministic=True):
    """One masked episode of `model` per seed (deterministic by default); `normalize` maps raw observations to the
    policy's (ObsNormalizer; None: raw). Uses vec_env (raw, unnormalised envs) if given, else builds and closes
    n_envs (default min(#seeds, 10)) envs from make_env / env_module. Returns {'per_seed': [summarize()],
    'summary': aggregate()}."""
    own = vec_env is None
    if own:
        vec_env = make_vec_env(env_function(env_module, 'make_env', make_env), env_cfg,
                               n_envs or min(len(seeds), 10), 0, subproc)
    try:
        rows = [summarize(m, d) for _, m, d in run_policy(model, normalize or ObsNormalizer(), vec_env, list(seeds),
                                                          deterministic)]
    finally:
        if own:
            vec_env.close()
    return {'per_seed': rows, 'summary': aggregate(rows)}


def run_scripted(make_env, scripted_fn, env_cfg, seed):
    """One episode on `seed` with scripted_fn(env) choosing every decision, in this process; its summarize() row."""
    env = make_env(env_cfg, 0, 0)()
    try:
        env.reset(seed=int(seed))
        decisions, done, info = 0, False, {}
        while not done:
            _, _, terminated, truncated, info = env.step(np.asarray(scripted_fn(env), dtype=np.int64))
            decisions, done = decisions+1, terminated or truncated
        if 'episode_metrics' not in info:
            raise RuntimeError("env did not report 'episode_metrics' at episode end")
        return summarize(info['episode_metrics'], decisions)
    finally:
        env.close()


def _call(thunk):
    return thunk.fn()


def baseline(env_cfg, seeds=EVAL_SEEDS, *, workers=1, make_env=None, scripted_fn=None, env_module=ENV_MODULE):
    """The scripted baseline (scripted_fn, default env_module's scripted_actions) on the same seeds as evaluate(),
    with the same summary: {'seeds', 'per_seed', 'summary'}. workers > 1: episodes in parallel spawned processes
    (the functions travel cloudpickled)."""
    make_env = env_function(env_module, 'make_env', make_env)
    scripted_fn = env_function(env_module, 'scripted_actions', scripted_fn)
    seeds = [int(s) for s in seeds]
    jobs = [functools.partial(run_scripted, make_env, scripted_fn, env_cfg, seed) for seed in seeds]
    if workers <= 1 or len(jobs) <= 1:
        rows = [job() for job in jobs]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        single_thread_workers()
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
            rows = list(pool.map(_call, [Thunk(job) for job in jobs]))
    return {'seeds': seeds, 'per_seed': rows, 'summary': aggregate(rows)}


# ---------------------------------------------------------------------------------------------- run files
def write_config(cfg, run_dir, extra=None):
    """config.json: the train config with the env config inlined, provenance, and `extra`."""
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    d['env'] = asdict(cfg.env)
    out = {'train': _jsonable(d), 'env_config_type': _qualname(type(cfg.env)), 'versions': versions(),
           'git': git_commit(), 'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'argv': sys.argv, **(extra or {})}
    (Path(run_dir)/'config.json').write_text(json.dumps(out, indent=2, default=str)+'\n', encoding='utf-8')


def _tupled(x):
    return tuple(_tupled(v) for v in x) if isinstance(x, list) else x


def load_train_config(run_dir):
    """YardTrainConfig from runs_rl/<run>/config.json (its env_module's YardEnvConfig; unknown keys dropped)."""
    saved = json.loads((Path(run_dir)/'config.json').read_text(encoding='utf-8'))['train']
    env_cls = env_function(saved.get('env_module', ENV_MODULE), 'YardEnvConfig')
    env_names = {f.name for f in fields(env_cls)}
    env = env_cls(**{k: _tupled(v) for k, v in saved.pop('env').items() if k in env_names})
    names = {f.name for f in fields(YardTrainConfig)}
    return YardTrainConfig(env=env, **{k: v for k, v in saved.items() if k in names})


def append_history(run_dir, row):
    """One row of eval_history.csv (a resumed run keeps its file's columns)."""
    path = Path(run_dir)/'eval_history.csv'
    new = not path.exists()
    columns = EVAL_FIELDS if new else path.read_text(encoding='utf-8').splitlines()[0].split(',')
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, columns, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def history_row(summary, timesteps, wall_s, policy, best=False):
    return {'timesteps': int(timesteps), 'wall_s': wall_s, 'policy': policy,
            **{k: summary.get(k) for k in EVAL_FIELDS[3:-1]}, 'best': int(best)}


def history_state(run_dir):
    """(best rank, last policy-eval timesteps, wall seconds, scripted baseline recorded) from eval_history.csv."""
    path = Path(run_dir)/'eval_history.csv'
    rows = list(csv.DictReader(path.open(encoding='utf-8'))) if path.exists() else []
    ppo = [r for r in rows if r.get('policy', 'ppo') == 'ppo']
    wall = max((float(r['wall_s']) for r in rows if r.get('wall_s')), default=0.0)
    scripted = any(r.get('policy') == 'scripted' for r in rows)
    if not ppo:
        return None, None, wall, scripted
    return max(rank(r) for r in ppo), int(ppo[-1]['timesteps']), wall, scripted


def record_baseline(cfg, run_dir, make_env, scripted_fn, timesteps, wall_s):
    """The scripted baseline on cfg.eval_seeds -> eval_history.csv (policy 'scripted') and baseline.json."""
    start = time.time()
    workers = min(cfg.eval_envs, len(cfg.eval_seeds)) if cfg.subproc else 1
    out = baseline(cfg.env, cfg.eval_seeds, workers=workers, make_env=make_env, scripted_fn=scripted_fn,
                   env_module=cfg.env_module)
    (Path(run_dir)/'baseline.json').write_text(json.dumps(out, indent=2, default=str)+'\n', encoding='utf-8')
    append_history(run_dir, history_row(out['summary'], timesteps, wall_s, 'scripted'))
    if cfg.verbose:
        print(f"[baseline] scripted on {len(cfg.eval_seeds)} seeds: {_line(out['summary'])} | "
              f"{time.time()-start:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------------------------------------- callback
def _make_callback_class():
    from stable_baselines3.common.callbacks import BaseCallback

    class YardMonitor(BaseCallback):
        """Training-episode metrics to TensorBoard, held-out evaluation, best-model selection and checkpoints."""

        def __init__(self, cfg, run_dir, eval_env, best=None, last_eval=None, t0_wall=0.0):
            super().__init__(verbose=cfg.verbose)
            self.cfg, self.run_dir, self.eval_env = cfg, Path(run_dir), eval_env
            self.best, self.last_eval, self.last_ckpt, self.episodes = best, last_eval, None, {}
            self.wall0, self.t0_wall = time.time(), t0_wall

        def _on_training_start(self):
            if self.last_eval is None:
                self.last_eval = self.num_timesteps
                if self.cfg.eval_at_start:
                    self._evaluate()
            if self.last_ckpt is None:
                self.last_ckpt = self.num_timesteps

        def _on_step(self):
            for info in self.locals.get('infos', ()):
                m = info.get('episode_metrics')
                if m is None:
                    continue
                s = summarize(m)
                s.update({reason: s['end_reason'] == reason for reason in ('timeout', 'stalled')})
                for k in EPISODE_TB_FIELDS:
                    if s.get(k) is not None:
                        self.episodes.setdefault(f'episode/{k}', []).append(float(s[k]))
            if self.num_timesteps-self.last_eval >= self.cfg.eval_every:
                self.last_eval = self.num_timesteps
                self._evaluate()
            if self.num_timesteps-self.last_ckpt >= self.cfg.checkpoint_every:
                self.last_ckpt = self.num_timesteps
                self.save(self.run_dir/'checkpoints', f'model_{self.num_timesteps}', f'vecnormalize_{self.num_timesteps}')
            return True

        def _on_rollout_start(self):
            import torch
            torch.set_num_threads(1)

        def _on_rollout_end(self):
            import torch
            torch.set_num_threads(self.cfg.torch_threads)
            for k, v in self.episodes.items():
                self.logger.record(k, float(np.mean(v)))
            if self.episodes:
                self.logger.record('episode/count', len(self.episodes['episode/success']))
            self.episodes = {}

        def save(self, folder, model_name, vecnorm_name):
            folder.mkdir(parents=True, exist_ok=True)
            self.model.save(folder/f'{model_name}.zip')
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm.save(str(folder/f'{vecnorm_name}.pkl'))

        def _evaluate(self):
            vecnorm = self.model.get_vec_normalize_env()
            normalize = ObsNormalizer.from_vecnormalize(vecnorm) if vecnorm is not None else ObsNormalizer()
            start = time.time()
            res = evaluate(self.model, normalize, self.cfg.env, self.cfg.eval_seeds, vec_env=self.eval_env)
            s = res['summary']
            improved = self.best is None or rank(s) > self.best
            if improved:
                self.best = rank(s)
                self.save(self.run_dir, 'best_model', 'best_vecnormalize')
            wall = round(self.t0_wall+time.time()-self.wall0, 1)
            append_history(self.run_dir, history_row(s, self.num_timesteps, wall, 'ppo', improved))
            for k in EVAL_FIELDS[4:-1]:
                if s.get(k) is not None:
                    self.logger.record(f'eval/{k}', s[k])
            if self.verbose:
                print(f"[eval t={self.num_timesteps}] {_line(s)} | {time.time()-start:.0f} s"
                      f"{' | new best' if improved else ''}", flush=True)
            return res

    return YardMonitor


# ---------------------------------------------------------------------------------------------- train
def train(cfg, make_env=None, scripted_fn=None):
    """Train (or resume) a MaskablePPO run and return the model; see the module docstring for the outputs.
    make_env / scripted_fn override cfg.env_module's make_env / scripted_actions (env injection, e.g. a fake env in
    tests); scripted_fn is only needed when cfg.baseline is set."""
    single_thread_workers()
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import VecNormalize
    from ..rl.watchdog import ENV_DIR
    torch.set_num_threads(cfg.torch_threads)
    make_env = env_function(cfg.env_module, 'make_env', make_env)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_DIR] = str(run_dir/'hangs')   # inherited by the workers: a make_env that runs watchdog.watch dumps there
    base_seed = 2000+1000*cfg.seed   # env RNG seed = base_seed + rank; episode seeds are drawn >= 2000
    venv = make_vec_env(make_env, cfg.env, cfg.n_envs, base_seed, cfg.subproc)
    eval_env, failed = None, True
    try:
        if cfg.resume and (run_dir/'config.json').exists():   # keep the original config.json; note the resume
            note = {'resumed_from': str(cfg.resume), 'total_timesteps': cfg.total_timesteps, 'git': git_commit(),
                    'versions': versions(), 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
            (run_dir/f'resume_{int(time.time())}.json').write_text(json.dumps(note, indent=2)+'\n', encoding='utf-8')
        else:
            write_config(cfg, run_dir, {
                'env_functions': {'make_env': _qualname(make_env),
                                  'scripted_fn': _qualname(scripted_fn) if scripted_fn is not None
                                  else f'{cfg.env_module}.scripted_actions'},
                'observation_dim': int(np.prod(venv.observation_space.shape)),
                'action_nvec': [int(n) for n in getattr(venv.action_space, 'nvec', ())]})
        if cfg.resume:
            resume = Path(cfg.resume)
            venv = VecNormalize.load(str(_vecnorm_path(resume)), venv)
            venv.training, venv.norm_reward, venv.clip_reward = True, cfg.norm_reward, cfg.clip_reward
            model = MaskablePPO.load(str(resume), env=venv, device=cfg.device, tensorboard_log=str(run_dir/'tb'))
            best, last_eval, wall, baseline_done = history_state(run_dir)
        else:
            venv = VecNormalize(venv, norm_obs=cfg.norm_obs, norm_reward=cfg.norm_reward, gamma=cfg.gamma,
                                clip_obs=cfg.clip_obs, clip_reward=cfg.clip_reward)
            policy_kwargs = dict(net_arch=dict(pi=list(cfg.net_pi), vf=list(cfg.net_vf)),
                                 activation_fn=torch.nn.Tanh if cfg.activation == 'tanh' else torch.nn.ReLU)
            model = MaskablePPO('MlpPolicy', venv, learning_rate=LinearSchedule(cfg.learning_rate, cfg.lr_final),
                                n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                                gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
                                ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
                                target_kl=cfg.target_kl, policy_kwargs=policy_kwargs,
                                tensorboard_log=str(run_dir/'tb'), seed=cfg.seed, device=cfg.device,
                                verbose=cfg.verbose)
            best, last_eval, wall, baseline_done = None, None, 0.0, False
        if cfg.baseline and not baseline_done:
            try:
                record_baseline(cfg, run_dir, make_env, scripted_fn, model.num_timesteps, wall)
            except Exception as e:   # noqa: BLE001 - a broken baseline must not cost the training run
                print(f'[baseline] failed, training goes on without it: {e!r}', flush=True)
        eval_env = make_vec_env(make_env, cfg.env, max(1, min(cfg.eval_envs, len(cfg.eval_seeds))), 0, cfg.subproc)
        callback = _make_callback_class()(cfg, run_dir, eval_env, best, last_eval, wall)
        remaining = cfg.total_timesteps-model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not cfg.resume,
                        tb_log_name='ppo')
        else:   # nothing to train: a (silent) logger and the callback bound for the final eval/save
            from stable_baselines3.common.logger import configure
            model.set_logger(configure(None, []))
            callback.init_callback(model)
        if callback.last_eval != model.num_timesteps:
            callback._evaluate()
        callback.save(run_dir, 'final_model', 'final_vecnormalize')
        if not (run_dir/'best_model.zip').exists():   # e.g. no evaluation ran
            callback.save(run_dir, 'best_model', 'best_vecnormalize')
        failed = False
    finally:
        close_vec_env(venv, failed)
        close_vec_env(eval_env, failed)
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id', required=True)
    p.add_argument('--timesteps', type=int, default=YardTrainConfig.total_timesteps)
    p.add_argument('--envs', type=int, default=YardTrainConfig.n_envs)
    p.add_argument('--n-steps', type=int, default=YardTrainConfig.n_steps)
    p.add_argument('--batch-size', type=int, default=YardTrainConfig.batch_size)
    p.add_argument('--lr', type=float, default=YardTrainConfig.learning_rate)
    p.add_argument('--lr-final', type=float, default=YardTrainConfig.lr_final)
    p.add_argument('--ent-coef', type=float, default=YardTrainConfig.ent_coef)
    p.add_argument('--eval-every', type=int, default=YardTrainConfig.eval_every)
    p.add_argument('--eval-envs', type=int, default=YardTrainConfig.eval_envs)
    p.add_argument('--checkpoint-every', type=int, default=YardTrainConfig.checkpoint_every)
    p.add_argument('--seed', type=int, default=YardTrainConfig.seed)
    p.add_argument('--torch-threads', type=int, default=YardTrainConfig.torch_threads)
    p.add_argument('--scenario', help='YardEnvConfig.scenario (default: the env config default)')
    p.add_argument('--device', default=YardTrainConfig.device, help="learner device: 'cpu' or 'cuda'")
    p.add_argument('--out-root', default=YardTrainConfig.out_root, help='folder for run outputs (relative: under the repo)')
    p.add_argument('--env-module', default=ENV_MODULE, help='module providing YardEnvConfig, make_env, scripted_actions')
    p.add_argument('--no-subproc', action='store_true', help='envs in-process (DummyVecEnv)')
    p.add_argument('--no-baseline', action='store_true', help='skip the scripted baseline')
    p.add_argument('--resume', help='checkpoint .zip; the run config is reloaded from <out-root>/<run-id>/config.json')
    a = p.parse_args(argv)
    if a.resume:
        root = Path(a.out_root)
        saved = load_train_config((root if root.is_absolute() else ROOT/root)/a.run_id)
        cfg = replace(saved, resume=a.resume, total_timesteps=a.timesteps, out_root=a.out_root,
                      subproc=saved.subproc and not a.no_subproc, baseline=saved.baseline and not a.no_baseline)
    else:
        env = env_function(a.env_module, 'YardEnvConfig')(**({'scenario': a.scenario} if a.scenario else {}))
        cfg = YardTrainConfig(run_id=a.run_id, env=env, env_module=a.env_module, n_envs=a.envs,
                              subproc=not a.no_subproc, total_timesteps=a.timesteps, n_steps=a.n_steps,
                              batch_size=a.batch_size, learning_rate=a.lr, lr_final=a.lr_final, ent_coef=a.ent_coef,
                              eval_every=a.eval_every, eval_envs=a.eval_envs, baseline=not a.no_baseline,
                              checkpoint_every=a.checkpoint_every, seed=a.seed, torch_threads=a.torch_threads,
                              device=a.device, out_root=a.out_root)
    train(cfg)


if __name__ == '__main__':
    main()
