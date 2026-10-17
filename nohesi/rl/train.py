"""MaskablePPO training for the No Hesi team env (milestone 3).

    python -m nohesi.rl.train --run-id ppo_v4_001 --timesteps 10000000 --envs 16
    python -m nohesi.rl.train --run-id ppo_v4_001 --resume runs_rl/ppo_v4_001/checkpoints/model_2000000.zip

Everything for a run goes to runs_rl/<run_id>/:
    config.json                 TrainConfig + EnvConfig, library versions, git commit
    tb/                         TensorBoard (SB3 train/* and rollout/*, plus reward_terms/*, aux/*, episode/*, eval/*)
    checkpoints/model_<t>.zip + vecnormalize_<t>.pkl   every checkpoint_every timesteps
    best_model.zip + best_vecnormalize.pkl             highest mean eval team score
    final_model.zip + final_vecnormalize.pkl
    eval_history.csv            one row per evaluation on the held-out seeds

Timesteps are env decisions (one per 0.5 s of sim time, summed over all envs).

Threads: each SubprocVecEnv worker is single-threaded (OMP/MKL/OpenBLAS = 1, set before the workers spawn and
inherited by them) so 16 workers do not oversubscribe the 32 hardware threads. The learner process runs torch
single-threaded while collecting rollouts (per-step inference on 16 observations; extra OpenMP threads would only
spin and steal cores from the workers) and with torch_threads (default 4) during the PPO update (~1.3 s per
8192-sample update at 4 threads vs ~2.8 s at 1).
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_FIELDS = ('timesteps', 'wall_s', 'team_score_mean', 'team_score_std', 'collision_rate', 'close_passes',
               'draft_time_s', 'maneuvers', 'fallbacks', 'rejections', 'episode_length', 'best')


def _env_config_default():
    from .config import EnvConfig
    return EnvConfig()


@dataclass
class TrainConfig:
    run_id: str = 'ppo_v4'
    env: object = field(default_factory=_env_config_default)   # nohesi.rl.config.EnvConfig
    n_envs: int = 16
    subproc: bool = True             # SubprocVecEnv (False: DummyVecEnv in-process, for tests/debugging)
    total_timesteps: int = 10_000_000
    # PPO
    n_steps: int = 512               # per env: 16 x 512 = 8192 decisions per update
    batch_size: int = 1024
    n_epochs: int = 10
    learning_rate: float = 3e-4      # linearly decayed to lr_final over the run
    lr_final: float = 0.0
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    target_kl: float = None
    net_pi: tuple = (256, 256)
    net_vf: tuple = (256, 256)
    activation: str = 'tanh'         # 'tanh' (SB3 PPO default; bounded features with normalised obs) or 'relu'
    # VecNormalize
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    # evaluation / checkpoints
    eval_every: int = 500_000        # timesteps (decisions) between held-out evaluations
    eval_seeds: tuple = tuple(range(1000, 1010))   # 10 of the 20 held-out seeds during training; final eval uses 1000-1019
    eval_envs: int = 10              # parallel eval envs
    eval_at_start: bool = True       # evaluate the untrained policy at t=0 (start of the learning curve)
    checkpoint_every: int = 1_000_000
    seed: int = 0                    # PPO/torch seed; training env seeds come from the env (random >= 2000)
    torch_threads: int = 4
    device: str = 'cpu'
    out_root: str = 'runs_rl'
    resume: str = None               # checkpoint .zip to continue from (its vecnormalize_<t>.pkl alongside)
    verbose: int = 1

    def __post_init__(self):
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


class LinearSchedule:
    """start -> end linearly over progress_remaining 1 -> 0 (picklable, unlike a lambda in some setups)."""

    def __init__(self, start, end=0.0):
        self.start, self.end = float(start), float(end)

    def __call__(self, progress_remaining):
        return self.end+(self.start-self.end)*progress_remaining


# ---------------------------------------------------------------------------------------------- provenance
def git_commit():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        return {'commit': commit or None, 'dirty': bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def versions():
    import gymnasium, sb3_contrib, stable_baselines3, torch
    return {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__,
            'torch': torch.__version__, 'stable_baselines3': stable_baselines3.__version__,
            'sb3_contrib': sb3_contrib.__version__, 'gymnasium': gymnasium.__version__}


def _jsonable(x):
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, Path):
        return str(x)
    return x


def write_config(cfg, run_dir, extra=None):
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    d['env'] = asdict(cfg.env)
    out = {'train': _jsonable(d), 'versions': versions(), 'git': git_commit(),
           'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'argv': sys.argv, **(extra or {})}
    try:
        from . import env as rl_env
        space = rl_env.NoHesiEnv(cfg.env).observation_space
        out['observation_dim'] = int(space.shape[0])
    except Exception:   # noqa: BLE001 - informational only
        pass
    (run_dir/'config.json').write_text(json.dumps(out, indent=2, default=str)+'\n', encoding='utf-8')


def load_train_config(run_dir):
    """TrainConfig from runs_rl/<run>/config.json."""
    from .config import EnvConfig
    saved = json.loads((Path(run_dir)/'config.json').read_text(encoding='utf-8'))['train']
    env_names = {f.name for f in fields(EnvConfig)}
    env = EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.pop('env').items() if k in env_names})
    names = {f.name for f in fields(TrainConfig)}
    return TrainConfig(env=env, **{k: v for k, v in saved.items() if k in names})


# ---------------------------------------------------------------------------------------------- envs
def single_thread_workers():
    """Inherited by the spawned SubprocVecEnv workers (numpy/scipy/OSQP BLAS pools)."""
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')


def make_vec_env(env_cfg, n_envs, base_seed, subproc=True):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from .env import make_env
    thunks = [make_env(env_cfg, rank, base_seed) for rank in range(n_envs)]
    return SubprocVecEnv(thunks) if subproc and n_envs > 1 else DummyVecEnv(thunks)


# ---------------------------------------------------------------------------------------------- callback
def _flatten_terms(prefix, d, out):
    """Accumulate dict entries (scalar or per-car arrays -> summed) into out[prefix/key]."""
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten_terms(f'{prefix}/{k}', v, out)
            continue
        try:
            out.setdefault(f'{prefix}/{k}', []).append(float(np.sum(v)))
        except (TypeError, ValueError):
            pass


def _make_callback_class():
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainingMonitor(BaseCallback):
        """Per-term reward / aux logging, held-out evaluation, best-model selection and checkpoints."""

        def __init__(self, cfg, run_dir, eval_env, best=-np.inf, last_eval=None, last_ckpt=None, t0_wall=0.0):
            super().__init__(verbose=cfg.verbose)
            self.cfg, self.run_dir, self.eval_env = cfg, Path(run_dir), eval_env
            self.best, self.terms, self.episodes = best, {}, {}
            self.last_eval, self.last_ckpt = last_eval, last_ckpt
            self.wall0, self.t0_wall = time.time(), t0_wall

        # --- logging
        def _on_training_start(self):
            if self.last_eval is None:
                self.last_eval = self.num_timesteps
                if self.cfg.eval_at_start:
                    self._evaluate()
            if self.last_ckpt is None:
                self.last_ckpt = self.num_timesteps

        def _on_step(self):
            for info in self.locals.get('infos', ()):
                _flatten_terms('reward_terms', info.get('score_terms'), self.terms)
                _flatten_terms('aux', info.get('aux'), self.terms)
                m = info.get('episode_metrics')
                if m is not None:
                    from .evaluate import summarize
                    s = summarize(m)
                    for k in ('team_score', 'collision', 'close_passes', 'maneuvers', 'fallbacks', 'rejections',
                              'duration_s', 'draft_time_s'):
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
            for k, v in self.terms.items():   # mean per decision (per env step) over the rollout
                self.logger.record(k, float(np.mean(v)))
            for k, v in self.episodes.items():
                self.logger.record(k, float(np.mean(v)))
            if self.episodes:
                self.logger.record('episode/count', len(next(iter(self.episodes.values()))))
            self.terms, self.episodes = {}, {}

        # --- saving
        def save(self, folder, model_name, vecnorm_name):
            folder.mkdir(parents=True, exist_ok=True)
            self.model.save(folder/f'{model_name}.zip')
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm.save(str(folder/f'{vecnorm_name}.pkl'))

        # --- evaluation
        def _evaluate(self):
            from .evaluate import ObsNormalizer, evaluate_model
            vecnorm = self.model.get_vec_normalize_env()
            normalize = ObsNormalizer.from_vecnormalize(vecnorm) if vecnorm is not None else ObsNormalizer()
            start = time.time()
            res = evaluate_model(self.model, normalize, self.cfg.env, self.cfg.eval_seeds, vec_env=self.eval_env)
            s = res['summary']
            improved = s['team_score'] > self.best
            if improved:
                self.best = s['team_score']
                self.save(self.run_dir, 'best_model', 'best_vecnormalize')
            row = {'timesteps': self.num_timesteps, 'wall_s': round(self.t0_wall+time.time()-self.wall0, 1),
                   'team_score_mean': s['team_score'], 'team_score_std': s['team_score_std'],
                   'collision_rate': s['collision_rate'], 'close_passes': s['close_passes'],
                   'draft_time_s': s['draft_time_s'], 'maneuvers': s['maneuvers'], 'fallbacks': s['fallbacks'],
                   'rejections': s['rejections'], 'episode_length': s['decisions'], 'best': int(improved)}
            path = self.run_dir/'eval_history.csv'
            new = not path.exists()
            with path.open('a', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, EVAL_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            for k in EVAL_FIELDS[2:-1]:
                if row[k] is not None:
                    self.logger.record(f'eval/{k}', row[k])
            if self.verbose:
                print(f"[eval t={self.num_timesteps}] team score {s['team_score']:.1f} +- {s['team_score_std']:.1f} | "
                      f"collisions {s['collision_rate']:.2f} | close passes {s['close_passes']:.1f} | "
                      f"len {s['decisions']:.0f} | {time.time()-start:.0f} s{' | new best' if improved else ''}", flush=True)
            return res

    return TrainingMonitor


def _history_state(run_dir):
    """(best eval score, last eval timesteps, wall seconds) from an existing eval_history.csv, for resuming."""
    path = Path(run_dir)/'eval_history.csv'
    if not path.exists():
        return -np.inf, None, 0.0
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    if not rows:
        return -np.inf, None, 0.0
    return (max(float(r['team_score_mean']) for r in rows), int(rows[-1]['timesteps']), float(rows[-1]['wall_s']))


# ---------------------------------------------------------------------------------------------- train
def train(cfg):
    """Train (or resume) a MaskablePPO run; returns the model. See the module docstring for outputs."""
    single_thread_workers()
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import VecNormalize
    torch.set_num_threads(cfg.torch_threads)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.resume and (run_dir/'config.json').exists():   # keep the original config.json; note the resume
        note = {'resumed_from': str(cfg.resume), 'total_timesteps': cfg.total_timesteps, 'git': git_commit(),
                'versions': versions(), 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
        (run_dir/f'resume_{int(time.time())}.json').write_text(json.dumps(note, indent=2)+'\n', encoding='utf-8')
    else:
        write_config(cfg, run_dir)

    base_seed = 2000+1000*cfg.seed   # env RNG seed = base_seed + rank; episode seeds are drawn >= 2000
    venv = make_vec_env(cfg.env, cfg.n_envs, base_seed, cfg.subproc)
    eval_env = None
    try:
        if cfg.resume:
            resume = Path(cfg.resume)
            vecnorm_path = resume.with_name(resume.stem.replace('model', 'vecnormalize')+'.pkl')
            venv = VecNormalize.load(str(vecnorm_path), venv)
            venv.training, venv.norm_reward = True, cfg.norm_reward
            model = MaskablePPO.load(str(resume), env=venv, device=cfg.device,
                                     tensorboard_log=str(run_dir/'tb'))
            best, last_eval, wall = _history_state(run_dir)
        else:
            venv = VecNormalize(venv, norm_obs=cfg.norm_obs, norm_reward=cfg.norm_reward, gamma=cfg.gamma,
                                clip_obs=cfg.clip_obs)
            policy_kwargs = dict(net_arch=dict(pi=list(cfg.net_pi), vf=list(cfg.net_vf)),
                                 activation_fn=torch.nn.Tanh if cfg.activation == 'tanh' else torch.nn.ReLU)
            model = MaskablePPO('MlpPolicy', venv, learning_rate=LinearSchedule(cfg.learning_rate, cfg.lr_final),
                                n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                                gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
                                ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
                                target_kl=cfg.target_kl, policy_kwargs=policy_kwargs,
                                tensorboard_log=str(run_dir/'tb'), seed=cfg.seed, device=cfg.device,
                                verbose=cfg.verbose)
            best, last_eval, wall = -np.inf, None, 0.0
        n_eval = max(1, min(cfg.eval_envs, len(cfg.eval_seeds)))
        eval_env = make_vec_env(cfg.env, n_eval, 0, cfg.subproc)
        callback = _make_callback_class()(cfg, run_dir, eval_env, best, last_eval, None, wall)
        remaining = cfg.total_timesteps-model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not cfg.resume,
                        tb_log_name='ppo')
        if callback.last_eval != model.num_timesteps:
            callback._evaluate()
        callback.save(run_dir, 'final_model', 'final_vecnormalize')
        if not (run_dir/'best_model.zip').exists():   # e.g. no evaluation ran
            callback.save(run_dir, 'best_model', 'best_vecnormalize')
    finally:
        venv.close()
        if eval_env is not None:
            eval_env.close()
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id', required=True)
    p.add_argument('--timesteps', type=int, default=TrainConfig.total_timesteps)
    p.add_argument('--envs', type=int, default=TrainConfig.n_envs)
    p.add_argument('--n-steps', type=int, default=TrainConfig.n_steps)
    p.add_argument('--batch-size', type=int, default=TrainConfig.batch_size)
    p.add_argument('--lr', type=float, default=TrainConfig.learning_rate)
    p.add_argument('--ent-coef', type=float, default=TrainConfig.ent_coef)
    p.add_argument('--activation', choices=('tanh', 'relu'), default=TrainConfig.activation)
    p.add_argument('--eval-every', type=int, default=TrainConfig.eval_every)
    p.add_argument('--eval-envs', type=int, default=TrainConfig.eval_envs)
    p.add_argument('--checkpoint-every', type=int, default=TrainConfig.checkpoint_every)
    p.add_argument('--seed', type=int, default=TrainConfig.seed)
    p.add_argument('--torch-threads', type=int, default=TrainConfig.torch_threads)
    p.add_argument('--scenario')
    p.add_argument('--score')
    p.add_argument('--mpc', choices=('linear', 'nonlinear'))
    p.add_argument('--vehicle', choices=('kinematic', 'dynamic'))
    p.add_argument('--resume', help='checkpoint .zip; the run config is reloaded from runs_rl/<run-id>/config.json')
    a = p.parse_args(argv)
    if a.resume:
        saved = load_train_config(ROOT/TrainConfig.out_root/a.run_id)
        cfg = replace(saved, resume=a.resume, total_timesteps=a.timesteps)
    else:
        from .config import EnvConfig
        env = replace(EnvConfig(), **{k: v for k, v in (('scenario', a.scenario), ('score', a.score), ('mpc', a.mpc),
                                                         ('vehicle_model', a.vehicle)) if v})
        cfg = TrainConfig(run_id=a.run_id, env=env, n_envs=a.envs, total_timesteps=a.timesteps, n_steps=a.n_steps,
                          batch_size=a.batch_size, learning_rate=a.lr, ent_coef=a.ent_coef, activation=a.activation,
                          eval_every=a.eval_every, eval_envs=a.eval_envs, checkpoint_every=a.checkpoint_every,
                          seed=a.seed, torch_threads=a.torch_threads)
    train(cfg)


if __name__ == '__main__':
    main()
