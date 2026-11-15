"""nohesi.yard.train on a fake env with YardEnv's API (the real env is injected the same way, by default)."""
import csv
import json
import sys
from dataclasses import dataclass, replace
import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip('sb3_contrib')
from nohesi.yard import train as yt   # noqa: E402

HOLD, CREEP, GO, BACK_OFF, REPLAN = range(5)
SLOTS, GOAL = 3, 6.0


@dataclass(frozen=True)
class FakeEnvConfig:
    version: str = 'fake1'
    scenario: str = 'fake'
    steps: int = 20          # decisions per episode at most
    followers: int = 2       # slot 3 is absent (HOLD only)


class FakeYardEnv(gymnasium.Env):
    """YardEnv in miniature: 3 slots x (HOLD, CREEP, GO, BACK_OFF, REPLAN); GO / CREEP move a follower 1 / 0.5 towards
    its spot, BACK_OFF 0.5 back (masked at the start); a follower parks at GOAL. The episode ends when all are parked
    or after cfg.steps decisions. A masked action raises, so a masking bug fails the run."""
    observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)
    action_space = spaces.MultiDiscrete([5]*SLOTS)

    def __init__(self, cfg=FakeEnvConfig(), seed_pool=None, rng_seed=None):
        self.cfg, self.seed_pool, self.pool_index = cfg, list(seed_pool or ()), 0
        if rng_seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(rng_seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            if self.seed_pool:
                seed = self.seed_pool[self.pool_index % len(self.seed_pool)]
                self.pool_index += 1
            else:
                seed = int(self.np_random.integers(2000, 2**31-1))
        self.episode_seed, self.t = int(seed), 0
        self.progress, self.replans, self.back_off = np.zeros(SLOTS), 0, 0.0
        self.active = np.arange(SLOTS) < self.cfg.followers
        return self._obs(), {'seed': self.episode_seed}

    def _obs(self):
        return np.concatenate([self.progress/GOAL, self.active, [self.t/self.cfg.steps, 1.0]]).astype(np.float32)

    def parked(self):
        return self.active & (self.progress >= GOAL)

    def action_masks(self):
        mask = np.zeros((SLOTS, 5), bool)
        moving = self.active & ~self.parked()
        mask[moving] = True
        mask[moving & (self.progress <= 0), BACK_OFF] = False
        mask[~moving, HOLD] = True
        return mask.ravel()

    def step(self, action):
        action = np.asarray(action, int)
        mask = self.action_masks().reshape(SLOTS, 5)
        if not all(mask[k, a] for k, a in enumerate(action)):
            raise ValueError(f'masked action {action}')
        gain = np.select([action == GO, action == CREEP, action == BACK_OFF], [1.0, 0.5, -0.5], 0.0)
        self.progress += gain
        self.replans += int(np.sum(action == REPLAN))
        self.back_off += 0.5*float(np.sum(action == BACK_OFF))
        self.t += 1
        success = bool(self.parked().sum() == self.cfg.followers)
        terminated = success or self.t >= self.cfg.steps
        info = {}
        if terminated:
            info['episode_metrics'] = {
                'seed': self.episode_seed, 'success': success, 'parked': int(self.parked().sum()),
                'followers': self.cfg.followers, 'collision': False, 'end_reason': 'parked' if success else 'timeout',
                'duration_s': 0.5*self.t, 'replans': self.replans, 'replans_failed': 0, 'safety_stops': 0,
                'back_off_m': self.back_off, 'picks_rejected': 0, 'reassigned': 0,
                'path_length_m': GOAL*self.cfg.followers, 'driven_m': float(np.abs(self.progress).sum())}
        return self._obs(), float(gain.sum())-0.1, terminated, False, info


# The env-module API (YardTrainConfig.env_module = __name__ resolves these, as it does nohesi.yard.env's).
YardEnvConfig = FakeEnvConfig
SCRIPTED_CALLS = []


def make_env(cfg, rank=0, base_seed=0, seed_pool=None):
    def thunk():
        return FakeYardEnv(cfg, seed_pool, rng_seed=base_seed+rank)
    return thunk


def scripted_actions(env):
    """GO for every follower still moving, HOLD otherwise: parks both followers in 6 decisions (3 s)."""
    SCRIPTED_CALLS.append(1)
    return np.where(env.active & ~env.parked(), GO, HOLD)


SEEDS = (1000, 1001)


def tiny(tmp_path, **kw):
    cfg = dict(run_id='tiny', env=FakeEnvConfig(), n_envs=2, subproc=False, total_timesteps=3072, n_steps=128,
               batch_size=128, n_epochs=2, eval_every=1024, eval_seeds=SEEDS, eval_envs=2, checkpoint_every=1024,
               torch_threads=1, out_root=str(tmp_path), verbose=0)
    cfg.update(kw)
    return yt.YardTrainConfig(**cfg)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    cfg = tiny(tmp_path_factory.mktemp('runs_rl'))
    imported = 'nohesi.yard.env' in sys.modules
    SCRIPTED_CALLS.clear()
    model = yt.train(cfg, make_env=make_env, scripted_fn=scripted_actions)
    if not imported:   # injected factories: the real env is never imported
        assert 'nohesi.yard.env' not in sys.modules
    assert SCRIPTED_CALLS, 'the scripted baseline did not run through the injected function'
    return cfg, model


def _history(cfg):
    return list(csv.DictReader((cfg.run_dir/'eval_history.csv').open(encoding='utf-8')))


def test_training_writes_outputs(trained):
    cfg, model = trained
    run = cfg.run_dir
    assert model.num_timesteps >= cfg.total_timesteps
    for name in ('config.json', 'eval_history.csv', 'baseline.json', 'best_model.zip', 'best_vecnormalize.pkl',
                 'final_model.zip', 'final_vecnormalize.pkl'):
        assert (run/name).exists(), name
    assert list((run/'checkpoints').glob('model_*.zip')) and list((run/'checkpoints').glob('vecnormalize_*.pkl'))
    assert list((run/'tb').rglob('events.out.tfevents.*'))
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    assert config['train']['env'] == {'version': 'fake1', 'scenario': 'fake', 'steps': 20, 'followers': 2}
    assert config['train']['eval_seeds'] == list(SEEDS) and config['train']['lr_final'] == 1e-5
    assert config['env_functions']['make_env'].endswith('test_yard_train.make_env')
    assert config['env_functions']['scripted_fn'].endswith('test_yard_train.scripted_actions')
    assert config['observation_dim'] == 8 and config['action_nvec'] == [5, 5, 5]
    assert 'sb3_contrib' in config['versions'] and 'commit' in config['git']


def test_eval_history_has_baseline_once_and_policy_evals(trained):
    cfg, _ = trained
    rows = _history(cfg)
    scripted = [r for r in rows if r['policy'] == 'scripted']
    ppo = [r for r in rows if r['policy'] == 'ppo']
    assert len(scripted) == 1 and rows[0]['policy'] == 'scripted'
    assert float(scripted[0]['success_rate']) == 1.0 and float(scripted[0]['duration_s']) == 3.0
    assert float(scripted[0]['episodes']) == len(SEEDS)
    # t=0, every eval_every, and the final one
    assert [int(r['timesteps']) for r in ppo][0] == 0 and int(ppo[-1]['timesteps']) >= cfg.total_timesteps
    assert len(ppo) >= 1+cfg.total_timesteps//cfg.eval_every
    assert ppo[0]['best'] == '1' and all(0 <= float(r['success_rate']) <= 1 for r in ppo)
    baseline = json.loads((cfg.run_dir/'baseline.json').read_text(encoding='utf-8'))
    assert [r['seed'] for r in baseline['per_seed']] == list(SEEDS)


def test_best_model_is_the_best_ranked_eval(trained):
    cfg, _ = trained
    ppo = [r for r in _history(cfg) if r['policy'] == 'ppo']
    best = [r for r in ppo if r['best'] == '1'][-1]
    assert yt.rank(best) == max(yt.rank(r) for r in ppo)


def test_evaluate_saved_best_model(trained):
    from sb3_contrib import MaskablePPO
    from nohesi.rl.evaluate import ObsNormalizer
    cfg, _ = trained
    model = MaskablePPO.load(str(cfg.run_dir/'best_model.zip'), device='cpu')
    normalize = ObsNormalizer.load(cfg.run_dir/'best_vecnormalize.pkl')
    out = yt.evaluate(model, normalize, cfg.env, SEEDS, make_env=make_env, subproc=False)
    assert [r['seed'] for r in out['per_seed']] == list(SEEDS)
    assert out['summary']['episodes'] == 2 and 0 <= out['summary']['success_rate'] <= 1
    assert all(r['decisions'] > 0 for r in out['per_seed'])


def test_baseline_summary_matches_scripted_policy():
    SCRIPTED_CALLS.clear()
    out = yt.baseline(FakeEnvConfig(), SEEDS, make_env=make_env, scripted_fn=scripted_actions)
    s = out['summary']
    assert out['seeds'] == list(SEEDS) and len(SCRIPTED_CALLS) == 6*len(SEEDS)
    assert (s['success_rate'], s['parked_frac'], s['collision_rate'], s['duration_s'], s['decisions']) == (1, 1, 0, 3, 6)
    assert s['success_duration_s'] == 3.0 and s['end_reasons'] == {'parked': 2}
    # the same through the env-module lookup (as nohesi.yard.env is found by default)
    assert yt.baseline(FakeEnvConfig(), SEEDS, env_module=__name__)['summary'] == s


def test_rank_prefers_success_then_shorter_duration_then_parked():
    a = {'success_rate': 0.5, 'success_duration_s': 200.0, 'parked_frac': 0.6}
    assert yt.rank(a) > yt.rank({**a, 'success_rate': 0.45, 'success_duration_s': 100.0})
    assert yt.rank(a) > yt.rank({**a, 'success_duration_s': 210.0})
    assert yt.rank({'success_rate': 0.0, 'parked_frac': 0.5}) > yt.rank({'success_rate': 0.0, 'parked_frac': 0.3})
    assert yt.rank({'success_rate': 0.05, 'success_duration_s': 900.0}) > yt.rank({'success_rate': 0, 'parked_frac': 1})


def test_resume_continues_and_keeps_one_baseline(trained, tmp_path):
    import shutil
    cfg, _ = trained
    run = tmp_path/cfg.run_id   # a copy: the other tests read the original run
    shutil.copytree(cfg.run_dir, run)
    ckpt = sorted((run/'checkpoints').glob('model_*.zip'), key=lambda p: int(p.stem.split('_')[1]))[-1]
    resumed = replace(cfg, out_root=str(tmp_path), resume=str(ckpt), total_timesteps=cfg.total_timesteps+512)
    model = yt.train(resumed, make_env=make_env, scripted_fn=scripted_actions)
    assert model.num_timesteps >= cfg.total_timesteps+512
    rows = list(csv.DictReader((run/'eval_history.csv').open(encoding='utf-8')))
    assert sum(r['policy'] == 'scripted' for r in rows) == 1
    assert int(rows[-1]['timesteps']) >= cfg.total_timesteps+512 and list(run.glob('resume_*.json'))


def test_subprocess_paths_take_the_injected_env(tmp_path):
    """Spawned workers: nohesi.rl.vec_worker's SubprocVecEnv with the injected thunks, and the parallel baseline."""
    from sb3_contrib import MaskablePPO
    venv = yt.make_vec_env(make_env, FakeEnvConfig(), 2, 0, subproc=True)
    try:
        assert type(venv).__name__ == 'LeanSubprocVecEnv'
        model = MaskablePPO('MlpPolicy', venv, n_steps=32, batch_size=64, n_epochs=1, verbose=0, seed=0)
        model.learn(64)
        out = yt.evaluate(model, None, FakeEnvConfig(), SEEDS, vec_env=venv)
    finally:
        venv.close()
    assert [r['seed'] for r in out['per_seed']] == list(SEEDS)
    par = yt.baseline(FakeEnvConfig(), SEEDS, workers=2, make_env=make_env, scripted_fn=scripted_actions)
    assert par['summary']['success_rate'] == 1.0 and [r['seed'] for r in par['per_seed']] == list(SEEDS)


def test_cli_builds_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(yt, 'train', lambda cfg: seen.setdefault('cfg', cfg))
    yt.main(['--run-id', 'yard_ppo_001', '--timesteps', '2000000', '--envs', '8', '--no-subproc', '--scenario', 'fake2',
             '--env-module', __name__, '--out-root', str(tmp_path)])
    cfg = seen['cfg']
    assert (cfg.total_timesteps, cfg.n_envs, cfg.subproc, cfg.env, cfg.env_module) == \
        (2_000_000, 8, False, FakeEnvConfig(scenario='fake2'), __name__)
    assert (cfg.n_steps, cfg.batch_size, cfg.lr_final, cfg.eval_seeds) == (256, 512, 1e-5, tuple(range(1000, 1020)))
    assert cfg.run_dir == tmp_path/'yard_ppo_001'
    # --resume reloads the saved config (env config rebuilt through env_module)
    cfg.run_dir.mkdir()
    yt.write_config(cfg, cfg.run_dir)
    seen.clear()
    yt.main(['--run-id', 'yard_ppo_001', '--out-root', str(tmp_path), '--timesteps', '3000000',
             '--resume', str(cfg.run_dir/'checkpoints'/'model_1000000.zip')])
    again = seen['cfg']
    assert again.env == cfg.env and again.subproc is False and again.total_timesteps == 3_000_000
    assert again.resume.endswith('model_1000000.zip')
