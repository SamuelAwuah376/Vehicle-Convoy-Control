"""Milestone 3 training tooling: a tiny MaskablePPO run, evaluation of the saved policy, and mask compliance."""
import csv
import json
import numpy as np
import pytest

pytest.importorskip('sb3_contrib')
rl_env = pytest.importorskip('nohesi.rl.env')
from nohesi.rl.config import EnvConfig   # noqa: E402
from nohesi.rl.evaluate import evaluate, run_policy, ObsNormalizer   # noqa: E402
from nohesi.rl.train import TrainConfig, train   # noqa: E402


def tiny(tmp_path, **kw):
    cfg = dict(run_id='tiny', env=EnvConfig(), n_envs=2, subproc=False, total_timesteps=256, n_steps=128,
               batch_size=64, n_epochs=1, eval_every=10**9, eval_at_start=False, eval_seeds=(1000,), eval_envs=1,
               checkpoint_every=128, torch_threads=1, out_root=str(tmp_path), verbose=0)
    cfg.update(kw)
    return TrainConfig(**cfg)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    cfg = tiny(tmp_path_factory.mktemp('runs_rl'))
    train(cfg)
    return cfg


def test_tiny_training_writes_outputs(trained):
    run = trained.run_dir
    for name in ('config.json', 'eval_history.csv', 'best_model.zip', 'best_vecnormalize.pkl', 'final_model.zip'):
        assert (run/name).exists(), name
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    assert config['train']['env']['score'] == trained.env.score and 'sb3_contrib' in config['versions']
    assert 'commit' in config['git']
    rows = list(csv.DictReader((run/'eval_history.csv').open(encoding='utf-8')))
    assert rows and int(rows[-1]['timesteps']) >= trained.total_timesteps
    assert np.isfinite(float(rows[-1]['team_score_mean']))
    assert list((run/'checkpoints').glob('model_*.zip')) and list((run/'checkpoints').glob('vecnormalize_*.pkl'))


def test_resume_continues_timesteps(trained):
    ckpt = sorted((trained.run_dir/'checkpoints').glob('model_*.zip'))[0]
    from dataclasses import replace
    model = train(replace(trained, resume=str(ckpt), total_timesteps=trained.total_timesteps+256))
    assert model.num_timesteps >= trained.total_timesteps+256


def test_evaluate_loads_saved_policy(trained):
    out = evaluate(trained.run_dir/'best_model.zip', trained.run_dir/'best_vecnormalize.pkl', trained.env,
                   seeds=[1000], n_envs=1, baseline=False, subproc=False)
    assert len(out['per_seed']) == 1 and out['per_seed'][0]['seed'] == 1000
    assert np.isfinite(out['summary']['team_score']) and out['summary']['decisions'] > 0


def test_actions_respect_masks(trained):
    """The env never receives a masked action, in training rollouts or in deterministic evaluation."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    seen = []

    class Checked(rl_env.NoHesiEnv):
        def step(self, action):
            mask = np.asarray(self.action_masks(), bool).reshape(3, -1)
            assert all(mask[k, a] for k, a in enumerate(np.asarray(action, int))), (action, mask)
            seen.append(1)
            return super().step(action)

    venv = DummyVecEnv([lambda: Checked(EnvConfig())])
    model = MaskablePPO('MlpPolicy', venv, n_steps=64, batch_size=32, n_epochs=1, verbose=0, seed=1)
    model.learn(64)                         # stochastic sampling with masks
    run_policy(model, ObsNormalizer(), venv, [1000])   # deterministic evaluation with masks
    assert len(seen) > 64
    venv.close()


def test_init_from_copies_policy_and_obs_norm_only(trained, tmp_path):
    import pickle
    import torch
    from sb3_contrib import MaskablePPO
    from dataclasses import replace
    src = trained.run_dir/'final_model.zip'
    # total_timesteps=0: no learning, so the new run's saved model is exactly its starting point.
    cfg = replace(trained, run_id='tiny_ft', env=replace(trained.env, score='v5'), total_timesteps=0,
                  out_root=str(tmp_path), init_from=str(src))
    model = train(cfg)
    assert model.num_timesteps == 0
    a, b = MaskablePPO.load(str(src)).policy.state_dict(), MaskablePPO.load(str(cfg.run_dir/'final_model.zip')).policy.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    old = pickle.load(open(trained.run_dir/'final_vecnormalize.pkl', 'rb'))
    new = pickle.load(open(cfg.run_dir/'final_vecnormalize.pkl', 'rb'))
    assert np.array_equal(old.obs_rms.mean, new.obs_rms.mean) and new.ret_rms.count < 1
    assert json.loads((cfg.run_dir/'config.json').read_text(encoding='utf-8'))['train']['init_from'] == str(src)


def test_vf_warmup_trains_value_network_only(trained, tmp_path):
    import torch
    from sb3_contrib import MaskablePPO
    from dataclasses import replace
    src = trained.run_dir/'final_model.zip'
    # The whole run is warm-up: the policy must come out unchanged, the value network trained.
    cfg = replace(trained, run_id='tiny_warm', env=replace(trained.env, score='v7'), total_timesteps=256, vf_warmup=256,
                  out_root=str(tmp_path), init_from=str(src))
    model = train(cfg)
    assert model.num_timesteps >= 256
    a, b = MaskablePPO.load(str(src)).policy.state_dict(), MaskablePPO.load(str(cfg.run_dir/'final_model.zip')).policy.state_dict()
    policy = [k for k in a if k.startswith(('mlp_extractor.policy_net', 'action_net'))]
    value = [k for k in a if k.startswith(('mlp_extractor.value_net', 'value_net'))]
    assert policy and value
    assert all(torch.equal(a[k], b[k]) for k in policy)
    assert not all(torch.equal(a[k], b[k]) for k in value)
    assert all(p.requires_grad for p in model.policy.parameters())   # unfrozen afterwards


def test_vf_warmup_then_policy_trains(trained, tmp_path):
    import torch
    from sb3_contrib import MaskablePPO
    from dataclasses import replace
    src = trained.run_dir/'final_model.zip'
    # lr_final > 0: this tiny run updates once after the warm-up, at the end of the schedule.
    cfg = replace(trained, run_id='tiny_warm2', total_timesteps=512, vf_warmup=256, lr_final=1e-4, out_root=str(tmp_path),
                  init_from=str(src))
    model = train(cfg)
    assert model.num_timesteps >= 512
    a, b = MaskablePPO.load(str(src), device='cpu').policy.state_dict(), model.policy.state_dict()
    assert not all(torch.equal(a[k], b[k].cpu()) for k in a if k.startswith('action_net'))


def test_cli_builds_fine_tune_config(monkeypatch, tmp_path):
    import nohesi.rl.train as train_module
    seen = {}
    monkeypatch.setattr(train_module, 'train', lambda cfg: seen.setdefault('cfg', cfg))
    train_module.main(['--run-id', 'cli', '--timesteps', '6000000', '--envs', '32', '--n-steps', '256', '--score', 'v7',
                       '--lr', '1e-4', '--lr-final', '0', '--init-from', 'models/x/final_model.zip', '--vf-warmup', '200000',
                       '--collision-extra', '0', '--device', 'cpu', '--out-root', str(tmp_path)])
    cfg = seen['cfg']
    assert (cfg.total_timesteps, cfg.n_envs, cfg.n_steps, cfg.learning_rate, cfg.vf_warmup) == (6_000_000, 32, 256, 1e-4, 200_000)
    assert (cfg.env.score, cfg.env.collision_extra, cfg.init_from, cfg.device) == ('v7', 0.0, 'models/x/final_model.zip', 'cpu')
    assert cfg.run_dir == tmp_path/'cli'
