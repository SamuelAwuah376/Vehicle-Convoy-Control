"""experiments/yard_eval.py: config.json layouts, model / VecNormalize resolution, and the paired summary (no rollouts)."""
import csv
import importlib.util
import json
from pathlib import Path
import pytest

pytest.importorskip('gymnasium')
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('yard_eval', ROOT/'experiments'/'yard_eval.py')
ye = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ye)


def test_seeds():
    assert ye.parse_seeds('1000-1002, 1005,1001') == [1000, 1001, 1002, 1005]
    assert ye.seed_text([1003, 1000, 1001, 1002, 1007]) == '1000-1003,1007'


def test_config_layouts(tmp_path):
    from nohesi.yard.env import YardEnvConfig
    ye.WARNINGS.clear()
    (tmp_path/'config.json').write_text(json.dumps({   # the trainer's layout (nohesi.yard.train.write_config)
        'train': {'run_id': 'r', 'env': {'version': 'ye1', 'scenario': 'yard1_clear', 'followers': [3, 3],
                                         'w_park': 7.0},
                  'env_module': 'nohesi.yard.env', 'eval_seeds': [1000, 1001], 'norm_obs': True},
        'env_config_type': 'nohesi.yard.env.YardEnvConfig', 'git': {'commit': None}}))
    cfg, module, info = ye.load_run_config(tmp_path)
    assert cfg == YardEnvConfig(version='ye1', scenario='yard1_clear', followers=(3, 3), w_park=7.0) and module == 'nohesi.yard.env'
    assert info['eval_seeds'] == [1000, 1001]
    assert any('missing' in w for w in ye.WARNINGS)   # w_time etc. took today's defaults
    ye.WARNINGS.clear()
    (tmp_path/'config.json').write_text(json.dumps({'scenario': 'yard1', 'w_unheard_of': 1}))   # a bare env config
    cfg, module, _ = ye.load_run_config(tmp_path)
    assert cfg.scenario == 'yard1' and module == 'nohesi.yard.env'
    cfg, module, info = ye.load_run_config(None)
    assert cfg == YardEnvConfig() and info == {} and any('no config.json' in w for w in ye.WARNINGS)


def test_model_resolution(tmp_path):
    (tmp_path/'checkpoints').mkdir()
    for name in ('best_model.zip', 'checkpoints/model_100.zip', 'checkpoints/vecnormalize_100.pkl',
                 'checkpoints/model_200.zip', 'config.json'):
        (tmp_path/name).write_bytes(b'')
    model, norm = ye.resolve_model(tmp_path, 'latest')
    assert model.name == 'model_200.zip' and norm is None and ye.model_label(model) == 'ckpt200'
    model, norm = ye.resolve_model(tmp_path, '100')
    assert norm.name == 'vecnormalize_100.pkl'
    assert ye.resolve_model(tmp_path, str(tmp_path/'checkpoints'/'model_100'))[0].name == 'model_100.zip'
    model, norm = ye.resolve_model(tmp_path, 'best')
    assert ye.model_label(model) == 'best' and norm is None
    assert ye.config_dir(tmp_path/'checkpoints'/'model_100.zip', tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError, match='best'):   # lists what there is
        ye.resolve_model(tmp_path, 'final')


def _row(seed, followers, success, duration, collision=False):
    end = 'parked' if success else ('collision' if collision else 'stalled')
    return {'seed': seed, 'followers': followers, 'success': success, 'parked': followers if success else 0,
            'collision': collision, 'end_reason': end, 'duration_s': duration, 'safety_stops': 0.0, 'replans': 1.0,
            'back_off_m': 0.0, 'decisions': 10, 'parked_frac': None, 'replans_failed': 0.0, 'picks_rejected': 0.0,
            'reassigned': 0.0, 'path_length_m': 1.0, 'driven_m': 1.0}


def test_paired_summary_and_cases(tmp_path):
    ppo = [_row(1, 1, True, 50.0), _row(2, 2, True, 90.0), _row(3, 3, True, 70.0), _row(4, 3, False, 300.0, True),
           _row(5, 2, False, 200.0)]
    scripted = [_row(1, 1, True, 60.0), _row(2, 2, True, 80.0), _row(3, 3, True, 70.0), _row(4, 3, True, 100.0),
                _row(5, 2, False, 200.0)]
    rows = ye.merge(ppo, scripted)
    d = ye.paired(rows)
    assert d['seeds_both_solved'] == 3 and d['delta_duration_mean_s'] == pytest.approx(0.0)
    assert (d['ppo_faster'], d['ppo_slower'], d['equal']) == (1, 1, 1)
    assert d['solved_by_scripted_only'] == [4] and d['solved_by_neither'] == [5]
    assert d['by_followers']['2']['delta_duration_mean_s'] == pytest.approx(10.0)
    assert ye.select_cases(rows) == [('ppo_wins_most', 1), ('ppo_loses_most', 2), ('ppo_fails', 4)]
    s = ye.policy_summary(ppo)
    assert s['by_followers']['3'] == {'episodes': 2, 'success_rate': 0.5, 'duration_s': 185.0,
                                      'success_duration_s': 70.0}
    ye.write_csv(tmp_path/'per_seed.csv', rows)
    out = list(csv.DictReader((tmp_path/'per_seed.csv').open(encoding='utf-8')))
    assert out[0]['ppo_success'] == '1' and out[0]['delta_duration_s'] == '-10' and out[3]['delta_duration_s'] == ''
