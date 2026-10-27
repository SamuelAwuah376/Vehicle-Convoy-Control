"""Training-only braking penalty (EnvConfig.aux_brake) and the braking_mps metric."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.intentions import encode
from nohesi.rl.config import EnvConfig
from nohesi.rl.env import NoHesiEnv


def test_brake_penalty_charges_speed_braked_away():
    env = NoHesiEnv(replace(EnvConfig(scenario='v7', score='v8'), aux_brake=-0.1))
    env.reset(seed=0)
    total = 0.0
    for _ in range(6):
        n = len(env.run.controls)
        _, reward, _, _, info = env.step([encode(0, -1, 0)]*3)   # slow down: some braking
        braked = -np.minimum(np.array(env.run.controls[n:])[:, :, 0], 0).sum()*env.scenario.timing.dt
        assert info['aux']['brake'] == pytest.approx(-0.1*braked)
        assert reward == pytest.approx(sum(info['score_terms'].values())/env.cfg.reward_scale+sum(info['aux'].values()))
        total += braked
    assert total > 0
    plain = NoHesiEnv(EnvConfig())   # default 0: no brake term
    plain.reset(seed=0)
    assert 'brake' not in plain.step([encode(0, 0, 0)]*3)[4]['aux']


def test_braking_metric():
    from nohesi.config import SCENARIOS
    from nohesi.evaluation.metrics import metrics
    from nohesi.evaluation.runner import run
    s = SCENARIOS['v7']()
    ep = run(replace(s, timing=replace(s.timing, duration=10.0)), 0, 'lane_keeping')
    m = metrics(ep)
    assert m['braking_mps'] == pytest.approx((-np.minimum(ep.controls[:, :, 0], 0).sum(axis=0)*s.timing.dt).tolist())
