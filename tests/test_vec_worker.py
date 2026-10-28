import functools
import os
import sys
import cloudpickle
import gymnasium
import numpy as np
from gymnasium import spaces
from nohesi.rl.evaluate import ObsNormalizer, run_policy
from nohesi.rl.vec_worker import subproc_vec_env


class CountEnv(gymnasium.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (1,), np.float32)
    action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(1, np.float32), {'reset': True}

    def step(self, action):
        self.t += 1
        return np.full(1, self.t, np.float32), float(action), self.t >= 3, False, {}

    def torch_loaded(self):
        return 'torch' in sys.modules


class CrashEnv(CountEnv):
    """Dies, as a native solver crash ends a worker, at its second step or in risky(), unless reset with a seed (the
    vec env resets a replacement worker with one)."""

    def reset(self, seed=None, options=None):
        self.seeded = seed is not None
        return super().reset(seed, options)

    def step(self, action):
        if self.t == 1 and not self.seeded:
            os._exit(3)
        return super().step(action)

    def risky(self):
        if not self.seeded:
            os._exit(3)
        return 'ok'


class EpisodeEnv(CountEnv):
    """3-step episodes reporting episode_metrics; the first worker to reach step 2 of seed 1001 dies (marker file)."""

    def __init__(self, marker):
        self.marker = marker

    def reset(self, seed=None, options=None):
        self.episode_seed = seed
        return super().reset(seed, options)

    def action_masks(self):
        return np.ones(2, bool)

    def step(self, action):
        if self.t == 1 and self.episode_seed == 1001 and not os.path.exists(self.marker):
            open(self.marker, 'w').close()
            os._exit(3)
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            info['episode_metrics'] = {'seed': self.episode_seed}
        return obs, reward, terminated, truncated, info


class Zeros:
    def predict(self, obs, action_masks=None, deterministic=True):
        return np.zeros(len(obs), int), None


def by_value(test):
    """Spawned workers cannot import this test module: pickle its classes by value for the test's duration."""
    @functools.wraps(test)
    def wrapped(*args, **kwargs):
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        try:
            return test(*args, **kwargs)
        finally:
            cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
    return wrapped


@by_value
def test_dead_worker_is_replaced_and_its_step_truncated():
    logs = []
    v = subproc_vec_env([CrashEnv, CountEnv], log=logs.append)
    try:
        v.reset()
        v.step(np.array([0, 1]))
        obs, rew, done, info = v.step(np.array([1, 1]))   # worker 0 dies in this step
        assert done[0] and info[0]['worker_restart'] and info[0]['TimeLimit.truncated'] and rew[0] == 0
        assert info[0]['terminal_observation'][0] == 1 and obs[0, 0] == 0   # last observation; a fresh episode
        assert not done[1] and obs[1, 0] == 2                                 # the other worker carries on
        obs, _, done, _ = v.step(np.array([1, 1]))
        assert obs[0, 0] == 1 and done[1]                                     # the replacement steps on
        assert v.restarts == 1 and logs == ['[vec env] worker 0 died (exit code 3): restarted (1 so far)']
        assert v.env_method('torch_loaded') == [False, False]
    finally:
        v.close()


@by_value
def test_env_method_on_a_dying_worker_is_sent_again_to_its_replacement():
    v = subproc_vec_env([CrashEnv, CrashEnv], log=lambda message: None)
    try:
        v.reset()
        assert v.env_method('risky') == ['ok', 'ok'] and v.restarts == 2
    finally:
        v.close()


@by_value
def test_evaluation_reruns_an_episode_whose_worker_died(tmp_path):
    make = functools.partial(EpisodeEnv, str(tmp_path/'died'))
    v = subproc_vec_env([make, make], log=lambda message: None)
    try:
        out = run_policy(Zeros(), ObsNormalizer(), v, [1000, 1001, 1002])
        assert [(seed, m['seed'], steps) for seed, m, steps in out] == [(1000, 1000, 3), (1001, 1001, 3), (1002, 1002, 3)]
        assert v.restarts == 1
    finally:
        v.close()


def test_lean_workers_step_reset_and_never_load_torch():
    cloudpickle.register_pickle_by_value(sys.modules[__name__])   # spawned workers cannot import this test module
    v = subproc_vec_env([CountEnv, CountEnv])
    try:
        assert v.reset().shape == (2, 1)
        for t in (1, 2):
            obs, rew, done, info = v.step(np.array([0, 1]))
            assert np.all(obs[:, 0] == t) and list(rew) == [0, 1] and not done.any()
        obs, _, done, info = v.step(np.array([1, 1]))
        assert done.all() and np.all(obs == 0) and info[0]['terminal_observation'][0] == 3
        assert v.reset_infos[0] == {'reset': True}
        assert v.env_method('torch_loaded') == [False, False]
        assert v.env_is_wrapped(gymnasium.Wrapper) == [False, False]
    finally:
        v.close()
        cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
