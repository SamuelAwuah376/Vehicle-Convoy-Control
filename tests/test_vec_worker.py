import sys
import cloudpickle
import gymnasium
import numpy as np
from gymnasium import spaces
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
