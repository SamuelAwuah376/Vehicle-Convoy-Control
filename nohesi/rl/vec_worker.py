"""Torch-free SubprocVecEnv worker. SB3's own worker lives in stable_baselines3, whose import loads torch and the CUDA
libraries (~0.5 GB RAM, ~0.8 GB commit per process) into every env worker, which never use them; 24 training + 4 eval
workers ran the machine out of RAM. This module imports nothing from stable_baselines3 or torch, so a spawned worker
loads only the env. Same pipe protocol as stable_baselines3.common.vec_env.subproc_vec_env._worker."""
import cloudpickle


class Thunk:
    """cloudpickle-wrapped env factory (spawned workers receive it pickled)."""

    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, state):
        self.fn = cloudpickle.loads(state)


def _is_wrapped(env, wrapper_class):
    while env is not None:
        if isinstance(env, wrapper_class):
            return True
        env = getattr(env, 'env', None)
    return False


def worker(remote, parent_remote, thunk):
    parent_remote.close()
    env = thunk.fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info['TimeLimit.truncated'] = truncated and not terminated
                reset_info = {}
                if done:
                    info['terminal_observation'] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == 'reset':
                options = {'options': data[1]} if data[1] else {}
                remote.send(env.reset(seed=data[0], **options))
            elif cmd == 'render':
                remote.send(env.render())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                remote.send(env.get_wrapper_attr(data[0])(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(env.get_wrapper_attr(data))
            elif cmd == 'has_attr':
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == 'is_wrapped':
                remote.send(_is_wrapped(env, data))
            else:
                raise NotImplementedError(f'`{cmd}` is not implemented in the worker')
        except (EOFError, KeyboardInterrupt):
            break


DEAD = (EOFError, BrokenPipeError, ConnectionResetError)   # a worker's pipe after the worker died


def _log(message):
    print(message, flush=True)


def subproc_vec_env(env_fns, log=_log):
    """SB3 SubprocVecEnv (spawn) whose workers run `worker` above instead of SB3's (imported lazily: the parent process
    has stable_baselines3 loaded anyway, the workers must not).

    A worker that dies (a native crash in the solver: the blasfeo access violations of batch 2c, a CPU fault the
    solver's assembly kernels give Windows no way to unwind, so ctypes can't turn it into an exception; or the step
    watchdog ending a hung step) is replaced instead of ending the run: its step returns done with
    TimeLimit.truncated (zero reward, the last observation as terminal_observation: PPO bootstraps it) and
    info['worker_restart'], and the new worker starts a fresh episode (env RNG reseeded, so it doesn't replay the dead
    one's seeds). env_method calls to it are sent again to the new worker. `log` gets one line per restart."""
    import multiprocessing as mp
    import numpy as np
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
    from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs

    class LeanSubprocVecEnv(SubprocVecEnv):
        def __init__(self, env_fns):   # SubprocVecEnv.__init__ with this module's worker as the process target
            self.waiting, self.closed = False, False
            self.ctx, self.env_fns, self.restarts = mp.get_context('spawn'), list(env_fns), 0
            self.rng, self.dead = np.random.default_rng(), set()
            self.remotes, self.processes = map(list, zip(*[self._spawn(i) for i in range(len(env_fns))]))
            self.remotes[0].send(('get_spaces', None))
            observation_space, action_space = self.remotes[0].recv()
            VecEnv.__init__(self, len(env_fns), observation_space, action_space)
            self.last_obs = [None]*len(env_fns)

        def _spawn(self, i):
            remote, work_remote = self.ctx.Pipe()
            process = self.ctx.Process(target=worker, args=(work_remote, remote, Thunk(self.env_fns[i])), daemon=True)
            process.start()
            work_remote.close()
            return remote, process

        def _restart(self, i):
            """Replace dead worker i with a fresh one on a new episode; returns its first observation."""
            old = self.processes[i]
            old.join(timeout=10)
            if old.is_alive():
                old.terminate()
            code = old.exitcode
            try:
                self.remotes[i].close()
            except OSError:
                pass
            self.remotes[i], self.processes[i] = self._spawn(i)
            self.restarts += 1
            log(f'[vec env] worker {i} died (exit code {code if code is None or code < 256 else hex(code)}): '
                f'restarted ({self.restarts} so far)')
            self.remotes[i].send(('reset', (int(self.rng.integers(2000, 2**31-1)), None)))
            obs, _ = self.remotes[i].recv()
            return obs

        def _send(self, i, message):
            try:
                self.remotes[i].send(message)
            except DEAD:
                self.dead.add(i)

        def step_async(self, actions):
            for i, action in enumerate(actions):
                self._send(i, ('step', action))
            self.waiting = True

        def step_wait(self):
            results = []
            for i, remote in enumerate(self.remotes):
                try:
                    if i in self.dead:
                        raise EOFError
                    results.append(remote.recv())
                except DEAD:
                    info = {'TimeLimit.truncated': True, 'terminal_observation': self.last_obs[i], 'worker_restart': True}
                    results.append((self._restart(i), 0.0, True, info, {}))
            self.dead.clear()
            self.waiting = False
            obs, rews, dones, infos, self.reset_infos = zip(*results)
            self.last_obs = list(obs)
            return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

        def reset(self):
            obs = super().reset()
            self.last_obs = list(obs)
            return obs

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            message = ('env_method', (method_name, method_args, method_kwargs))
            indices = list(self._get_indices(indices))
            for i in indices:
                self._send(i, message)
            out = []
            for i in indices:
                try:
                    if i in self.dead:
                        raise EOFError
                    out.append(self.remotes[i].recv())
                except DEAD:   # the call again, on the replacement (once: a second death raises)
                    self.dead.discard(i)
                    self.last_obs[i] = self._restart(i)
                    self.remotes[i].send(message)
                    out.append(self.remotes[i].recv())
            return out

    return LeanSubprocVecEnv(env_fns)
