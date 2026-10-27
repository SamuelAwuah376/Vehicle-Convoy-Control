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


def subproc_vec_env(env_fns):
    """SB3 SubprocVecEnv (spawn) whose workers run `worker` above instead of SB3's (imported lazily: the parent process
    has stable_baselines3 loaded anyway, the workers must not)."""
    import multiprocessing as mp
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

    class LeanSubprocVecEnv(SubprocVecEnv):
        def __init__(self, env_fns):   # SubprocVecEnv.__init__ with this module's worker as the process target
            self.waiting, self.closed = False, False
            ctx = mp.get_context('spawn')
            self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns])
            self.processes = []
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
                process = ctx.Process(target=worker, args=(work_remote, remote, Thunk(env_fn)), daemon=True)
                process.start()
                self.processes.append(process)
                work_remote.close()
            self.remotes[0].send(('get_spaces', None))
            observation_space, action_space = self.remotes[0].recv()
            VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    return LeanSubprocVecEnv(env_fns)
