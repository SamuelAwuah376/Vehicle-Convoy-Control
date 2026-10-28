"""Step watchdog for training workers.

A solver hang inside compiled code never returns (seen twice: HPIPM on a QP linearised around a diverged iterate,
now guarded by NonlinearMPC._sane), so
the worker stops answering and the whole run waits on it. A daemon thread in each worker checks how long the env's
current step has run; past the timeout it writes the Python stack and the nonlinear MPC's last QP (acados'
dump_last_qp_to_json, for an offline repro) to the run's hangs/ folder, then ends the worker. The trainer's vec env
then replaces the worker (vec_worker.subproc_vec_env: its episode is cut short) and training carries on.

The trainer sets NOHESI_HANG_DIR (run_dir/hangs) before spawning workers; NOHESI_STEP_TIMEOUT overrides the timeout
(s, default 300: a normal decision takes well under 10 s). Time the whole process was frozen (suspended to pause a run,
or the machine asleep: the watchdog's own poll then comes back late) doesn't count as step time.
"""
import faulthandler
import json
import os
from pathlib import Path
import threading
import time

ENV_DIR, ENV_TIMEOUT = 'NOHESI_HANG_DIR', 'NOHESI_STEP_TIMEOUT'


def watch(env, poll=5.0):
    """Start the watchdog for `env` (an object with step_started: monotonic time or None) when NOHESI_HANG_DIR is set."""
    folder = os.environ.get(ENV_DIR)
    if not folder:
        return None
    timeout = float(os.environ.get(ENV_TIMEOUT, 300))
    thread = threading.Thread(target=run, args=(env, Path(folder), timeout, poll), daemon=True, name='step-watchdog')
    thread.start()
    return thread


def run(env, folder, timeout, poll, exit=os._exit, once=False):
    """Watch until a step outlasts `timeout`; then dump (stack, last QP, episode info) and call exit(3)."""
    last, current, frozen = time.monotonic(), None, 0.0
    while True:
        time.sleep(poll)
        now = time.monotonic()
        gap, last = now-last, now
        started = getattr(env, 'step_started', None)
        if started != current:   # a new step (or none): its own frozen time starts at 0
            current, frozen = started, 0.0
        if gap > 3*poll+5:   # this thread was frozen too: the process was paused or the machine slept
            frozen += gap-poll
        if started is None or now-started-frozen < timeout:
            if once:
                return None
            continue
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder/f'hang_{os.getpid()}_{int(time.time())}'
        info = {'pid': os.getpid(), 'episode_seed': getattr(env, 'episode_seed', None),
                'step_seconds': round(time.monotonic()-started, 1)}
        try:
            info['sim_time'] = float(env.run.world.time)
        except Exception:
            pass
        with open(f'{stem}_stack.txt', 'w', encoding='utf-8') as f:
            faulthandler.dump_traceback(f, all_threads=True)
        try:   # the QP the solver is stuck on (qp_in is only read by the solver)
            env.team.mpc.solver.dump_last_qp_to_json(f'{stem}_qp.json', overwrite=True)
            info['qp'] = f'{stem}_qp.json'
        except Exception as e:
            info['qp_error'] = repr(e)
        Path(f'{stem}.json').write_text(json.dumps(info, indent=1, default=str), encoding='utf-8')
        return exit(3)
