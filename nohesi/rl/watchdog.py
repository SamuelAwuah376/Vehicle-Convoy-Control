"""Step watchdog for training workers.

A solver hang inside compiled code never returns (seen once: HPIPM iterating without end in an emergency solve), so
the worker stops answering and the whole run waits on it. A daemon thread in each worker checks how long the env's
current step has run; past the timeout it writes the Python stack and the nonlinear MPC's last QP (acados'
dump_last_qp_to_json, for an offline repro) to the run's hangs/ folder, then ends the worker. The trainer then fails,
and its supervisor (train_rl_supervised.py) resumes from the latest checkpoint.

The trainer sets NOHESI_HANG_DIR (run_dir/hangs) before spawning workers; NOHESI_STEP_TIMEOUT overrides the timeout
(s, default 300: a normal decision takes well under 10 s).
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
    while True:
        time.sleep(poll)
        started = getattr(env, 'step_started', None)
        if started is None or time.monotonic()-started < timeout:
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
