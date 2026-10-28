"""Run train_rl.py's run under a supervisor, for long runs: if training exits with an error, resume it from its latest
checkpoint, up to MAX_RESTARTS times (an env worker that dies, a native solver crash or the step watchdog ending a hung
solver, no longer ends the run: the vec env replaces the worker, see nohesi/rl/vec_worker.py). Launch this file
instead of train_rl.py (same settings, read from train_rl.py); output goes to the same stdout.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import train_rl as T

MAX_RESTARTS = 30


def latest_checkpoint(run_dir):
    """The checkpoint .zip with the most timesteps, or None."""
    ckpts = sorted(run_dir.glob('checkpoints/model_*.zip'), key=lambda p: int(re.search(r'model_(\d+)', p.name).group(1)))
    return ckpts[-1] if ckpts else None


def main():
    root = Path(__file__).resolve().parent
    run_dir = root/'runs_rl'/T.RUN_ID
    cmd = [sys.executable, str(root/'train_rl.py')]
    for attempt in range(MAX_RESTARTS+1):
        code = subprocess.call(cmd, cwd=root)
        if code == 0 or (run_dir/'final_model.zip').exists():
            return code
        ckpt = latest_checkpoint(run_dir)
        print(f'[supervisor] training exited with code {code}; restart {attempt+1}/{MAX_RESTARTS} from '
              f'{ckpt or "the start (no checkpoint yet)"}', flush=True)
        if ckpt is not None:
            cmd = [sys.executable, '-m', 'nohesi.rl.train', '--run-id', T.RUN_ID, '--resume', str(ckpt),
                   '--timesteps', str(T.TOTAL_TIMESTEPS)]
        time.sleep(10)
    return 1


if __name__ == '__main__':   # required: SubprocVecEnv spawns workers that re-import the training script on Windows
    sys.exit(main())
