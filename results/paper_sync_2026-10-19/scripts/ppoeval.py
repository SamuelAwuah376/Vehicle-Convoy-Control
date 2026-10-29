import sys, json, os
sys.path.insert(0, '/tmp/np')
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from nppolicy import NumpyPolicy
from nohesi.rl.env import NoHesiEnv
from nohesi.rl.evaluate import load_env_cfg, resolve, summarize

def episode(args):
    run, which, score, seed = args
    fn = f'/tmp/out/ppo_{os.path.basename(run)}_{which}_{score}_{seed}.json'
    if os.path.exists(fn): return fn
    cfg = replace(load_env_cfg(run), score=score)
    pol = NumpyPolicy(*resolve(run, which))
    env = NoHesiEnv(cfg)
    obs, _ = env.reset(seed=seed); n = 0
    while True:
        obs, r, term, trunc, info = env.step(pol(obs, env.action_masks())); n += 1
        if term or trunc: break
    m = info['episode_metrics']; m.pop('scenario', None); m.pop('planner_config', None)
    m['summary'] = summarize(m, n)
    json.dump(m, open(fn, 'w'), default=float)
    return fn

if __name__ == '__main__':
    jobs = [tuple(j.split(':')[:3])+(int(j.split(':')[3]),) for j in sys.argv[1:]]
    with ProcessPoolExecutor(2) as ex:
        for f in ex.map(episode, jobs): print(f, flush=True)
