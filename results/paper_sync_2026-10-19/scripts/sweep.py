import json, sys, os
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from nohesi.config import SCENARIOS, SCORES, Planner
from nohesi.evaluation.runner import run
from nohesi.evaluation.metrics import metrics

def job(a):
    scen, seed, pol, score, plant = a
    fn = f'/tmp/out/{scen}_{pol}_{score}_{seed}_{plant}.json'
    if os.path.exists(fn): return fn
    s = SCENARIOS[scen](); s = replace(s, vehicle=replace(s.vehicle, model=plant))
    ep = run(s, seed, pol, Planner(), score=SCORES[score]())
    m = metrics(ep); m.pop('scenario'); m.pop('planner_config')
    json.dump(m, open(fn, 'w'))
    return fn

jobs = []
for pol in ('overtaker', 'lane_keeping'):
    for sc in ('v2', 'v1', 'v0'):
        for seed in range(10): jobs.append(('v1', seed, pol, sc, 'kinematic'))
for scen in ('v1', 'v4'):
    for seed in range(5): jobs.append((scen, seed, 'overtaker', 'v3', 'dynamic'))
with ProcessPoolExecutor(2) as ex:
    for f in ex.map(job, jobs): print(f, flush=True)
