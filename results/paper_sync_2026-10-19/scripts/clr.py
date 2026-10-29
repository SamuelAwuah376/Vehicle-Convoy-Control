import numpy as np, json
from concurrent.futures import ProcessPoolExecutor
from nohesi.config import SCENARIOS, SCORES, Planner
from nohesi.evaluation.runner import run
def f(seed):
    ep = run(SCENARIOS['v1'](), seed, 'overtaker', Planner(), score=SCORES['v2']())
    c = [o.min_clearance for o, _ in ep.awarded]
    allc = [o.min_clearance for o in ep.overtakes]
    k = int(np.nanargmin(np.nanmin(ep.traffic_clearance, axis=1))); i = int(np.nanargmin(ep.traffic_clearance[k]))
    phases = [r for r in ep.intention_records if r.vehicle == i and abs(r.time-ep.time[k]) < 4]
    return seed, c, allc, float(ep.time[k]), i, float(np.nanmin(ep.traffic_clearance)), [(round(r.time,1), r.kind) for r in phases]
with ProcessPoolExecutor(2) as ex:
    res = list(ex.map(f, range(10)))
C = np.concatenate([r[1] for r in res]); A = np.concatenate([r[2] for r in res])
print('scored pass clearance median %.3f mean %.3f min %.3f' % (np.median(C), C.mean(), C.min()), 'closeness mean %.3f' % np.mean(1+(1.5-C)/1.5))
print('n scored', len(C), 'n overtakes', len(A), 'frac scored below 1.0m', np.mean(C < 1.0))
for r in res: print(r[0], 'minclr %.3f at t=%.2f car %d' % (r[5], r[3], r[4]), r[6])
