from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from ..config import Scenario
from ..rl.scripted import LaneKeeping
from ..scoring.events import EventDetector
from ..scoring.score import Scorer, TERMS
from ..world import World


@dataclass
class Episode:
    scenario: Scenario
    seed: int
    policy: str
    time: np.ndarray            # (T,)
    team: np.ndarray            # (T, 3, 4)
    team_s: np.ndarray          # (T, 3)
    team_d: np.ndarray
    traffic_s: np.ndarray       # (T, N)
    traffic_d: np.ndarray
    traffic_v: np.ndarray
    traffic_id: np.ndarray
    controls: np.ndarray        # (T-1, 3, 2) applied [acceleration, steering]
    score: dict                 # term -> (T-1, 3) per-step increments
    pair_clearance: np.ndarray  # (T-1, 3)
    traffic_clearance: np.ndarray
    compute_time: np.ndarray    # (T-1,) policy seconds per step
    overtakes: list = field(default_factory=list)
    awarded: list = field(default_factory=list)
    collisions: list = field(default_factory=list)
    end_reason: str = 'time'


def run(scenario=None, seed=0, policy=None):
    scenario = scenario or Scenario()
    world = World(scenario, seed)
    policy = policy or LaneKeeping(scenario, world.road, world.plant)
    detector = EventDetector(world.plant.params, world.road)
    scorer = Scorer(scenario.score)
    snap = world.snapshot
    detector.update(snap, 0.0)
    snaps, controls, increments, pair_c, traffic_c, compute, overtakes = [snap], [], [], [], [], [], []
    steps = int(round(scenario.timing.duration/scenario.timing.dt))
    for _ in range(steps):
        start = perf_counter()
        command = policy(snap)
        compute.append(perf_counter()-start)
        snap, applied = world.step(command)
        m = detector.update(snap, scenario.timing.dt)
        increments.append(scorer.update(m))
        snaps.append(snap); controls.append(applied)
        pair_c.append(m.pair_clearance); traffic_c.append(m.traffic_clearance)
        overtakes.extend(m.overtakes)
        if snap.done:
            break
    stack = lambda name: np.array([getattr(s, name) for s in snaps])
    return Episode(scenario, seed, getattr(policy, 'name', type(policy).__name__),
                   stack('time'), stack('team'), stack('team_s'), stack('team_d'),
                   stack('traffic_s'), stack('traffic_d'), stack('traffic_v'), stack('traffic_id'),
                   np.array(controls), {t: np.array([inc[t] for inc in increments]) for t in TERMS},
                   np.array(pair_c), np.array(traffic_c), np.array(compute),
                   overtakes, scorer.awarded, list(snap.collisions), 'collision' if snap.done else 'time')
