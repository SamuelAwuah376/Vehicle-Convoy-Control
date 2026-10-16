from dataclasses import dataclass, field, replace
from time import perf_counter
import numpy as np
from ..config import Planner, Scenario
from ..coordination.centralized import CentralizedTeam
from ..rl.scripted import LaneKeeping, Overtaker
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
    planner: Planner = None
    planner_log: object = None
    intention_records: list = field(default_factory=list)
    combo_multiplier: np.ndarray = None   # (T-1,) team multiplier after each step
    max_combo: int = 0
    maneuvers: list = field(default_factory=list)   # Scorer.maneuvers award dicts
    team_lat: np.ndarray = None   # (T, 3, 2) body [vy, yaw rate] (kinematic relations for the kinematic plant)


POLICIES = ('lane_keeping', 'overtaker')


def make_controller(name, scenario, world, planner):
    if name == 'lane_keeping':
        return LaneKeeping(scenario, world.road, world.plant)
    if name == 'overtaker':
        return CentralizedTeam(scenario, world, Overtaker(world.road, world.plant.params), planner)
    raise ValueError(f'unknown policy {name!r}')


def run(scenario=None, seed=0, policy='lane_keeping', planner=None, score=None):
    """policy: a name from POLICIES, or a callable world -> controller (snapshot -> (3, 2) controls).
    score: a Score replacing scenario.score (None keeps it)."""
    scenario, planner = scenario or Scenario(), planner or Planner()
    if score is not None:
        scenario = replace(scenario, score=score)
    world = World(scenario, seed)
    policy = make_controller(policy, scenario, world, planner) if isinstance(policy, str) else policy(world)
    detector = EventDetector(world.plant.params, world.road)
    scorer = Scorer(scenario.score)
    snap = world.snapshot
    detector.update(snap, 0.0)
    snaps, controls, increments, pair_c, traffic_c, compute, overtakes, combo = [snap], [], [], [], [], [], [], []
    steps = int(round(scenario.timing.duration/scenario.timing.dt))
    for _ in range(steps):
        start = perf_counter()
        command = policy(snap)
        compute.append(perf_counter()-start)
        snap, applied = world.step(command)
        m = detector.update(snap, scenario.timing.dt)
        increments.append(scorer.update(m)); combo.append(scorer.multiplier)
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
                   overtakes, scorer.awarded, list(snap.collisions), 'collision' if snap.done else 'time',
                   planner if hasattr(policy, 'log') else None, getattr(policy, 'log', None),
                   list(policy.intentions.records) if hasattr(policy, 'intentions') else [],
                   np.array(combo), scorer.max_combo, scorer.maneuvers,
                   np.stack((stack('team_vy'), stack('team_r')), -1))
