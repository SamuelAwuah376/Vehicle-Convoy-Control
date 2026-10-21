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
    draft_pass: list = field(default_factory=list)   # per awarded pass: was it a draft pass
    train_pass: list = field(default_factory=list)   # per awarded pass: did it get the draft-train bonus
    train_time: float = 0.0                          # s the three-car draft train held


POLICIES = ('lane_keeping', 'overtaker')


def make_controller(name, scenario, world, planner):
    if name == 'lane_keeping':
        return LaneKeeping(scenario, world.road, world.plant)
    if name == 'overtaker':
        return CentralizedTeam(scenario, world, Overtaker(world.road, world.plant.params, desired=scenario.overtaker_speed,
                                                          look=scenario.overtaker_look), planner)
    raise ValueError(f'unknown policy {name!r}')


class EpisodeRun:
    """run() one sim step at a time: step() advances one dt, episode() builds the Episode so far."""
    def __init__(self, scenario=None, seed=0, policy='lane_keeping', planner=None, score=None):
        scenario, planner = scenario or Scenario(), planner or Planner()
        planner = replace(planner, speed_max=scenario.team_speed_max)   # the scenario owns the team speed cap
        if score is not None:
            scenario = replace(scenario, score=score)
        self.scenario, self.seed, self.planner = scenario, seed, planner
        self.world = World(scenario, seed)
        self.policy = make_controller(policy, scenario, self.world, planner) if isinstance(policy, str) else policy(self.world)
        self.detector = EventDetector(self.world.plant.params, self.world.road)
        self.scorer = Scorer(scenario.score)
        self.snap = self.world.snapshot
        self.detector.update(self.snap, 0.0)
        self.snaps, self.controls, self.increments, self.pair_c, self.traffic_c = [self.snap], [], [], [], []
        self.compute, self.overtakes, self.combo = [], [], []
        self.steps = int(round(scenario.timing.duration/scenario.timing.dt))

    @property
    def done(self):
        return self.snap.done or len(self.controls) >= self.steps

    def step(self):
        """One sim step; returns the Scorer increments (term -> (3,))."""
        start = perf_counter()
        command = self.policy(self.snap)
        self.compute.append(perf_counter()-start)
        self.snap, applied = self.world.step(command)
        m = self.detector.update(self.snap, self.scenario.timing.dt)
        inc = self.scorer.update(m)
        self.increments.append(inc); self.combo.append(self.scorer.multiplier)
        self.snaps.append(self.snap); self.controls.append(applied)
        self.pair_c.append(m.pair_clearance); self.traffic_c.append(m.traffic_clearance)
        self.overtakes.extend(m.overtakes)
        return inc

    def episode(self):
        policy, snap, scorer = self.policy, self.snap, self.scorer
        stack = lambda name: np.array([getattr(s, name) for s in self.snaps])
        return Episode(self.scenario, self.seed, getattr(policy, 'name', type(policy).__name__),
                       stack('time'), stack('team'), stack('team_s'), stack('team_d'),
                       stack('traffic_s'), stack('traffic_d'), stack('traffic_v'), stack('traffic_id'),
                       np.array(self.controls), {t: np.array([inc[t] for inc in self.increments]) for t in TERMS},
                       np.array(self.pair_c), np.array(self.traffic_c), np.array(self.compute),
                       list(self.overtakes), list(scorer.awarded), list(snap.collisions), 'collision' if snap.done else 'time',
                       self.planner if hasattr(policy, 'log') else None, getattr(policy, 'log', None),
                       list(policy.intentions.records) if hasattr(policy, 'intentions') else [],
                       np.array(self.combo), scorer.max_combo, list(scorer.maneuvers),
                       np.stack((stack('team_vy'), stack('team_r')), -1), list(scorer.draft_pass),
                       list(scorer.train_pass), scorer.train_time)


def run(scenario=None, seed=0, policy='lane_keeping', planner=None, score=None):
    """policy: a name from POLICIES, or a callable world -> controller (snapshot -> (3, 2) controls).
    score: a Score replacing scenario.score (None keeps it)."""
    r = EpisodeRun(scenario, seed, policy, planner, score)
    while not r.done:
        r.step()
    return r.episode()
