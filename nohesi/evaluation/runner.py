from dataclasses import dataclass, field, replace
from time import perf_counter
import numpy as np
from ..config import Planner, Scenario
from ..coordination.centralized import CentralizedTeam
from ..perception.system import make_perception
from ..rl.scripted import LaneKeeping, OvertakeMacro, Overtaker
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
    gaps: list = None   # with EpisodeRun(record_gaps=True): per sim step, per car (phase, lane, front, rear,
                        # overtake stage or None, overtaken car or None): the intention after that step's decision.
                        # Its keys are true ids (under perception, each track id is mapped to its matched true car,
                        # None for a track matched to none), so they resolve against traffic_id
    views: list = None  # with record_gaps and perception (not shadow): per frame, the perceived traffic the decision
                        # read, (s, d, track id, matched true id or -1): planner_log.conflicts and the intention
                        # records ('pass <id>') carry those track ids


POLICIES = ('lane_keeping', 'overtaker', 'overtake_macro')


def make_controller(name, scenario, world, planner):
    if name == 'lane_keeping':
        return LaneKeeping(scenario, world.road, world.plant)
    if name in ('overtaker', 'overtake_macro'):
        policy = (Overtaker if name == 'overtaker' else OvertakeMacro)(
            world.road, world.plant.params, desired=scenario.overtaker_speed, look=scenario.overtaker_look)
        return CentralizedTeam(scenario, world, policy, planner)
    raise ValueError(f'unknown policy {name!r}')


class EpisodeRun:
    """run() one sim step at a time: step() advances one dt, episode() builds the Episode so far.
    world: (scenario, seed) -> world (World, or carla.world.CarlaWorld from a CarlaSession). record_gaps: keep the
    intention layer's state per step (Episode.gaps, for drawing the locked gaps). perception: (scenario, world, seed)
    -> perception system or None (perception.system; the default builds the scenario's, None when it is off).

    snap is the true snapshot: scoring, the event detector, collisions, done and the Episode arrays read it. view is
    what the policy, and so the coordinator, intentions and planners, is called with: the perception system's output,
    or snap itself with perception off (or in mode 'shadow')."""
    def __init__(self, scenario=None, seed=0, policy='lane_keeping', planner=None, score=None, world=World,
                 record_gaps=False, perception=make_perception):
        scenario, planner = scenario or Scenario(), planner or Planner()
        planner = replace(planner, speed_max=scenario.team_speed_max)   # the scenario owns the team speed cap
        if score is not None:
            scenario = replace(scenario, score=score)
        self.scenario, self.seed, self.planner = scenario, seed, planner
        self.world = world(scenario, seed)
        self.policy = make_controller(policy, scenario, self.world, planner) if isinstance(policy, str) else policy(self.world)
        self.detector = EventDetector(self.world.plant.params, self.world.road)
        self.scorer = Scorer(scenario.score)
        self.snap = self.world.snapshot
        self.perception = perception(scenario, self.world, seed) if perception is not None else None
        self.shadow = scenario.perception.mode == 'shadow'
        self._view = self.perception.reset(self.snap) if self.perception is not None else None
        self.detector.update(self.snap, 0.0)
        self.snaps, self.controls, self.increments, self.pair_c, self.traffic_c = [self.snap], [], [], [], []
        self.compute, self.overtakes, self.combo = [], [], []
        self.steps = int(round(scenario.timing.duration/scenario.timing.dt))
        self.gaps = [] if record_gaps and hasattr(self.policy, 'intentions') else None
        self.views = [] if record_gaps and self.perception is not None and not self.shadow else None

    @property
    def done(self):
        return self.snap.done or len(self.controls) >= self.steps

    @property
    def view(self):
        """The snapshot the policy sees: the perceived one, or the true one (perception off or shadow)."""
        return self.snap if self.perception is None or self.shadow else self._view

    def step(self):
        """One sim step; returns the Scorer increments (term -> (3,))."""
        start = perf_counter()
        command = self.policy(self.view)
        self.compute.append(perf_counter()-start)
        truth = None
        if self.views is not None:
            truth = _matched_ids(self.view, self.snap)
            self.views.append((self.view.traffic_s.copy(), self.view.traffic_d.copy(), self.view.traffic_id.copy(),
                               truth))
        if self.gaps is not None:
            self.gaps.append([(it.phase, it.lane, _true_key(it.front, self.view, truth),
                               _true_key(it.rear, self.view, truth), it.overtake.stage if it.overtake else None,
                               _true_key(it.overtake.car, self.view, truth) if it.overtake else None)
                              for it in self.policy.intentions.intentions])
        self.snap, applied = self.world.step(command)
        if self.perception is not None:
            self._view = self.perception.step(self.snap, applied[:, 0].copy())
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
                       list(scorer.train_pass), scorer.train_time, self.gaps, self.views)


MATCH_GATE = 3.0   # m: a track within this of a true car (one to one, nearest first) is that car's (Episode.gaps)


def _matched_ids(view, snap):
    """(n,) the true id of each of the perceived snapshot's tracks: the true cars assigned one to one by distance
    (Hungarian, within MATCH_GATE), -1 for a track assigned to none (a false or duplicate track)."""
    from scipy.optimize import linear_sum_assignment
    out = np.full(len(view.traffic_s), -1)
    if len(out) == 0 or len(snap.traffic_s) == 0:
        return out
    D = np.hypot(view.traffic_s[:, None]-snap.traffic_s[None], view.traffic_d[:, None]-snap.traffic_d[None])
    rows, cols = linear_sum_assignment(np.where(D < MATCH_GATE, D, 1e6))
    ok = D[rows, cols] < MATCH_GATE
    out[rows[ok]] = np.asarray(snap.traffic_id)[cols[ok]]
    return out


def _true_key(key, view, truth):
    """A ('traffic', id) gap key of the view with its true id (None when the track matched no car); team keys and
    keys without perception (truth None) as they are."""
    if truth is None or key is None or key[0] != 'traffic':
        return key
    j = np.flatnonzero(view.traffic_id == key[1])
    return (key[0], int(truth[j[0]])) if len(j) and truth[j[0]] >= 0 else None


def run(scenario=None, seed=0, policy='lane_keeping', planner=None, score=None, world=World, record_gaps=False,
        perception=make_perception):
    """policy: a name from POLICIES, or a callable world -> controller (snapshot -> (3, 2) controls).
    score: a Score replacing scenario.score (None keeps it). world, record_gaps, perception: as EpisodeRun."""
    r = EpisodeRun(scenario, seed, policy, planner, score, world, record_gaps, perception)
    while not r.done:
        r.step()
    return r.episode()
