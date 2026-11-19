"""The perception seam: what a run calls to turn the true snapshot into the one the team plans on (design sections 3,
7 and 11).

A perception system implements two calls (PerceptionProtocol; nohesi.perception.pipeline.PerceptionSystem is the real
one, PassThrough the identity):
- reset(snap) -> Snapshot: once per episode, with the initial true snapshot, before the first decision (the place
  for the synthetic pre-roll);
- step(snap, accel) -> Snapshot: after every world step, with the new true snapshot and the (3,) accelerations the
  plant applied in that step (camera pitch, ego motion).
Both return the perceived snapshot: the team fields as given (each car knows its own state, its teammates' over V2V),
the traffic fields from the belief (track ids in traffic_id; traffic_target_lane the most likely lane; the optional
traffic_sig_s, traffic_sig_d, traffic_sig_v and traffic_p_exist), and collisions and done copied from the truth.
Return the same object when nothing should change, and never modify the snapshot passed in (its arrays are the
world's).

evaluation.runner.EpisodeRun keeps the true snapshot in run.snap (scoring, the event detector, collisions, done and
the Episode arrays) and calls the policy, so the coordinator, the intentions and the planners, with run.view. The RL
env (rl.env.NoHesiEnv) builds its observation, masks and escapes from run.view and its reward from run.snap. In mode
'shadow' the run still calls the system every step, but the view stays the true snapshot.
"""
from typing import Protocol

__all__ = ['PerceptionProtocol', 'PassThrough', 'make_perception']


class PerceptionProtocol(Protocol):
    def reset(self, snap):
        """Initial true snapshot -> initial perceived snapshot."""

    def step(self, snap, accel):
        """New true snapshot and the (3,) applied accelerations -> perceived snapshot."""


def make_perception(scenario, world, seed):
    """The episode's perception system: None when scenario.perception is off (the run then reads the true snapshot,
    exactly as before), else the pipeline on the world's road and vehicle parameters, its RNG seeded from `seed`."""
    cfg = scenario.perception
    if not cfg.enabled:
        return None
    from .pipeline import PerceptionSystem
    return PerceptionSystem(cfg, scenario, world.road, world.plant.params, seed)


class PassThrough:
    """The identity system: the view is the true snapshot itself (seam tests, the byte-identical check)."""
    def reset(self, snap):
        return snap

    def step(self, snap, accel):
        return snap
