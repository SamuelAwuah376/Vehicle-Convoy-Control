"""Planner m3.4 safety layer: sub-sampled footprint checks and escape rollouts that start later on a plan."""
import numpy as np
from nohesi.config import SCENARIOS, Planner
from nohesi.planning.fallback import dense, first_conflict, lane_keep_plan
from nohesi.world import World

VEH = SCENARIOS['v4']().vehicle


def test_dense_interpolates_between_samples():
    assert dense([0., 1., 3.], 2).tolist() == [0., 0.5, 1., 2., 3.]
    assert dense(np.array([[0., 2.], [10., 20.]]), 2).tolist() == [[0., 1., 2.], [10., 15., 20.]]
    assert dense([1., 2.], 1).tolist() == [1., 2.]


def test_substeps_catch_a_fast_pass_between_plan_nodes():
    ego = np.zeros((4, 4))                              # parked at s = 0, d = 0
    s = np.array([-30., -10., 10., 30.])                # passes straight through the ego between nodes 1 and 2
    other = [(('team', 1), s, np.zeros(4), np.zeros(4))]
    assert first_conflict(ego, other, VEH, 0.1, substeps=1) is None
    assert first_conflict(ego, other, VEH, 0.1, substeps=5) == (('team', 1), 2)


def test_lane_keep_plan_can_start_later_on_a_plan():
    sc = SCENARIOS['v4']()
    world = World(sc, 0)
    snap, cfg = world.snapshot, Planner()
    i, lane = 0, sc.team_lane
    full = lane_keep_plan(snap, world.road, world.plant, i, lane, 28.0, cfg)
    k0 = 5
    tail = lane_keep_plan(snap, world.road, world.plant, i, lane, 28.0, cfg, start=(full.z[k0], full.u[k0-1, 1], k0))
    assert tail.z.shape == (cfg.horizon+1-k0, 4) and tail.u.shape == (cfg.horizon-k0, 2)
    assert np.isclose(tail.time, snap.time+k0*cfg.step)
    assert np.allclose(tail.z[0], full.z[k0])           # starts where the given plan state is
    assert np.allclose(tail.z[1:, 0], full.z[k0+1:, 0], atol=0.5)   # and follows the same lane-keeping law
