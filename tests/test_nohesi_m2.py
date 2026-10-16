import numpy as np
import pytest
from dataclasses import replace
from nohesi.config import Planner, Scenario, Timing
from nohesi.coordination.centralized import CentralizedTeam
from nohesi.dynamics import KinematicPlant
from nohesi.evaluation.metrics import metrics
from nohesi.evaluation.runner import run
from nohesi.geometry import clearance, footprint
from nohesi.intentions import ACTIONS, IntentionManager, encode
from nohesi.planning.base import Obstacles, Target, constant_speed_plan
from nohesi.planning.single import LaneChangeMPC
from nohesi.rl.scripted import Overtaker
from nohesi.road import Road
from tests.test_nohesi import snapshot

SC, CFG = Scenario(), Planner()
ROAD, PLANT = Road(SC.road), KinematicPlant(SC.vehicle, SC.timing.dt)
P = PLANT.params


def drive(target, cars=(), state=(0., 0., 0., 28.), seconds=8.0, keys=None):
    """Closed-loop MPC on one vehicle; cars are (s0, d, v) at constant speed. Returns per-step history."""
    mpc = LaneChangeMPC(CFG, PLANT, ROAD)
    state, steer, accel, plan = np.array(state), 0.0, 0.0, None
    cars = np.array(cars, float).reshape(-1, 3)
    out = {'state': [], 'cars': [], 'plans': [], 'next': []}
    for k in range(int(round(seconds/SC.timing.dt))):
        t = k*SC.timing.dt
        cars_s = cars[:, 0]+cars[:, 2]*t
        if k % 2 == 0:
            horizon = np.arange(CFG.horizon+1)[:, None]*CFG.step
            obs = Obstacles(keys or tuple(('traffic', j) for j in range(len(cars))), cars_s+cars[:, 2]*horizon,
                            np.repeat(cars[None, :, 1], CFG.horizon+1, 0), np.repeat(cars[None, :, 2], CFG.horizon+1, 0))
            ref = plan.shifted(1) if plan is not None else constant_speed_plan(t, CFG.step, CFG.horizon, *state[[0, 1, 3]])
            plan = mpc.solve(t, state.copy(), steer, accel, target, obs, ref)
            assert plan.status == 'solved', (t, plan.status)
            out['plans'].append(plan); out['next'].append((k+2, plan.z[1]))
        state, accel, steer = PLANT.step(state, plan.u[0, 0], plan.u[0, 1], steer)
        out['state'].append(np.r_[state, steer]); out['cars'].append(cars_s+cars[:, 2]*SC.timing.dt)
    out['state'], out['cars'] = np.array(out['state']), np.array(out['cars'])
    return out


def lateral_accel(st):
    """Steady-state CG lateral acceleration v^2 cos(beta) tan(steer)/L from history rows [x, y, yaw, v, steer]."""
    beta = np.arctan(P.cg_to_rear_axle/P.wheelbase*np.tan(st[:, 4]))
    return st[:, 3]**2*np.cos(beta)*np.tan(st[:, 4])/P.wheelbase


def min_clearance(out, cars):
    worst = np.inf
    for st, cs in zip(out['state'], out['cars']):
        ego = footprint(st[:4], P)
        for (s0, d, v), s in zip(cars, cs):
            worst = min(worst, clearance(ego, footprint(np.array([s, d, 0, 0]), P)))
    return worst


def test_empty_road_lane_change_within_limits():
    out = drive(Target(d=3.6, speed=32.0))
    st, t = out['state'], np.arange(1, len(out['state'])+1)*SC.timing.dt
    assert np.all(np.abs(st[t > 4, 1]-3.6) < 0.1)        # done within 4 s
    assert st[-1, 3] == pytest.approx(32, abs=0.1)
    lateral = lateral_accel(st)
    assert np.max(np.abs(lateral)) < 1.05*P.lateral_accel_max   # cap applies before the rate limit
    assert np.max(np.abs(st[:, 1])) < ROAD.width/2-P.width/2
    # Model mismatch: planned next MPC state vs executed state.
    errors = [np.abs(st[k-1, :2]-z[:2]).max() for k, z in out['next'] if k <= len(st)]
    assert max(errors) < 0.05


def test_follows_slower_car_without_violating_margin():
    cars = [(40.0, 0.0, 20.0)]
    out = drive(Target(d=0.0, speed=32.0), cars, seconds=12)
    gap = out['cars'][:, 0]-out['state'][:, 0]-P.length
    assert gap.min() >= CFG.margin_long-0.05
    assert out['state'][-1, 3] == pytest.approx(20, abs=0.5)
    assert gap[-1] > 0.8*CFG.headway*20                    # soft headway pulls back toward headway*v


def test_drafts_teammate_at_shorter_headway():
    cars = [(40.0, 0.0, 20.0)]
    gaps = {}
    for key in (('team', 1), ('traffic', 0)):
        out = drive(Target(d=0.0, speed=32.0), cars, seconds=12, keys=(key,))
        gap = out['cars'][:, 0]-out['state'][:, 0]-P.length
        assert gap.min() >= CFG.margin_long-0.05
        assert out['state'][-1, 3] == pytest.approx(20, abs=0.5)
        gaps[key[0]] = gap[-1]
    # Settled bumper gap tracks each soft headway at 20 m/s; teammates are followed closer.
    assert gaps['team'] == pytest.approx(CFG.teammate_headway*20, abs=1.0)
    assert gaps['traffic'] == pytest.approx(CFG.headway*20, abs=1.0)
    assert gaps['team'] < gaps['traffic']


def test_lean_target_settles_off_centre():
    out = drive(Target(d=CFG.lean_offset, speed=28.0), seconds=6)
    assert np.all(np.abs(out['state'][-20:, 1]-CFG.lean_offset) < 0.05)


def test_merge_into_locked_gap_keeps_clearance():
    cars = [(-20.0, 3.6, 25.0), (35.0, 3.6, 25.0), (60.0, 0.0, 22.0)]
    out = drive(Target(d=3.6, speed=30.0, front=('traffic', 1), rear=('traffic', 0)), cars, (0., 0., 0., 25.), 8)
    assert abs(out['state'][-1, 1]-3.6) < 0.1
    s = out['state'][-1, 0]
    assert out['cars'][-1, 0]+P.length < s < out['cars'][-1, 1]-P.length
    assert min_clearance(out, cars) >= CFG.margin_long-0.1


def manager():
    return IntentionManager(CFG, ROAD, P, lane=1, speed=28.0)


def test_gap_rule_selects_gap_and_rejects():
    im = manager()
    team = [(100, 0, 28), (0, 0, 28), (-100, 0, 28)]
    # Car 1 predicted end: 0 + 28*3 = 84. Lane 2 cars predicted at 3 s: 60 and 120.
    snap = snapshot(0.0, team, [(5, 60-25*3, 3.6, 25), (6, 120-25*3, 3.6, 25)])
    gap, reason = im.gap_rule(snap, [None]*3, 1, 2, 28.0)
    assert gap == (('traffic', 6), ('traffic', 5)) and reason == ''
    assert im.gap_rule(snap, [None]*3, 1, 3, 28.0) == (None, 'no_lane')
    tight = snapshot(0.0, team, [(5, 82-25*3, 3.6, 25), (6, 120-25*3, 3.6, 25)])
    assert im.gap_rule(tight, [None]*3, 1, 2, 28.0) == (None, 'gap_too_short')


def test_second_teammate_merges_behind_first_not_into_same_gap():
    im = manager()
    snap = snapshot(0.0, [(40, 0, 28), (0, 0, 28), (-200, 0, 28)], [(5, -200, 3.6, 25)])
    actions = np.full(3, encode(0, 0)); actions[[0, 1]] = encode(1, 0)
    im.decide(snap, actions, [None]*3, order=[0, 1, 2])
    a, b = im.intentions[0], im.intentions[1]
    assert a.phase == b.phase == 'pending'
    assert b.front == ('team', 0) and a.front is None


def test_commitment_masks_lane_actions_and_pending_can_abort():
    im = manager()
    snap = snapshot(0.0, [(40, 0, 28), (0, 0, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)])
    im.decide(snap, np.array([encode(0, 0), encode(1, 0), encode(1, 0)]), [None]*3, order=[0, 1, 2])
    assert im.intentions[1].phase == im.intentions[2].phase == 'pending'
    im.decide(snap, np.array([encode(0, 0)]*3), [None]*3, order=[0, 1, 2])    # keep -> abort while pending
    assert im.intentions[1].phase == 'keep' and im.intentions[1].lane == 1
    im.decide(snap, np.array([encode(0, 0), encode(1, 0), encode(0, 0)]), [None]*3, order=[0, 1, 2])
    crossing = snapshot(0.1, [(40, 0, 28), (3, 1.0, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)])
    im.update_phases(crossing)
    assert im.intentions[1].phase == 'committed'
    im.decide(crossing, np.array([encode(0, 0), encode(-1, 1), encode(0, 0)]), [None]*3, order=[0, 1, 2])
    it = im.intentions[1]
    assert it.phase == 'committed' and it.lane == 2 and it.speed == 30.0  # lane action masked, speed applied
    kinds = [r.kind for r in im.records if r.vehicle == 1]
    assert kinds == ['accepted', 'aborted', 'accepted', 'committed', 'masked']


def test_lean_action_shifts_target_and_lane_change_completion():
    assert len(ACTIONS) == 45 and all(ACTIONS[encode(*a)] == a for a in ACTIONS)
    im = manager()
    snap = snapshot(0.0, [(40, 0, 28), (0, 0, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)])
    im.decide(snap, np.array([encode(0, 0, 1), encode(1, 0, -1), encode(0, 0)]), [None]*3, order=[0, 1, 2])
    assert im.target(0).d == pytest.approx(CFG.lean_offset) and im.target(2).d == 0.0
    assert im.target(1).d == pytest.approx(3.6-CFG.lean_offset)
    im.update_phases(snapshot(0.1, [(40, 0, 28), (3, 3.6, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)]))
    assert im.intentions[1].phase == 'committed'    # at the lane centre but 0.6 m from the leaned target
    im.update_phases(snapshot(0.2, [(40, 0, 28), (3, 3.0, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)]))
    assert im.intentions[1].phase == 'keep' and im.intentions[1].lane == 2
    im.decide(snap, np.array([encode(0, 0)]*3), [None]*3, order=[0, 1, 2])
    assert im.target(0).d == 0.0


def test_speed_target_is_clipped():
    im = manager()
    snap = snapshot(0.0, [(40, 0, 28), (0, 0, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)])
    for _ in range(10):
        im.decide(snap, np.array([encode(0, 1), encode(0, -1), encode(0, 0)]), [None]*3, order=[0, 1, 2])
    assert [it.speed for it in im.intentions] == [CFG.speed_max, CFG.speed_min, 28.0]


def test_injected_solver_failures_use_shifted_then_lane_keep():
    def factory(world):
        team = CentralizedTeam(SC, world, Overtaker(world.road, world.plant.params), CFG)
        solve = team.mpc.solve

        def failing(time, *args):
            plan = solve(time, *args)
            return replace(plan, status='infeasible') if time >= 2.0 else plan
        team.mpc.solve = failing
        return team
    ep = run(replace(SC, timing=Timing(duration=4)), 0, factory)
    levels = [f[2] for f in ep.planner_log.fallbacks if f[1] == 0]
    assert levels[:CFG.max_shifted_steps] == ['shifted']*CFG.max_shifted_steps
    assert set(levels[CFG.max_shifted_steps:]) == {'lane_keep'}
    assert ep.end_reason == 'time'


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_overtaker_v0(seed):
    m = metrics(run(Scenario(), seed, 'overtaker'))
    p = m['planner']
    assert m['end_reason'] == 'time' and m['collisions'] == []
    assert p['lane_changes_completed'] >= 1 and m['overtakes'] >= 1
    assert m['min_clearance_traffic_m'] >= CFG.margin_lat-0.1
    assert m['min_clearance_teammates_m'] >= CFG.margin_lat-0.1


def test_overtaker_reproducible():
    def strip(m):
        m = {k: v for k, v in m.items() if k != 'policy_compute_ms'}
        m['planner'] = {k: v for k, v in m['planner'].items() if k != 'solve_ms'}
        return m
    short = replace(SC, timing=Timing(duration=15))
    assert strip(metrics(run(short, 7, 'overtaker'))) == strip(metrics(run(short, 7, 'overtaker')))


def test_scenario_v1_definition():
    from nohesi.config import SCENARIOS
    v1 = SCENARIOS['v1']()
    assert (v1.version, v1.traffic.count, v1.timing.duration) == ('v1', 10, 120.0)
    assert v1.road.length >= 35*120+v1.team_start_s+v1.traffic.spawn_ahead+v1.traffic.respawn_ahead


def test_overtaker_v1_seed0():
    from nohesi.config import scenario_v1
    m = metrics(run(scenario_v1(), 0, 'overtaker'))
    assert m['end_reason'] == 'time' and m['collisions'] == []
    assert m['planner']['solve_ms']['mean'] < 10   # loose regression guard; typically ~1.3 ms
    assert m['min_clearance_traffic_m'] < 1.5      # leaning produces close passes
    assert m['min_clearance_teammates_m'] >= CFG.margin_lat-0.05   # hard margin, sampled


def test_cancel_returns_to_origin_lane():
    im = manager()
    snap = snapshot(0.0, [(40, 0, 28), (0, 0, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)])
    im.decide(snap, np.array([encode(0, 0), encode(1, 0), encode(0, 0)]), [None]*3, order=[0, 1, 2])
    im.update_phases(snapshot(0.1, [(40, 0, 28), (3, 2.5, 28), (-40, 0, 28)], [(5, -500, 3.6, 25)]))
    assert im.intentions[1].phase == 'committed'
    im.cancel(1, 0.2, 'fallback')
    assert im.intentions[1].lane == 1 and im.intentions[1].phase == 'keep'   # origin, not lane_of(d) == 2


def test_teammate_merging_from_other_side_rejected():
    im = manager()
    snap = snapshot(0.0, [(2, 3.6, 19), (0, -3.6, 19), (-200, 0, 28)], [(5, -500, 3.6, 25)])
    im.intentions[0].lane, im.intentions[1].lane = 2, 0
    im.decide(snap, np.array([encode(-1, 0), encode(1, 0), encode(0, 0)]), [None]*3, order=[0, 1, 2])
    assert im.intentions[0].phase == 'pending'
    assert im.intentions[1].phase == 'keep' and im.records[-1].reason == 'teammate_merging'


def test_first_conflict_uses_footprints_and_margin():
    from nohesi.planning.fallback import first_conflict
    N = 10
    z = np.column_stack((np.arange(N+1)*2.0, np.zeros(N+1), np.zeros(N+1), np.full(N+1, 20.0)))
    side = ('team', 1), z[:, 0], np.full(N+1, 2.0), np.zeros(N+1)        # 0.2 m side clearance
    assert first_conflict(z, [side], P, 0.25) == (('team', 1), 1)
    assert first_conflict(z, [side], P, 0.1) is None
    crossing = ('traffic', 3), np.full(N+1, 12.0), np.zeros(N+1), np.zeros(N+1)
    hit = first_conflict(z, [crossing], P, 0.25)
    assert hit[0] == ('traffic', 3) and z[hit[1], 0] > 12-P.length-1


def test_lane_keep_plan_matches_execution():
    from nohesi.planning.fallback import lane_keep_plan
    from nohesi.world import World
    world = World(SC, 0)
    snap = world.snapshot
    plan = lane_keep_plan(snap, world.road, world.plant, 0, 1, 28.0, CFG)
    from nohesi.rl.scripted import lane_keep_control
    for _ in range(10):   # 0.5 s
        controls = np.array([lane_keep_control(snap, world.road, world.plant.params, i, 1, 28.0) for i in range(3)])
        snap, _ = world.step(controls)
    np.testing.assert_allclose(plan.z[5, :2], [snap.team_s[0], snap.team_d[0]], atol=0.05)


def test_conflict_makes_lower_priority_vehicle_yield(monkeypatch):
    import nohesi.coordination.centralized as C
    calls = {'n': 0}
    real = C.first_conflict

    def fake(z, others, params, margin):
        keys = [o[0] for o in others]
        if ('team', 0) in keys and calls['n'] == 0:    # first check that involves teammate 0
            calls['n'] += 1
            return ('team', 0), 3
        return real(z, others, params, margin)
    monkeypatch.setattr(C, 'first_conflict', fake)
    ep = run(replace(SC, timing=Timing(duration=1)), 0, 'overtaker')
    (time, vehicle, other, step, resolution), = ep.planner_log.conflicts
    assert other == ('team', 0) and vehicle != 0 and resolution == 'replanned'


def test_scenario_v2_definition():
    from nohesi.config import SCENARIOS
    v1, v2 = SCENARIOS['v1'](), SCENARIOS['v2']()
    assert (v2.version, v2.traffic.count) == ('v2', 20)
    assert (v2.timing, v2.road, v2.vehicle) == (v1.timing, v1.road, v1.vehicle)
