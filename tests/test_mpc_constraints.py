import numpy as np
from nohesi.config import Planner, Scenario
from nohesi.dynamics import KinematicPlant
from nohesi.planning.base import Obstacles, Target, constant_speed_plan
from nohesi.planning.single import LaneChangeMPC, _Rows
from nohesi.road import Road

SC, CFG = Scenario(), Planner()
ROAD, PLANT = Road(SC.road), KinematicPlant(SC.vehicle, SC.timing.dt)
P = PLANT.params
N, T = CFG.horizon, np.arange(CFG.horizon+1)*CFG.step
LAT_GAP, LONG_GAP = P.width+CFG.margin_lat, P.length+CFG.margin_long


def solve(obs_s, obs_d, v, key, target, cfg=CFG, state=(0., 0., 0., 28.)):
    """One QP from a constant-speed reference against a single obstacle given per horizon step."""
    ref = constant_speed_plan(0.0, cfg.step, cfg.horizon, state[0], state[1], state[3])
    obs = Obstacles((key,), obs_s[:, None], obs_d[:, None], np.full((cfg.horizon+1, 1), float(v)))
    return LaneChangeMPC(cfg, PLANT, ROAD).solve(0.0, np.array(state), 0.0, 0.0, target, obs, ref)


def test_teammate_on_side_boundary_is_feasible():
    # Teammate beside the ego at lat_gap +- 3 mm (alternating along the horizon) while ds crosses 0:
    # an exact threshold flickers side/behind/ahead and is infeasible; the tolerance keeps it beside.
    wiggle = 0.003*(-1)**np.arange(N+1); wiggle[:3] = 0.003
    d = -(LAT_GAP+wiggle)
    for s0, v in ((-2.0, 30.0), (2.0, 26.0), (1.4, 28.0)):
        args = (s0+v*T, d, v, ('team', 1), Target(0.0, 28.0))
        assert solve(*args).status == 'solved', (s0, v)
        assert solve(*args, cfg=Planner(lateral_tolerance=1e-9)).status == 'infeasible', (s0, v)


def test_stale_rear_gap_car_ahead_crossing_lane_is_feasible():
    # The locked rear gap car has already passed the ego (20+ m ahead) and drifts across its d:
    # the gap rule's "beside" would flip sides; as an ordinary obstacle it is simply ahead.
    plan = solve(20.0+29.0*T, np.linspace(-2.0, 2.0, N+1), 29.0, ('team', 1), Target(0.0, 28.0, rear=('team', 1)))
    assert plan.status == 'solved'
    assert np.max(np.abs(plan.z[:, 1])) < 0.5


def test_axis_choice_consistent_along_horizon():
    # Reference sits on an in-lane car with ds alternating sign: behind/ahead must not alternate.
    mpc = LaneChangeMPC(CFG, PLANT, ROAD)
    zr = constant_speed_plan(0.0, CFG.step, N, 0.0, 0.0, 28.0).z
    obs_s = (zr[:, 0]+0.1*(-1)**np.arange(N+1))[:, None]
    rows = _Rows()
    mpc._obstacle_rows(rows, zr, obs_s, np.zeros((N+1, 1)), np.full((N+1, 1), 28.0), (('traffic', 0),), Target(0.0, 28.0))
    lo, hi = rows.lo[0], rows.hi[0]
    assert len(rows.lo) == 3 and len(lo) == N        # one hard block, one headway block plus its slack bounds
    assert np.all(np.isfinite(lo)) and np.all(np.isinf(hi))   # all "ahead", the first step's choice (ds > 0)
