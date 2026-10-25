"""Planner m4.1: the emergency escape planned by the nonlinear MPC in emergency mode. Skipped without acados."""
from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)

from nohesi.config import Planner, SCENARIOS   # noqa: E402
from nohesi.evaluation.runner import EpisodeRun   # noqa: E402


def dynamic_run(steps=10, **planner):
    sc = SCENARIOS['v5']()
    sc = replace(sc, vehicle=replace(sc.vehicle, model='dynamic'), timing=replace(sc.timing, duration=10.0))
    r = EpisodeRun(sc, 1010, 'overtaker', Planner(mpc='nonlinear', **planner))
    for _ in range(steps):
        r.step()
    return r


def test_emergency_escape_is_an_mpc_plan_within_grip():
    r = dynamic_run()
    team, snap = r.policy, r.snap
    plan = team._emergency(snap, 0, {}, [1, 2])
    assert plan.source == 'emergency' and plan.status == 'solved' and plan.lateral is not None
    assert team.mode[0] == 'emergency' and team._plan_driven(0)
    assert team.log.fallbacks[-1][1:] == (0, 'emergency')
    assert any(s[5] == 'emergency' for s in team.log.solves)
    # The first candidate is braking in the car's lane: with nothing in the way it is taken, braking within the budget.
    assert plan.u[:, 0].min() < -1.0 and plan.u[:, 0].min() >= -team.brake-1e-6
    # The escaping car plans first next cycle.
    assert team.order(snap)[0] == 0


def test_emergency_off_keeps_the_law_escape():
    r = dynamic_run(nmpc_emergency=False)
    team, snap = r.policy, r.snap
    team._lane_keep(snap, 0, 'emergency', {}, [1, 2])
    assert team.mode[0] == 'emergency' and not team._plan_driven(0)
    assert team.order(snap) == sorted(range(3), key=lambda i: -snap.team_s[i])


def test_plan_verification_rule():
    """A plan is accepted when its true grip use stays within the limit; a car starting past it gets the first
    nmpc_recover_steps steps to come back, as long as they don't exceed its starting use."""
    import numpy as np
    team = dynamic_run(steps=1).policy
    mpc, c, N = team.mpc, team.mpc.cfg, team.cfg.horizon
    use = []
    mpc.grip = lambda X, U: np.vstack((np.array(use), np.array(use)))   # stand-in: the same use on both axles
    X, U = np.zeros((N+1, 8)), np.zeros((N, 2))
    use[:] = [0.5]*(N+1)
    assert mpc._grip_ok(X, U)
    use[:] = [0.5]*5+[c.nmpc_grip_limit+0.05]+[0.5]*(N-5)
    assert not mpc._grip_ok(X, U)
    use[:] = [0.98]+[0.97]*c.nmpc_recover_steps+[0.85]*(N-c.nmpc_recover_steps)   # sliding start, recovering
    assert mpc._grip_ok(X, U)
    use[:] = [0.98]+[0.99]+[0.85]*(N-1)   # sliding start, getting worse
    assert not mpc._grip_ok(X, U)


def test_diverged_iterates_are_never_linearised():
    """An iterate past physical bounds (the cause of the HPIPM hangs: a QP with Jacobian entries up to 1e43) fails
    that guess instead of being solved on."""
    import numpy as np
    mpc = dynamic_run(steps=1).policy.mpc
    N = mpc.cfg.horizon
    X, U = np.zeros((N+1, 8)), np.zeros((N, 2))
    X[:, 3] = 30.0
    assert mpc._sane(X, U)
    for j, value in ((3, 1e6), (5, 50.0), (6, 2.0), (4, np.nan)):
        bad = X.copy()
        bad[7, j] = value
        assert not mpc._sane(bad, U), j


def test_braking_rollout_guess_stops_at_zero_speed():
    """The emergency braking candidate's rollout guess from 10 m/s stops instead of braking on into reverse (which
    started QPs from a guess breaking vx >= 0, in the stiff low-speed tyre region: most diverged iterates)."""
    import numpy as np
    from nohesi.planning.base import Plan
    mpc = dynamic_run(steps=1).policy.mpc
    N, h = mpc.cfg.horizon, mpc.cfg.step
    x0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -8.0])
    ref = Plan(0.0, h, np.zeros((N+1, 4)), np.tile([-8.24, 0.0], (N, 1)), 'emergency')
    X, U = mpc._rollout_guess(x0, 0.0, ref)
    assert X[:, 3].min() >= 0 and X[-1, 3] < 0.5   # stopped, never reversing
    assert mpc._sane(X, U)
