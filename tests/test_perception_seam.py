"""The perception seam (docs/perception_design.md sections 3, 7 and 11; phase P0): with perception off the milestone 8
runs reproduce exactly, a pass-through system changes nothing, and a perceived view reaches the policy, the planner and
the RL observation and masks while collisions, scoring and the reward stay on the true state."""
import hashlib
import json
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pytest
from nohesi.config import SCENARIOS, SCORES, Planner, Scenario, scenario_dict
from nohesi.evaluation.metrics import metrics
from nohesi.evaluation.runner import EpisodeRun, run
from nohesi.perception.config import Perception
from nohesi.perception.system import PassThrough, make_perception
from nohesi.world import Collision, World

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models'/'ppo_v10_e5_dyn_b11c_2.75m'
# Recorded on the code before the seam (commit 87f246e): (digest, team score).
NMPC_V9 = ('8fe28f7b9916ae6b', 28162.087880277053)    # v9 seed 1000, 20 s, overtaker, run_sim.py's planner, dynamic car
LINEAR_V8 = ('03b77692909384d0', 56406.189827797294)  # v8 seed 2, 10 s, overtaker, Planner(), kinematic car
RL_V9 = ('e433d196cb951bbd', 447940.59311542165)      # milestone 8 model, v9 seed 1000, 20 deterministic decisions
SCENARIO_JSON = {'v0': 'bcb1767d103a0b77', 'v1': 'af4f245dccc041c8', 'v2': '94d9db642dc103d8', 'v3': '7f7682a57fe8fea6',
                 'v4': '23c2deb241a49c97', 'v5': '8230d37d3ddf430c', 'v6': '44aa894f42f939fe', 'v7': '104fc5403615138b',
                 'v8': 'cda3cc37f844681a', 'v9': 'e851695f092ffa39'}
ARRAYS = ('time', 'team', 'team_s', 'team_d', 'traffic_s', 'traffic_d', 'traffic_v', 'traffic_id', 'controls',
          'pair_clearance', 'traffic_clearance', 'combo_multiplier', 'team_lat')
NMPC = replace(Planner(), mpc='nonlinear', retry_cooldown=1.0, w_lateral=10.0)
DS = 5.0


def digest(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(f'{a.dtype}{a.shape}'.encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def episode_digest(ep):
    return digest(*(getattr(ep, k) for k in ARRAYS), *(ep.score[t] for t in sorted(ep.score)))


def team_score(ep):
    return float(sum(v.sum() for v in ep.score.values()))


def scenario(version, duration, model='kinematic', **kw):
    s = SCENARIOS[version]()
    return replace(s, vehicle=replace(s.vehicle, model=model), timing=replace(s.timing, duration=duration), **kw)


class Shift:
    """Every traffic car DS metres further along the road in the view; records the accelerations it is given."""
    def __init__(self, *args):
        self.args, self.accel = args, []

    def reset(self, snap):
        return replace(snap, traffic_s=snap.traffic_s+DS)

    def step(self, snap, accel):
        self.accel.append(accel)
        return replace(snap, traffic_s=snap.traffic_s+DS)


class Crashed:
    """A view that claims a crash the true state does not have."""
    def reset(self, snap):
        return replace(snap, collisions=(Collision(0, 'traffic', -1),), done=True)

    def step(self, snap, accel):
        return self.reset(snap)


class Fixed:
    """Constant full-ahead controls whatever it sees; records the snapshots it is called with."""
    def __init__(self, world):
        self.seen = []

    def __call__(self, snap):
        self.seen.append(snap)
        return np.array([[3.0, 0.0]]*3)


def truth_at(run_, snap):
    return next(t for t in reversed(run_.snaps) if t.time == snap.time)


def is_view(snap, run_):
    """snap is the Shift view of the true snapshot at its time."""
    true = truth_at(run_, snap)
    return snap is not true and np.array_equal(snap.traffic_s, true.traffic_s+DS)


def is_truth(snap, run_):
    return any(snap is t for t in run_.snaps)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """nohesi.perception.pipeline replaced by one whose PerceptionSystem is Shift (the real one lands separately)."""
    module = types.ModuleType('nohesi.perception.pipeline')
    module.PerceptionSystem = Shift
    monkeypatch.setitem(sys.modules, 'nohesi.perception.pipeline', module)
    return module


@pytest.fixture(scope='module')
def ppo():
    pytest.importorskip('sb3_contrib')
    torch = pytest.importorskip('torch')
    from nohesi.rl.evaluate import load_env_cfg, load_policy, resolve
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield load_env_cfg(MODEL), *load_policy(*resolve(MODEL, 'final'))
    torch.set_num_threads(threads)


# ---------------------------------------------------------------------------------------------- configuration
def test_default_scenario_serializes_as_before():
    assert Scenario().perception == Perception() and not Scenario().perception.enabled
    for version, make in SCENARIOS.items():
        s = make()
        d = scenario_dict(s)
        assert 'perception' not in d and d == {k: v for k, v in asdict(s).items() if k != 'perception'}
        assert hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16] == SCENARIO_JSON[version]


def test_enabled_perception_is_recorded():
    s = scenario('v9', 0.2, perception=Perception(enabled=True, mode='ideal'))
    d = scenario_dict(s)
    assert d['perception']['enabled'] and d['perception']['mode'] == 'ideal'
    assert [c['name'] for c in d['perception']['cameras']] == [c.name for c in s.perception.cameras]
    assert {k: v for k, v in d.items() if k != 'perception'} == scenario_dict(replace(s, perception=Perception()))
    ep = run(s, 0, 'lane_keeping', perception=lambda *a: PassThrough())
    assert json.dumps(metrics(ep)['scenario']) == json.dumps(d)
    with pytest.raises(ValueError):
        Scenario(perception=None)


def test_snapshot_is_certain_without_perception():
    snap = World(SCENARIOS['v9'](), 0).snapshot
    assert snap.traffic_sig_s is None and snap.traffic_sig_d is None and snap.traffic_sig_v is None
    assert snap.traffic_p_exist is None


def test_make_perception(fake_pipeline):
    s = SCENARIOS['v9']()
    world = World(s, 3)
    assert make_perception(s, world, 3) is None
    on = replace(s, perception=Perception(enabled=True))
    system = make_perception(on, world, 3)
    assert isinstance(system, Shift)
    assert system.args == (on.perception, on, world.road, world.plant.params, 3)


# ---------------------------------------------------------------------------------------------- reproduction
def test_perception_off_reproduces_milestone8_nmpc():
    pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)
    ep = run(scenario('v9', 20.0, 'dynamic'), 1000, 'overtaker', NMPC, score=SCORES['v10']())
    assert (episode_digest(ep), team_score(ep)) == NMPC_V9


def test_perception_off_and_pass_through_reproduce():
    s = scenario('v8', 10.0)
    for perception in (make_perception, None, lambda *a: PassThrough()):
        r = EpisodeRun(s, 2, 'overtaker', Planner(), SCORES['v10'](), perception=perception)
        while not r.done:
            assert r.view is r.snap
            r.step()
        ep = r.episode()
        assert (episode_digest(ep), team_score(ep)) == LINEAR_V8


def rl_digest(ppo, perception, decisions=20, seed=1000):
    from nohesi.rl.env import NoHesiEnv
    env_cfg, model, normalize = ppo
    env = NoHesiEnv(env_cfg, perception=perception)
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for _ in range(decisions):
        assert env.snapshot is env.true_snapshot
        masks = env.action_masks()
        action, _ = model.predict(normalize(obs[None]), action_masks=masks[None], deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action[0])
        out += [masks, action[0], np.array([reward]), obs]
    return digest(*out), float(env.run.scorer.team())


def test_rl_env_reproduces_milestone8_model(ppo):
    pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)
    assert rl_digest(ppo, make_perception) == RL_V9
    assert rl_digest(ppo, lambda *a: PassThrough()) == RL_V9


# ---------------------------------------------------------------------------------------------- view vs truth
def test_policy_sees_the_view_while_collisions_and_score_use_the_truth():
    s = scenario('v9', 20.0)
    true_run = EpisodeRun(s, 1000, Fixed)
    shift_run = EpisodeRun(s, 1000, Fixed, perception=Shift)
    for r in (true_run, shift_run):
        while not r.done:
            r.step()
    truth, shifted = true_run.episode(), shift_run.episode()
    # Full-ahead controls crash car 0 into traffic in the truth; the view puts that car 5 m further away.
    assert truth.end_reason == 'collision' and truth.collisions[0].kind == 'traffic'
    assert episode_digest(shifted) == episode_digest(truth) and team_score(shifted) == team_score(truth)
    assert shifted.collisions == truth.collisions and shifted.end_reason == truth.end_reason
    assert metrics(shifted)['team_score'] == metrics(truth)['team_score']
    seen_true, seen_view = true_run.policy.seen, shift_run.policy.seen
    assert len(seen_view) == len(truth.time)-1
    for k, (a, b) in enumerate(zip(seen_true, seen_view)):
        np.testing.assert_array_equal(a.traffic_s, truth.traffic_s[k])
        np.testing.assert_array_equal(b.traffic_s, truth.traffic_s[k]+DS)
        np.testing.assert_array_equal(b.team, truth.team[k])
    np.testing.assert_array_equal(np.array(shift_run.perception.accel), truth.controls[:, :, 0])


def test_a_crash_in_the_view_alone_does_not_end_the_run():
    r = EpisodeRun(scenario('v9', 1.0), 1000, 'lane_keeping', perception=lambda *a: Crashed())
    while not r.done:
        assert r.view.done and not r.snap.done
        r.step()
    ep = r.episode()
    assert ep.end_reason == 'time' and not ep.collisions and len(ep.time) == 21


def test_planner_plans_on_the_view(monkeypatch):
    import nohesi.coordination.centralized as centralized
    predicted, gaps = [], []
    original = centralized.traffic_prediction

    def recorded(snap, road, times):
        predicted.append(snap)
        return original(snap, road, times)
    monkeypatch.setattr(centralized, 'traffic_prediction', recorded)
    r = EpisodeRun(scenario('v9', 1.0), 1000, 'overtaker', Planner(), perception=Shift)
    im = r.policy.intentions
    lane_gap = im.lane_gap
    im.lane_gap = lambda snap, *a, **k: (gaps.append(snap), lane_gap(snap, *a, **k))[1]
    while not r.done:
        r.step()
    assert predicted and all(is_view(s, r) for s in predicted)   # the MPC obstacles
    assert all(is_view(s, r) for s in gaps)


@pytest.fixture
def env_cfg():
    pytest.importorskip('gymnasium')
    from nohesi.rl.config import EnvConfig
    return EnvConfig(version='e5', scenario='v9', score='v10', overtakes=True, follow=True, crash_penalty=-10000.0,
                     aux_stuck=-20.0, stuck_grace=0.5, stuck_escape_only=True, stuck_pass_lanes=True,
                     obs_clock=False)


def test_rl_env_observes_and_masks_the_view_and_rewards_the_truth(env_cfg):
    from nohesi.intentions import encode
    from nohesi.rl.env import EGO, TRAFFIC, NoHesiEnv, merge_possible
    env = NoHesiEnv(env_cfg, perception=Shift)
    env.scenario = replace(env.scenario, timing=replace(env.scenario.timing, duration=3.0))
    obs, _ = env.reset(seed=1000)
    view, truth = env.snapshot, env.true_snapshot
    assert view is env.run.view and truth is env.run.snap and is_view(view, env.run)
    # Car 0's traffic slots: nearest by |ds| in the view, with the view's merge flags.
    ds = view.traffic_s-view.team_s[0]
    near = np.flatnonzero((ds >= -env_cfg.behind) & (ds <= env_cfg.ahead))
    near = near[np.argsort(np.abs(ds[near]), kind='stable')][:env_cfg.traffic_k]
    slots = obs[EGO+env.lanes:EGO+env.lanes+TRAFFIC*env_cfg.traffic_k].reshape(-1, TRAFFIC)[:len(near)]
    np.testing.assert_allclose(slots[:, 0], ds[near]/100, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(slots[:, 5:], merge_possible(view, env.team.road, env.team.params,
                                                               env.scenario.traffic)[near])
    # Every intention check (masks, escapes, the coordinator's decisions) reads the view; stuck time reads the truth.
    im, checked, stuck = env.team.intentions, [], []
    for name in ('lane_gap', 'overtake_check', 'follow_check', '_corridor', '_lead', 'update_phases', 'passed_car',
                 'return_gap'):
        method = getattr(im, name)
        setattr(im, name, lambda snap, *a, _m=method, **k: (checked.append(snap), _m(snap, *a, **k))[1])
    env_stuck = env.stuck
    env.stuck = lambda snap: (stuck.append(snap), env_stuck(snap))[1]
    env._options = None
    env.action_masks()
    env.escapes()
    env._obs()
    assert checked and all(s is env.snapshot for s in checked)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step([encode(0, 0, 0)]*3)
        done = terminated or truncated
        assert terminated == env.true_snapshot.done
    assert len(checked) > 100 and all(is_view(s, env.run) for s in checked)
    assert len(stuck) >= 60 and all(is_truth(s, env.run) for s in stuck)


def test_rl_env_crash_penalty_reads_the_truth(env_cfg):
    from nohesi.intentions import encode
    from nohesi.rl.env import NoHesiEnv
    env = NoHesiEnv(env_cfg, perception=lambda *a: Crashed())
    env.scenario = replace(env.scenario, timing=replace(env.scenario.timing, duration=1.0))
    env.reset(seed=1000)
    assert env.snapshot.done and not env.true_snapshot.done
    _, _, terminated, truncated, info = env.step([encode(0, 0, 0)]*3)
    assert not terminated and not truncated and info['aux']['crash'] == 0


def test_shadow_mode_runs_the_system_but_keeps_the_truth(fake_pipeline):
    s = scenario('v9', 0.5, perception=Perception(enabled=True, mode='shadow'))
    r = EpisodeRun(s, 1000, Fixed)
    assert isinstance(r.perception, Shift)
    while not r.done:
        r.step()
    assert len(r.perception.accel) == 10
    assert all(snap is t for snap, t in zip(r.policy.seen, r.snaps))
