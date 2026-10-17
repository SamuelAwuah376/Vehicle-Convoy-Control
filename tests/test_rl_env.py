import warnings
from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip('gymnasium')
from gymnasium.utils.env_checker import check_env   # noqa: E402
from nohesi.config import SCORES, Timing   # noqa: E402
from nohesi.evaluation.metrics import metrics   # noqa: E402
from nohesi.evaluation.runner import run   # noqa: E402
from nohesi.intentions import encode   # noqa: E402
from nohesi.rl.config import EnvConfig   # noqa: E402
from nohesi.rl.env import N_ACTIONS, NoHesiEnv, obs_dim   # noqa: E402
from nohesi.rl.scripted import Overtaker   # noqa: E402

KEEP = [encode(0, 0, 0)]*3


def config(**kw):
    score = 'v4'
    if score not in SCORES:   # score v4 lands separately
        warnings.warn('score v4 missing, testing with v3')
        score = 'v3'
    return EnvConfig(score=score, **kw)


def short(duration=10.0, **kw):
    env = NoHesiEnv(config(**kw))
    env.scenario = replace(env.scenario, timing=Timing(env.scenario.timing.dt, duration))
    return env


def random_episode(env, seed, rng):
    obs, info = env.reset(seed=seed)
    out = [obs]
    while True:
        m = env.action_masks().reshape(3, N_ACTIONS)
        obs, r, term, trunc, info = env.step([rng.choice(np.flatnonzero(row)) for row in m])
        out += [obs, r]
        if term or trunc:
            return out, info


def test_check_env():
    check_env(short(2.0), skip_render_check=True)


def test_spaces_and_obs():
    env = short(5.0)
    assert env.observation_space.shape == (obs_dim(env.cfg),) == (273,)
    assert env.action_space.nvec.tolist() == [45]*3
    obs, info = env.reset(seed=1000)
    assert info['seed'] == 1000
    for _ in range(3):
        assert obs.dtype == np.float32 and obs.shape == (273,) and np.all(np.isfinite(obs))
        assert np.abs(obs).max() < 50
        obs, *_ = env.step(KEEP)


def test_seed_pool_and_rng():
    env = NoHesiEnv(config(), seed_pool=[1000, 1001])
    assert [env.reset()[1]['seed'] for _ in range(3)] == [1000, 1001, 1000]
    a, b = NoHesiEnv(config(), rng_seed=5), NoHesiEnv(config(), rng_seed=5)
    seeds = [a.reset()[1]['seed'] for _ in range(3)]
    assert seeds == [b.reset()[1]['seed'] for _ in range(3)] and min(seeds) >= 2000


def test_masks():
    # Road-geometry and phase rules only; the gap-rule mask is tested separately.
    env = short(5.0, mask_gap_rule=False)
    env.reset(seed=1)
    lane = np.array([a // 9 - 2 for a in range(N_ACTIONS)])
    m = env.action_masks()
    assert m.shape == (135,) and m.dtype == np.bool_
    assert np.array_equal(m.reshape(3, 45)[0], np.abs(lane) <= 1)   # middle lane: +-2 off the road
    it = env.intentions[0]
    it.lane = 0
    assert np.array_equal(env.action_masks().reshape(3, 45)[0], (lane >= 0) & (lane <= 2))
    it.phase, it.origin, it.lane = 'pending', 0, 1   # pending: every valid command (others abort)
    assert np.array_equal(env.action_masks().reshape(3, 45)[0], (lane >= 0) & (lane <= 2))
    it.phase, it.lane = 'committed', 2   # committed: only the locked change
    assert np.array_equal(env.action_masks().reshape(3, 45)[0], lane == 2)
    assert env.action_masks().reshape(3, 45).any(axis=1).all()


def test_deterministic():
    a, ia = random_episode(short(), 7, np.random.default_rng(0))
    b, ib = random_episode(short(), 7, np.random.default_rng(0))
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert ia['team_score'] == ib['team_score']


def test_reward_sums_to_score_and_truncates():
    env = NoHesiEnv(config())   # full v4 episode: 240 decisions
    env.reset(seed=4)
    total, aux, n = 0.0, 0.0, 0
    while True:
        _, r, term, trunc, info = env.step(KEEP)
        total, aux, n = total+r, aux+sum(info['aux'].values()), n+1
        if term or trunc:
            break
    assert trunc and not term and n == 240
    assert total == pytest.approx(info['team_score']/env.cfg.reward_scale+aux, abs=1e-9)
    em = info['episode_metrics']
    assert em['end_reason'] == 'time' and em['team_score'] == pytest.approx(info['team_score'], abs=1e-9)
    with pytest.raises(RuntimeError):
        env.step(KEEP)


def test_overtaker_reproduces_run():
    env = short(40.0)
    env.reset(seed=3)
    policy = Overtaker(env.team.road, env.team.params)
    policy.bind_gap_check(env.gap_check)
    while True:
        _, _, term, trunc, info = env.step(policy(env.snapshot, env.intentions))
        if term or trunc:
            break
    ep = run(env.scenario, 3, 'overtaker')
    ref = metrics(ep)
    em = info['episode_metrics']
    assert em['team_score'] == ref['team_score'] and em['score_by_term'] == ref['score_by_term']
    assert em['planner']['lane_changes_accepted'] == ref['planner']['lane_changes_accepted'] > 0
    assert np.array_equal(env.episode().team, ep.team)


def test_collision_costs_five_times_the_game_penalty():
    from nohesi.rl.config import EnvConfig
    from nohesi.rl.env import NoHesiEnv
    cfg = EnvConfig()
    env = NoHesiEnv(cfg)
    env.reset(seed=3)
    # Force a crash: put car 0 onto the nearest traffic car, then step once with keep-lane / hold actions.
    w = env.run.world
    t = w.traffic
    w.team[0, :2] = w.road.to_xy(t.s[0], t.d[0]); w.team[0, 3] = t.v[0]
    from nohesi.intentions import encode
    obs, reward, terminated, truncated, info = env.step([encode(0, 0, 0)]*3)
    assert terminated and info['score_terms']['collision'] < 0
    assert info['aux']['collision'] == pytest.approx(cfg.collision_extra*info['score_terms']['collision']/cfg.reward_scale)
    total_collision = info['score_terms']['collision']/cfg.reward_scale+info['aux']['collision']
    assert total_collision == pytest.approx(5*info['score_terms']['collision']/cfg.reward_scale)


def test_gap_rule_mask_prevents_rejections():
    """With the gap-rule mask, random masked actions produce almost no rejected lane requests."""
    from dataclasses import replace
    from nohesi.rl.config import EnvConfig
    from nohesi.rl.env import NoHesiEnv
    counts = {}
    for masked in (False, True):
        env = NoHesiEnv(replace(EnvConfig(), mask_gap_rule=masked))
        env.reset(seed=5)
        rng, rejected = np.random.default_rng(0), 0
        for _ in range(60):
            m = env.action_masks().reshape(3, -1)
            a = [rng.choice(np.flatnonzero(m[i])) for i in range(3)]
            _, _, term, trunc, info = env.step(a)
            rejected += round(-info['aux']['rejected']/0.1)
            if term or trunc:
                break
        counts[masked] = rejected
    assert counts[True] <= max(2, counts[False]//5), counts


def test_merge_possible_flags():
    from nohesi.config import SCENARIOS
    from nohesi.road import Road
    from nohesi.rl.env import merge_possible
    from nohesi.world import Snapshot
    sc = SCENARIOS['v4']()
    road, p = Road(sc.road), sc.vehicle
    L = p.length
    c = [road.lane_center(l) for l in range(3)]
    # Traffic 0 in the middle lane; traffic 1 alongside it in lane 0 (blocks -d); team car 0 in lane 2 just beyond
    # the room rule (free +d), team cars 1/2 far away. Traffic 2 in lane 2 has a team car in the gap rule zone.
    traffic_s = np.array([100.0, 100.0+0.5*L, 300.0])
    traffic_d = np.array([c[1], c[0], c[2]])
    team_s = np.array([100.0+1.6*L, 500.0, 300.0-L])
    team_d = np.array([c[2], c[0], c[1]+0.5])
    team = np.column_stack((team_s, team_d, np.zeros(3), np.full(3, 30.0)))
    snap = Snapshot(0.0, team, np.zeros(3), team_s, team_d, traffic_s, traffic_d, np.full(3, 25.0),
                    np.arange(3), (), False)
    m = merge_possible(snap, road, p, sc.traffic)
    assert m[0].tolist() == [0.0, 1.0]     # lane 0 blocked alongside, lane 2 clear
    assert m[1].tolist() == [0.0, 0.0]     # no lane -1; lane 1 occupied alongside by traffic 0
    assert m[2].tolist() == [0.0, 0.0]     # team car in lane 1 within the room rule; no lane 3
