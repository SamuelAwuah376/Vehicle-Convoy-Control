"""Integration tests of the yard parking stack through its episode loop and RL env (nohesi/yard/episode.py, env.py;
contract in docs/yard_interfaces.md): spaces and masks, seeding and reproducibility, the scripted baseline, reversing,
safety under arbitrary (masked) actions, BACK_OFF, REPLAN masking, stalling, the YardEpisode record and the reward.

Episodes are real (scan, driver picks, Hybrid A*, coordination, MPC, supervisor), a few seconds each, so the expensive
rollouts are module fixtures shared between tests."""
from dataclasses import dataclass, field, replace
import numpy as np
import pytest
from gymnasium import spaces
from nohesi.yard import viz
from nohesi.yard.env import SLOT, SLOTS, YardEnv, YardEnvConfig, make_env, obs_dim, scripted_actions
from nohesi.yard.episode import ACTIONS, BACK_OFF, BACK_OFF_DISTANCE, GO, HOLD, MOVING, REPLAN, YardEpisode
from nohesi.yard.paths import Path
from nohesi.yard.sensors import FREE, OCCUPIED, OccupancyGrid

METRIC_KEYS = {'seed', 'success', 'parked', 'followers', 'collision', 'end_reason', 'duration_s', 'replans',
               'replans_failed', 'safety_stops', 'back_off_m', 'picks_rejected', 'reassigned', 'path_length_m',
               'driven_m'}
EVENT_KINDS = {'parked', 'collision', 'replan', 'replan_failed', 'safety_stop', 'reassigned', 'no_spot'}
REPLAN_KINDS = ('replan', 'replan_failed', 'reassigned')
NA = len(ACTIONS)
SCRIPTED_SEEDS = tuple(range(6))
LOCK_SEED = 6                     # a short scripted episode (2 followers, ~60 s simulated)
RANDOM_SEEDS = (10, 13, 15)       # 3 followers each
RANDOM_DECISIONS = 240            # 120 s simulated per random episode
RANDOM_WEIGHTS = np.array([1.0, 1.0, 2.0, 1.0, 0.3])   # HOLD, CREEP, GO, BACK_OFF, REPLAN (where allowed)
RANDOM_REPLANS = 2                # per episode: a REPLAN that falls through to the driver's re-pick costs seconds
BACK_SEED, BACK_GO = 2, 14        # single follower (no coordination effects), GO decisions before the back-off
OFF_PATH = 2.5                    # m: sideways shift that puts a truck off its path (> Planning.track_tolerance)


@dataclass
class Rollout:
    env: YardEnv
    seed: int
    remaining0: list                                # run.remaining() right after reset
    obs: list = field(default_factory=list)         # reset observation, then one per step
    masks: list = field(default_factory=list)       # action_masks() after reset and after each step
    parked: list = field(default_factory=list)      # (SLOTS,) bool parked flags, aligned with masks
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    tallies: list = field(default_factory=list)
    info: dict = field(default_factory=dict)        # last step's info
    terminated: bool = False
    truncated: bool = False
    ep: YardEpisode = None

    @property
    def run(self):
        return self.env.run


def _parked(env):
    return np.array([f.parked for f in env.run.f]+[False]*(SLOTS-env.run.n))


def rollout(env, seed, policy, max_steps=2000):
    """Reset env at seed and step policy(env) until the episode ends or max_steps decisions."""
    obs, info = env.reset(seed=seed)
    assert info['seed'] == seed
    r = Rollout(env, seed, list(env.run.remaining()), [obs], [env.action_masks()], [_parked(env)])
    for _ in range(max_steps):
        a = np.asarray(policy(env), int)
        obs, reward, terminated, truncated, info = env.step(a)
        r.actions.append(a); r.obs.append(obs); r.rewards.append(reward); r.tallies.append(info['tally'])
        r.masks.append(env.action_masks()); r.parked.append(_parked(env))
        r.info, r.terminated, r.truncated = info, terminated, truncated
        if terminated or truncated:
            break
    r.ep = env.episode()
    return r


def random_policy(seed, weights=RANDOM_WEIGHTS, replans=RANDOM_REPLANS):
    """Random masked policy: per slot a weighted draw among the allowed actions (REPLAN at most `replans` times)."""
    rng, left = np.random.default_rng([seed, 99]), [replans]

    def policy(env):
        m = env.action_masks().reshape(SLOTS, NA).copy()
        if left[0] <= 0:
            m[:, REPLAN] = False
        a = np.array([rng.choice(NA, p=weights*row/np.sum(weights*row)) for row in m])
        left[0] -= int(np.sum(a[:env.run.n] == REPLAN))
        return a
    return policy


def assert_obs_ok(r):
    n = r.run.n
    for k, o in enumerate(r.obs):
        assert o.shape == (obs_dim(),) == (152,) and o.dtype == np.float32
        assert np.all(np.isfinite(o)), f'seed {r.seed}: non-finite observation at decision {k}'
        assert r.env.observation_space.contains(o)
        assert not np.any(o[n*SLOT:SLOTS*SLOT]), f'seed {r.seed}: absent slots must be zeros'
        assert o[-1] == pytest.approx(n/SLOTS)
    assert all(isinstance(x, float) and np.isfinite(x) for x in r.rewards)


def assert_mask_invariants(r):
    n = r.run.n
    for k, (m, parked) in enumerate(zip(r.masks, r.parked)):
        assert m.shape == (SLOTS*NA,) and m.dtype == bool
        m = m.reshape(SLOTS, NA)
        assert m[:, HOLD].all(), f'seed {r.seed}: HOLD masked at decision {k}'
        assert not m[n:, 1:].any(), f'seed {r.seed}: an absent slot allows more than HOLD at decision {k}'
        assert not m[parked, 1:].any(), f'seed {r.seed}: a parked follower allows more than HOLD at decision {k}'


def safety_onsets(ep):
    """Safety-stop onsets in the record: rising edges of ep.safety_stop per follower."""
    s = ep.safety_stop.astype(int)
    return int(s[0].sum()+np.sum(np.diff(s, axis=0) == 1))


def reverse_metres(ep, i, t0=-np.inf, t1=np.inf):
    """Distance follower i drove in reverse (per dt, signed by the mean speed) between record times t0 and t1."""
    moved = np.hypot(*np.diff(ep.states[:, i, :2], axis=0).T)
    v = ep.states[:, i, 3]
    window = (ep.time[1:] > t0+1e-9) & (ep.time[1:] <= t1+1e-9)
    return float(np.sum(moved[window & (v[1:]+v[:-1] < 0)]))


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def scripted():
    """The scripted baseline on SCRIPTED_SEEDS: {seed: Rollout}."""
    return {seed: rollout(YardEnv(), seed, scripted_actions) for seed in SCRIPTED_SEEDS}


@pytest.fixture(scope='module')
def lockstep():
    """Three envs stepped with the same actions (env 0's scripted ones) from the same seed: two default envs built
    by make_env with different env RNG seeds, and one whose w_park and w_progress are raised (dynamics unchanged)."""
    cfg = YardEnvConfig()
    bonus = replace(cfg, w_park=cfg.w_park+100.0, w_progress=cfg.w_progress+1.0)
    envs = [make_env(cfg, rank=0)(), make_env(cfg, rank=1, base_seed=7)(), YardEnv(bonus)]
    resets = [env.reset(seed=LOCK_SEED) for env in envs]
    remaining0 = envs[0].run.remaining()
    steps = []                                       # per decision: [(obs, reward, terminated, truncated, info)] x 3
    for _ in range(2000):
        a = scripted_actions(envs[0])
        steps.append([env.step(a) for env in envs])
        if any(s[2] or s[3] for s in steps[-1]):
            break
    return envs, resets, steps, remaining0, bonus


@pytest.fixture(scope='module')
def randoms():
    """The random masked policy on RANDOM_SEEDS, RANDOM_DECISIONS decisions at most: {seed: Rollout}."""
    out = {}
    for seed in RANDOM_SEEDS:
        r = rollout(YardEnv(), seed, random_policy(seed), max_steps=RANDOM_DECISIONS)
        r.metrics = r.run.metrics()
        out[seed] = r
    return out


@dataclass
class BackOffRun:
    env: YardEnv
    go: list = field(default_factory=list)          # per decision: dict(progress, v, back_off_m, backing, time, mask)
    stop: list = field(default_factory=list)
    back: list = field(default_factory=list)
    resume: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    parked: list = field(default_factory=list)
    ended: bool = False                             # the episode ended during the manoeuvre (it should not)
    started: bool = False                           # the back-off engaged
    k0: int = 0                                     # progress when BACK_OFF was first chosen
    mask_before: np.ndarray = None
    t_back: float = 0.0
    t_back_end: float = 0.0
    seed: int = BACK_SEED
    ep: YardEpisode = None

    @property
    def run(self):
        return self.env.run


@pytest.fixture(scope='module')
def back_off():
    """One follower: GO for BACK_GO decisions, HOLD to a standstill, BACK_OFF until the back-off is over (40 decisions
    at most), then GO for 12."""
    env = YardEnv(YardEnvConfig(followers=(1, 1)))
    env.reset(seed=BACK_SEED)
    r = BackOffRun(env, masks=[env.action_masks()], parked=[_parked(env)])
    run, f = env.run, env.run.f[0]

    def step(a, log):
        _, _, terminated, truncated, info = env.step([a, HOLD, HOLD])
        log.append({'progress': f.progress, 'v': float(run.world.states[0, 3]), 'back_off_m': info['tally']['back_off_m'],
                    'backing': f.backing is not None, 'time': run.time,
                    'mask': env.action_masks().reshape(SLOTS, NA)[0].copy()})
        r.masks.append(env.action_masks()); r.parked.append(_parked(env))
        r.ended |= terminated or truncated
        return not r.ended

    for _ in range(BACK_GO):
        step(GO, r.go)
    for _ in range(20):
        if abs(run.world.states[0, 3]) < MOVING or not step(HOLD, r.stop):
            break
    r.k0, r.mask_before, r.t_back = f.progress, env.action_masks().reshape(SLOTS, NA)[0].copy(), run.time
    for _ in range(40):
        if not step(BACK_OFF, r.back):
            break
        r.started |= f.backing is not None
        if r.started and f.backing is None:
            break
    r.t_back_end = run.time
    for _ in range(12):
        if not step(GO, r.resume):
            break
    r.ep = env.episode()
    return r


# ------------------------------------------------------------------------------------------------------- tests
def test_spaces_masks_and_observations(scripted, back_off):
    env = YardEnv()
    assert isinstance(env.action_space, spaces.MultiDiscrete) and env.action_space.nvec.tolist() == [5, 5, 5]
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (152,) and env.observation_space.dtype == np.float32
    m = env.action_masks().reshape(SLOTS, NA)                   # before reset: HOLD only everywhere
    assert m[:, HOLD].all() and not m[:, 1:].any()
    with pytest.raises(RuntimeError):
        env.step(np.zeros(SLOTS, int))
    for r in scripted.values():                                 # finite observations, masks after every step
        assert_obs_ok(r)
        assert_mask_invariants(r)
        assert r.parked[-1][:r.run.n].all()                     # ... parked followers' masks included
    r = back_off                                                # a single follower: slots 1 and 2 absent
    assert r.run.n == 1
    assert_mask_invariants(r)


def test_reset_seeding():
    """reset(seed=None) cycles through seed_pool, else draws a seed >= 2000 from the env RNG."""
    env = YardEnv(seed_pool=[LOCK_SEED, 3])
    assert [env.reset()[1]['seed'] for _ in range(3)] == [LOCK_SEED, 3, LOCK_SEED]
    _, info = make_env(YardEnvConfig(), rank=2, base_seed=5)().reset()
    assert info['seed'] >= 2000


def test_reproducible_with_same_seed_and_actions(lockstep):
    envs, resets, steps, _, _ = lockstep
    (o0, i0), (o1, i1) = resets[0], resets[1]
    assert i0 == i1 == {'seed': LOCK_SEED} and np.array_equal(o0, o1)
    assert steps[-1][0][2] or steps[-1][0][3]                    # the episode ran to its end
    for k, (a, b, _) in enumerate(steps):
        assert np.array_equal(a[0], b[0]), f'observations differ at decision {k}'
        assert a[1] == b[1], f'rewards differ at decision {k}'
        assert a[2:4] == b[2:4]
    assert steps[-1][0][4]['episode_metrics'] == steps[-1][1][4]['episode_metrics']
    ea, eb = envs[0].episode(), envs[1].episode()
    assert np.array_equal(ea.states, eb.states) and np.array_equal(ea.actions, eb.actions)
    assert [(e['kind'], e['follower'], e['time']) for e in ea.events] == \
        [(e['kind'], e['follower'], e['time']) for e in eb.events]


def test_scripted_baseline_parks_everyone(scripted):
    for seed, r in scripted.items():
        m = r.info['episode_metrics']
        assert METRIC_KEYS <= set(m), f'seed {seed}: missing {METRIC_KEYS-set(m)}'
        assert m['seed'] == seed and m['followers'] == r.run.n
        assert m['end_reason'] == 'parked' and m['success'] and m['parked'] == r.run.n, (seed, m)
        assert not m['collision'] and not r.run.world.collisions()
        assert r.terminated and not r.truncated
        kinds = [e['kind'] for e in r.ep.events]
        assert 'collision' not in kinds and set(kinds) <= EVENT_KINDS
        assert m['duration_s'] == pytest.approx(r.run.time) and m['duration_s'] == pytest.approx(r.ep.time[-1])
        assert m['replans'] == kinds.count('replan') and m['reassigned'] == kinds.count('reassigned')
        assert m['safety_stops'] == sum(t['safety_stops'] for t in r.tallies) == safety_onsets(r.ep)
        assert m['back_off_m'] == pytest.approx(sum(t['back_off_m'] for t in r.tallies))
        assert m['picks_rejected'] == sum(not p['accepted'] for p in r.ep.picks)
        if all(x is not None for x in r.remaining0):            # 'path_length_m': the first paths
            assert m['path_length_m'] == pytest.approx(sum(r.remaining0))
        assert m['path_length_m'] > 0 and m['driven_m'] > 0.5*m['path_length_m']


def test_reverse_driving_happens(scripted):
    """Hybrid A* plans cusps and the MPC drives the reverse segments: some follower reverses along a path with a cusp."""
    found = []
    for seed, r in scripted.items():
        ep, truck = r.ep, r.run.truck
        assert ep.states[:, :, 3].min() >= -truck.v_reverse_max-0.1     # reversing limit (MPC soft bound)
        assert ep.states[:, :, 3].max() <= truck.v_forward_max+0.1
        for i in range(ep.n):
            cusped = [p for _, j, p in ep.paths if j == i and len(p.cusps)]
            if np.any(ep.states[:, i, 3] < 0) and cusped:
                assert all(set(np.unique(p.direction)) == {-1, 1} for p in cusped)
                found.append((seed, i, float(ep.states[:, i, 3].min())))
    assert found, 'no follower reversed along a path with a cusp'


def test_random_masked_policy_never_collides(randoms):
    stops = {}
    for seed, r in randoms.items():
        m, ep = r.metrics, r.ep
        assert r.run.end_reason != 'collision', (seed, [e for e in ep.events if e['kind'] == 'collision'])
        assert not m['collision'] and not r.run.world.collisions()
        assert 'collision' not in [e['kind'] for e in ep.events]
        assert_obs_ok(r)
        assert_mask_invariants(r)
        assert m['safety_stops'] == sum(t['safety_stops'] for t in r.tallies) == safety_onsets(ep)
        assert m['driven_m'] > 20.0                                      # the trucks did move around
        stops[seed] = m['safety_stops']
    assert sum(r.metrics['back_off_m'] for r in randoms.values()) > 0    # BACK_OFF was exercised
    print('random masked policy: safety stops per seed', stops)


def test_back_off_reverses_then_go_resumes(back_off):
    r = back_off
    assert not r.ended and r.go[-1]['v'] > 1.0                  # it was driving forward
    assert r.mask_before[BACK_OFF] and r.k0 > 0 and r.started
    v = np.array([d['v'] for d in r.back])
    progress = np.array([d['progress'] for d in r.back])
    assert v.min() < -0.3, 'BACK_OFF never reversed'
    assert progress.min() < r.k0, 'progress did not decrease while backing off'
    s = r.run.f[0].path.s
    backed = s[r.k0]-s[progress.min()]
    assert backed <= BACK_OFF_DISTANCE+1e-9
    if s[r.k0] >= BACK_OFF_DISTANCE+1.0:                         # room for the full manoeuvre
        assert backed >= BACK_OFF_DISTANCE-1.0
    back_m = sum(d['back_off_m'] for d in r.back)
    assert back_m > 1.0 and r.run.metrics()['back_off_m'] == pytest.approx(back_m)
    assert back_m == pytest.approx(reverse_metres(r.ep, 0, r.t_back, r.t_back_end), abs=0.3)   # metres reversed
    # While backing off: BACK_OFF stays allowed (to continue), REPLAN is masked.
    for d in r.back:
        if d['backing']:
            assert d['mask'][BACK_OFF] and not d['mask'][REPLAN]
    assert not r.back[-1]['backing'], 'the back-off did not end within 40 decisions'
    assert (r.ep.actions[:, 0] == BACK_OFF).any()
    # GO afterwards drives forward again.
    p_end = r.back[-1]['progress']
    assert r.resume[-1]['progress'] >= p_end+4
    assert max(d['v'] for d in r.resume) > 0.5 and min(d['v'] for d in r.resume) > -MOVING
    assert not any(d['backing'] for d in r.resume)


def test_replan_masking_and_events():
    """REPLAN is masked while the follower moves (|v| >= 0.2) and for replan_cooldown after a replan, and allowed once
    it stands still with a reason to replan (here: shifted OFF_PATH m sideways, tracker status 'off_path'). A masked
    REPLAN acts as HOLD."""
    env = YardEnv()
    env.reset(seed=0)
    run, cfg, f = env.run, env.run.cfg, env.run.f[0]

    def mask0():
        return env.action_masks().reshape(SLOTS, NA)[0]

    def replan_events(n0):
        return [e for e in run.events[n0:] if e['follower'] == 0 and e['kind'] in REPLAN_KINDS]

    for _ in range(10):
        env.step([GO, HOLD, HOLD])
    assert abs(run.world.states[0, 3]) > 1.0
    run.world.states[0, 1] += OFF_PATH                           # off its path, still rolling (wide access road)
    moving = 0
    for _ in range(30):
        m, v = mask0(), abs(run.world.states[0, 3])
        if v >= 0.2:
            assert not m[REPLAN], f'REPLAN allowed at {v:.2f} m/s'
            moving += 1
        if m[REPLAN]:
            break
        n0 = len(run.events)
        env.step([REPLAN, HOLD, HOLD])                           # masked: a HOLD, no replan
        assert f.action == HOLD and not replan_events(n0)
    assert moving >= 3 and mask0()[REPLAN], 'REPLAN never allowed after stopping off the path'
    assert abs(run.world.states[0, 3]) < 0.2 and f.status == 'off_path'

    def do_replan():
        n0, t0, paths0 = len(run.events), run.time, len(run.paths)
        _, _, _, _, info = env.step([REPLAN, HOLD, HOLD])
        ev = replan_events(n0)
        assert ev and ev[0]['kind'] in REPLAN_KINDS and all(e['time'] == t0 for e in ev)
        if ev[0]['kind'] == 'replan':
            assert info['tally']['replans'] == 1 and len(run.paths) == paths0+1 and run.paths[-1][:2] == (t0, 0)
        elif ev[0]['kind'] == 'replan_failed':
            assert info['tally']['replans_failed'] == 1
        return t0, ev[0]['kind']

    t0, kind = do_replan()
    if kind == 'replan':                                         # on the new path: off it again, at a standstill
        run.world.states[0, 1] -= OFF_PATH
    cooled = 0
    while run.time-t0 < cfg.replan_cooldown-1e-6:
        assert not mask0()[REPLAN], 'REPLAN allowed within the cooldown'
        cooled += f.status == 'off_path' and abs(run.world.states[0, 3]) < 0.2
        n0 = len(run.events)
        env.step([REPLAN, HOLD, HOLD])
        assert not replan_events(n0)
    assert cooled >= 1 and f.status == 'off_path' and mask0()[REPLAN]   # the cooldown alone was masking it
    _, kind = do_replan()
    if kind == 'replan':                                         # and the new path is driven
        p0 = f.progress
        for _ in range(10):
            env.step([GO, HOLD, HOLD])
        assert f.progress > p0+4 and not run.world.collisions()


def test_hold_everyone_stalls():
    env = YardEnv()
    env.reset(seed=1)
    n, cfg, pl = env.run.n, env.cfg, env.run.cfg
    rewards = []
    for _ in range(1000):
        _, reward, terminated, truncated, info = env.step(np.full(SLOTS, HOLD))
        rewards.append(reward)
        if terminated or truncated:
            break
    m = info['episode_metrics']
    assert m['end_reason'] == 'stalled' and truncated and not terminated
    assert not m['collision'] and not env.run.world.collisions() and m['parked'] == 0
    assert m['duration_s'] == pytest.approx(pl.stall_time)       # nobody ever moved
    assert len(rewards) == round(pl.stall_time/pl.decision_interval)
    assert m['driven_m'] < 0.05
    assert sum(rewards) < 0 and rewards[-1] <= -cfg.w_stall
    assert sum(rewards) == pytest.approx(-cfg.w_time*n*pl.stall_time-cfg.w_stall, abs=1e-3)


def test_episode_record(scripted, tmp_path):
    r = scripted[SCRIPTED_SEEDS[0]]
    ep, run = r.ep, r.run
    assert isinstance(ep, YardEpisode) and ep.seed == r.seed and ep.n == run.n
    T, n = len(ep.time), ep.n
    assert T > 1 and ep.time[0] == 0.0 and np.allclose(np.diff(ep.time), run.cfg.dt)
    assert ep.states.shape == (T, n, 4) and ep.steer.shape == (T, n)
    assert ep.actions.shape == (T, n) and np.issubdtype(ep.actions.dtype, np.integer)
    assert ep.actions.min() >= 0 and ep.actions.max() < NA
    assert ep.safety_stop.shape == (T, n) and ep.safety_stop.dtype == bool
    assert np.array_equal(ep.states[-1], run.world.states)
    K = len(ep.obstacles)
    assert ep.missed.shape == ep.revealed_at.shape == (K,) and ep.missed.dtype == bool
    assert np.all(np.isinf(ep.revealed_at[~ep.missed]))          # only missed obstacles get revealed
    assert ep.leader.shape == (4,)
    assert isinstance(ep.grid0, OccupancyGrid)
    assert np.any(ep.grid0.cells == FREE) and np.any(ep.grid0.cells == OCCUPIED)
    assert ep.end_reason == 'parked'
    assert len(ep.spots) == n
    parked = {e['follower'] for e in ep.events if e['kind'] == 'parked'}
    assert parked == set(range(n)) and sum(e['kind'] == 'parked' for e in ep.events) == n
    for i in parked:
        assert ep.spots[i] is not None and np.shape(ep.spots[i]) == (3,)
        assert np.hypot(*(ep.states[-1, i, :2]-ep.spots[i][:2])) <= run.cfg.park_tolerance
    assert ep.picks
    for p in ep.picks:
        assert {'time', 'follower', 'pose', 'accepted', 'reason'} <= set(p)
        assert np.shape(p['pose']) == (3,) and isinstance(p['reason'], str) and 0 <= p['follower'] < n
        assert p['accepted'] == (p['reason'] == ''), 'a rejected pick names its reason, an accepted one none'
    assert {p['follower'] for p in ep.picks if p['accepted']} == set(range(n))
    assert {i for _, i, _ in ep.paths} == set(range(n))
    assert all(isinstance(p, Path) and 0 <= t <= ep.time[-1] for t, _, p in ep.paths)
    assert [t for t, _, _ in ep.paths] == sorted(t for t, _, _ in ep.paths)
    for e in ep.events:
        assert {'time', 'follower', 'kind', 'detail'} <= set(e) and e['kind'] in EVENT_KINDS
    out = tmp_path/'yard.png'
    viz.plot(ep, out)
    assert out.exists() and out.stat().st_size > 1000


def test_reward_park_bonus_and_potential_progress(lockstep, scripted):
    envs, _, steps, remaining0, bonus = lockstep
    cfg = envs[0].cfg
    d_park, d_progress = bonus.w_park-cfg.w_park, bonus.w_progress-cfg.w_progress
    # Same dynamics, other weights: the reward differs by exactly the park and progress terms.
    parked = progress = 0.0
    for k, (a, _, c) in enumerate(steps):
        assert np.array_equal(a[0], c[0])
        t = a[4]['tally']
        assert c[1]-a[1] == pytest.approx(d_park*t['parked']+d_progress*t['progress_m'], abs=1e-9), k
        parked += t['parked']
        progress += t['progress_m']
    m = steps[-1][0][4]['episode_metrics']
    assert m['end_reason'] == 'parked'
    assert parked == m['parked'] == envs[0].run.n                # w_park once per parked follower
    assert sum(1 for e in envs[0].run.events if e['kind'] == 'parked') == m['parked']
    # Potential-based progress: the tallies telescope to the initial remaining path length (0 left when parked).
    assert progress == pytest.approx(sum(x for x in remaining0 if x is not None), abs=1e-6)
    assert progress == pytest.approx(m['path_length_m'], rel=0.02, abs=1.0)
    for seed, r in scripted.items():                             # with replans and reassignments (seeds 0, 2, 5)
        total = sum(t['progress_m'] for t in r.tallies)
        if all(x is not None for x in r.remaining0):
            assert total == pytest.approx(sum(r.remaining0), abs=1e-6), seed
        assert total == pytest.approx(r.info['episode_metrics']['path_length_m'], rel=0.05, abs=2.0), seed


def test_collision_penalised_and_ends_episode():
    env = YardEnv()
    env.reset(seed=LOCK_SEED)
    run = env.run
    for _ in range(4):
        env.step(scripted_actions(env))
    assert not run.done
    i = run.n-1                                                  # still queued on the access road
    run.world.states[i] = np.r_[run.world.leader[:3], 0.0]       # moved into the leader's footprint between steps
    obs, reward, terminated, truncated, info = env.step(scripted_actions(env))
    assert reward < -env.cfg.w_collision/2
    assert terminated and not truncated and run.end_reason == 'collision'
    assert info['tally']['collision'] and np.all(np.isfinite(obs))
    m = info['episode_metrics']
    assert m['collision'] and m['end_reason'] == 'collision' and not m['success']
    hits = [e for e in run.events if e['kind'] == 'collision']
    assert len(hits) == 1 and hits[0]['follower'] == i and hits[0]['detail'].startswith('leader')
    assert env.episode().end_reason == 'collision'
    with pytest.raises(RuntimeError):
        env.step(np.zeros(SLOTS, int))
