"""Follow mode in the RL env (EnvConfig.follow, batch 9b): the FOLLOW slot, its mask, the appended observations, the
missed-escape and dropped-lane-change charges, and the warm start's copied FOLLOW logit."""
from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip('gymnasium')
from nohesi.intentions import ACTIONS, FOLLOW, Record, encode   # noqa: E402
from nohesi.rl.config import EnvConfig   # noqa: E402
from nohesi.rl.env import FOLLOW_FEATURES, LANE_CMD, N_ACTIONS, NoHesiEnv, obs_dim   # noqa: E402

KEEP = encode(0, 0, 0)
E5 = EnvConfig(version='e5', overtakes=True, scenario='v8', score='v10')


def follow_env(**kw):
    env = NoHesiEnv(replace(E5, follow=True, **kw))
    env.scenario = replace(env.scenario, team_start_headway=0.4)   # cars within follow_window of each other
    return env


def test_follow_slot_mask_and_observations():
    with pytest.raises(ValueError):
        EnvConfig(follow=True)                                     # e4 has no FOLLOW slot
    env = follow_env()
    assert env.planner.follow and not NoHesiEnv(E5).planner.follow
    assert env.observation_space.shape == (obs_dim(E5)+3*FOLLOW_FEATURES,) == (312,)
    obs, _ = env.reset(seed=1000)
    m = env.action_masks().reshape(3, -1)
    assert m[:, FOLLOW].tolist() == [False, True, True]             # car 0 leads; 1 and 2 have a teammate right ahead
    assert not env._overtake_options()[:, FOLLOW-N_ACTIONS].any()  # that overtake never starts
    assert obs[-6:].tolist() == [0, 0, 0, 1, 0, 1]                  # (following, follow possible) per car
    obs, *_ = env.step([KEEP, FOLLOW, FOLLOW])
    assert [it.follow for it in env.intentions] == [None, 0, 1]
    assert obs[-6:].tolist() == [0, 0, 1, 1, 1, 1]
    m = env.action_masks().reshape(3, -1)
    assert m[1, FOLLOW] and m[1, :N_ACTIONS].any()                 # continue, or any other action to break off
    env.step([KEEP, KEEP, FOLLOW])
    assert [it.follow for it in env.intentions] == [None, None, 1]
    assert [r.reason for r in env.team.intentions.records if r.kind == 'follow_ended'] == ['policy']


def test_a_follower_takes_its_leaders_way_out():
    """Missed escapes: car 1, stuck behind a slow car with a faster lane open, follows car 0. It is charged while its
    leader keeps its lane, not when the leader starts (or is making) a lane change it will join."""
    env = follow_env(aux_stuck=-1.0, stuck_escape_only=True)
    env.reset(seed=1000)
    road, snap0 = env.team.intentions.road, env.run.snap
    lane, s1 = env.intentions[1].lane, snap0.team_s[1]
    traffic = [(s1+60.0, lane, 15.0), (s1+150.0, lane+1, 40.0)]
    s, l, v = (np.array(x, float) for x in zip(*traffic))
    env.run.snap = replace(snap0, traffic_s=s, traffic_d=np.array([road.lane_center(int(k)) for k in l]),
                           traffic_v=v, traffic_id=np.arange(len(s)), traffic_vd=None, traffic_target_lane=None)
    env._lane_mask = lambda i, it: np.isin(LANE_CMD, (0, 1))
    env.intentions[1].follow = 0
    one = np.array([False, True, False])
    assert env.escapes(one, [KEEP, KEEP, KEEP])[1]                  # not following: a missed escape
    assert env.escapes(one, [KEEP, FOLLOW, KEEP])[1]                # following a leader that stays: still missed
    assert not env.escapes(one, [encode(1, 0, 0), FOLLOW, KEEP])[1]  # the leader moves over now: taken with it
    env.intentions[0].phase = 'pending'
    assert not env.escapes(one, [KEEP, FOLLOW, KEEP])[1]            # the leader is changing lanes: taken with it


def test_dropped_follow_join_is_not_a_policy_abort():
    env = follow_env(aux_lc_abort=-5.0)
    env.reset(seed=1000)
    man = env.team.intentions
    decide = man.decide

    def with_records(snap, *a):
        man.records += [Record(snap.time, 1, 'aborted', 'follow', 1), Record(snap.time, 2, 'aborted', '', 1)]
        return decide(snap, *a)
    man.decide = with_records
    info = env.step([KEEP]*3)[4]
    assert info['aux']['lc_abort'] == pytest.approx(-5.0)            # only the policy's own drop


def test_warm_start_copies_the_follow_logit():
    """e5 (306 inputs) into e5 + follow (312): old logits unchanged, new inputs start at zero weight, and each car's
    FOLLOW logit restarts as the mean of its lane-keep logits plus the shift."""
    import gymnasium
    import torch
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from nohesi.rl.train import load_policy_weights

    class Spaces(gymnasium.Env):
        def __init__(self, n):
            self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
            self.action_space = spaces.MultiDiscrete([N_ACTIONS+6]*3)

        def reset(self, *, seed=None, options=None):
            return np.zeros(self.observation_space.shape, np.float32), {}

        def step(self, action):
            return np.zeros(self.observation_space.shape, np.float32), 0.0, True, False, {}
    make = lambda n: MaskablePPO('MlpPolicy', DummyVecEnv([lambda: Spaces(n)]), seed=0, device='cpu')
    src, dst = make(306), make(312)
    keep = [a for a in range(N_ACTIONS) if ACTIONS[a][0] == 0]
    load_policy_weights(dst, src, ((FOLLOW, keep, 8.0),))
    x = torch.randn(5, 312)

    def logits(policy, obs):
        return policy.action_net(policy.mlp_extractor.forward_actor(policy.extract_features(obs, policy.pi_features_extractor)))
    with torch.no_grad():
        a, b = logits(src.policy, x[:, :306]).reshape(5, 3, -1), logits(dst.policy, x).reshape(5, 3, -1)
    others = [k for k in range(N_ACTIONS+6) if k != FOLLOW]
    assert torch.allclose(a[..., others], b[..., others], atol=1e-5)
    assert torch.allclose(b[..., FOLLOW], a[..., keep].mean(-1)+8.0, atol=1e-4)
