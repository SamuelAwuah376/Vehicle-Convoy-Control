"""Interacting multiple model (IMM) filter over models.MODELS = (keep, left, right), each a UKF (design section 6).

API (the tracker and belief.py build on it):
- IMMState(means (3, 4), covs (3, 4, 4), mu (3,)): per-model means and covariances of x = [s, d, v_s, v_d] and the
  model probabilities, in MODELS order. A plain dataclass; .copy() copies the arrays (the tracker's replay history).
- init_state(mean, cov, mu=TrackerConfig.imm_init) -> IMMState: every model starts at (mean, cov).
- imm_transition(cfg, dt) -> (3, 3) Markov matrix, Pi[i, j] = P(model j after dt | model i now): the exact
  continuous-time chain with keep -> each change at rate cfg.imm_switch per second and change -> keep at 10x that
  (no direct left <-> right), so any dt (the tracker's irregular measurement times) composes: Pi(a + b) = Pi(a) Pi(b).
- imm_predict(state, dt, cfg, road) -> IMMState: mixing (Blom and Bar-Shalom), then each model's UKF prediction with
  its own motion and process noise. mu becomes the predicted model probabilities. dt = 0 returns a copy.
- imm_update(state, z, R, h, residual=None) -> (IMMState, loglik, nis (3,)): each model's UKF update with the same
  measurement; mu reweighted by the models' likelihoods (log-sum-exp safe, floored at MU_FLOOR); loglik =
  log sum_j mu_j N(z; zhat_j, S_j), the mixture's measurement likelihood (existence and association); nis per model.
- imm_combine(state) -> (mean (4,), cov (4, 4)): the moment-matched single Gaussian.
  imm_combine and lane_probability of stacked tracks (P, 3, ...) run compiled (_tracking_jit.combine, lane_rows)
  when numba is on (perception/_jit.py); imm_combine_numpy and lane_probability_numpy are the numpy reference.
- lane_probability(state, road) -> (lanes,): probability of the lane the car is heading to, sum_j mu_j [lane of
  model j], the lane of each model from its mean d by models.target_lane: keep -> lane_of(d), left -> lane_of(d) + 1
  and right -> lane_of(d) - 1 (until the car has reached the centre of the lane it is changing into: that lane),
  clipped to the road. It replaces Snapshot.traffic_target_lane.

Batching: every function also takes stacked tracks, means (..., 3, 4), covs (..., 3, 4, 4), mu (..., 3); z (m,) is
shared by all tracks or given per track as (..., m) (R likewise (m, m) or (..., m, m)), so one call can gate or update
every track against one detection.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from .. import _jit
from ..config import TrackerConfig
from . import _tracking_jit as K
from .models import MODELS, _STEP, motion, process_noise, target_lane, _road
from .ukf import sigma_points, ukf_update, symmetrize

__all__ = ['IMMState', 'MU_FLOOR', 'init_state', 'imm_transition', 'imm_predict', 'imm_update', 'imm_combine',
           'imm_combine_numpy', 'lane_probability', 'lane_probability_numpy']

MU_FLOOR = 1e-12
_STEPS = np.array([_STEP[m] for m in MODELS], np.int64)     # the models' lateral steps, for _tracking_jit.lane_rows


@dataclass
class IMMState:
    """Per-model means (..., 3, 4), covariances (..., 3, 4, 4) and model probabilities (..., 3), in MODELS order."""
    means: np.ndarray
    covs: np.ndarray
    mu: np.ndarray

    def copy(self):
        return IMMState(self.means.copy(), self.covs.copy(), self.mu.copy())


def init_state(mean, cov, mu=None):
    """IMMState with every model at N(mean, cov) (mean (..., 4), cov (..., 4, 4)) and probabilities mu (3,)
    (default TrackerConfig.imm_init), normalized."""
    mean, cov = np.asarray(mean, float), np.asarray(cov, float)
    mu = np.asarray(TrackerConfig.imm_init if mu is None else mu, float)
    k = len(MODELS)
    means = np.repeat(mean[..., None, :], k, axis=-2)
    covs = np.repeat(np.broadcast_to(symmetrize(cov), mean.shape[:-1]+cov.shape[-2:])[..., None, :, :], k, axis=-3)
    mu = np.broadcast_to(mu/mu.sum(-1, keepdims=True), means.shape[:-1]).copy()
    return IMMState(means, covs, mu)


@lru_cache(maxsize=512)
def _transition(rate, dt):
    a, b = rate, 10*rate
    G = np.array([[-2*a, a, a], [b, -b, 0.0], [b, 0.0, -b]])
    P = np.clip(expm(G*dt), 0.0, None)
    P /= P.sum(1, keepdims=True)
    P.setflags(write=False)
    return P


def imm_transition(cfg, dt):
    """(3, 3) model transition matrix over dt (rows sum to 1)."""
    if dt < 0:
        raise ValueError('dt must be >= 0')
    return _transition(float(cfg.imm_switch), float(dt)).copy()


@lru_cache(maxsize=64)
def _motions(cfg, road):
    return tuple(motion(model, cfg, road) for model in MODELS)


def imm_predict(state, dt, cfg, road):
    """Mix the models, then predict each one dt ahead with its UKF."""
    if dt < 0:
        raise ValueError('dt must be >= 0 (late measurements are replayed from the history)')
    if dt == 0:
        return state.copy()
    Pi = _transition(float(cfg.imm_switch), float(dt))
    mu, means, covs = np.asarray(state.mu, float), np.asarray(state.means, float), np.asarray(state.covs, float)
    cbar = np.maximum(np.einsum('...i,ij->...j', mu, Pi), MU_FLOOR)
    w = mu[..., :, None]*Pi/cbar[..., None, :]                               # mixing weights mu_{i|j}
    m0 = np.einsum('...ij,...in->...jn', w, means)
    dm = means[..., :, None, :]-m0[..., None, :, :]                          # (..., i, j, n)
    P0 = np.einsum('...ij,...ink->...jnk', w, covs)+np.einsum('...ij,...ijn,...ijk->...jnk', w, dm, dm)
    X, wm, wc = sigma_points(m0, P0)                                         # (..., 3, 2n+1, n)
    Y = np.empty_like(X)
    for j, f in enumerate(_motions(cfg, road)):
        Xj = X[..., j, :, :]
        Y[..., j, :, :] = f(Xj.reshape(-1, Xj.shape[-1]), dt).reshape(Xj.shape)
    m = wm @ Y
    dY = Y-m[..., None, :]
    Q = np.stack([process_noise(model, cfg, dt) for model in MODELS])
    P = symmetrize(np.swapaxes(dY*wc[:, None], -1, -2) @ dY+Q)
    return IMMState(m, P, cbar/cbar.sum(-1, keepdims=True))


def imm_update(state, z, R, h, residual=None):
    """Update every model with measurement z ~ N(h(x), R); returns (IMMState, mixture loglik, per-model nis)."""
    z, R = np.asarray(z, float), np.asarray(R, float)
    r = ukf_update(state.means, state.covs, z[..., None, :], R[..., None, :, :], h, residual)
    a = np.log(np.maximum(np.asarray(state.mu, float), MU_FLOOR))+r.loglik
    top = a.max(-1, keepdims=True)
    total = top+np.log(np.exp(a-top).sum(-1, keepdims=True))
    mu = np.maximum(np.exp(a-total), MU_FLOOR)
    return IMMState(r.mean, r.cov, mu/mu.sum(-1, keepdims=True)), total[..., 0][()], r.nis


def imm_combine(state):
    """Moment-matched (mean, cov) of the model mixture."""
    mu, means = np.asarray(state.mu, float), np.asarray(state.means, float)
    if _jit.ENABLED and means.ndim == 3:
        c = np.ascontiguousarray
        return K.combine(c(means), c(np.asarray(state.covs, float)), c(mu))
    return imm_combine_numpy(state)


def imm_combine_numpy(state):
    """imm_combine in numpy (the reference of the compiled kernel)."""
    mu, means = np.asarray(state.mu, float), np.asarray(state.means, float)
    m = np.einsum('...j,...jn->...n', mu, means)
    dm = means-m[..., None, :]
    P = np.einsum('...j,...jnk->...nk', mu, state.covs)+np.einsum('...j,...jn,...jk->...nk', mu, dm, dm)
    return m, symmetrize(P)


def lane_probability(state, road):
    """(..., lanes) probability of the lane the car is heading to."""
    d = np.asarray(state.means, float)[..., 1]
    if _jit.ENABLED and d.ndim == 2 and d.shape[1] == len(MODELS):
        lanes, w = _road(road)
        return K.lane_rows(np.ascontiguousarray(d), np.ascontiguousarray(state.mu, float), _STEPS, lanes, w,
                           lanes*w/2)
    return lane_probability_numpy(state, road)


def lane_probability_numpy(state, road):
    """lane_probability in numpy (the reference of the compiled kernel)."""
    lanes, _ = _road(road)
    d = np.asarray(state.means, float)[..., 1]
    target = np.stack([target_lane(model, d[..., j], road) for j, model in enumerate(MODELS)], -1)
    onehot = target[..., None] == np.arange(lanes)
    return np.einsum('...j,...jl->...l', np.asarray(state.mu, float), onehot.astype(float))
