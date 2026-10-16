"""Per-vehicle road-frame MPC solved as a QP with OSQP.

The kinematic bicycle (small-angle progress s' = v) is linearised around a reference plan
(normally the previous plan shifted by one step), giving one QP per replan. Collision
avoidance is convexified per obstacle and horizon step by choosing one separating
half-plane from the reference: stay behind, stay ahead, or stay beside. For the locked
gap's bounding cars the choice is "on the gap side" longitudinally once the reference
reaches the gap, and "beside" before that, so the vehicle can wait in its own lane until
the gap is reachable; a gap car already past the vehicle the wrong way is treated as an
ordinary obstacle. "Beside" is picked when the reference is laterally clear to within
`lateral_tolerance` (the side bound keeps the full margin), and the choice per obstacle is
kept consistent along the horizon: no side flip and no behind/ahead jump between adjacent
steps (the earlier step's choice is kept). A soft time headway (slack-penalised) is added for cars the reference
shares a lane with: `headway` to traffic and the shorter `teammate_headway` to teammates
(drafting). For teammates in `target.yield_to` (changing into the vehicle's lane) a soft longitudinal
get-clear row per step (weight `w_yield`) opens a slot regardless of lateral offset. These are
constraints on predicted traffic and an approximate model: they do not guarantee safety.

Variables x = [z_0..z_N (4 each: s, d, heading error, v), u_0..u_{N-1} (a, steer), slacks].
Matrices are assembled with vectorised NumPy; the cost Hessian is fixed apart from slacks.
"""
from time import perf_counter
import numpy as np
import osqp
import scipy.sparse as sp
from .base import Plan

STATUS = {1: 'solved', 7: 'timeout', 3: 'infeasible', 4: 'infeasible'}


class _Rows:
    """Accumulates sparse constraint blocks given as (local_row, col, value) triplet arrays."""
    def __init__(self):
        self.r, self.c, self.v, self.lo, self.hi, self.n = [], [], [], [], [], 0

    def block(self, count, entries, lo, hi):
        for local, col, val in entries:
            local, col, val = np.broadcast_arrays(local, col, val)
            self.r.append(local.ravel()+self.n); self.c.append(col.ravel()); self.v.append(val.ravel().astype(float))
        self.lo.append(np.broadcast_to(lo, count)); self.hi.append(np.broadcast_to(hi, count))
        self.n += count

    def matrix(self, n):
        A = sp.csc_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))), shape=(self.n, n))
        return A, np.concatenate(self.lo).astype(float), np.concatenate(self.hi).astype(float)


class LaneChangeMPC:
    def __init__(self, cfg, plant, road):
        self.cfg, self.plant, self.road = cfg, plant, road
        self.nz, self.nu = 4*(cfg.horizon+1), 2*cfg.horizon
        self.base_hessian = self._hessian()

    def zi(self, k, j):
        return 4*np.asarray(k)+j

    def ui(self, k, j):
        return self.nz+2*np.asarray(k)+j

    def solve(self, time, z0, steer_prev, accel_prev, target, obstacles, reference):
        start = perf_counter()
        c, p, N, h = self.cfg, self.plant.params, self.cfg.horizon, self.cfg.step
        nz, nu, zi, ui = self.nz, self.nu, self.zi, self.ui
        # Work in s relative to the vehicle for conditioning.
        s0 = z0[0]
        zr = reference.z.copy(); zr[:, 0] -= s0
        ur = reference.u
        rows = _Rows()
        z0r = np.array(z0, float); z0r[0] = 0.0
        j4 = np.arange(4)
        rows.block(4, [(j4, zi(0, j4), 1.0)], z0r, z0r)

        # Dynamics: z_{k+1} = z_k + h*(f + Fz (z_k - zr_k) + Fu (u_k - ur_k)).
        k = np.arange(N)
        psi, v, delta = zr[:-1, 2], zr[:-1, 3], ur[:, 1]
        steer_gain = v/(p.wheelbase*np.cos(delta)**2)
        r = 4*k
        entries = [(r[:, None]+j4, zi(k+1, 0)[:, None]+j4, 1.0),     # +z_{k+1}
                   (r[:, None]+j4, zi(k, 0)[:, None]+j4, -1.0),      # -z_k
                   (r, zi(k, 3), -h),                                 # s' = v
                   (r+1, zi(k, 2), -h*v*np.cos(psi)), (r+1, zi(k, 3), -h*np.sin(psi)),
                   (r+2, zi(k, 3), -h*np.tan(delta)/p.wheelbase), (r+2, ui(k, 1), -h*steer_gain),
                   (r+3, ui(k, 0), -h)]
        rhs = np.zeros((N, 4))
        rhs[:, 1] = -h*v*np.cos(psi)*psi
        rhs[:, 2] = -h*steer_gain*delta
        rows.block(4*N, entries, rhs.ravel(), rhs.ravel())

        # Inputs: acceleration, steering cap, steering rate (first step relative to the current steer).
        speed = np.maximum(zr[:-1, 3], 0.0)
        rows.block(N, [(k, ui(k, 0), 1.0)], -p.brake_max, self.plant.accel_limit(speed))
        cap = self.plant.steer_limit(speed)
        rows.block(N, [(k, ui(k, 1), 1.0)], -cap, cap)
        rate = p.steer_rate*h
        rows.block(1, [(0, ui(0, 1), 1.0)], steer_prev-rate, steer_prev+rate)
        rows.block(N-1, [(k[1:]-1, ui(k[1:], 1), 1.0), (k[1:]-1, ui(k[1:]-1, 1), -1.0)], -rate, rate)

        # States: road edges and non-negative speed.
        ks = np.arange(1, N+1)
        edge = self.road.width/2-p.width/2-c.edge_margin
        rows.block(N, [(ks-1, zi(ks, 1), 1.0)], -edge, edge)
        rows.block(N, [(ks-1, zi(ks, 3), 1.0)], 0.0, np.inf)

        w_slack = self._obstacle_rows(rows, zr, obstacles.s-s0, obstacles.d, obstacles.v, obstacles.keys, target)
        n_slack = len(w_slack)
        n = nz+nu+n_slack
        A, lo, hi = rows.matrix(n)
        P, q = self._cost(n, w_slack, target, steer_prev, accel_prev)
        solver = osqp.OSQP(algebra='builtin')   # skips per-call probing for optional CUDA/MKL backends
        solver.setup(P, q, A, lo, hi, verbose=False, max_iter=c.max_iter, eps_abs=1e-4, eps_rel=1e-4,
                     polishing=False, warm_starting=True)
        solver.warm_start(x=np.r_[zr.ravel(), ur.ravel(), np.zeros(n_slack)])
        result = solver.solve(raise_error=False)
        status = STATUS.get(int(result.info.status_val), 'failed')
        elapsed = perf_counter()-start
        if status != 'solved':
            return Plan(time, h, reference.z, reference.u, 'qp', status, int(result.info.iter), elapsed)
        z = result.x[:nz].reshape(N+1, 4).copy(); z[:, 0] += s0
        return Plan(time, h, z, result.x[nz:nz+nu].reshape(N, 2).copy(), 'qp', status,
                    int(result.info.iter), elapsed)

    def _obstacle_rows(self, rows, zr, obs_s, obs_d, obs_v, keys, target):
        """Add one separating constraint per relevant (step, obstacle) and the soft rows; return the slack weights."""
        c, p, N = self.cfg, self.plant.params, self.cfg.horizon
        if not keys:
            return np.zeros(0)
        long_gap, lat_gap = p.length+c.margin_long, p.width+c.margin_lat
        os_, od, ov = obs_s[1:], obs_d[1:], obs_v[1:]
        ds, dd = zr[1:, :1]-os_, zr[1:, 1:2]-od
        relevant = (np.abs(ds) <= c.relevance) & (np.abs(dd) <= 2*self.road.cfg.lane_width)
        front = np.array([key == target.front for key in keys])[None, :]
        rear = np.array([key == target.rear for key in keys])[None, :]
        tau = np.where([key[0] == 'team' for key in keys], c.teammate_headway, c.headway)[None, :]
        lateral_clear = np.abs(dd) >= lat_gap-c.lateral_tolerance
        # A gap car already past the ego the wrong way (rear car ahead, front car behind) is an ordinary obstacle.
        stale = (rear & (ds < -long_gap)) | (front & (ds > long_gap))
        front, rear = front & ~stale, rear & ~stale
        side = np.where(front, ds > -long_gap,
                        np.where(rear, ds < long_gap, lateral_clear & (np.abs(ds) < long_gap+c.side_zone)))
        behind = ~side & (front | (~rear & (ds < 0)))
        # Axis code per (step, obstacle): +1/-1 beside (left/right of it), 2 behind, -2 ahead, 0 not relevant.
        # Along the horizon a side flip or a behind<->ahead jump keeps the previous step's choice.
        code = np.where(side, np.where(dd >= 0, 1, -1), np.where(behind, 2, -2))*relevant
        for k in range(1, N):
            flip = (code[k] == -code[k-1]) & (code[k] != 0)
            code[k, flip] = code[k-1, flip]
        behind, ahead = code == 2, code == -2
        kk = np.broadcast_to(np.arange(1, N+1)[:, None], ds.shape)
        full = lambda x: np.broadcast_to(x, ds.shape)
        for mask, col, lo, hi in ((code == 1, 1, od+lat_gap, np.inf), (code == -1, 1, -np.inf, od-lat_gap),
                                  (behind, 0, -np.inf, os_-long_gap), (ahead, 0, os_+long_gap, np.inf)):
            m = int(mask.sum())
            if m:
                rows.block(m, [(np.arange(m), self.zi(kk[mask], col), 1.0)], full(lo)[mask], full(hi)[mask])
        # Soft headway (slack >= 0) only for cars the reference shares a lane with, tau per obstacle:
        # behind: s + tau*v - slack <= obs - gap;  ahead: s + slack >= obs + gap + tau*v_obs.
        slack = 0
        for mask, sign in ((behind & ~lateral_clear, -1.0), (ahead & ~lateral_clear, 1.0)):
            m = int(mask.sum())
            if not m:
                continue
            local, cols = np.arange(m), self.nz+self.nu+slack+np.arange(m)
            entries = [(local, self.zi(kk[mask], 0), 1.0), (local, cols, sign)]
            if sign < 0:
                entries.append((local, self.zi(kk[mask], 3), full(tau)[mask]))
                rows.block(m, entries, -np.inf, (os_-long_gap)[mask])
            else:
                rows.block(m, entries, (os_+long_gap+tau*ov)[mask], np.inf)
            rows.block(m, [(local, cols, 1.0)], 0.0, np.inf)
            slack += m
        weights = [np.full(slack, c.w_slack)]
        # Soft yield to teammates changing into the ego's lane, any lateral offset: get clear along the road the way
        # the ego already is (behind or level: drop back; ahead: pull ahead), past the changer's stay-beside zone
        # (side_zone) so its MPC may enter. One slack per step.
        ks, local = np.arange(1, N+1), np.arange(N)
        clear = long_gap+c.side_zone+c.yield_margin
        for o in [keys.index(key) for key in target.yield_to if key in keys]:
            cols, sign = self.nz+self.nu+slack+local, 1.0 if obs_s[0, o] < 0 else -1.0
            bound = os_[:, o]+sign*clear
            rows.block(N, [(local, self.zi(ks, 0), 1.0), (local, cols, sign)],
                       *((bound, np.inf) if sign > 0 else (-np.inf, bound)))
            rows.block(N, [(local, cols, 1.0)], 0.0, np.inf)
            slack += N; weights.append(np.full(N, c.w_yield))
        return np.concatenate(weights)

    def _hessian(self):
        """Upper-triangular cost Hessian over states and inputs as COO (rows, cols, values)."""
        c, N, nz = self.cfg, self.cfg.horizon, self.nz
        ks, k = np.arange(1, N+1), np.arange(N)
        diag_idx = [4*ks+1, 4*ks+2, 4*ks+3]
        diag_val = [np.full(N, 2*c.w_lateral), np.full(N, 2*c.w_heading), np.full(N, 2*c.w_speed)]
        off_idx, off_val = [], []
        for j, (weight, rate) in enumerate(((c.w_accel, c.w_accel_rate), (c.w_steer, c.w_steer_rate))):
            # weight*u_k^2 + rate*(u_k - u_{k-1})^2 with u_{-1} fixed (its linear part goes in q).
            idx = nz+2*k+j
            diag_idx.append(idx); diag_val.append(2*weight+2*rate+2*rate*(k < N-1))
            off_idx.append(idx[:-1]); off_val.append(np.full(N-1, -2*rate))   # (k-1, k), upper triangle
        diag, off = np.concatenate(diag_idx), np.concatenate(off_idx)
        return np.r_[diag, off], np.r_[diag, off+2], np.concatenate(diag_val+off_val)

    def _cost(self, n, w_slack, target, steer_prev, accel_prev):
        """w_slack: per-slack weight w, costing w*(slack^2 + slack)."""
        c, N, nz = self.cfg, self.cfg.horizon, self.nz
        rows, cols, vals = self.base_hessian
        slack = nz+self.nu+np.arange(len(w_slack))
        P = sp.csc_matrix((np.r_[vals, 2*w_slack], (np.r_[rows, slack], np.r_[cols, slack])),
                          shape=(n, n))
        q = np.zeros(n)
        ks = np.arange(1, N+1)
        q[4*ks+1] = -2*c.w_lateral*target.d
        q[4*ks+3] = -2*c.w_speed*target.speed
        q[nz] = -2*c.w_accel_rate*accel_prev
        q[nz+1] = -2*c.w_steer_rate*steer_prev
        q[slack] = w_slack
        return P, q
