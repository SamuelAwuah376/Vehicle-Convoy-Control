"""Per-vehicle road-frame MPC solved as a QP with OSQP.

The kinematic bicycle referenced at the centre of gravity (slip angle beta = atan(l_r/L tan(steer)),
small-angle progress s' = v) is linearised around a reference plan
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
The constraint selection lives in constraints.py (solver independent, shared with other MPCs).

Variables x = [z_0..z_N (4 each: s, d, heading error, v), u_0..u_{N-1} (a, steer), slacks].
Matrices are assembled with vectorised NumPy; the cost Hessian is fixed apart from slacks.
"""
from time import perf_counter
import numpy as np
import osqp
import scipy.sparse as sp
from .base import Obstacles, Plan
from .constraints import obstacle_constraints, state_bounds

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

    def solve(self, time, z0, steer_prev, accel_prev, target, obstacles, reference, lateral=None):
        """lateral: (vy, r) of the plant, unused by this kinematic model (accepted for a common interface)."""
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

        self._dynamics_rows(rows, zr, ur)

        # Inputs: acceleration, steering cap, steering rate (first step relative to the current steer).
        k = np.arange(N)
        speed = np.maximum(zr[:-1, 3], 0.0)
        rows.block(N, [(k, ui(k, 0), 1.0)], -p.brake_max, self.plant.accel_limit(speed))
        cap = self.plant.steer_limit(speed)
        rows.block(N, [(k, ui(k, 1), 1.0)], -cap, cap)
        rate = p.steer_rate*h
        rows.block(1, [(0, ui(0, 1), 1.0)], steer_prev-rate, steer_prev+rate)
        rows.block(N-1, [(k[1:]-1, ui(k[1:], 1), 1.0), (k[1:]-1, ui(k[1:]-1, 1), -1.0)], -rate, rate)

        # States: road edges and non-negative speed.
        self._constraint_rows(rows, state_bounds(c, p, self.road))

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

    def _dynamics_rows(self, rows, zr, ur):
        """Dynamics rows linearised around the reference (zr: (N+1, 4), ur: (N, 2)).

        z_{k+1} = z_k + h*(f + Fz (z_k - zr_k) + Fu (u_k - ur_k)); CG bicycle, slip beta = atan(kappa tan(steer)),
        kappa = l_r/L:  d' = v sin(psi+beta),  psi' = v g(steer),  g = cos(beta) tan(steer)/L = tan/(L sqrt(1+kappa^2 tan^2)).
        """
        p, N, h, zi, ui = self.plant.params, self.cfg.horizon, self.cfg.step, self.zi, self.ui
        k, j4 = np.arange(N), np.arange(4)
        psi, v, delta = zr[:-1, 2], zr[:-1, 3], ur[:, 1]
        L = p.wheelbase
        kappa = getattr(p, 'cg_to_rear_axle', L/2)/L
        tan, sec2 = np.tan(delta), 1/np.cos(delta)**2
        root = 1+(kappa*tan)**2
        beta, dbeta = np.arctan(kappa*tan), kappa*sec2/root
        g, dg = tan/(L*np.sqrt(root)), sec2/(L*root**1.5)
        cos_head, sin_head = np.cos(psi+beta), np.sin(psi+beta)
        r = 4*k
        entries = [(r[:, None]+j4, zi(k+1, 0)[:, None]+j4, 1.0),     # +z_{k+1}
                   (r[:, None]+j4, zi(k, 0)[:, None]+j4, -1.0),      # -z_k
                   (r, zi(k, 3), -h),                                 # s' = v
                   (r+1, zi(k, 2), -h*v*cos_head), (r+1, zi(k, 3), -h*sin_head), (r+1, ui(k, 1), -h*v*cos_head*dbeta),
                   (r+2, zi(k, 3), -h*g), (r+2, ui(k, 1), -h*v*dg),
                   (r+3, ui(k, 0), -h)]
        rhs = np.zeros((N, 4))   # h*(f - Fz zr - Fu ur)
        rhs[:, 1] = -h*v*cos_head*(psi+dbeta*delta)
        rhs[:, 2] = -h*v*dg*delta
        rows.block(4*N, entries, rhs.ravel(), rhs.ravel())

    def _obstacle_rows(self, rows, zr, obs_s, obs_d, obs_v, keys, target):
        """Add the separating, headway and yield rows (constraints.obstacle_constraints; zr and obs_s already
        relative to s0); return the slack weights."""
        if not keys:
            return np.zeros(0)
        spec = obstacle_constraints(self.cfg, self.plant.params, self.road, zr, Obstacles(keys, obs_s, obs_d, obs_v),
                                    target)
        return self._constraint_rows(rows, spec)

    def _constraint_rows(self, rows, spec, slack=0):
        """OSQP rows for a StageConstraints: hard rows in one block, then each run of soft rows (a kind; a yield
        teammate) with its slack columns from nz+nu+slack and sigma >= 0 rows. Returns the new slacks' weights."""
        def block(idx, extra=()):
            local, entries = np.arange(len(idx)), list(extra)
            for col, coef in ((0, spec.cs[idx]), (1, spec.cd[idx]), (3, spec.cv[idx])):
                nz = coef != 0   # no stored zeros: keeps the sparsity pattern
                entries.append((local[nz], self.zi(spec.k[idx][nz], col), coef[nz]))
            rows.block(len(idx), entries, spec.lo[idx], spec.hi[idx])
        hard, soft = np.flatnonzero(~spec.soft), np.flatnonzero(spec.soft)
        if len(hard):
            block(hard)
        kind, j = spec.kind[soft], spec.j[soft]
        cut = np.flatnonzero((kind[1:] != kind[:-1]) | ((kind[1:] == 'yield') & (j[1:] != j[:-1])))+1
        for run in np.split(soft, cut) if len(soft) else []:
            local, cols = np.arange(len(run)), self.nz+self.nu+slack+np.arange(len(run))
            block(run, [(local, cols, np.where(np.isfinite(spec.lo[run]), 1.0, -1.0))])   # slack relaxes the finite bound
            rows.block(len(run), [(local, cols, 1.0)], 0.0, np.inf)
            slack += len(run)
        return spec.slack_weight[soft]

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
