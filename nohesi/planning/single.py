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
With Planner.qp_reuse (default) one OSQP instance is kept with a fixed sparsity pattern and only its data updated
per solve (see _solve_fixed); a solve whose rows don't fit falls back to a fresh setup.
"""
from time import perf_counter
import numpy as np
import osqp
import scipy.sparse as sp
from .base import Obstacles, Plan
from .constraints import obstacle_constraints, state_bounds
from ..dynamics import brake_limit

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
        self.state_bounds = state_bounds(cfg, plant.params, road)
        self.brake = brake_limit(plant, getattr(cfg, 'grip_brake', 1.0))   # brake_max on the kinematic plant
        self._solver, self._fixed, self.fallbacks = None, None, 0   # qp_reuse: persistent OSQP, pattern, setups

    def zi(self, k, j):
        return 4*np.asarray(k)+j

    def ui(self, k, j):
        return self.nz+2*np.asarray(k)+j

    def solve(self, time, z0, steer_prev, accel_prev, target, obstacles, reference, lateral=None):
        """lateral: (vy, r) of the plant, unused by this kinematic model (accepted for a common interface)."""
        start = perf_counter()
        c, N, h, nz, nu = self.cfg, self.cfg.horizon, self.cfg.step, self.nz, self.nu
        # Work in s relative to the vehicle for conditioning.
        s0 = z0[0]
        zr = reference.z.copy(); zr[:, 0] -= s0
        ur = reference.u
        result = self._solve_fixed(z0, zr, ur, obstacles, s0, target, steer_prev, accel_prev) if c.qp_reuse else None
        if result is None:
            rows = _Rows()
            self._base_rows(rows, z0, zr, ur, steer_prev)
            w_slack = self._obstacle_rows(rows, zr, obstacles.s-s0, obstacles.d, obstacles.v, obstacles.keys, target)
            n_slack = len(w_slack)
            n = nz+nu+n_slack
            A, lo, hi = rows.matrix(n)
            P, q = self._cost(n, w_slack, target, steer_prev, accel_prev)
            solver = self._setup(P, q, A, lo, hi)
            solver.warm_start(x=np.r_[zr.ravel(), ur.ravel(), np.zeros(n_slack)])
            result = solver.solve(raise_error=False)
        status = STATUS.get(int(result.info.status_val), 'failed')
        elapsed = perf_counter()-start
        if status != 'solved':
            return Plan(time, h, reference.z, reference.u, 'qp', status, int(result.info.iter), elapsed)
        z = result.x[:nz].reshape(N+1, 4).copy(); z[:, 0] += s0
        return Plan(time, h, z, result.x[nz:nz+nu].reshape(N, 2).copy(), 'qp', status,
                    int(result.info.iter), elapsed)

    def _setup(self, P, q, A, lo, hi):
        c = self.cfg
        solver = osqp.OSQP(algebra='builtin')   # skips per-call probing for optional CUDA/MKL backends
        solver.setup(P, q, A, lo, hi, verbose=False, max_iter=c.max_iter, eps_abs=c.qp_eps, eps_rel=c.qp_eps,
                     polishing=False, warm_starting=True)
        return solver

    def _base_rows(self, rows, z0, zr, ur, steer_prev):
        """Initial state, dynamics, input and state-bound rows: a fixed pattern (values and bounds vary)."""
        p, N, h, zi, ui = self.plant.params, self.cfg.horizon, self.cfg.step, self.zi, self.ui
        z0r = np.array(z0, float); z0r[0] = 0.0
        j4 = np.arange(4)
        rows.block(4, [(j4, zi(0, j4), 1.0)], z0r, z0r)

        self._dynamics_rows(rows, zr, ur)

        # Inputs: acceleration, steering cap, steering rate (first step relative to the current steer).
        k = np.arange(N)
        accel, cap = self._input_bounds(zr)
        rows.block(N, [(k, ui(k, 0), 1.0)], -self.brake, accel)
        rows.block(N, [(k, ui(k, 1), 1.0)], -cap, cap)
        rate = p.steer_rate*h
        rows.block(1, [(0, ui(0, 1), 1.0)], steer_prev-rate, steer_prev+rate)
        rows.block(N-1, [(k[1:]-1, ui(k[1:], 1), 1.0), (k[1:]-1, ui(k[1:]-1, 1), -1.0)], -rate, rate)

        # States: road edges and non-negative speed.
        self._constraint_rows(rows, self.state_bounds)

    def _input_bounds(self, zr):
        """Acceleration upper bound and steering cap per step at the reference speed."""
        speed = np.maximum(zr[:-1, 3], 0.0)
        return self.plant.accel_limit(speed), self.plant.steer_limit(speed)

    def _solve_fixed(self, z0, zr, ur, obstacles, s0, target, steer_prev, accel_prev):
        """Solve on the persistent OSQP instance (qp_reuse, fixed sparsity: values updated, no setup); None (counted
        in self.fallbacks) when the obstacle rows don't fit. Rows: the base rows as in the setup path, one s_k row per
        step, then per step qp_soft_slots soft rows on (s_k, v_k, own slack) and their sigma >= 0 rows.
        Hard obstacle rows bound a single state (s or d) each, so at step k they intersect to one interval: s bounds
        go in the s_k row, d bounds tighten the edge row. The feasible set and cost are the setup path's, so the
        optimum is the same; OSQP's iterates differ (fewer rows), so solutions agree within its tolerance. An
        unused soft slot has zero coefficients and infinite bounds, its slack fixed to [0, 0] at zero cost."""
        c, N, nz, nu, h = self.cfg, self.cfg.horizon, self.nz, self.nu, self.cfg.step
        S = c.qp_soft_slots
        spec = obstacle_constraints(c, self.plant.params, self.road, zr, Obstacles(
            obstacles.keys, obstacles.s-s0, obstacles.d, obstacles.v), target) if obstacles.keys else None
        if self._fixed is None:
            rows = _Rows()
            self._base_rows(rows, z0, zr, ur, steer_prev)
            self._fixed = self._fixed_pattern(rows)
        f = self._fixed
        Av, lo, hi, Pv, q = f['Av'].copy(), f['lo'].copy(), f['hi'].copy(), f['Pv'].copy(), np.zeros(f['n'])
        # Base rows: initial state, dynamics coefficients and right-hand side, input bounds.
        lo[1:4] = hi[1:4] = z0[1:]
        coef, rhs = self._linearise(zr, ur)
        Av[f['dyn']:f['dyn']+5*N] = coef.ravel()
        lo[4:4+4*N] = hi[4:4+4*N] = rhs.ravel()
        b = 4+4*N
        hi[b:b+N], cap = self._input_bounds(zr)
        lo[b+N:b+2*N], hi[b+N:b+2*N] = -cap, cap
        rate = self.plant.params.steer_rate*h
        lo[b+2*N], hi[b+2*N] = steer_prev-rate, steer_prev+rate
        if spec is not None and len(spec):
            hard, soft = ~spec.soft, np.flatnonzero(spec.soft)
            on_s, on_d = hard & (spec.cs != 0), hard & (spec.cd != 0)
            ss = self._slots(spec.k[soft], S)
            if ss is None or (on_s & on_d).any() or (hard & (spec.cv != 0)).any() or spec.cd[soft].any():
                self.fallbacks += 1
                return None
            for mask, row in ((on_s, f['s']), (on_d, f['edge'])):   # intersect the intervals per step
                a, k = (spec.cs+spec.cd)[mask], row+spec.k[mask]-1
                np.maximum.at(lo, k, np.where(a > 0, spec.lo[mask], spec.hi[mask])/a)
                np.minimum.at(hi, k, np.where(a > 0, spec.hi[mask], spec.lo[mask])/a)
            if (lo > hi).any():   # empty interval: let the setup path report the infeasibility
                self.fallbacks += 1
                return None
            r, a, slo = f['soft']+ss, f['soft_v']+3*ss, spec.lo[soft]
            Av[a], Av[a+1], Av[a+2] = spec.cs[soft], spec.cv[soft], np.where(np.isfinite(slo), 1.0, -1.0)
            lo[r], hi[r] = slo, spec.hi[soft]   # the slack relaxes the finite bound
            hi[f['soft']+N*S+ss] = np.inf
            w = spec.slack_weight[soft]
            Pv[f['nP']+ss], q[nz+nu+ss] = 2*w, w
        self._linear_cost(q, target, steer_prev, accel_prev)
        Ax, Px = Av[f['perm_A']], Pv[f['perm_P']]
        if self._solver is None:
            n, m = f['n'], len(lo)
            self._solver = self._setup(sp.csc_matrix((Px, f['P_idx'], f['P_ptr']), shape=(n, n)), q,
                                       sp.csc_matrix((Ax, f['A_idx'], f['A_ptr']), shape=(m, n)), lo, hi)
            self._rho = self._solver.settings.rho
        else:
            self._solver.update(q=q, l=lo, u=hi, Px=Px, Ax=Ax)
            self._solver.update_settings(rho=self._rho)   # a fresh setup's rho (adaptive rho moved it)
        self._solver.warm_start(x=np.r_[zr.ravel(), ur.ravel(), np.zeros(N*S)], y=np.zeros(len(lo)))
        return self._solver.solve(raise_error=False)

    @staticmethod
    def _slots(k, width):
        """Slot index (k-1)*width + rank among the rows at step k, or None when a step has more than width rows."""
        order = np.argsort(k, kind='stable')
        ks = k[order]
        rank = np.arange(len(ks))-np.searchsorted(ks, ks)
        if len(ks) and rank.max() >= width:
            return None
        out = np.empty(len(k), int); out[order] = (ks-1)*width+rank
        return out

    def _fixed_pattern(self, rows):
        """Templates (COO values, bounds, P values), row offsets and CSC orderings of the fixed pattern; rows: the
        base rows (their last 2N are state_bounds: edge rows then speed rows, k = 1..N)."""
        N, nz, nu, S = self.cfg.horizon, self.nz, self.nu, self.cfg.qp_soft_slots
        n, base = nz+nu+N*S, rows.n
        sb = self.state_bounds
        assert list(sb.kind) == ['edge']*N+['speed']*N and np.array_equal(sb.k, np.r_[1:N+1, 1:N+1])
        slack, ks = nz+nu+np.arange(N*S), np.arange(1, N+1)
        kr = np.repeat(ks, S)
        r = np.concatenate(rows.r+[base+np.arange(N), np.repeat(base+N+np.arange(N*S), 3), base+N+N*S+np.arange(N*S)])
        col = np.concatenate(rows.c+[4*ks, np.c_[4*kr[:, None]+np.array([0, 3]), slack].ravel(), slack])
        Av = np.concatenate(rows.v+[np.ones(N), np.zeros(3*N*S), np.ones(N*S)])
        perm_A = np.lexsort((r, col))
        pr, pc = np.r_[self.base_hessian[0], slack], np.r_[self.base_hessian[1], slack]
        perm_P = np.lexsort((pr, pc))
        return dict(
            Av=Av, dyn=4+9*N, soft_v=len(Av)-4*N*S, s=base, edge=base-2*N, soft=base+N, n=n,
            nP=len(self.base_hessian[2]),
            lo=np.concatenate(rows.lo+[np.full(N*(1+S), -np.inf), np.zeros(N*S)]),
            hi=np.concatenate(rows.hi+[np.full(N*(1+S), np.inf), np.zeros(N*S)]),
            Pv=np.r_[self.base_hessian[2], np.full(N*S, 2*self.cfg.w_slack)], perm_A=perm_A, perm_P=perm_P,
            A_idx=r[perm_A], A_ptr=np.r_[0, np.cumsum(np.bincount(col, minlength=n))],
            P_idx=pr[perm_P], P_ptr=np.r_[0, np.cumsum(np.bincount(pc, minlength=n))])

    def _linearise(self, zr, ur):
        """Varying dynamics coefficients (5, N): d row on psi, v, steer; psi row on v, steer (in the _dynamics_rows
        entry order, which _fixed_pattern's 'dyn' offset relies on); and the right-hand side (N, 4)."""
        p, N, h = self.plant.params, self.cfg.horizon, self.cfg.step
        psi, v, delta = zr[:-1, 2], zr[:-1, 3], ur[:, 1]
        L = p.wheelbase
        kappa = getattr(p, 'cg_to_rear_axle', L/2)/L
        tan, sec2 = np.tan(delta), 1/np.cos(delta)**2
        root = 1+(kappa*tan)**2
        beta, dbeta = np.arctan(kappa*tan), kappa*sec2/root
        g, dg = tan/(L*np.sqrt(root)), sec2/(L*root**1.5)
        cos_head, sin_head = np.cos(psi+beta), np.sin(psi+beta)
        rhs = np.zeros((N, 4))   # h*(f - Fz zr - Fu ur)
        rhs[:, 1] = -h*v*cos_head*(psi+dbeta*delta)
        rhs[:, 2] = -h*v*dg*delta
        return np.array([-h*v*cos_head, -h*sin_head, -h*v*cos_head*dbeta, -h*g, -h*v*dg]), rhs

    def _dynamics_rows(self, rows, zr, ur):
        """Dynamics rows linearised around the reference (zr: (N+1, 4), ur: (N, 2)).

        z_{k+1} = z_k + h*(f + Fz (z_k - zr_k) + Fu (u_k - ur_k)); CG bicycle, slip beta = atan(kappa tan(steer)),
        kappa = l_r/L:  d' = v sin(psi+beta),  psi' = v g(steer),  g = cos(beta) tan(steer)/L = tan/(L sqrt(1+kappa^2 tan^2)).
        """
        N, h, zi, ui = self.cfg.horizon, self.cfg.step, self.zi, self.ui
        k, j4 = np.arange(N), np.arange(4)
        coef, rhs = self._linearise(zr, ur)
        r = 4*k
        entries = [(r[:, None]+j4, zi(k+1, 0)[:, None]+j4, 1.0),     # +z_{k+1}
                   (r[:, None]+j4, zi(k, 0)[:, None]+j4, -1.0),      # -z_k
                   (r, zi(k, 3), -h),                                 # s' = v
                   (r+1, zi(k, 2), coef[0]), (r+1, zi(k, 3), coef[1]), (r+1, ui(k, 1), coef[2]),
                   (r+2, zi(k, 3), coef[3]), (r+2, ui(k, 1), coef[4]),
                   (r+3, ui(k, 0), -h)]
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
        nz = self.nz
        rows, cols, vals = self.base_hessian
        slack = nz+self.nu+np.arange(len(w_slack))
        P = sp.csc_matrix((np.r_[vals, 2*w_slack], (np.r_[rows, slack], np.r_[cols, slack])),
                          shape=(n, n))
        q = self._linear_cost(np.zeros(n), target, steer_prev, accel_prev)
        q[slack] = w_slack
        return P, q

    def _linear_cost(self, q, target, steer_prev, accel_prev):
        """Fill the state and input part of q (targets; rate costs against the previous inputs)."""
        c, N, nz = self.cfg, self.cfg.horizon, self.nz
        ks = np.arange(1, N+1)
        q[4*ks+1] = -2*c.w_lateral*target.d
        q[4*ks+3] = -2*c.w_speed*target.speed
        q[nz] = -2*c.w_accel_rate*accel_prev
        q[nz+1] = -2*c.w_steer_rate*steer_prev
        return q
