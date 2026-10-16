"""Per-vehicle road-frame nonlinear MPC: dynamic bicycle with Fiala tyres, acados SQP-RTI (HPIPM).

Same solve() interface and Plan output as LaneChangeMPC (single.py), so the coordinator, fallbacks and viz are
unchanged. States x = [s, d, psi, vx, vy, r, steer, a] (s relative to s0), inputs u = [steer rate, accel rate].
Straight road: s' = vx cos psi - vy sin psi, d' = vx sin psi + vy cos psi, psi' = r; vx', vy', r' are
DynamicPlant.derivatives (Fiala brush tyres, friction ellipse with Fx = m a split 50/50 over static axle loads).
Deviations from the plant: slip angles use max(vx, 1) (plant 0.5; team speeds are >= 18 m/s), the axle peak floors
the ellipse at 5 % of mu Fz (never reached: |Fx| <= 4.5 kN vs mu Fz ~ 5.9 kN), no low-speed kinematic blend and no
vx >= 0 clamp, and the plant holds the commanded steer (rate limited) over each 0.1 s while the model ramps it.
The Fiala piecewise is exact (CasADi if_else): it is C1 at the saturation slip, enough for Gauss-Newton.

Cost (NONLINEAR_LS, cost scaling 1) mirrors LaneChangeMPC: y = [d - target.d, psi, s' - target.speed, steer, a,
vy, r, steer rate, accel rate]. The command for interval k is (a, steer) at stage k+1, so w_accel/w_steer weigh those
states at stages 1..N and the rate weights act on u_k scaled by step^2 (Delta = step*rate); the stage-0 state is
fixed, so its terms are constants; the terminal cost has the state terms. Two deviations from the linear MPC's cost:
the speed term is on the progress rate s' (as its s' = v), since one on body vx lets the plan yaw across the road to
gain vx while a "behind" row holds s; and small weights on vy and r (Planner.nmpc_w_vy/_yaw_rate): Gauss-Newton
ignores the bilinear r*vy coupling, and without them the full-step SQP diverges (plans drift sideways to "gain"
speed through the linearisation). A soft bound |tan slip| <= nmpc_slip_limit*tan(saturation slip) per axle (stages
1..N) keeps plans off the Fiala plateau, where the force has no gradient.
Box bounds: road edges on d and vx >= 0 (constraints.state_bounds), |steer| <= steer_max (optional grip cap:
Planner.nmpc_steer_cap), -brake_max <= a <= accel_limit(reference speed), |steer rate| <= steer_rate. Obstacle rows
(constraints.obstacle_constraints from the reference, v -> vx) fill fixed slots of general linear constraints at
stages 1..N (C at 1..N-1, C_e at N; none at stage 0): hard slots then soft slots (idxsg, Z = 2w, z = w; the other
side of a one-sided soft row is masked by its infinite bound); unused slots have zero rows and +-ACADOS_INFTY
bounds. Excess rows keep the nearest obstacles per stage and are counted in `overflow`.

Planner.nmpc_sqp_iter SQP-RTI iterations per call, linearised at the previous solution shifted by the steps between
its time and the reference's (matched on the reference trajectory, so one instance serves all three vehicles). If
there is none or the QP fails, retried from a model rollout of the reference commands, then from the kinematic
LaneChangeMPC solution (`stats` counts them). Status from the acados QP status: MAXITER 'timeout', MINSTEP/QP
failure 'infeasible', NaN 'failed'. Plan.u[k] = [a, steer] at stage k+1, z[:, 3] = hypot(vx, vy).

Code is generated once per configuration hash under NOHESI_ACADOS_BUILD (default C:/dev/acados_build: no spaces)
behind a file lock, then loaded with generate=False, build=False (safe across ProcessPoolExecutor workers).
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter, sleep
import warnings
import numpy as np

os.environ.setdefault('ACADOS_SOURCE_DIR', 'C:/acados')
_PATHS = [r'C:\msys64\ucrt64\bin', os.path.join(os.environ['ACADOS_SOURCE_DIR'], 'bin'),
          os.path.join(os.environ['ACADOS_SOURCE_DIR'], 'lib'), os.path.dirname(sys.executable)]
_have = os.environ.get('PATH', '').split(os.pathsep)
os.environ['PATH'] = os.pathsep.join([p for p in _PATHS if p not in _have]+_have)
import casadi as ca   # noqa: E402
from acados_template import (ACADOS_INFTY, AcadosModel, AcadosOcp, AcadosOcpSolver,   # noqa: E402
                             ocp_get_default_cmake_builder)
from .base import Plan   # noqa: E402
from .constraints import obstacle_constraints, state_bounds   # noqa: E402
from .single import LaneChangeMPC   # noqa: E402
from ..dynamics import kinematic_body   # noqa: E402

CODE_VERSION = 'nmpc-4'   # bump when the generated OCP structure changes
NX, NU, G = 8, 2, 9.81
S, D, PSI, VX, VY, R, STEER, ACCEL = range(NX)
NE = 7                          # terminal cost outputs [d, psi, s', steer, a, vy, r]
BOXES = (D, VX, STEER, ACCEL)   # idxbx at stages 1..N
COND_N = 5                      # HPIPM partial condensing horizon
BUILDS = 0                      # code generations + compilations in this process
INF = ACADOS_INFTY
QP_STATUS = {1: 'failed', 2: 'timeout', 3: 'infeasible', 4: 'infeasible'}   # acados status codes


def build_root():
    return Path(os.environ.get('NOHESI_ACADOS_BUILD', 'C:/dev/acados_build'))


def fiala(alpha, stiffness, peak):
    """CasADi copy of dynamics.fiala (exact piecewise)."""
    z = ca.tan(alpha)
    cubic = stiffness*z-stiffness**2*ca.fabs(z)*z/(3*peak)+stiffness**3*z**3/(27*peak**2)
    return ca.if_else(ca.fabs(z) >= 3*peak/stiffness, ca.sign(z)*peak, cubic)


def model_dynamics(p):
    """(x, u, xdot, slip) CasADi symbols/expressions of the road-frame dynamic bicycle for vehicle params p;
    slip = tan(slip angle)/tan(saturation slip angle) per axle (front, rear): |slip| >= 1 is the Fiala plateau."""
    x, u = ca.SX.sym('x', NX), ca.SX.sym('u', NU)
    psi, vx, vy, r, delta, a = x[PSI], x[VX], x[VY], x[R], x[STEER], x[ACCEL]
    l_r = p.cg_to_rear_axle
    l_f = p.wheelbase-l_r
    loads = (p.mass*G*l_r/p.wheelbase, p.mass*G*l_f/p.wheelbase)
    fx = 0.5*p.mass*a
    peak_f, peak_r = (ca.sqrt(ca.fmax((p.friction*fz)**2-fx**2, (0.05*p.friction*fz)**2)) for fz in loads)
    vxs = ca.fmax(vx, 1.0)
    alpha_f, alpha_r = delta-ca.atan((vy+l_f*r)/vxs), -ca.atan((vy-l_r*r)/vxs)
    fyf = fiala(alpha_f, p.cornering_stiffness_front, peak_f)
    fyr = fiala(alpha_r, p.cornering_stiffness_rear, peak_r)
    slip = ca.vertcat(ca.tan(alpha_f)*p.cornering_stiffness_front/(3*peak_f),
                      ca.tan(alpha_r)*p.cornering_stiffness_rear/(3*peak_r))
    xdot = ca.vertcat(vx*ca.cos(psi)-vy*ca.sin(psi), vx*ca.sin(psi)+vy*ca.cos(psi), r,
                      a+r*vy-fyf*ca.sin(delta)/p.mass, (fyf*ca.cos(delta)+fyr)/p.mass-r*vx,
                      (l_f*fyf*ca.cos(delta)-l_r*fyr)/p.yaw_inertia, u[0], u[1])
    return x, u, xdot, slip


def rk4(p, step, substeps):
    """CasADi Function (x, u) -> x after `step` s: the same RK4 the ERK integrator uses."""
    x, u, xdot, _ = model_dynamics(p)
    f = ca.Function('f', [x, u], [xdot])
    h, xk = step/substeps, x
    for _ in range(substeps):
        k1 = f(xk, u); k2 = f(xk+h/2*k1, u); k3 = f(xk+h/2*k2, u); k4 = f(xk+h*k3, u)
        xk = xk+h/6*(k1+2*k2+2*k3+k4)
    return ca.Function('rk4', [x, u], [xk])


@contextmanager
def _file_lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        if os.name == 'nt':
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    sleep(0.2)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == 'nt':
            os.lseek(fd, 0, 0)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)


def config_hash(cfg, p):
    key = {'code': CODE_VERSION, 'cond_n': COND_N,
           'planner': [cfg.horizon, cfg.step, cfg.nmpc_hard_slots, cfg.nmpc_soft_slots, cfg.nmpc_integrator_steps,
                       cfg.nmpc_slip_limit],
           'vehicle': [float(getattr(p, k)) for k in ('wheelbase', 'cg_to_rear_axle', 'mass', 'yaw_inertia',
                                                      'cornering_stiffness_front', 'cornering_stiffness_rear',
                                                      'friction')]}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


class NonlinearMPC:
    def __init__(self, cfg, plant, road):
        with warnings.catch_warnings():   # acados_template's own deprecated-option accessors
            warnings.simplefilter('ignore', DeprecationWarning)
            self._setup(cfg, plant, road)

    def _setup(self, cfg, plant, road):
        self.cfg, self.plant, self.road = cfg, plant, road
        N, h, p = cfg.horizon, cfg.step, plant.params
        self.H, self.Ssoft = cfg.nmpc_hard_slots, cfg.nmpc_soft_slots
        self.ng = self.H+self.Ssoft
        self.hash = config_hash(cfg, p)
        self.dir = build_root()/self.hash
        self.overflow = {'hard': 0, 'soft': 0}   # obstacle rows dropped for lack of slots
        self.cache = []                          # recent solutions for warm starts: (time, z, X_abs, U)
        self.stats = dict.fromkeys(('warm', 'rollout', 'linear', 'retry', 'retry_solved'), 0)   # guesses tried
        self.linear = None                       # LaneChangeMPC for the last-resort initial guess
        self.compiled = False                    # True when this construction generated and compiled the code
        ocp = self._ocp()
        builder = ocp_get_default_cmake_builder()
        builder.generator = 'MinGW Makefiles'
        builder.additional_cmake_options = '-DCMAKE_C_COMPILER=gcc -DCMAKE_POLICY_VERSION_MINIMUM=3.5'
        ready = self.dir/'ready'
        if not ready.exists():
            build_root().mkdir(parents=True, exist_ok=True)
            with _file_lock(str(build_root()/f'{self.hash}.lock')):
                if not ready.exists():
                    global BUILDS
                    shutil.rmtree(self.dir, ignore_errors=True)
                    self.dir.mkdir(parents=True)
                    AcadosOcpSolver.generate(ocp, cmake_builder=builder, verbose=False)
                    AcadosOcpSolver.build(ocp.code_gen_options.code_export_directory, cmake_builder=builder,
                                          verbose=False)
                    ready.write_text(ocp.name)
                    BUILDS += 1
                    self.compiled = True
        self.solver = AcadosOcpSolver(ocp, generate=False, build=False, check_reuse_possible=False, verbose=False)
        # Values fixed per configuration but not compiled in (the shared library may come from other weights).
        c = cfg
        W = 2*np.diag([c.w_lateral, c.w_heading, c.w_speed, c.w_steer, c.w_accel, c.nmpc_w_vy, c.nmpc_w_yaw_rate,
                       c.w_steer_rate*h*h, c.w_accel_rate*h*h])
        for k in range(N):
            self.solver.cost_set(k, 'W', W)
            self.solver.constraints_set(k, 'lbu', np.array([-p.steer_rate]))
            self.solver.constraints_set(k, 'ubu', np.array([p.steer_rate]))
        self.solver.cost_set(N, 'W', W[:NE, :NE])
        self.solver.options_set('levenberg_marquardt', float(c.nmpc_levenberg_marquardt))
        self.rk4 = rk4(p, h, cfg.nmpc_integrator_steps)

    def _ocp(self):
        c, p, N = self.cfg, self.plant.params, self.cfg.horizon
        x, u, xdot, slip = model_dynamics(p)
        model = AcadosModel()
        model.name = f'nohesi_nmpc_{self.hash}'
        model.x, model.u, model.f_expl_expr = x, u, xdot
        model.xdot = ca.SX.sym('xdot', NX)
        model.f_impl_expr = model.xdot-xdot
        ocp = AcadosOcp()
        ocp.model, ocp.name = model, model.name
        ocp.code_gen_options.code_export_directory = str(self.dir/'c_generated_code').replace('\\', '/')
        ocp.code_gen_options.json_file = 'nmpc.json'   # acados writes it into code_export_directory
        o = ocp.solver_options
        o.N_horizon, o.tf = N, N*c.step
        o.integrator_type, o.sim_method_num_stages, o.sim_method_num_steps = 'ERK', 4, c.nmpc_integrator_steps
        o.nlp_solver_type, o.qp_solver = 'SQP_RTI', 'PARTIAL_CONDENSING_HPIPM'
        o.qp_solver_cond_N, o.qp_solver_iter_max, o.hessian_approx = min(COND_N, N), 200, 'GAUSS_NEWTON'
        o.cost_scaling = np.ones(N+1)
        o.print_level = 0
        # Cost (module docstring): y = [d, psi, s', steer, a, vy, r, steer rate, accel rate]; terminal the first NE.
        cost = ocp.cost
        cost.cost_type = cost.cost_type_e = 'NONLINEAR_LS'
        y = ca.vertcat(x[D], x[PSI], xdot[S], x[STEER], x[ACCEL], x[VY], x[R], u)
        model.cost_y_expr, model.cost_y_expr_e = y, y[:NE]
        cost.W, cost.yref, cost.W_e, cost.yref_e = np.eye(NE+2), np.zeros(NE+2), np.eye(NE), np.zeros(NE)
        con = ocp.constraints
        con.x0 = np.zeros(NX)
        con.idxbu, con.lbu, con.ubu = np.array([0]), np.array([-p.steer_rate]), np.array([p.steer_rate])
        con.idxbx = con.idxbx_e = np.array(BOXES)
        con.lbx = con.lbx_e = np.array([-INF, 0.0, -p.steer_max, -p.brake_max])
        con.ubx = con.ubx_e = np.array([INF, INF, p.steer_max, p.accel_max])
        ng, soft = self.ng, np.arange(self.H, self.ng)
        con.C, con.D, con.lg, con.ug = np.zeros((ng, NX)), np.zeros((ng, NU)), -INF*np.ones(ng), INF*np.ones(ng)
        con.C_e, con.lg_e, con.ug_e = np.zeros((ng, NX)), -INF*np.ones(ng), INF*np.ones(ng)
        con.idxsg, con.lsg, con.usg = soft, np.zeros(len(soft)), np.zeros(len(soft))
        con.idxsg_e, con.lsg_e, con.usg_e = soft, np.zeros(len(soft)), np.zeros(len(soft))
        # Soft slip limit |slip| <= nmpc_slip_limit at stages 1..N (none at stage 0): keeps plans off the tyre
        # plateau, where the force has no gradient and RTI loses steering authority.
        model.con_h_expr, model.con_h_expr_e = slip, slip
        lim = c.nmpc_slip_limit*np.ones(2)
        con.lh, con.uh, con.lh_e, con.uh_e = -lim, lim, -lim, lim
        con.idxsh = con.idxsh_e = np.arange(2)
        con.lsh = con.ush = con.lsh_e = con.ush_e = np.zeros(2)
        ones, ones_h = np.ones(len(soft)), np.r_[np.ones(len(soft)), np.full(2, c.nmpc_w_slip)]
        cost.zl_0 = cost.zu_0 = ones
        cost.Zl_0 = cost.Zu_0 = 2*ones
        cost.zl = cost.zu = cost.zl_e = cost.zu_e = ones_h
        cost.Zl = cost.Zu = cost.Zl_e = cost.Zu_e = 2*ones_h
        return ocp

    def solve(self, time, z0, steer_prev, accel_prev, target, obstacles, reference, lateral=None):
        """lateral: (vy, r) of the plant (zeros if None)."""
        start = perf_counter()
        c, p, N, h, sv = self.cfg, self.plant.params, self.cfg.horizon, self.cfg.step, self.solver
        s0 = float(z0[0])
        vy0, r0 = (0.0, 0.0) if lateral is None else (float(lateral[0]), float(lateral[1]))
        x0 = np.array([0.0, z0[1], z0[2], np.sqrt(max(z0[3]**2-vy0**2, 0.0)), vy0, r0, steer_prev, accel_prev], float)
        sv.constraints_set(0, 'lbx', x0)
        sv.constraints_set(0, 'ubx', x0)

        # Per-stage bounds and references.
        zr = reference.z
        speed = np.maximum(zr[:-1, 3], 0.0)
        a_hi = self.plant.accel_limit(speed)
        cap = self.plant.steer_limit(speed) if c.nmpc_steer_cap else np.full(N, p.steer_max)
        bounds = state_bounds(c, p, self.road)
        edge = bounds.hi[bounds.kind == 'edge']
        for k in range(1, N+1):
            sv.constraints_set(k, 'lbx', np.array([-edge[k-1], 0.0, -cap[k-1], -p.brake_max]))
            sv.constraints_set(k, 'ubx', np.array([edge[k-1], INF, cap[k-1], a_hi[k-1]]))
        yref = np.r_[target.d, 0.0, target.speed, np.zeros(NE-1)]
        for k in range(N):
            sv.cost_set(k, 'yref', yref)
        sv.cost_set(N, 'yref', yref[:NE])
        self._obstacle_slots(zr, obstacles, target, s0)

        # Linearisation points tried in turn (RTI: the QP is only as good as its linearisation): the previous
        # solution shifted if the reference is that plan, a model rollout of the reference commands, then the
        # kinematic LaneChangeMPC solution.
        warm = self._warm_guess(x0, s0, reference)
        guesses = (['warm'] if warm is not None else [])+['rollout', 'linear']
        args, iters = (time, z0, steer_prev, accel_prev, target, obstacles, reference), 0
        for n, kind in enumerate(guesses):
            X, U = warm if kind == 'warm' else getattr(self, f'_{kind}_guess')(x0, s0, reference, args)
            self.stats[kind] += 1
            self.stats['retry'] += n > 0
            sv.set_flat('x', X.ravel())
            sv.set_flat('u', U.ravel())
            status = 'solved'
            for _ in range(c.nmpc_sqp_iter):
                code = sv.solve()
                iters += int(np.sum(sv.get_stats('qp_iter')))
                if code != 0:
                    qp = np.atleast_1d(sv.get_stats('qp_stat'))
                    status = QP_STATUS.get(int(qp[-1]) if qp.size else code, 'failed')
                    break
            X = sv.get_flat('x').reshape(N+1, NX)
            U = sv.get_flat('u').reshape(N, NU)
            if status == 'solved' and not (np.all(np.isfinite(X)) and np.all(np.isfinite(U))):
                status = 'failed'
            if status == 'solved':
                self.stats['retry_solved'] += n > 0
                break
            sv.reset()
        elapsed = perf_counter()-start
        if status != 'solved':
            return Plan(time, h, reference.z, reference.u, 'qp', status, iters, elapsed)
        z = np.column_stack((X[:, S]+s0, X[:, D], X[:, PSI], np.hypot(X[:, VX], X[:, VY])))
        z[0] = [s0, z0[1], z0[2], z0[3]]
        u = X[1:, [ACCEL, STEER]].copy()
        Xa = X.copy(); Xa[:, S] += s0
        self.cache = [(time, z, Xa, U)]+self.cache[:7]
        return Plan(time, h, z, u, 'qp', status, iters, elapsed)

    def _warm_guess(self, x0, s0, reference):
        """Previous solution shifted to the reference's time when the reference is that plan shifted, else None."""
        N, h = self.cfg.horizon, self.cfg.step
        for t, z, Xa, U in self.cache:
            m = int(round((reference.time-t)/h))
            if 1 <= m <= N//2 and abs(reference.time-t-m*h) < 1e-6 and \
                    np.allclose(reference.z[:N+1-m], z[m:], atol=1e-6):
                X = np.vstack((Xa[m:], np.repeat(Xa[-1:], m, 0)))
                X[N+1-m:, S] = Xa[-1, S]+Xa[-1, VX]*h*np.arange(1, m+1)
                X[:, S] -= s0
                X[0] = x0
                return X, np.vstack((U[m:], np.zeros((m, NU))))
        return None

    def _linear_guess(self, x0, s0, reference, args):
        """States of the kinematic LaneChangeMPC solution (vy, r from the kinematic relations), else the reference."""
        N, h, p = self.cfg.horizon, self.cfg.step, self.plant.params
        if self.linear is None:
            self.linear = LaneChangeMPC(self.cfg, self.plant, self.road)
        plan = self.linear.solve(*args)
        z, u = plan.z, plan.u
        X = np.zeros((N+1, NX))
        X[:, S], X[:, D], X[:, PSI] = z[:, 0]-s0, z[:, 1], z[:, 2]
        X[1:, STEER], X[1:, ACCEL] = u[:, 1], u[:, 0]
        for k in range(1, N+1):
            X[k, VX], X[k, VY], X[k, R] = kinematic_body(z[k, 3], X[k, STEER], p)
        X[0] = x0
        U = np.column_stack((np.diff(X[:, STEER])/h, np.diff(X[:, ACCEL])/h))
        return X, U

    def _rollout_guess(self, x0, s0, reference, args=None):
        """Model rollout from x0 with rates that track the reference commands [a, steer] (rate-limited steer)."""
        N, h, rate = self.cfg.horizon, self.cfg.step, self.plant.params.steer_rate
        X, U = np.zeros((N+1, NX)), np.zeros((N, NU))
        X[0] = x0
        for k in range(N):
            a, delta = reference.u[k]
            U[k] = [np.clip((delta-X[k, STEER])/h, -rate, rate), (a-X[k, ACCEL])/h]
            X[k+1] = np.asarray(self.rk4(X[k], U[k])).ravel()
        return X, U

    def _obstacle_slots(self, zr, obstacles, target, s0):
        """Map obstacle_constraints rows onto the fixed hard/soft slots of stages 1..N."""
        N, H, ns, sv = self.cfg.horizon, self.H, self.Ssoft, self.solver
        C = np.zeros((N+1, self.ng, NX))
        lg, ug = np.full((N+1, self.ng), -INF), np.full((N+1, self.ng), INF)
        zl = np.tile(np.r_[np.ones(ns), np.full(2, self.cfg.nmpc_w_slip)], (N+1, 1))   # soft slots, then slip
        Zl = 2*zl
        if obstacles.keys:
            spec = obstacle_constraints(self.cfg, self.plant.params, self.road, zr, obstacles, target, s0)
            # Relevance for overflow: distance along the road (plus lateral) from the reference to obstacle j.
            jj = np.maximum(spec.j, 0)
            dist = np.abs(zr[spec.k, 0]-obstacles.s[spec.k, jj])+np.abs(zr[spec.k, 1]-obstacles.d[spec.k, jj])
            for soft, offset, n, name in ((False, 0, H, 'hard'), (True, H, ns, 'soft')):
                idx = np.flatnonzero(spec.soft == soft)
                idx = idx[np.lexsort((dist[idx], spec.k[idx]))]   # by stage, then nearest first
                k = spec.k[idx]
                first = np.searchsorted(k, k)                      # rank within stage
                slot = np.arange(len(idx))-first
                keep = slot < n
                self.overflow[name] += int(np.sum(~keep))
                idx, k, slot = idx[keep], k[keep], slot[keep]+offset
                C[k, slot, S], C[k, slot, D], C[k, slot, VX] = spec.cs[idx], spec.cd[idx], spec.cv[idx]
                lg[k, slot] = np.maximum(spec.lo[idx], -INF)
                ug[k, slot] = np.minimum(spec.hi[idx], INF)
                if soft:
                    zl[k, slot-H], Zl[k, slot-H] = spec.slack_weight[idx], 2*spec.slack_weight[idx]
        for k in range(1, N+1):
            sv.constraints_set(k, 'C', C[k], api='new')
            sv.constraints_set(k, 'lg', lg[k])
            sv.constraints_set(k, 'ug', ug[k])
            # One finite bound per soft row; the other side is masked (infinite), so both slacks share the weight.
            for name, val in (('zl', zl[k]), ('zu', zl[k]), ('Zl', Zl[k]), ('Zu', Zl[k])):
                sv.cost_set(k, name, val)
