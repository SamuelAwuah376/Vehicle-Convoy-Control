"""Per-vehicle road-frame nonlinear MPC: dynamic bicycle with Fiala tyres, acados SQP-RTI (HPIPM).

Same solve() interface and Plan output as LaneChangeMPC (single.py), so the coordinator, fallbacks and viz are
unchanged. States x = [s, d, psi, vx, vy, r, steer, a] (s relative to s0), inputs u = [steer rate, accel rate].
Straight road: s' = vx cos psi - vy sin psi, d' = vx sin psi + vy cos psi, psi' = r; vx', vy', r' are
DynamicPlant.derivatives (Fiala brush tyres, friction ellipse; axle loads with longitudinal load transfer, Fx = m a
split by the brake or drive bias and capped at mu Fz per axle, as DynamicPlant.axle_forces).
Deviations from the plant: slip angles use max(vx, nmpc_vx_floor) (plant 0.5, kinematic below 3-6 m/s: near standstill
the tyre dynamics are stiffer than the RK4 can integrate, which made emergency stops diverge), the axle peak floors
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
1..N) keeps plans off the Fiala plateau, where the force has no gradient, and a HARD friction-circle bound
sqrt(Fx^2 + Fy^2) <= nmpc_grip_limit mu Fz per axle (stages 1..N) keeps every plan within what the tyres can do, with
grip in reserve when braking (or accelerating) and turning.
Physics is hard, avoidance soft: obstacle rows and the road edges are soft with weight nmpc_w_obstacle (linear and
quadratic slack cost, large enough to act as an exact penalty: a plan that avoids everything within grip is returned
as with hard constraints; only when none exists is the least violation taken), so a solve fails only when the grip
bound can't be met (e.g. the car is already sliding past it). `stats['violations']` counts solved plans that use an
obstacle or edge slack (> 1 cm).
Box bounds: road edges on d (soft, as above) and vx >= 0 (constraints.state_bounds), |steer| <= steer_max (optional grip cap:
Planner.nmpc_steer_cap), -brake_limit <= a <= accel_limit(reference speed) (dynamics.brake_limit: grip_brake mu g on
the dynamic plant), |steer rate| <= steer_rate. Obstacle rows
(constraints.obstacle_constraints from the reference, v -> vx) fill fixed slots of general linear constraints at
stages 1..N (C at 1..N-1, C_e at N; none at stage 0): the hard rows' slots (weight nmpc_w_obstacle) then the soft
rows' (headway, yield: their own weights), all soft (idxsg, Z = 2w, z = w; the other side of a one-sided row is
masked by its infinite bound); unused slots have zero rows and +-ACADOS_INFTY bounds. Excess rows keep the nearest
obstacles per stage and are counted in `overflow`. Slack order per stage 1..N: road edge, obstacle slots, slip.

Emergency solves (solve(..., emergency=True), CentralizedTeam._emergency): only the contact rows (soft headway and
yield rows dropped), nmpc_emergency_sqp_iter iterations, and the smoothness weights (steer, accel, their rates) scaled
by nmpc_emergency_comfort; `last_slack` is the largest obstacle/edge slack of the last solved plan (m).
Plan verification (Planner.nmpc_verify): one SQP-RTI iteration satisfies the grip limit only to first order, so a
solved plan's true grip use (the model's nonlinear value at its states, stages 1..N) is checked; while it breaks
nmpc_grip_limit + nmpc_grip_tol, up to nmpc_verify_iter more iterations run (and, when nmpc_verify_budget > 0, only
within that much wall-clock time). If the car starts past the limit, the first nmpc_recover_steps stages need only
not exceed its starting use. A plan that still breaks it, or whose extra iterations fail, has status 'grip' (not verified): the next guess is tried, and
the coordinator hands a car with no plan to the emergency mode. `stats['verify_iter']` counts the extra iterations,
`stats['grip_rejected']` the guesses rejected.
Iterates are checked before every solver call (_sane: finite, and speed, lateral speed, yaw rate, heading, position,
steer and acceleration within physical bounds, SANE). An iterate past them (a bad guess's first step can jump to an
absurd trajectory) would be linearised into a QP with Jacobian entries up to 1e43, on which HPIPM never returns (seen
twice in training): that guess fails instead (status 'failed', `stats['insane']`) and the next one is tried.
Per-solve data goes in through _Setter: the same acados C setters AcadosOcpSolver.constraints_set/cost_set call, without
their per-call Python checks (type and shape checks, a dims query, three casts), and skipping values a stage already
holds. About 360 such sets per solve cost as much as the solve itself (profiled); the values passed are identical.
Planner.nmpc_sqp_iter SQP-RTI iterations per call, linearised at the previous solution shifted by the steps between
its time and the reference's (matched on the reference trajectory, so one instance serves all three vehicles). If
there is none or the QP fails, retried from a model rollout of the reference commands, then from the kinematic
LaneChangeMPC solution (`stats` counts them). Status from the acados QP status: MAXITER 'timeout', MINSTEP/QP
failure 'infeasible', NaN 'failed'. Plan.u[k] = [a, steer] at stage k+1, z[:, 3] = hypot(vx, vy).

Code is generated once per configuration hash under NOHESI_ACADOS_BUILD (default C:/dev/acados_build: no spaces)
behind a file lock, then loaded with generate=False, build=False (safe across ProcessPoolExecutor workers).
"""
from contextlib import contextmanager
from ctypes import c_int, c_void_p
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
from ..dynamics import brake_limit, kinematic_body   # noqa: E402

CODE_VERSION = 'nmpc-8'   # bump when the generated OCP structure changes
NX, NU, G = 8, 2, 9.81
S, D, PSI, VX, VY, R, STEER, ACCEL = range(NX)
NE = 7                          # terminal cost outputs [d, psi, s', steer, a, vy, r]
BOXES = (D, VX, STEER, ACCEL)   # idxbx at stages 1..N
COND_N = 5                      # HPIPM partial condensing horizon
BUILDS = 0                      # code generations + compilations in this process
INF = ACADOS_INFTY
QP_STATUS = {1: 'failed', 2: 'timeout', 3: 'infeasible', 4: 'infeasible'}   # acados status codes
# Physical bounds of a plan iterate (relative s, d, heading error, vx, vy, yaw rate; steer and accel are checked against
# the vehicle): anything past them is a diverged iterate, never linearised (see the module docstring).
SANE = {S: (-100.0, 600.0), D: (-30.0, 30.0), PSI: (-np.pi, np.pi), VX: (-5.0, 100.0), VY: (-40.0, 40.0), R: (-4.0, 4.0)}


class _Setter:
    """Direct acados setters for one solver: constraints and cost fields of one stage, as constraints_set/cost_set
    (column-major matrices), skipping the Python wrapper's checks; `cached` fields skip a value equal to the last one
    set at that stage. Callers pass arrays of the right size."""
    def __init__(self, solver, cached=('lbx', 'ubx', 'zl', 'zu', 'Zl', 'Zu')):
        self.lib, self.cached, self.last = solver.acados_lib, frozenset(cached), {}
        self.c_args = (solver.nlp_config, solver.nlp_dims, solver.nlp_in, solver.nlp_out)
        self.k_args = (solver.nlp_config, solver.nlp_dims, solver.nlp_in)
        self.names = {}

    def _skip(self, stage, field, value):
        if field not in self.cached:
            return False
        key = (stage, field)
        old = self.last.get(key)
        if old is not None and np.array_equal(old, value):
            return True
        self.last[key] = value.copy()
        return False

    def constraints(self, stage, field, value):
        value = np.ascontiguousarray(value, dtype=float)
        if not self._skip(stage, field, value):
            self.lib.ocp_nlp_constraints_model_set(*self.c_args, c_int(stage), self._name(field), c_void_p(value.ctypes.data))

    def cost(self, stage, field, value):
        value = np.ascontiguousarray(value, dtype=float)
        if not self._skip(stage, field, value):
            self.lib.ocp_nlp_cost_model_set(*self.k_args, c_int(stage), self._name(field), c_void_p(value.ctypes.data))

    def _name(self, field):
        name = self.names.get(field)
        if name is None:
            name = self.names[field] = field.encode('utf-8')
        return name


def build_root():
    return Path(os.environ.get('NOHESI_ACADOS_BUILD', 'C:/dev/acados_build'))


def fiala(alpha, stiffness, peak):
    """CasADi copy of dynamics.fiala (exact piecewise)."""
    z = ca.tan(alpha)
    cubic = stiffness*z-stiffness**2*ca.fabs(z)*z/(3*peak)+stiffness**3*z**3/(27*peak**2)
    return ca.if_else(ca.fabs(z) >= 3*peak/stiffness, ca.sign(z)*peak, cubic)


def model_dynamics(p, vx_floor=1.0):
    """(x, u, xdot, slip, grip) CasADi symbols/expressions of the road-frame dynamic bicycle for vehicle params p;
    slip = tan(slip angle)/tan(saturation slip angle) per axle (front, rear): |slip| >= 1 is the Fiala plateau;
    grip = sqrt(Fx^2 + Fy^2)/(mu Fz) per axle (friction-circle use)."""
    x, u = ca.SX.sym('x', NX), ca.SX.sym('u', NU)
    psi, vx, vy, r, delta, a = x[PSI], x[VX], x[VY], x[R], x[STEER], x[ACCEL]
    l_r = p.cg_to_rear_axle
    l_f = p.wheelbase-l_r
    loads = (p.mass*G*l_r/p.wheelbase, p.mass*G*l_f/p.wheelbase)
    # As DynamicPlant.axle_forces: load transfer, brake/drive split, Fx capped at mu Fz per axle.
    mu_g = p.friction*G
    shift = p.mass*ca.fmin(ca.fmax(a, -mu_g), mu_g)*p.cg_height/p.wheelbase
    fz_f, fz_r = loads[0]-shift, loads[1]+shift
    share = ca.if_else(a < 0, p.brake_bias_front, p.drive_bias_front)
    raw_f, raw_r = share*p.mass*a, (1-share)*p.mass*a
    fx_f = ca.fmin(ca.fmax(raw_f, -p.friction*fz_f), p.friction*fz_f)
    fx_r = ca.fmin(ca.fmax(raw_r, -p.friction*fz_r), p.friction*fz_r)
    ax = a-(raw_f-fx_f+raw_r-fx_r)/p.mass
    peak_f, peak_r = (ca.sqrt(ca.fmax((p.friction*fz)**2-fx**2, (0.05*p.friction*fz)**2))
                      for fz, fx in ((fz_f, fx_f), (fz_r, fx_r)))
    vxs = ca.fmax(vx, vx_floor)   # slip angles: well conditioned near standstill (Planner.nmpc_vx_floor)
    alpha_f, alpha_r = delta-ca.atan((vy+l_f*r)/vxs), -ca.atan((vy-l_r*r)/vxs)
    fyf = fiala(alpha_f, p.cornering_stiffness_front, peak_f)
    fyr = fiala(alpha_r, p.cornering_stiffness_rear, peak_r)
    slip = ca.vertcat(ca.tan(alpha_f)*p.cornering_stiffness_front/(3*peak_f),
                      ca.tan(alpha_r)*p.cornering_stiffness_rear/(3*peak_r))
    grip = ca.vertcat(ca.sqrt(fx_f**2+fyf**2+1e-6)/(p.friction*fz_f), ca.sqrt(fx_r**2+fyr**2+1e-6)/(p.friction*fz_r))
    xdot = ca.vertcat(vx*ca.cos(psi)-vy*ca.sin(psi), vx*ca.sin(psi)+vy*ca.cos(psi), r,
                      ax+r*vy-fyf*ca.sin(delta)/p.mass, (fyf*ca.cos(delta)+fyr)/p.mass-r*vx,
                      (l_f*fyf*ca.cos(delta)-l_r*fyr)/p.yaw_inertia, u[0], u[1])
    return x, u, xdot, slip, grip


def rk4(p, step, substeps, vx_floor=1.0):
    """CasADi Function (x, u) -> x after `step` s: the same RK4 the ERK integrator uses."""
    x, u, xdot, _, _ = model_dynamics(p, vx_floor)
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
                       cfg.nmpc_slip_limit, cfg.nmpc_grip_limit, cfg.nmpc_vx_floor],
           'vehicle': [float(getattr(p, k)) for k in ('wheelbase', 'cg_to_rear_axle', 'mass', 'yaw_inertia',
                                                      'cornering_stiffness_front', 'cornering_stiffness_rear',
                                                      'friction', 'cg_height', 'brake_bias_front', 'drive_bias_front')]}
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
        self.stats = dict.fromkeys(('warm', 'rollout', 'linear', 'retry', 'retry_solved', 'violations', 'verify_iter',
                                    'grip_rejected', 'insane'), 0)
        self.last_slack = np.inf                 # largest obstacle/edge slack of the last solve (inf: not solved)
        self.weights = None                      # 'normal' or 'emergency': the cost weights now set
        self.linear = None                       # LaneChangeMPC for the last-resort initial guess
        self.compiled = False                    # True when this construction generated and compiled the code
        self.brake = brake_limit(plant, cfg.grip_brake)
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
        self.set = _Setter(self.solver)
        # Values fixed per configuration but not compiled in (the shared library may come from other weights).
        c = cfg
        for k in range(N):
            self.solver.constraints_set(k, 'lbu', np.array([-p.steer_rate]))
            self.solver.constraints_set(k, 'ubu', np.array([p.steer_rate]))
        self._set_weights('normal')
        self.solver.options_set('levenberg_marquardt', float(c.nmpc_levenberg_marquardt))
        self.rk4 = rk4(p, h, cfg.nmpc_integrator_steps, cfg.nmpc_vx_floor)
        x, u, _, _, grip = model_dynamics(p, cfg.nmpc_vx_floor)
        self.grip = ca.Function('grip', [x, u], [grip]).map(N+1)   # (NX, N+1), (NU, N+1) -> (2, N+1)

    def _set_weights(self, kind):
        """Cost weights for normal or emergency solves (smoothness terms scaled by nmpc_emergency_comfort)."""
        if self.weights == kind:
            return
        c, N, h = self.cfg, self.cfg.horizon, self.cfg.step
        f = c.nmpc_emergency_comfort if kind == 'emergency' else 1.0
        W = 2*np.diag([c.w_lateral, c.w_heading, c.w_speed, f*c.w_steer, f*c.w_accel, c.nmpc_w_vy, c.nmpc_w_yaw_rate,
                       f*c.w_steer_rate*h*h, f*c.w_accel_rate*h*h])
        for k in range(N):
            self.solver.cost_set(k, 'W', W)
        self.solver.cost_set(N, 'W', W[:NE, :NE])
        self.weights = kind

    def _ocp(self):
        c, p, N = self.cfg, self.plant.params, self.cfg.horizon
        x, u, xdot, slip, grip = model_dynamics(p, c.nmpc_vx_floor)
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
        con.lbx = con.lbx_e = np.array([-INF, 0.0, -p.steer_max, -self.brake])
        con.ubx = con.ubx_e = np.array([INF, INF, p.steer_max, p.accel_max])
        ng = self.ng
        soft = np.arange(ng)   # every obstacle slot is soft (the hard rows' slots weighted nmpc_w_obstacle)
        con.idxsbx = con.idxsbx_e = np.array([0])   # BOXES[0] = d: road edges soft too
        con.lsbx = con.usbx = con.lsbx_e = con.usbx_e = np.zeros(1)
        con.C, con.D, con.lg, con.ug = np.zeros((ng, NX)), np.zeros((ng, NU)), -INF*np.ones(ng), INF*np.ones(ng)
        con.C_e, con.lg_e, con.ug_e = np.zeros((ng, NX)), -INF*np.ones(ng), INF*np.ones(ng)
        con.idxsg, con.lsg, con.usg = soft, np.zeros(len(soft)), np.zeros(len(soft))
        con.idxsg_e, con.lsg_e, con.usg_e = soft, np.zeros(len(soft)), np.zeros(len(soft))
        # Soft slip limit |slip| <= nmpc_slip_limit at stages 1..N (none at stage 0): keeps plans off the tyre
        # plateau, where the force has no gradient and RTI loses steering authority. Hard friction-circle limit
        # grip <= nmpc_grip_limit per axle: braking or accelerating hard leaves less grip for turning.
        hx = ca.vertcat(slip, grip)
        model.con_h_expr, model.con_h_expr_e = hx, hx
        lo = np.r_[-c.nmpc_slip_limit*np.ones(2), np.zeros(2)]
        hi = np.r_[c.nmpc_slip_limit*np.ones(2), np.full(2, c.nmpc_grip_limit)]
        con.lh, con.uh, con.lh_e, con.uh_e = lo, hi, lo, hi
        con.idxsh = con.idxsh_e = np.arange(2)   # slip soft; grip (rows 2, 3) hard
        con.lsh = con.ush = con.lsh_e = con.ush_e = np.zeros(2)
        ones, ones_h = np.ones(ng), self._slack_weights(np.ones(self.Ssoft))   # stage 0: obstacle slots only
        cost.zl_0 = cost.zu_0 = ones
        cost.Zl_0 = cost.Zu_0 = 2*ones
        cost.zl = cost.zu = cost.zl_e = cost.zu_e = ones_h
        cost.Zl = cost.Zu = cost.Zl_e = cost.Zu_e = self._slack_quad(2*np.ones(self.Ssoft))
        return ocp

    def solve(self, time, z0, steer_prev, accel_prev, target, obstacles, reference, lateral=None, emergency=False):
        """lateral: (vy, r) of the plant (zeros if None); emergency: an emergency solve (module docstring)."""
        start = perf_counter()
        c, p, N, h, sv = self.cfg, self.plant.params, self.cfg.horizon, self.cfg.step, self.solver
        s0 = float(z0[0])
        vy0, r0 = (0.0, 0.0) if lateral is None else (float(lateral[0]), float(lateral[1]))
        x0 = np.array([0.0, z0[1], z0[2], np.sqrt(max(z0[3]**2-vy0**2, 0.0)), vy0, r0, steer_prev, accel_prev], float)
        st = self.set
        st.constraints(0, 'lbx', x0)
        st.constraints(0, 'ubx', x0)

        # Per-stage bounds and references.
        zr = reference.z
        speed = np.maximum(zr[:-1, 3], 0.0)
        a_hi = self.plant.accel_limit(speed)
        cap = self.plant.steer_limit(speed) if c.nmpc_steer_cap else np.full(N, p.steer_max)
        bounds = state_bounds(c, p, self.road)
        edge = bounds.hi[bounds.kind == 'edge']
        for k in range(1, N+1):
            st.constraints(k, 'lbx', np.array([-edge[k-1], 0.0, -cap[k-1], -self.brake]))
            st.constraints(k, 'ubx', np.array([edge[k-1], INF, cap[k-1], a_hi[k-1]]))
        yref = np.r_[target.d, 0.0, target.speed, np.zeros(NE-1)]
        for k in range(N):
            st.cost(k, 'yref', yref)
        st.cost(N, 'yref', yref[:NE])
        self._obstacle_slots(zr, obstacles, target, s0, contact_only=emergency)
        self._set_weights('emergency' if emergency else 'normal')
        self.last_slack = np.inf

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
            for _ in range(c.nmpc_emergency_sqp_iter if emergency else c.nmpc_sqp_iter):
                if not self._sane(X, U):   # never linearise a diverged iterate
                    status = 'failed'
                    self.stats['insane'] += 1
                    break
                code = sv.solve()
                iters += int(np.sum(sv.get_stats('qp_iter')))
                if code != 0:
                    qp = np.atleast_1d(sv.get_stats('qp_stat'))
                    status = QP_STATUS.get(int(qp[-1]) if qp.size else code, 'failed')
                    break
                X, U = sv.get_flat('x').reshape(N+1, NX), sv.get_flat('u').reshape(N, NU)
            X = sv.get_flat('x').reshape(N+1, NX)
            U = sv.get_flat('u').reshape(N, NU)
            if status == 'solved' and not self._sane(X, U):
                status = 'failed'
                self.stats['insane'] += 1
            if status == 'solved' and c.nmpc_verify:
                status, X, U, extra = self._verify(X, U)
                iters += extra
            if status == 'solved':
                self.stats['retry_solved'] += n > 0
                self.last_slack = self._max_slack()
                self.stats['violations'] += self.last_slack > 0.01
                break
            sv.reset()
        elapsed = perf_counter()-start
        if status != 'solved':
            return Plan(time, h, reference.z, reference.u, 'qp', status, iters, elapsed)
        z = np.column_stack((X[:, S]+s0, X[:, D], X[:, PSI], np.hypot(X[:, VX], X[:, VY])))
        z[0] = [s0, z0[1], z0[2], z0[3]]
        u = X[1:, [ACCEL, STEER]].copy()
        lat = X[:, [VY, R]].copy()
        lat[0] = vy0, r0
        Xa = X.copy(); Xa[:, S] += s0
        self.cache = [(time, z, Xa, U)]+self.cache[:7]
        return Plan(time, h, z, u, 'qp', status, iters, elapsed, lateral=lat)

    def _sane(self, X, U):
        """The iterate is finite and within physical bounds (SANE; steer within 1.5 steer_max, accel within 3x the
        braking and drive limits): safe to linearise."""
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(U))):
            return False
        p = self.plant.params
        if np.abs(X[:, STEER]).max() > 1.5*p.steer_max or np.abs(X[:, ACCEL]).max() > 3*max(p.brake_max, p.accel_max):
            return False
        return all(lo <= X[:, j].min() and X[:, j].max() <= hi for j, (lo, hi) in SANE.items())

    def _grip_ok(self, X, U):
        """The plan's true grip use is within the limit at stages 1..N (if the car starts past it, the first
        nmpc_recover_steps stages need only not exceed its starting use)."""
        c = self.cfg
        use = np.asarray(self.grip(X.T, np.vstack((U, U[-1:])).T)).max(axis=0)
        limit = np.full(len(use), c.nmpc_grip_limit+c.nmpc_grip_tol)
        if use[0] > limit[0]:
            limit[1:c.nmpc_recover_steps+1] = max(use[0], limit[0])
        return bool(np.all(use[1:] <= limit[1:]))

    def _verify(self, X, U):
        """More SQP iterations while the plan breaks the grip limit, within nmpc_verify_iter (and the wall-clock
        budget when set): (status, X, U, QP iterations)."""
        c, sv, N = self.cfg, self.solver, self.cfg.horizon
        start, n, iters = perf_counter(), 0, 0
        while not self._grip_ok(X, U):
            if n >= c.nmpc_verify_iter or (c.nmpc_verify_budget > 0 and perf_counter()-start > c.nmpc_verify_budget):
                self.stats['grip_rejected'] += 1
                return 'grip', X, U, iters
            code = sv.solve()   # X was checked sane by the caller or by the previous pass
            n += 1
            self.stats['verify_iter'] += 1
            iters += int(np.sum(sv.get_stats('qp_iter')))
            X, U = sv.get_flat('x').reshape(N+1, NX), sv.get_flat('u').reshape(N, NU)
            if code != 0 or not self._sane(X, U):   # couldn't be verified (and a diverged iterate isn't iterated on)
                self.stats['grip_rejected'] += 1
                return 'grip', X, U, iters
        return 'solved', X, U, iters

    def _max_slack(self):
        """Largest obstacle or road-edge slack (m) of the current solution over stages 1..N (one flat get each for the
        lower and upper slacks; stage 0 holds ng slacks, stages 1..N each 1 + ng + 2, the edge and obstacle slots first)."""
        sv, N = self.solver, self.cfg.horizon
        per = 1+self.ng+2
        idx = self.ng+np.arange(N)[:, None]*per+np.arange(1+self.H)[None, :]
        return float(max(sv.get_flat('sl')[idx].max(), sv.get_flat('su')[idx].max()))

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
        """Model rollout from x0 with rates that track the reference commands [a, steer] (rate-limited steer). The car
        stops at zero speed (the model alone would brake on into reverse, a guess that breaks vx >= 0)."""
        N, h, rate = self.cfg.horizon, self.cfg.step, self.plant.params.steer_rate
        X, U = np.zeros((N+1, NX)), np.zeros((N, NU))
        X[0] = x0
        for k in range(N):
            a, delta = reference.u[k]
            if X[k, VX] <= 0.1:   # stopped: no more braking
                a = max(a, 0.0)
            U[k] = [np.clip((delta-X[k, STEER])/h, -rate, rate), (a-X[k, ACCEL])/h]
            X[k+1] = np.asarray(self.rk4(X[k], U[k])).ravel()
            if X[k+1, VX] < 0:
                X[k+1, [VX, VY, R, ACCEL]] = 0.0
        return X, U

    def _slack_quad(self, soft):
        """Quadratic slack weights (Z) of one stage 1..N, in the order of _slack_weights: nmpc_obstacle_quad for the
        road edge and the hard rows' obstacle slots, `soft` for the soft rows' slots, 2 nmpc_w_slip for the slip rows."""
        c = self.cfg
        return np.r_[c.nmpc_obstacle_quad, np.full(self.H, c.nmpc_obstacle_quad), soft, np.full(2, 2*c.nmpc_w_slip)]

    def _slack_weights(self, soft):
        """Linear slack weights of one stage 1..N: road edge, the hard rows' obstacle slots, the soft rows' slots
        (`soft`), then the slip rows."""
        c = self.cfg
        return np.r_[c.nmpc_w_obstacle, np.full(self.H, c.nmpc_w_obstacle), soft, np.full(2, c.nmpc_w_slip)]

    def _obstacle_slots(self, zr, obstacles, target, s0, contact_only=False):
        """Map obstacle_constraints rows onto the fixed hard/soft slots of stages 1..N; contact_only drops the soft
        (headway, yield) rows."""
        N, H, ns, st = self.cfg.horizon, self.H, self.Ssoft, self.set
        C = np.zeros((N+1, self.ng, NX))
        lg, ug = np.full((N+1, self.ng), -INF), np.full((N+1, self.ng), INF)
        zl = np.tile(self._slack_weights(np.ones(ns)), (N+1, 1))   # edge, obstacle slots, slip
        Zl = np.tile(self._slack_quad(2*np.ones(ns)), (N+1, 1))
        if obstacles.keys:
            spec = obstacle_constraints(self.cfg, self.plant.params, self.road, zr, obstacles, target, s0)
            if contact_only:
                spec = spec.select(~spec.soft)
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
                    zl[k, 1+slot], Zl[k, 1+slot] = spec.slack_weight[idx], 2*spec.slack_weight[idx]
        for k in range(1, N+1):
            st.constraints(k, 'C', np.ravel(C[k], order='F'))   # column-major, as constraints_set(..., api='new')
            st.constraints(k, 'lg', lg[k])
            st.constraints(k, 'ug', ug[k])
            # One finite bound per soft row; the other side is masked (infinite), so both slacks share the weight.
            for name, val in (('zl', zl[k]), ('zu', zl[k]), ('Zl', Zl[k]), ('Zu', Zl[k])):
                st.cost(k, name, val)
