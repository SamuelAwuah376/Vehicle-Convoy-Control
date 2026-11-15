"""Yard path tracking (docs/yard_interfaces.md, `mpc.py`): a linear time-varying MPC on OSQP that tracks one
constant-direction segment of a Path at a time, forward or in reverse. It stops at every cusp, shifts and starts the
next segment, holds short of a stop_index and parks at the path's end. A pure-pursuit tracker (valid in both directions)
takes over when OSQP fails.

Model, per MPC step h = Planning.mpc_step, at the rear axle with signed speed:
    state  z = [x, y, yaw, v, steer, a]    a: the plant's lagged acceleration (first order, Truck.accel_lag)
    input  u = [a_cmd, steer_rate]          a_cmd held over h, steer_cmd = steer + steer_rate*h
The (v, a) part is the plant's lag discretised exactly over the Planning.dt sub-steps of h, so the distance D driven
over a step is linear in (v, a, a_cmd). Heading and position use the midpoint rule
    yaw+ = yaw + D tan(steer_mid)/L,   [x, y]+ = [x, y] + D [cos, sin](yaw + (yaw+ - yaw)/2),
linearised around the reference (x_r, y_r, yaw_r, v_r, steer_r). The reference is the current segment's samples
interpolated at arc lengths that advance from the truck's projection by the speed profile. steer_r is the path's
steering atan(L dyaw/ds) per sample interval (ds signed by the direction of motion) averaged over STEER_WINDOW m of path:
a step in the path's steering (Hybrid A* joins arcs of different curvature) becomes a ramp centred on it. The steering
takes 3.8 s from lock to lock, longer than the 3 s horizon, and at full lock an outward error cannot be corrected, so
the MPC also tracks this ramp (W_STEER) instead of turning late.

Speed profile (magnitude, signed by the segment direction): min(speed_limit, v_forward_max or v_reverse_max,
sqrt(lat_accel_max/|curvature|), the steering-rate cap below), decelerating at brake_comfort to 0 at the segment end (a
cusp or the path end) and HOLD_MARGIN short of stop_index; in time from the truck's speed it accelerates at
<= accel_max and decelerates at brake_comfort (harder, up to brake_max, only when a stop point is closer than the comfort
stopping distance). The steering-rate cap keeps the speed where the reference steering changes sharply (an S-bend, a
switch between full-lock arcs) low enough that the hydraulic steering (Truck.steer_rate) can follow within
STEER_WINDOW metres of path; it binds only on such bends.

QP: variables z_0..z_N, u_0..u_{N-1} and one slack per step on the speed bounds. Costs w_pos (x, y error), w_yaw
(yaw error; the yaw is unwrapped against the reference, so the error is wrapped), w_speed, w_accel and w_steer_rate
(deviations of a_cmd and steer_rate from the reference's). Hard bounds: |steer| <= steer_max, |steer_rate| <=
Truck.steer_rate, a_cmd in the gear's range ([-brake_max, accel_max] forward, [-accel_max, brake_max] reverse). Soft
bounds (slack, heavy weight): v in [0, v_forward_max] on forward segments, [-v_reverse_max, 0] on reverse ones; soft
because at a stop the lagged acceleration carries the linear model through zero speed where the plant's gear clamps it.
One OSQP instance is set up once with a fixed sparsity pattern; each call updates its data (q, l, u, A values).

A (nearly) stopped truck whose steering is off the reference where it stands (at a cusp after a lock-to-lock shift, at
the path's start) first steers in place: the reference waits at zero speed for the steering time. A truck standing at
(or past) the path's end outside park_tolerance reports 'off_path' (this path cannot park it: replan), as it does when
the lateral error exceeds track_tolerance.
"""
from math import atan, cos, hypot, sin, sqrt
from time import perf_counter
import numpy as np
import osqp
import scipy.sparse as sp
from .truck import wrap

STOPPED = 0.05          # m/s: standing still (a cusp shift, arrival)
CUSP_REACH = 0.4        # m: a cusp is reached within this of the segment's last sample
PARK_YAW = 0.1          # rad: arrival heading tolerance (Planning.park_tolerance is the position one)
HOLD_MARGIN = 0.3       # m: a hold at stop_index aims this far short of that sample
PROGRESS_WINDOW = 6.0   # m: progress is searched this far ahead of the previous progress
END_REACH = 2.0         # m: a cusp or the path end counts as reached only with progress this close to it
END_ALONG = 0.1         # m: standing still no more than this short of a segment's end (along its heading) is "at" it
ALIGN_SPEED = 0.1       # m/s: below this, a truck whose steering is off the reference steers before driving on
ALIGN_TOL = 0.05        # rad: ... until within this of the reference steering where it stands
ALIGN_SLACK = 1.0       # s: a standstill spends at most the full-lock-to-full-lock steering time plus this aligning
STEER_WINDOW = 3.0      # m of path over which the steering reference spreads a change of the path's steering
STEER_REACH = 3.0       # m: the steering-rate speed cap applies this far either side of such a change
W_STEER = 8.0           # cost weight on the steering angle's deviation from the (smoothed) reference steering
SPEED_MARGIN = 0.02     # m/s kept under v_forward_max / v_reverse_max (the accel lag overshoots slightly)
W_SLACK = (2e3, 2e2)    # speed-bound slack weights: quadratic, linear
SOLVED = (1, 2)         # OSQP status_val accepted: solved, solved inaccurate (if finite)
FALLBACK_STEPS = 3      # the fallback's speed command reaches the reference speed this many MPC steps ahead
NX, NU = 6, 2


class _Segment:
    """One constant-direction run of path samples i0..i1 (absolute arc length s), with the reference steering per
    interval and the static speed profile (direction limit, curvature, steering rate; brake_comfort to 0 at i1)."""
    def __init__(self, truck, path, i0, i1):
        self.i0, self.i1 = int(i0), int(i1)
        self.direction = d = int(path.direction[i0])
        self.s = path.s[i0:i1+1].copy()
        self.x, self.y = path.x[i0:i1+1].copy(), path.y[i0:i1+1].copy()
        self.yaw = np.unwrap(path.yaw[i0:i1+1])
        ds = np.diff(self.s)
        kappa = np.where(ds > 1e-6, d*np.diff(self.yaw)/np.maximum(ds, 1e-6), 0.0)   # dyaw/ds, ds = d*|ds|
        self.steer = np.clip(np.arctan(truck.wheelbase*kappa), -truck.steer_max, truck.steer_max)
        self.kappa_end = float(np.tan(self.steer[-1])/truck.wheelbase)
        self.steer_ref = self._smoothed_steer()
        ak = np.abs(kappa)
        k_sample = np.maximum(np.r_[ak[0], ak], np.r_[ak, ak[-1]])
        self.vmax = vmax = (truck.v_forward_max if d > 0 else truck.v_reverse_max)-SPEED_MARGIN
        cap = np.minimum(vmax, np.sqrt(truck.lat_accel_max/np.maximum(k_sample, 1e-9)))
        cap = np.minimum(cap, self._steer_rate_cap(truck))
        prof = cap.copy()
        prof[-1] = 0.0
        b = truck.brake_comfort
        for i in range(len(prof)-2, -1, -1):
            prof[i] = min(prof[i], sqrt(prof[i+1]**2+2*b*ds[i]))
        self.profile = prof

    def _smoothed_steer(self):
        """(n,) the interval steering averaged over STEER_WINDOW m of path centred on each sample (the window shrinks
        at the segment's ends): a steering reference that starts a change STEER_WINDOW/2 early and is rate-feasible
        at the steering-rate speed cap."""
        cum = np.r_[0.0, np.cumsum(self.steer*np.diff(self.s))]   # integral of the piecewise-constant steering
        a = np.clip(self.s-STEER_WINDOW/2, self.s[0], self.s[-1])
        b = np.clip(self.s+STEER_WINDOW/2, self.s[0], self.s[-1])
        width = b-a
        avg = (np.interp(b, self.s, cum)-np.interp(a, self.s, cum))/np.maximum(width, 1e-9)
        return np.where(width > 1e-9, avg, np.r_[self.steer, self.steer[-1:]])

    def _steer_rate_cap(self, truck):
        """(n,) speed at which the steering, at Truck.steer_rate, covers the path's steering change within
        STEER_WINDOW m of path (applied within STEER_REACH of the change)."""
        n, st = len(self.s), self.steer
        if n < 3:
            return np.full(n, np.inf)
        mid = 0.5*(self.s[:-1]+self.s[1:])                      # interval centres carry st
        lo = np.clip(np.searchsorted(mid, self.s-STEER_REACH, 'left'), 0, n-2)
        hi = np.clip(np.searchsorted(mid, self.s+STEER_REACH, 'right')-1, 0, n-2)
        hi = np.maximum(hi, lo)
        width = int(np.max(hi-lo))
        idx = np.minimum(lo[:, None]+np.arange(width+1)[None], hi[:, None])
        change = st[idx].max(1)-st[idx].min(1)
        return np.where(change > 1e-6, truck.steer_rate*STEER_WINDOW/np.maximum(change, 1e-6), np.inf)

    def at(self, s):
        """Interpolated (x, y, yaw) at arc lengths s (clamped to the segment)."""
        return np.interp(s, self.s, self.x), np.interp(s, self.s, self.y), np.interp(s, self.s, self.yaw)

    def steer_at(self, s):
        """Reference steering (smoothed) at arc lengths s."""
        return np.interp(s, self.s, self.steer_ref)

    def point_ahead(self, s):
        """(x, y) at arc length s, extended beyond the segment's end along its last arc (pure-pursuit target)."""
        if s <= self.s[-1]:
            return float(np.interp(s, self.s, self.x)), float(np.interp(s, self.s, self.y))
        sig, p0 = self.direction*(s-self.s[-1]), self.yaw[-1]         # signed arc length past the end
        k = self.kappa_end
        if abs(k*sig) < 1e-6:
            return self.x[-1]+sig*cos(p0), self.y[-1]+sig*sin(p0)
        return self.x[-1]+(sin(p0+k*sig)-sin(p0))/k, self.y[-1]-(cos(p0+k*sig)-cos(p0))/k


class PathTracker:
    """Tracks a Path (forward and reverse segments) with an LTV MPC; see the module docstring.

    Attributes: path, progress (sample index, monotone within a segment), segment (index into path.segments()),
    arrived, prediction ((N+1, 6) last QP state trajectory [x, y, yaw, v, steer, a] or None) and stats
    ({'solves', 'solve_time' (s, total), 'fallbacks'})."""
    def __init__(self, truck, planning):
        self.truck, self.planning = truck, planning
        self.N, self.h = int(planning.mpc_horizon), float(planning.mpc_step)
        if self.N < 2:
            raise ValueError('mpc_horizon must be at least 2')
        self.path, self.progress, self.segment, self.arrived = None, 0, 0, False
        self.prediction, self._aligning = None, 0.0
        self.stats = {'solves': 0, 'solve_time': 0.0, 'fallbacks': 0}
        self.last_error = None                              # repr of the last exception a solve raised
        self._segs = []
        self._lag = self._lag_model()
        self._qp = self._pattern()
        self._solver = None

    # ----------------------------------------------------------------------------------------------- path handling
    def set_path(self, path, progress=0):
        """New path (or a replan / a back-off retrace); progress = the sample the truck starts at."""
        self.path = path
        self._segs = [_Segment(self.truck, path, a, b) for a, b in path.segments()]
        self.progress = int(np.clip(progress, 0, len(path)-1))
        self.segment = path.segment_of(self.progress)
        self.arrived = False
        self.prediction, self._aligning = None, 0.0

    def _update_progress(self, seg, x, y):
        lo = max(self.progress, seg.i0)
        hi = int(np.searchsorted(self.path.s, self.path.s[lo]+PROGRESS_WINDOW, 'right'))-1
        hi = max(lo, min(seg.i1, hi))
        self.progress = max(self.progress, self.path.nearest((x, y), lo, hi+1))

    def _project(self, seg, x, y, yaw):
        """(s0, lateral error, heading error) of the rear axle against the segment polyline next to progress."""
        j, n, best = self.progress-seg.i0, len(seg.s), None
        for i in (j-1, j):
            if 0 <= i < n-1:
                ex, ey = seg.x[i+1]-seg.x[i], seg.y[i+1]-seg.y[i]
                l2 = ex*ex+ey*ey
                u = min(max(((x-seg.x[i])*ex+(y-seg.y[i])*ey)/l2, 0.0), 1.0) if l2 > 1e-12 else 0.0
                px, py = seg.x[i]+u*ex, seg.y[i]+u*ey
                d2 = (x-px)**2+(y-py)**2
                if best is None or d2 < best[0]:
                    best = (d2, i, u, px, py)
        _, i, u, px, py = best
        s0 = seg.s[i]+u*(seg.s[i+1]-seg.s[i])
        yaw_p = seg.yaw[i]+u*(seg.yaw[i+1]-seg.yaw[i])
        lateral = -sin(yaw_p)*(x-px)+cos(yaw_p)*(y-py)
        return float(s0), float(lateral), float(wrap(yaw-yaw_p))

    def _near_end(self, seg):
        return self.path.s[seg.i1]-self.path.s[self.progress] <= END_REACH

    def _reached_end(self, seg, x, y):
        """At the segment's last sample: within CUSP_REACH of it, or level with or past it along its heading (a
        stop beside it or an overshoot) within track_tolerance."""
        if not self._near_end(seg):
            return False
        dx, dy = x-seg.x[-1], y-seg.y[-1]
        along = seg.direction*(dx*cos(seg.yaw[-1])+dy*sin(seg.yaw[-1]))
        dist = hypot(dx, dy)
        return dist <= CUSP_REACH or (along > -END_ALONG and dist < self.planning.track_tolerance)

    # ---------------------------------------------------------------------------------------------------- command
    def command(self, state, steer, accel, speed_limit, stop_index=None):
        """(steer_cmd, accel_cmd, gear, info) for the truck at state [x, y, yaw, v] with the plant's steer and lagged
        accel. speed_limit >= 0 (m/s; 0: stop now at brake_comfort); stop_index: a sample the rear axle must not
        pass (None: no limit)."""
        if self.path is None:
            raise RuntimeError('PathTracker.command needs set_path first')
        t, pl = self.truck, self.planning
        x, y, yaw, v = (float(a) for a in state[:4])
        steer, accel = float(steer), float(accel)
        speed_limit = np.inf if speed_limit is None else max(float(speed_limit), 0.0)
        stop_index = None if stop_index is None else max(int(stop_index), 0)
        seg = self._segs[self.segment]
        last = self.segment == len(self._segs)-1
        if self.arrived:
            return self._hold(seg, steer, 'arrived', *self._project(seg, x, y, yaw)[1:])
        self._update_progress(seg, x, y)
        status = None
        if not last and not self._held(seg, stop_index) and abs(v) < STOPPED and self._reached_end(seg, x, y):
            self.segment += 1
            seg, last, status = self._segs[self.segment], self.segment == len(self._segs)-1, 'cusp'
            self.progress = seg.i0
            self._update_progress(seg, x, y)
        s0, lateral, heading = self._project(seg, x, y, yaw)
        if last and abs(v) < STOPPED and self._near_end(seg) and \
                hypot(x-seg.x[-1], y-seg.y[-1]) <= pl.park_tolerance and abs(wrap(yaw-seg.yaw[-1])) <= PARK_YAW:
            self.arrived = True
            return self._hold(seg, steer, 'arrived', lateral, heading)
        if abs(lateral) > pl.track_tolerance:
            return self._stop(seg, v, steer, lateral, heading)
        if last and abs(v) < STOPPED and speed_limit > 0 and not self._held(seg, stop_index) and \
                self._reached_end(seg, x, y):
            # Standing at (or past) the path's end outside park_tolerance: this path cannot park it; replan.
            return self._stop(seg, v, steer, lateral, heading)

        ref = self._reference(seg, s0, v, steer, speed_limit, stop_index)
        if status is None and ref['wait'] > 0 and self.segment > 0 and s0-seg.s[0] < CUSP_REACH:
            status = 'cusp'                                  # steering at the cusp before driving off
        if status is None:
            u0 = max(seg.direction*v, 0.0)
            near = ref['s_lim']-s0 <= u0*u0/(2*t.brake_comfort)+0.5
            if ref['hold'] and (near or speed_limit == 0.0):
                status = 'holding'
            elif not last and seg.s[-1]-s0 <= u0*u0/(2*t.brake_comfort)+0.5:
                status = 'cusp'
            else:
                status = 'tracking'
        z0 = np.array([x, y, yaw, v, steer, accel])
        sol = self._solve(seg, z0, ref)
        if sol is None:
            self.stats['fallbacks'] += 1
            steer_cmd, accel_cmd = self._pursuit(seg, s0, z0, ref)
            status = 'fallback'
        else:
            accel_cmd, rate = sol
            steer_cmd = float(np.clip(steer+rate*self.h, -t.steer_max, t.steer_max))
        return steer_cmd, accel_cmd, seg.direction, self._info(lateral, heading, status)

    @staticmethod
    def _held(seg, stop_index):
        """stop_index lies in (or before) this segment: the truck may not pass the segment's end (no shift)."""
        return stop_index is not None and stop_index <= seg.i1

    def _info(self, lateral, heading, status):
        return {'progress': self.progress, 'segment': self.segment, 'lateral_error': lateral,
                'heading_error': heading, 'status': status}

    def _hold(self, seg, steer, status, lateral, heading):
        """Standing still: brakes on (accel against the gear, which the plant clamps at zero speed), steering held."""
        return steer, -seg.direction*self.truck.brake_comfort, seg.direction, self._info(lateral, heading, status)

    def _stop(self, seg, v, steer, lateral, heading):
        """off_path: brake to a stop at brake_comfort, steering held."""
        sign = np.sign(v) if abs(v) > 1e-3 else seg.direction
        return steer, -float(sign)*self.truck.brake_comfort, seg.direction, self._info(lateral, heading, 'off_path')

    # -------------------------------------------------------------------------------------------------- reference
    def _reference(self, seg, s0, v, steer, speed_limit, stop_index):
        """Reference over the horizon from the truck's projection s0: arc lengths sr (N+1,), speed magnitudes ur,
        the stop point s_lim (segment end or HOLD_MARGIN short of stop_index) and the wait (s) at zero speed for the
        steering to come within ALIGN_TOL of the reference before a (nearly) stopped truck drives on."""
        t, N, h, d = self.truck, self.N, self.h, seg.direction
        b = t.brake_comfort
        s_lim, hold = float(seg.s[-1]), speed_limit == 0.0
        if self._held(seg, stop_index):
            s_stop = float(self.path.s[stop_index])-HOLD_MARGIN
            if s_stop < s_lim:
                s_lim, hold = s_stop, True
        u0 = max(d*v, 0.0)
        gap = s_lim-s0
        if gap <= 1e-3:
            dec = t.brake_max
        else:
            dec = min(max(b, u0*u0/(2*gap)), t.brake_max)
        s_end = max(s_lim, s0)
        static = seg.profile
        wait = 0.0
        if u0 < ALIGN_SPEED:
            wait = max(abs(float(seg.steer_at(s0))-steer)-ALIGN_TOL, 0.0)/t.steer_rate
            self._aligning = self._aligning+h if wait > 0 else 0.0
            if self._aligning > 2*t.steer_max/t.steer_rate+ALIGN_SLACK:
                wait = 0.0                                   # budget spent: drive on (no standstill deadlock)
        else:
            self._aligning = 0.0

        def target(s):
            return min(float(np.interp(s, seg.s, static)), speed_limit, sqrt(2*b*max(s_lim-s, 0.0)))
        sr, ur = np.empty(N+1), np.empty(N+1)
        sr[0], ur[0] = s0, u0
        for k in range(N):
            u = ur[k]
            if (k+1)*h <= wait:
                sr[k+1], ur[k+1] = sr[k], 0.0
                continue
            un = min(max(target(min(sr[k]+h*u, s_end)), u-dec*h), u+t.accel_max*h)
            un = max(un, 0.0)
            sn = sr[k]+0.5*h*(u+un)
            if sn >= s_end:
                sn, un = s_end, 0.0
            sr[k+1], ur[k+1] = sn, un
        return {'s': sr, 'u': ur, 's_lim': s_lim, 'hold': hold, 'wait': wait}

    # ---------------------------------------------------------------------------------------------------- the QP
    def _lag_model(self):
        """Exact (linear) response of the plant's lag over one MPC step of Planning.dt sub-steps: coefficients of
        (v, a, a_cmd) in v+, a+ and the distance D driven, and the steering midpoint weight."""
        t, h = self.truck, self.h
        dt = float(self.planning.dt)
        m = max(int(round(h/dt)), 1)
        dt = h/m
        lam = 1.0-min(dt/t.accel_lag, 1.0)

        def run(v, a, c, steps=1):
            dist = 0.0
            for _ in range(m*steps):
                a = lam*a+(1-lam)*c
                vn = v+dt*a
                dist += 0.5*dt*(v+vn)
                v = vn
            return v, a, dist
        _, _, dv = run(1.0, 0.0, 0.0)
        va, aa, da = run(0.0, 1.0, 0.0)
        vc, ac, dc = run(0.0, 0.0, 1.0)
        # The plant reaches a steering command within one sub-step unless rate-limited; at the rate limit the mean
        # steering over the step is steer + (m+1)/(2m) steer_rate*h.
        fa, fc = run(0.0, 1.0, 0.0, FALLBACK_STEPS)[0], run(0.0, 0.0, 1.0, FALLBACK_STEPS)[0]
        return dict(dv=dv, da=da, dc=dc, va=va, vc=vc, aa=aa, ac=ac, w=(m+1)/(2*m), fa=fa, fc=fc)

    def _pattern(self):
        """Fixed sparsity pattern (rows, cols, CSC order), constant values and the cost Hessian."""
        N, pl = self.N, self.planning
        nz, nu = NX*(N+1), NU*N
        n = nz+nu+N
        X = lambda k, j: NX*np.asarray(k)+j
        U = lambda k, j: nz+NU*np.asarray(k)+j
        S = lambda k: nz+nu+np.asarray(k)-1          # k = 1..N
        k, ks = np.arange(N), np.arange(1, N+1)
        rows, cols = [], []

        def add(r, c):
            r, c = np.broadcast_arrays(np.asarray(r), np.asarray(c))
            rows.append(r.ravel()); cols.append(c.ravel())
        add(np.arange(NX), X(0, np.arange(NX)))                               # initial state
        dyn = lambda j: (NX+NX*k+j)[:, None]
        add(dyn(0), np.stack([X(k+1, 0), X(k, 0), X(k, 3), X(k, 5), U(k, 0), X(k, 2), X(k, 4), U(k, 1)], 1))
        add(dyn(1), np.stack([X(k+1, 1), X(k, 1), X(k, 3), X(k, 5), U(k, 0), X(k, 2), X(k, 4), U(k, 1)], 1))
        add(dyn(2), np.stack([X(k+1, 2), X(k, 2), X(k, 3), X(k, 5), U(k, 0), X(k, 4), U(k, 1)], 1))
        add(dyn(3), np.stack([X(k+1, 3), X(k, 3), X(k, 5), U(k, 0)], 1))
        add(dyn(4), np.stack([X(k+1, 4), X(k, 4), U(k, 1)], 1))
        add(dyn(5), np.stack([X(k+1, 5), X(k, 5), U(k, 0)], 1))
        r = NX*(N+1)
        add(r+k, U(k, 0)); add(r+N+k, U(k, 1))                                # inputs
        add(r+2*N+k, X(ks, 4))                                                # steering angle
        add((r+3*N+k)[:, None], np.stack([X(ks, 3), S(ks)], 1))              # v + slack >= lo
        add((r+4*N+k)[:, None], np.stack([X(ks, 3), S(ks)], 1))              # v - slack <= hi
        add(r+5*N+k, S(ks))                                                   # slack >= 0
        m = r+6*N
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        lag = self._lag
        const = dict(
            init=np.ones(NX),
            v=np.tile([1.0, -1.0, -lag['va'], -lag['vc']], N),
            d=np.tile([1.0, -1.0, -self.h], N),
            a=np.tile([1.0, -lag['aa'], -lag['ac']], N),
            tail=np.r_[np.ones(3*N), np.tile([1.0, 1.0], N), np.tile([1.0, -1.0], N), np.ones(N)])
        perm = np.lexsort((rows, cols))
        diag = np.zeros(n)
        diag[X(ks, 0)] = diag[X(ks, 1)] = 2*pl.w_pos
        diag[X(ks, 2)] = 2*pl.w_yaw
        diag[X(ks, 3)] = 2*pl.w_speed
        diag[X(ks, 4)] = 2*W_STEER
        diag[U(k, 0)] = 2*pl.w_accel
        diag[U(k, 1)] = 2*pl.w_steer_rate
        diag[S(ks)] = 2*W_SLACK[0]
        nzd = np.flatnonzero(diag)
        P = sp.csc_matrix((diag[nzd], (nzd, nzd)), shape=(n, n))
        return dict(n=n, m=m, nz=nz, nu=nu, perm=perm, const=const, P=P, X=X, U=U, S=S,
                    A_idx=rows[perm], A_ptr=np.r_[0, np.cumsum(np.bincount(cols, minlength=n))])

    def _solve(self, seg, z0, ref):
        """Build and solve the QP; (accel_cmd, steer_rate) of the first step, or None (solver failure)."""
        start = perf_counter()
        try:
            out = self._solve_qp(seg, z0, ref)
        except (ValueError, TypeError, ArithmeticError, RuntimeError, osqp.OSQPException) as err:
            self.last_error, out = repr(err), None
        self.stats['solves'] += 1
        self.stats['solve_time'] += perf_counter()-start
        return out

    def _solve_qp(self, seg, z0, ref):
        t, pl, N, h, q_ = self.truck, self.planning, self.N, self.h, self._qp
        lag, d = self._lag, seg.direction
        X, U, S = q_['X'], q_['U'], q_['S']
        k, ks = np.arange(N), np.arange(1, N+1)
        sr, ur = ref['s'], ref['u']
        xr, yr, yawr = seg.at(sr)
        ox, oy = z0[0], z0[1]                               # positions relative to the truck (conditioning)
        xr, yr = xr-ox, yr-oy
        vr = d*ur
        ar = np.diff(vr)/h                                  # (N,) reference acceleration
        steer_r = seg.steer_at(sr)
        rate_r = np.clip(np.diff(steer_r)/h, -t.steer_rate, t.steer_rate)
        # Linearisation per step (around the reference).
        Dr = d*np.diff(sr)
        dm = steer_r[:-1]+lag['w']*h*rate_r
        G, Gp = np.tan(dm)/t.wheelbase, 1.0/(t.wheelbase*np.cos(dm)**2)
        th = yawr[:-1]+0.5*Dr*G
        c, s = np.cos(th), np.sin(th)
        Cd, Sd = c-0.5*Dr*s*G, s+0.5*Dr*c*G
        Q, Qp = 0.5*Dr**2*s*Gp, 0.5*Dr**2*c*Gp
        wh = lag['w']*h
        one, mone = np.ones(N), -np.ones(N)
        dx, da, dc = lag['dv'], lag['da'], lag['dc']
        Vx = np.stack([one, mone, -Cd*dx, -Cd*da, -Cd*dc, Dr*s, Q, Q*wh], 1)
        Vy = np.stack([one, mone, -Sd*dx, -Sd*da, -Sd*dc, -Dr*c, -Qp, -Qp*wh], 1)
        Vp = np.stack([one, mone, -G*dx, -G*da, -G*dc, -Dr*Gp, -Dr*Gp*wh], 1)
        C = q_['const']
        Av = np.concatenate([C['init'], Vx.ravel(), Vy.ravel(), Vp.ravel(), C['v'], C['d'], C['a'], C['tail']])
        rhs = np.zeros((N, NX))
        rhs[:, 0] = Dr*s*th+Q*dm
        rhs[:, 1] = -Dr*c*th-Qp*dm
        rhs[:, 2] = -Dr*Gp*dm
        # Initial state: yaw unwrapped against the reference, speed clamped to the gear's sign (the plant does).
        v0 = max(z0[3], 0.0) if d > 0 else min(z0[3], 0.0)
        init = np.array([0.0, 0.0, yawr[0]+float(wrap(z0[2]-yawr[0])), v0, z0[4], z0[5]])
        if d > 0:
            c_lo, c_hi, v_lo, v_hi = -t.brake_max, t.accel_max, 0.0, seg.vmax
        else:
            c_lo, c_hi, v_lo, v_hi = -t.accel_max, t.brake_max, -seg.vmax, 0.0
        inf = np.full(N, np.inf)
        lo = np.concatenate([init, rhs.ravel(), np.full(N, c_lo), np.full(N, -t.steer_rate), np.full(N, -t.steer_max),
                             np.full(N, v_lo), -inf, np.zeros(N)])
        hi = np.concatenate([init, rhs.ravel(), np.full(N, c_hi), np.full(N, t.steer_rate), np.full(N, t.steer_max),
                             inf, np.full(N, v_hi), inf])
        qv = np.zeros(q_['n'])
        qv[X(ks, 0)] = -2*pl.w_pos*xr[1:]
        qv[X(ks, 1)] = -2*pl.w_pos*yr[1:]
        qv[X(ks, 2)] = -2*pl.w_yaw*yawr[1:]
        qv[X(ks, 3)] = -2*pl.w_speed*vr[1:]
        qv[X(ks, 4)] = -2*W_STEER*steer_r[1:]
        qv[U(k, 0)] = -2*pl.w_accel*np.clip(ar, c_lo, c_hi)
        qv[U(k, 1)] = -2*pl.w_steer_rate*rate_r
        qv[S(ks)] = W_SLACK[1]
        Ax = Av[q_['perm']]
        if self._solver is None:
            self._solver = osqp.OSQP(algebra='builtin')
            A = sp.csc_matrix((Ax, q_['A_idx'], q_['A_ptr']), shape=(q_['m'], q_['n']))
            self._solver.setup(q_['P'], qv, A, lo, hi, verbose=False, max_iter=4000, eps_abs=1e-4, eps_rel=1e-4,
                               polishing=False, warm_starting=True)
            self._rho = self._solver.settings.rho
        else:
            self._solver.update(q=qv, l=lo, u=hi, Ax=Ax)
            self._solver.update_settings(rho=self._rho)
        zr = np.column_stack([xr, yr, yawr, vr, steer_r, np.r_[ar, ar[-1:]]])
        zr[0] = init
        self._solver.warm_start(x=np.r_[zr.ravel(), np.column_stack([ar, rate_r]).ravel(), np.zeros(N)])
        res = self._solver.solve(raise_error=False)
        sol = res.x
        if int(res.info.status_val) not in SOLVED or sol is None or not np.all(np.isfinite(sol)):
            self.prediction = None
            return None
        z = sol[:q_['nz']].reshape(N+1, NX).copy()
        z[:, 0] += ox
        z[:, 1] += oy
        self.prediction = z
        u0 = sol[U(0, 0)], sol[U(0, 1)]
        return float(np.clip(u0[0], c_lo, c_hi)), float(np.clip(u0[1], -t.steer_rate, t.steer_rate))

    # ------------------------------------------------------------------------------------------------- fallback
    def _pursuit(self, seg, s0, z0, ref):
        """Pure pursuit at the rear axle, either direction: the circle tangent to the truck's heading through the
        lookahead point has curvature 2 y_l/(x_l^2 + y_l^2) in the truck frame whether the point is ahead (forward)
        or behind (reverse: the lookahead is mirrored behind the rear axle), and the rear axle drives that circle at
        steer = atan(L*curvature) in both directions. Speed: the constant acceleration command that brings the
        plant (its lag included) to the reference speed FALLBACK_STEPS steps ahead; the reference starts from the
        truck's speed and position, so it carries the feedback of the profile."""
        t, h = self.truck, self.h
        x, y, yaw, v, steer, accel = z0
        look = float(np.clip(2.5+1.0*abs(v), 2.5, 6.0))
        px, py = seg.point_ahead(s0+look)
        dx, dy = px-x, py-y
        xl, yl = cos(yaw)*dx+sin(yaw)*dy, -sin(yaw)*dx+cos(yaw)*dy
        kappa = 2*yl/max(xl*xl+yl*yl, 1e-6)
        target = float(np.clip(atan(t.wheelbase*kappa), -t.steer_max, t.steer_max))
        steer_cmd = steer+float(np.clip(target-steer, -t.steer_rate*h, t.steer_rate*h))
        d, lag = seg.direction, self._lag
        target_v = d*ref['u'][min(FALLBACK_STEPS, len(ref['u'])-1)]
        accel_cmd = (target_v-v-lag['fa']*accel)/lag['fc']
        lo, hi = (-t.brake_max, t.accel_max) if d > 0 else (-t.accel_max, t.brake_max)
        return float(steer_cmd), float(np.clip(accel_cmd, lo, hi))
