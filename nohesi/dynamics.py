from dataclasses import asdict
import math
from types import SimpleNamespace
import numpy as np
from .controllers import wrap


def kinematic_step(state, acceleration, steering, previous_steering, cfg):
    """CG state [x, y, yaw, v]; bounded controls; midpoint integration at the CG.

    beta = atan(l_r/L tan delta); x' = v cos(yaw+beta), y' = v sin(yaw+beta), yaw' = v cos(beta) tan(delta)/L.
    Velocity clips at zero without allowing artificial reverse travel.
    """
    a = float(np.clip(acceleration, -cfg.brake_max, cfg.accel_max))
    delta = float(np.clip(steering, previous_steering-cfg.steer_rate*cfg.dt,
                          previous_steering+cfg.steer_rate*cfg.dt))
    delta = float(np.clip(delta, -cfg.steer_max, cfg.steer_max))
    x, y, yaw, v = state
    beta = np.arctan(getattr(cfg, 'cg_to_rear_axle', cfg.wheelbase/2)/cfg.wheelbase*np.tan(delta))
    active_dt = min(cfg.dt, v/-a) if a < 0 else cfg.dt
    vm = max(0, v + 0.5*a*active_dt)
    yaw_rate = vm*np.cos(beta)*np.tan(delta)/cfg.wheelbase
    heading = yaw + 0.5*yaw_rate*active_dt + beta
    result = np.array([x+vm*np.cos(heading)*active_dt,
                       y+vm*np.sin(heading)*active_dt,
                       wrap(yaw+yaw_rate*active_dt), max(0, v+a*cfg.dt)])
    return result, a, delta


class KinematicPlant:
    """Stage A plant: kinematic bicycle at the centre of gravity plus speed-dependent drive and grip limits.

    State is [x, y, yaw, v] at the CG (cg_to_rear_axle = l_r ahead of the rear axle), v the CG speed. The grip
    cap clips the steering command before the rate limit, so lateral acceleration can briefly exceed it while
    steering unwinds after a speed increase.
    """
    def __init__(self, vehicle, dt):
        self.params = SimpleNamespace(**asdict(vehicle), dt=dt)

    # Limits accept scalars or arrays (the MPC evaluates them along the horizon).
    def accel_limit(self, speed):
        p = self.params
        return p.accel_max*np.minimum(1.0, p.power_speed/np.maximum(speed, 1e-6))

    def steer_limit(self, speed):
        p = self.params
        return np.minimum(p.steer_max, np.arctan(p.lateral_accel_max*p.wheelbase/np.maximum(speed, 1.0)**2))

    def step(self, state, acceleration, steering, previous_steering):
        a = min(acceleration, self.accel_limit(state[3]))
        cap = self.steer_limit(state[3])
        return kinematic_step(state, a, float(np.clip(steering, -cap, cap)), previous_steering, self.params)

    def predict_step(self, state, acceleration, steering, previous_steering):
        """Prediction model for rollouts (lane-keep fallback): this kinematic step, whatever the plant."""
        return KinematicPlant.step(self, state, acceleration, steering, previous_steering)


G = 9.81


def kinematic_body(v, delta, p):
    """Body-frame CG velocity and yaw rate (vx, vy, r) of the kinematic bicycle at speed v and steer delta."""
    beta = math.atan(p.cg_to_rear_axle/p.wheelbase*math.tan(delta))
    return v*math.cos(beta), v*math.sin(beta), v*math.cos(beta)*math.tan(delta)/p.wheelbase


def fiala(alpha, stiffness, peak):
    """Brush-model lateral force (N): cubic in tan(alpha), saturating at `peak` for |alpha| >= atan(3 peak/C)."""
    if peak <= 0:
        return 0.0
    z = math.tan(alpha)
    if abs(z) >= 3*peak/stiffness:
        return math.copysign(peak, alpha)
    return stiffness*z-stiffness**2*abs(z)*z/(3*peak)+stiffness**3*z**3/(27*peak**2)


class DynamicPlant(KinematicPlant):
    """Dynamic bicycle with Fiala tyres at the CG, blended into the kinematic model at low speed.

    step() takes [x, y, yaw, v] or [x, y, yaw, v, vy, r] (v = CG speed, vy and r body lateral speed and yaw
    rate) and returns the same length; a 4-vector starts from the kinematic vy, r at the previous steer.
    Controls: power-limited accel and brake as KinematicPlant, steering rate and steer_max limits but no grip
    cap (the tyres saturate). Longitudinal force (axle_forces): the axle loads shift by m a h / L (quasi-static load
    transfer, cg_height h), Fx = m a is split by drive_bias_front or brake_bias_front, and each axle's Fx is capped
    at its grip mu Fz, so the car can't brake or accelerate harder than its tyres allow. Each axle's lateral peak is
    sqrt((mu Fz)^2 - Fx_axle^2) (friction ellipse). Midpoint substeps of <= dynamic_substep.
    Between dynamic_blend_low and _high the substep result is blended linearly with a kinematic substep;
    below low it is purely kinematic (vy, r from the kinematic relations), which also stops without reversing.
    accel_limit/steer_limit/predict_step stay kinematic for the MPC and fallback rollouts.
    """
    def __init__(self, vehicle, dt):
        super().__init__(vehicle, dt)
        p = self.params
        self.substeps = max(1, math.ceil(dt/p.dynamic_substep-1e-9))
        self._sub = SimpleNamespace(**vars(p))
        self._sub.dt = dt/self.substeps
        self.substep_dt = self._sub.dt
        self.l_f = p.wheelbase-p.cg_to_rear_axle
        self.loads = (p.mass*G*p.cg_to_rear_axle/p.wheelbase, p.mass*G*self.l_f/p.wheelbase)

    def axle_forces(self, a):
        """(Fz front, Fz rear, Fx front, Fx rear, resulting acceleration) at commanded acceleration a: loads shifted
        by m a h / L (a clipped to +-mu g for the shift), Fx = m a split by brake_bias_front (a < 0) or
        drive_bias_front, each axle's Fx capped at +-mu Fz; the acceleration is a unless a cap binds."""
        p = self.params
        mu_g = p.friction*G
        shift = p.mass*min(max(a, -mu_g), mu_g)*p.cg_height/p.wheelbase
        fz_f, fz_r = self.loads[0]-shift, self.loads[1]+shift
        share = p.brake_bias_front if a < 0 else p.drive_bias_front
        fx_f, fx_r = share*p.mass*a, (1-share)*p.mass*a
        cap_f, cap_r = p.friction*fz_f, p.friction*fz_r
        cf, cr = min(max(fx_f, -cap_f), cap_f), min(max(fx_r, -cap_r), cap_r)
        return fz_f, fz_r, cf, cr, (a if (cf == fx_f and cr == fx_r) else (cf+cr)/p.mass)

    def derivatives(self, s, a, delta):
        """d/dt of [x, y, yaw, vx, vy, r] (body velocities) at constant accel and steer."""
        p, l_f, l_r = self.params, self.l_f, self.params.cg_to_rear_axle
        _, _, yaw, vx, vy, r = s
        u = max(vx, 0.5)
        fz_f, fz_r, fx_f, fx_r, ax = self.axle_forces(a)
        peak_f, peak_r = (math.sqrt(max((p.friction*fz)**2-fx**2, 0.0)) for fz, fx in ((fz_f, fx_f), (fz_r, fx_r)))
        fyf = fiala(delta-math.atan((vy+l_f*r)/u), p.cornering_stiffness_front, peak_f)
        fyr = fiala(-math.atan((vy-l_r*r)/u), p.cornering_stiffness_rear, peak_r)
        c, sn = math.cos(yaw), math.sin(yaw)
        return (vx*c-vy*sn, vx*sn+vy*c, r, ax+r*vy-fyf*math.sin(delta)/p.mass,
                (fyf*math.cos(delta)+fyr)/p.mass-r*vx, (l_f*fyf*math.cos(delta)-l_r*fyr)/p.yaw_inertia)

    def step(self, state, acceleration, steering, previous_steering, inner=None):
        """inner: optional callback (substep n, [x, y, yaw, vx, vy, r], current steer, commanded acceleration,
        commanded steer) -> (acceleration, steer target) run every substep (the stability layer); its steer target is
        then rate limited per substep. Returns (state, acceleration applied (substep mean with inner), steer)."""
        p, sub = self.params, self._sub
        a = float(np.clip(min(acceleration, self.accel_limit(state[3])), -p.brake_max, p.accel_max))
        if inner is not None:
            return self._step_inner(state, a, float(steering), float(previous_steering), inner)
        delta = float(np.clip(steering, previous_steering-p.steer_rate*p.dt, previous_steering+p.steer_rate*p.dt))
        delta = float(np.clip(delta, -p.steer_max, p.steer_max))
        x, y, yaw, v = (float(q) for q in state[:4])
        if len(state) > 4:
            vy, r = float(state[4]), float(state[5])
            vx = math.sqrt(max(v*v-vy*vy, 0.0))
        else:
            vx, vy, r = kinematic_body(v, previous_steering, p)
        h, span = sub.dt, p.dynamic_blend_high-p.dynamic_blend_low
        for _ in range(self.substeps):
            w = min(max((math.hypot(vx, vy)-p.dynamic_blend_low)/span, 0.0), 1.0)
            kin = kinematic_step((x, y, yaw, math.hypot(vx, vy)), a, delta, delta, sub)[0]
            kvx, kvy, kr = kinematic_body(kin[3], delta, p)
            if w == 0:
                x, y, yaw, vx, vy, r = kin[0], kin[1], kin[2], kvx, kvy, kr
                continue
            s = (x, y, yaw, vx, vy, r)
            k1 = self.derivatives(s, a, delta)
            k2 = self.derivatives(tuple(q+0.5*h*dq for q, dq in zip(s, k1)), a, delta)
            dx, dy, dyaw, dvx, dvy, dr = (q+h*dq for q, dq in zip(s, k2))
            dvx = max(dvx, 0.0)
            x, y = kin[0]+w*(dx-kin[0]), kin[1]+w*(dy-kin[1])
            yaw = float(wrap(kin[2]+w*wrap(dyaw-kin[2])))
            vx, vy, r = kvx+w*(dvx-kvx), kvy+w*(dvy-kvy), kr+w*(dr-kr)
        out = [x, y, yaw, math.hypot(vx, vy)] + ([vy, r] if len(state) > 4 else [])
        return np.array(out), a, delta

    def _step_inner(self, state, a_cmd, steer_cmd, steer, inner):
        """step() with the inner loop: per substep the callback sets acceleration and steer target."""
        p, sub = self.params, self._sub
        x, y, yaw, v = (float(q) for q in state[:4])
        if len(state) > 4:
            vy, r = float(state[4]), float(state[5])
            vx = math.sqrt(max(v*v-vy*vy, 0.0))
        else:
            vx, vy, r = kinematic_body(v, steer, p)
        h, span, rate = sub.dt, p.dynamic_blend_high-p.dynamic_blend_low, p.steer_rate*sub.dt
        applied = []
        for n in range(self.substeps):
            a, target = inner(n, (x, y, yaw, vx, vy, r), steer, a_cmd, steer_cmd)
            a = float(np.clip(min(a, self.accel_limit(math.hypot(vx, vy))), -p.brake_max, p.accel_max))
            steer = float(np.clip(np.clip(target, steer-rate, steer+rate), -p.steer_max, p.steer_max))
            applied.append(a)
            w = min(max((math.hypot(vx, vy)-p.dynamic_blend_low)/span, 0.0), 1.0)
            kin = kinematic_step((x, y, yaw, math.hypot(vx, vy)), a, steer, steer, sub)[0]
            kvx, kvy, kr = kinematic_body(kin[3], steer, p)
            if w == 0:
                x, y, yaw, vx, vy, r = kin[0], kin[1], kin[2], kvx, kvy, kr
                continue
            s = (x, y, yaw, vx, vy, r)
            k1 = self.derivatives(s, a, steer)
            k2 = self.derivatives(tuple(q+0.5*h*dq for q, dq in zip(s, k1)), a, steer)
            dx, dy, dyaw, dvx, dvy, dr = (q+h*dq for q, dq in zip(s, k2))
            dvx = max(dvx, 0.0)
            x, y = kin[0]+w*(dx-kin[0]), kin[1]+w*(dy-kin[1])
            yaw = float(wrap(kin[2]+w*wrap(dyaw-kin[2])))
            vx, vy, r = kvx+w*(dvx-kvx), kvy+w*(dvy-kvy), kr+w*(dr-kr)
        out = [x, y, yaw, math.hypot(vx, vy)] + ([vy, r] if len(state) > 4 else [])
        return np.array(out), float(np.mean(applied)), steer


def brake_limit(plant, grip_fraction):
    """Braking (m/s^2, positive) the planner may command: brake_max, and on the dynamic plant at most grip_fraction
    of mu g, so hard braking leaves the tyres grip to steer with."""
    p = plant.params
    if isinstance(plant, DynamicPlant):
        return min(p.brake_max, grip_fraction*p.friction*G)
    return p.brake_max


def make_plant(vehicle, dt):
    return (DynamicPlant if vehicle.model == 'dynamic' else KinematicPlant)(vehicle, dt)
