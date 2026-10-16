from dataclasses import asdict
import math
from types import SimpleNamespace
import numpy as np
from convoy.dynamics import wrap


def kinematic_step(state, acceleration, steering, previous_steering, cfg):
    """CG state [x, y, yaw, v]; bounded controls; midpoint integration (convoy.dynamics.step at the CG).

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
    cap (the tyres saturate). Fx = m a is split 50/50 between axles; each axle's lateral peak is
    sqrt((mu Fz)^2 - Fx_axle^2) on static loads (friction ellipse). Midpoint substeps of <= dynamic_substep.
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
        self.l_f = p.wheelbase-p.cg_to_rear_axle
        self.loads = (p.mass*G*p.cg_to_rear_axle/p.wheelbase, p.mass*G*self.l_f/p.wheelbase)

    def derivatives(self, s, a, delta):
        """d/dt of [x, y, yaw, vx, vy, r] (body velocities) at constant accel and steer."""
        p, l_f, l_r = self.params, self.l_f, self.params.cg_to_rear_axle
        _, _, yaw, vx, vy, r = s
        u, fx = max(vx, 0.5), 0.5*p.mass*a
        peak_f, peak_r = (math.sqrt(max((p.friction*fz)**2-fx**2, 0.0)) for fz in self.loads)
        fyf = fiala(delta-math.atan((vy+l_f*r)/u), p.cornering_stiffness_front, peak_f)
        fyr = fiala(-math.atan((vy-l_r*r)/u), p.cornering_stiffness_rear, peak_r)
        c, sn = math.cos(yaw), math.sin(yaw)
        return (vx*c-vy*sn, vx*sn+vy*c, r, a+r*vy-fyf*math.sin(delta)/p.mass,
                (fyf*math.cos(delta)+fyr)/p.mass-r*vx, (l_f*fyf*math.cos(delta)-l_r*fyr)/p.yaw_inertia)

    def step(self, state, acceleration, steering, previous_steering):
        p, sub = self.params, self._sub
        a = float(np.clip(min(acceleration, self.accel_limit(state[3])), -p.brake_max, p.accel_max))
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


def make_plant(vehicle, dt):
    return (DynamicPlant if vehicle.model == 'dynamic' else KinematicPlant)(vehicle, dt)
