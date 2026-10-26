"""CARLA's Tesla Model 3 tuned to our car's figures (config.Vehicle, e.g. bmw_m3): mass, the launch and power
limits, brakes with our front/rear split, and a throttle without engine braking. CARLA's tyres, suspension and
all-wheel drive (its split is not exposed) stay its own: that is what the CARLA tests put our controller against.

The stock car (measured on a test road): 1845 kg, one gear (final ratio 9, 0.37 m wheels: 232.3 rpm per m/s, 15000
rpm at 64.6 m/s), 743 Nm flat to 9000 rpm (about 6 m/s^2), about 17.2 N of drive force per Nm of engine torque net
of the engine's damping (0.15 Nm s/rad at full throttle), 2.64 N of brake force per Nm of wheel brake torque (700 Nm a
wheel: about 4 m/s^2), and engine damping of 2.0 at zero throttle with the clutch engaged, so partial throttle brakes
the car at speed (zero throttle disengages the clutch: coasting). The automatic gearbox (one gear) disengages the
drivetrain at zero throttle and re-engages it only above ~0.25 with a jolt that can spin the car at speed, so the
tuned car drives in manual first gear (autobox=False).
"""
import math
import numpy as np

RPM_PER_MS = 232.3          # engine rpm per m/s (final ratio 9, wheel radius 0.37 m)
DRIVE_N_PER_NM = 17.25      # drive force (N) per Nm of engine torque, net of engine damping
BRAKE_N_PER_NM = 2.64       # brake force (N) per Nm of wheel brake torque, summed over the wheels
# CARLA tyre friction per unit of our mu: on a 100 m skidpad at 1.3 the tuned car holds ~10 m/s^2 and its rear lets
# go at ~10.5 m/s^2, as mu = 1.05 does (stock 3.5 still holds 16 m/s^2).
FRICTION_PER_MU = 1.3/1.05


def resistance(v):
    """Stock car's coasting deceleration (m/s^2) at speed v: rolling plus aerodynamic drag, measured."""
    return 0.065+1.74e-4*np.asarray(v, float)**2


def tune(pc, vehicle, headroom=1.15, max_speed=75.0, damping=0.05, tire_friction=None, autobox=False):
    """Modify CARLA PhysicsControl pc in place to our vehicle's figures and return it. The drive covers
    headroom x (our acceleration limit min(accel_max, accel_max power_speed / v) plus the car's resistance) up
    to max_speed; full brake reaches headroom x brake_max, split brake_bias_front to the front axle; the engine
    damping is `damping` at every throttle, so easing off barely brakes. Tyre friction: tire_friction, by default
    FRICTION_PER_MU x vehicle.friction (CARLA's scale is not mu)."""
    import carla
    pc.mass = vehicle.mass
    pc.use_gear_autobox = autobox
    pc.max_rpm = max_speed*RPM_PER_MS
    pc.damping_rate_full_throttle = damping
    pc.damping_rate_zero_throttle_clutch_engaged = damping
    pc.damping_rate_zero_throttle_clutch_disengaged = damping
    rpm = np.linspace(0.0, pc.max_rpm, 16)
    v = np.maximum(rpm/RPM_PER_MS, 0.1)
    accel = np.minimum(vehicle.accel_max, vehicle.accel_max*vehicle.power_speed/v)
    torque = headroom*vehicle.mass*(accel+resistance(v))/DRIVE_N_PER_NM+damping*rpm*2*math.pi/60
    pc.torque_curve = [carla.Vector2D(float(r), float(t)) for r, t in zip(rpm, torque)]
    total = headroom*vehicle.mass*vehicle.brake_max/BRAKE_N_PER_NM
    wheels = pc.wheels
    for k, w in enumerate(wheels):
        share = vehicle.brake_bias_front if k < 2 else 1-vehicle.brake_bias_front
        w.max_brake_torque = float(total*share/2)
        w.tire_friction = float(FRICTION_PER_MU*vehicle.friction if tire_friction is None else tire_friction)
    pc.wheels = wheels
    return pc
