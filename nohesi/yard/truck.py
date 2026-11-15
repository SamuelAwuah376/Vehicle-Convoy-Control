"""Rigid truck geometry and motion at the rear axle centre: state [x, y, yaw, v] (v signed, < 0 reversing), plus the
steering angle and the lagged acceleration the plant carries between steps."""
import numpy as np

DISKS = 4   # footprint cover: this many equal disks along the centreline


def corners(truck, states):
    """Footprint corners (M, 4, 2) of rear-axle states (M, >= 3), counter-clockwise from rear right."""
    states = np.atleast_2d(np.asarray(states, float))
    rear, front, half = -truck.rear_overhang, truck.length-truck.rear_overhang, truck.width/2
    local = np.array([[rear, -half], [front, -half], [front, half], [rear, half]])
    c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
    x = local[None, :, 0]*c[:, None]-local[None, :, 1]*s[:, None]+states[:, None, 0]
    y = local[None, :, 0]*s[:, None]+local[None, :, 1]*c[:, None]+states[:, None, 1]
    return np.stack((x, y), -1)


def disk_offsets(truck, n=DISKS):
    """(n,) offsets along the heading from the rear axle of the cover disks' centres, and their radius: n equal disks
    whose union contains the footprint rectangle."""
    seg = truck.length/n
    centres = -truck.rear_overhang+seg*(np.arange(n)+0.5)
    return centres, float(np.hypot(truck.width/2, seg/2))


def disk_centres(truck, states, n=DISKS):
    """(M, n, 2) cover-disk centres of rear-axle states (M, >= 3)."""
    states = np.atleast_2d(np.asarray(states, float))
    off, _ = disk_offsets(truck, n)
    return np.stack((states[:, None, 0]+off[None]*np.cos(states[:, None, 2]),
                     states[:, None, 1]+off[None]*np.sin(states[:, None, 2])), -1)


def body_centre(truck, states):
    states = np.atleast_2d(np.asarray(states, float))
    mid = truck.length/2-truck.rear_overhang
    return states[:, :2]+mid*np.stack((np.cos(states[:, 2]), np.sin(states[:, 2])), -1)


class TruckPlant:
    """Kinematic bicycle at the rear axle with signed speed, steering-rate and steering-angle limits, a first-order lag
    on the commanded acceleration (drivetrain and brakes) and a direction selector: in gear +1 the truck never rolls
    backward, in gear -1 never forward, in gear 0 (neutral, brakes held) it stops. A change of direction therefore needs
    a stop, as on a real truck. Low yard speeds keep the tyres far from their grip limit: no slip model is needed."""
    def __init__(self, truck, dt):
        self.truck, self.dt = truck, dt

    def step(self, state, steer, accel, steer_cmd, accel_cmd, gear):
        """One dt from (state [x, y, yaw, v], steer, lagged accel) under the commands (accel along +x of the truck,
        so reversing faster is a negative accel); returns (state, steer, accel)."""
        t, dt = self.truck, self.dt
        steer_cmd = float(np.clip(steer_cmd, -t.steer_max, t.steer_max))
        steer = steer+float(np.clip(steer_cmd-steer, -t.steer_rate*dt, t.steer_rate*dt))
        # Driving effort in the gear's direction is capped at accel_max, braking at brake_emergency.
        lo, hi = (-t.brake_emergency, t.accel_max) if gear >= 0 else (-t.accel_max, t.brake_emergency)
        accel_cmd = float(np.clip(accel_cmd, lo, hi))
        accel = accel+(accel_cmd-accel)*min(dt/t.accel_lag, 1.0)
        x, y, yaw, v = state
        v_new = v+accel*dt
        if gear > 0:
            v_new = max(v_new, 0.0)
        elif gear < 0:
            v_new = min(v_new, 0.0)
        else:
            v_new = max(abs(v)-t.brake_max*dt, 0.0)*np.sign(v)
        v_mid = 0.5*(v+v_new)
        yaw_rate = v_mid*np.tan(steer)/t.wheelbase
        yaw_mid = yaw+0.5*yaw_rate*dt
        return np.array([x+v_mid*np.cos(yaw_mid)*dt, y+v_mid*np.sin(yaw_mid)*dt, float(wrap(yaw+yaw_rate*dt)), v_new]), \
            steer, accel


def wrap(angle):
    return (np.asarray(angle)+np.pi) % (2*np.pi)-np.pi
