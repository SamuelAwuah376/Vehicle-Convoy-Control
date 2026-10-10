"""Optional nominal leader steering MPC; followers retain baseline control."""
import numpy as np


class SteeringMPC:
    def __init__(self, cfg):
        try:
            import casadi as ca
        except ImportError as exc:
            raise RuntimeError('Install the mpc extra: pip install -e ".[mpc]"') from exc
        self.cfg, self.failures = cfg, 0
        self.n, self.h = 12, 0.15
        o = self.opti = ca.Opti()
        self.x = o.variable(3, self.n+1)
        self.u = o.variable(self.n)
        self.initial = o.parameter(3)
        self.speed = o.parameter()
        self.previous = o.parameter()
        self.reference = o.parameter(3, self.n+1)
        o.subject_to(self.x[:, 0] == self.initial)
        cost = 0
        for k in range(self.n):
            yaw = self.x[2, k]
            yaw_rate = self.speed*ca.tan(self.u[k])/cfg.wheelbase
            # Match midpoint plant integration at the prediction step.
            mid_yaw = yaw+0.5*self.h*yaw_rate
            nxt = self.x[:, k]+self.h*ca.vertcat(self.speed*ca.cos(mid_yaw),
                                               self.speed*ca.sin(mid_yaw), yaw_rate)
            o.subject_to(self.x[:, k+1] == nxt)
            prior = self.previous if k == 0 else self.u[k-1]
            allowed = cfg.steer_rate*(cfg.dt if k == 0 else self.h)
            o.subject_to(o.bounded(-allowed, self.u[k]-prior, allowed))
            error = self.x[:, k+1]-self.reference[:, k+1]
            cost += 8*ca.sumsqr(error[:2])+2*(1-ca.cos(error[2]))+0.1*self.u[k]**2+2*(self.u[k]-prior)**2
        o.subject_to(o.bounded(-cfg.steer_max, self.u, cfg.steer_max))
        o.minimize(cost)
        o.solver('ipopt', {'expand': True, 'print_time': False},
                 {'print_level': 0, 'sb': 'yes', 'max_iter': 80, 'tol': 1e-6})
        self.last = None

    def command(self, state, progress, route, previous, fallback):
        o = self.opti
        xy, yaw = route.at(progress+state[3]*self.h*np.arange(self.n+1))
        reference = np.vstack((xy.T, yaw))
        o.set_value(self.initial, state[:3]); o.set_value(self.speed, state[3])
        o.set_value(self.previous, previous); o.set_value(self.reference, reference)
        o.set_initial(self.x, reference)
        o.set_initial(self.u, self.last if self.last is not None else previous)
        try:
            solution = o.solve()
            values = np.asarray(solution.value(self.u)).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise RuntimeError('nonfinite MPC solution')
            self.last = np.r_[values[1:], values[-1]]
            return float(values[0])
        except RuntimeError:
            self.failures += 1
            return fallback
