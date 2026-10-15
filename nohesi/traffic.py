import numpy as np


class Traffic:
    """Lane-keeping background cars with IDM car following, seeded spawning and recycling.

    Positions are rear-axle Frenet coordinates at lane centres. Each spawn gets a new id,
    so scoring can tell a recycled car from the one it replaced.
    """
    def __init__(self, cfg, road, rng, team_s, team_lane):
        self.cfg, self.road, self.rng = cfg, road, rng
        self.s = np.empty(0); self.lane = np.empty(0, int)
        self.v = np.empty(0); self.v0 = np.empty(0); self.id = np.empty(0, int)
        self.next_id = 0
        rear, front = np.min(team_s), np.max(team_s)
        for _ in range(cfg.count):
            for _ in range(1000):
                s = rng.uniform(rear-cfg.spawn_behind, front+cfg.spawn_ahead)
                lane = int(rng.integers(road.cfg.lanes))
                if lane == team_lane and rear-cfg.team_lane_clear_behind < s < front+cfg.team_lane_clear_ahead:
                    continue
                if self._free(s, lane):
                    self._add(s, lane)
                    break
            else:
                raise ValueError('could not place traffic; reduce count or widen spawn window')

    @property
    def d(self):
        return np.array([self.road.lane_center(l) for l in self.lane])

    def _free(self, s, lane, skip=None):
        same = (self.lane == lane)
        if skip is not None:
            same[skip] = False
        return not np.any(np.abs(self.s[same]-s) < self.cfg.min_gap)

    def _add(self, s, lane):
        v0 = self.rng.uniform(self.cfg.speed_min, self.cfg.speed_max)
        self.s = np.r_[self.s, s]; self.lane = np.r_[self.lane, lane]
        self.v = np.r_[self.v, v0]; self.v0 = np.r_[self.v0, v0]
        self.id = np.r_[self.id, self.next_id]; self.next_id += 1

    def idm(self, others_s, others_lane, others_v, length):
        """IDM acceleration; leaders include other traffic and the given (team) vehicles."""
        c = self.cfg
        all_s = np.r_[self.s, others_s]; all_lane = np.r_[self.lane, others_lane]; all_v = np.r_[self.v, others_v]
        accel = np.empty(len(self.s))
        for i in range(len(self.s)):
            ahead = (all_lane == self.lane[i]) & (all_s > self.s[i])
            free = 1-(self.v[i]/self.v0[i])**4
            if np.any(ahead):
                j = np.flatnonzero(ahead)[np.argmin(all_s[ahead])]
                gap = max(all_s[j]-self.s[i]-length, 0.1)
                desired = c.idm_gap+self.v[i]*c.idm_headway+self.v[i]*(self.v[i]-all_v[j])/(2*np.sqrt(c.idm_accel*c.idm_brake))
                free -= (max(desired, 0)/gap)**2
            accel[i] = np.clip(c.idm_accel*free, -c.idm_brake_max, c.idm_accel)
        return accel

    def step(self, dt, others_s, others_lane, others_v, length):
        a = self.idm(others_s, others_lane, others_v, length)
        stop = np.where(a < 0, np.minimum(dt, self.v/np.maximum(-a, 1e-9)), dt)
        self.s = self.s+np.maximum(self.v+0.5*a*stop, 0)*stop
        self.v = np.maximum(self.v+a*dt, 0)

    def recycle(self, team_s):
        """Respawn cars that fell behind the team at a free spot ahead (skipped if none is free)."""
        rear, front = np.min(team_s), np.max(team_s)
        for i in np.flatnonzero(self.s < rear-self.cfg.recycle_behind):
            s = front+self.cfg.respawn_ahead+self.rng.uniform(0, 50)
            lanes = [l for l in self.rng.permutation(self.road.cfg.lanes) if self._free(s, l, skip=i)]
            if lanes:
                v0 = self.rng.uniform(self.cfg.speed_min, self.cfg.speed_max)
                self.s[i], self.lane[i], self.v[i], self.v0[i] = s, lanes[0], v0, v0
                self.id[i] = self.next_id; self.next_id += 1

    def predict(self, horizon):
        """Constant-speed prediction of s at the given future times, shape (len(horizon), count)."""
        return self.s[None, :]+np.asarray(horizon)[:, None]*self.v[None, :]
