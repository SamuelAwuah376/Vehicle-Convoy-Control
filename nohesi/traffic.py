import numpy as np


def _profile(tau):
    """Quintic smoothstep from 0 to 1 with zero slope and curvature at both ends, and its derivative."""
    tau = np.clip(tau, 0, 1)
    return tau**3*(10-15*tau+6*tau**2), 30*tau**2*(1-tau)**2


class _Groups:
    """Union-find over cars with the OR of their lane bits per group."""
    def __init__(self, bits):
        self.parent, self.cover = list(range(len(bits))), [int(b) for b in bits]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[a] = b
            self.cover[b] |= self.cover[a]


def _touching(bits):
    """(n, n) bool: cars sharing or occupying adjacent lanes (lane bitmasks)."""
    bits = np.asarray(bits, np.int64)
    return ((bits | bits << 1 | bits >> 1)[:, None] & bits[None, :]) != 0


class Traffic:
    """Background cars with IDM car following, optional MOBIL lane changes, seeded spawning and recycling.

    Positions are CG Frenet coordinates. `lane` is the lane a car is in (its origin lane while
    changing); a change moves d from lane centre to lane centre over cfg.lane_change_time, during which
    the car occupies both lanes. Each spawn gets a new id, so scoring can tell a recycled car from the
    one it replaced.
    """
    def __init__(self, cfg, road, rng, team_s, team_lane):
        self.cfg, self.road, self.rng = cfg, road, rng
        self.s = np.empty(0); self.lane = np.empty(0, int)
        self.v = np.empty(0); self.v0 = np.empty(0); self.id = np.empty(0, int)
        self.target = np.empty(0, int)      # lane being changed to (only meaningful while changing)
        self.lc_t = np.empty(0)             # time into the current change; nan when not changing
        self.lc_end = np.empty(0)           # time the last change finished (-inf: none yet)
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
        n = len(self.s)
        self.time = 0.0
        # MOBIL evaluation schedule per slot (kept across recycling); no rng use unless enabled.
        self.next_eval = rng.uniform(0, cfg.lane_change_interval, n) if cfg.lane_changes else np.full(n, np.inf)

    @property
    def changing(self):
        return ~np.isnan(self.lc_t)

    @property
    def target_lane(self):
        return np.where(self.changing, self.target, self.lane)

    def _progress(self):
        return _profile(np.nan_to_num(self.lc_t)/self.cfg.lane_change_time)

    @property
    def d(self):
        centre = np.array([self.road.lane_center(l) for l in self.lane], float)
        if not np.any(self.changing):
            return centre
        p, _ = self._progress()
        return centre+np.where(self.changing, (self.target-self.lane)*self.road.cfg.lane_width*p, 0)

    @property
    def vd(self):
        _, dp = self._progress()
        return np.where(self.changing, (self.target-self.lane)*self.road.cfg.lane_width*dp/self.cfg.lane_change_time, 0.)

    @property
    def heading(self):
        """Heading relative to the road while changing lanes (0 otherwise)."""
        return np.arctan2(self.vd, np.maximum(self.v, 1e-3))

    def _occupies(self, lane):
        """Traffic cars occupying a lane (in it, or changing into or out of it)."""
        return (self.lane == lane) | (self.changing & (self.target == lane))

    def _free(self, s, lane, skip=None):
        same = self._occupies(lane)
        if skip is not None:
            same[skip] = False
        if np.any(np.abs(self.s[same]-s) < self.cfg.min_gap):
            return False
        if self.cfg.no_blockades and self._seals(s, {lane}, skip):
            return False
        return not (self.cfg.no_walls and self._forms_wall(s, lane, skip))

    def _bits(self):
        """Lane bitmask per traffic car (its lane, plus its target while changing)."""
        return (1 << self.lane) | np.where(self.changing, 1 << self.target, 0)

    def _seals(self, s, lanes, skip=None):
        """Would a car at s occupying `lanes` (car `skip` left out) join a group covering every lane, connecting within
        blockade_gap + blockade_buffer?"""
        keep = np.arange(len(self.s)) != (-1 if skip is None else skip)
        pos = np.r_[self.s[keep], s]
        bits = np.r_[self._bits()[keep], sum(1 << int(l) for l in set(lanes))]
        near = (np.abs(pos[:, None]-pos[None, :]) < self.cfg.blockade_gap+self.cfg.blockade_buffer) & _touching(bits)
        groups = _Groups(bits)
        for a, b in zip(*np.nonzero(np.triu(near, 1))):
            groups.union(a, b)
        return groups.cover[groups.find(len(pos)-1)] == (1 << self.road.cfg.lanes)-1

    def blockade_leaders(self, reach=80.0):
        """(n,) virtual leader per car (-1: none) and its virtual bumper gap: car r holds blockade_gap behind car f
        ahead in a touching lane (within `reach` m of connecting) when, at the point where r would connect to f, its
        group and the groups of every touching car it would then be connected to cover every lane. Groups are the
        current connections (closer than blockade_gap) plus approaches within blockade_buffer of connecting, taken
        nearest first, so two approaches cannot complete a blockade in the same step."""
        c, n = self.cfg, len(self.s)
        lead, gap = np.full(n, -1), np.full(n, np.inf)
        if n < 2:
            return lead, gap
        bits = self._bits()
        full = (1 << self.road.cfg.lanes)-1
        ds = self.s[None, :]-self.s[:, None]       # ds[r, f] = s_f - s_r
        touching = _touching(bits)
        np.fill_diagonal(touching, False)
        groups = _Groups(bits)
        for a, b in zip(*np.nonzero(np.triu(touching & (np.abs(ds) < c.blockade_gap), 1))):
            groups.union(a, b)
        rr, ff = np.nonzero(touching & (ds >= c.blockade_gap) & (ds < c.blockade_gap+reach))
        for k in np.argsort(ds[rr, ff], kind='stable'):
            r, f = rr[k], ff[k]
            if groups.find(r) == groups.find(f):
                continue
            at = self.s[f]-c.blockade_gap           # where r would connect to f
            joined = np.flatnonzero(touching[r] & (np.abs(self.s-at) < c.blockade_gap))
            cover = groups.cover[groups.find(r)] | groups.cover[groups.find(f)]
            for j in joined:
                cover |= groups.cover[groups.find(j)]
            if cover == full:
                if ds[r, f]-c.blockade_gap < gap[r]:
                    lead[r], gap[r] = f, ds[r, f]-c.blockade_gap
            elif ds[r, f] < c.blockade_gap+c.blockade_buffer:
                groups.union(r, f)
        return lead, gap

    def blockades(self):
        """Current blockades: groups of car indices covering every lane, connected closer than blockade_gap."""
        bits = self._bits()
        groups = _Groups(bits)
        near = (np.abs(self.s[:, None]-self.s[None, :]) < self.cfg.blockade_gap) & _touching(bits)
        for a, b in zip(*np.nonzero(np.triu(near, 1))):
            groups.union(a, b)
        full, out = (1 << self.road.cfg.lanes)-1, {}
        for k in range(len(self.s)):
            out.setdefault(groups.find(k), []).append(k)
        return [cars for root, cars in out.items() if groups.cover[root] == full]

    def _lane_sets(self, skip=None):
        """Lanes each traffic car occupies (its lane, plus its target while changing)."""
        changing = self.changing
        return [set() if k == skip else ({int(self.lane[k]), int(self.target[k])} if changing[k] else {int(self.lane[k])})
                for k in range(len(self.s))]

    def _wall_in(self, cars, lanes_of, window=None):
        """A wall among `cars` (indices): distinct cars covering every lane within `window` (default wall_window).
        Returns one or None."""
        n_lanes, window = self.road.cfg.lanes, window or self.cfg.wall_window

        def cover(lane, chosen):
            if lane == n_lanes:
                return chosen
            for k in cars:
                if k in chosen or lane not in lanes_of(k):
                    continue
                pos = [self._pos(c) for c in chosen]+[self._pos(k)]
                if max(pos)-min(pos) < window:
                    found = cover(lane+1, chosen+[k])
                    if found:
                        return found
            return None
        return cover(0, [])

    def _forms_wall(self, s, lane, skip=None):
        """Would a car at (s, lane) complete a wall with the other traffic (car `skip` left out)?"""
        sets = self._lane_sets(skip)
        near = [k for k in range(len(self.s)) if sets[k] and abs(self.s[k]-s) < self.cfg.wall_window]
        self._probe = s
        lanes_of = lambda k: {lane} if k == -1 else sets[k]
        found = self._wall_in([-1]+near, lanes_of)
        self._probe = None
        return found is not None

    def _pos(self, k):
        return self._probe if k == -1 else self.s[k]

    def walls(self, window=None):
        """Current walls: lists of car indices (one per lane, within `window`, default wall_window)."""
        window = window or self.cfg.wall_window
        if len(self.s) < self.road.cfg.lanes:
            return []
        n_lanes, out, used = self.road.cfg.lanes, [], set()
        order = np.argsort(self.s)
        s_sorted = self.s[order]
        sets = None
        for a in order:
            if a in used:
                continue
            # Cars at 0 <= s - s_a < window, in sorted order (the same list a scan of `order` gives).
            lo, hi = np.searchsorted(s_sorted, self.s[a]), np.searchsorted(s_sorted, self.s[a]+window)
            near = [k for k in order[lo:hi] if k not in used]
            if len(near) < n_lanes:   # a wall needs a distinct car per lane
                continue
            sets = sets if sets is not None else self._lane_sets()
            found = self._wall_in(near, lambda k: sets[k], window)
            if found:
                out.append(found); used.update(found)
        return out

    def _add(self, s, lane):
        v0 = self.rng.uniform(self.cfg.speed_min, self.cfg.speed_max)
        self.s = np.r_[self.s, s]; self.lane = np.r_[self.lane, lane]
        self.v = np.r_[self.v, v0]; self.v0 = np.r_[self.v0, v0]
        self.target = np.r_[self.target, lane]; self.lc_t = np.r_[self.lc_t, np.nan]
        self.lc_end = np.r_[self.lc_end, -np.inf]
        self.id = np.r_[self.id, self.next_id]; self.next_id += 1

    def _acc(self, v, v0, lead_gap=None, lead_v=None):
        """IDM acceleration for speed v toward v0 behind a leader at bumper gap lead_gap (None: free road)."""
        c = self.cfg
        free = 1-(v/v0)**4
        if lead_gap is not None:
            desired = c.idm_gap+v*c.idm_headway+v*(v-lead_v)/(2*np.sqrt(c.idm_accel*c.idm_brake))
            free -= (max(desired, 0)/max(lead_gap, 0.1))**2
        return float(np.clip(c.idm_accel*free, -c.idm_brake_max, c.idm_accel))

    def idm(self, others_s, others_lane, others_v, length):
        """IDM acceleration; leaders include other traffic and the given (team) vehicles. A changing
        car leads in both lanes and follows the nearer leader of its two lanes."""
        c = self.cfg
        tgt = self.target_lane
        all_s = np.r_[self.s, others_s]; all_lane = np.r_[self.lane, others_lane]; all_v = np.r_[self.v, others_v]
        all_lane2 = np.r_[tgt, others_lane]
        # All cars at once: row i masks the vehicles leading car i (either of its lanes, ahead of it).
        mine, theirs = np.c_[self.lane, tgt], np.c_[all_lane, all_lane2]
        ahead = (mine[:, :, None, None] == theirs[None, None, :, :]).any(axis=(1, 3)) & (all_s[None, :] > self.s[:, None])
        v0 = self.v0
        if self.cfg.no_walls:   # break walls before they close: the rearmost car of a forming wall drops back
            v0 = self.v0.copy()
            for wall in self.walls(2*self.cfg.wall_window):
                rear = min(wall, key=lambda k: self.s[k])
                v0[rear] = max(min(v0[rear], self.v[rear]-self.cfg.wall_slowdown), 1.0)
        free = 1-(self.v/v0)**4
        has = ahead.any(axis=1)
        j = np.argmin(np.where(ahead, all_s[None, :], np.inf), axis=1)
        gap = np.maximum(all_s[j]-self.s-length, 0.1)
        desired = c.idm_gap+self.v*c.idm_headway+self.v*(self.v-all_v[j])/(2*np.sqrt(c.idm_accel*c.idm_brake))
        free = np.where(has, free-(np.maximum(desired, 0)/gap)**2, free)
        acc = np.clip(c.idm_accel*free, -c.idm_brake_max, c.idm_accel)
        if c.no_blockades:   # hold back from completing a blockade (IDM behind a virtual leader blockade_gap ahead)
            lead, vgap = self.blockade_leaders()
            held = lead >= 0
            if held.any():
                v, lv = self.v[held], self.v[lead[held]]
                desired = v*c.blockade_headway+v*(v-lv)/(2*np.sqrt(c.idm_accel*c.idm_brake))
                virtual = c.idm_accel*(1-(v/v0[held])**4-(np.maximum(desired, 0)/np.maximum(vgap[held], 0.1))**2)
                acc[held] = np.minimum(acc[held], np.clip(virtual, -c.idm_brake_max, c.idm_accel))
        return acc

    def _mobil(self, others_s, others_lane, others_v, length, others_lane2):
        """Start lane changes for cars due an evaluation (in index order, so earlier starts are seen)."""
        c, lanes = self.cfg, self.road.cfg.lanes
        n, m = len(self.s), len(others_s)
        all_s = np.r_[self.s, others_s]; all_v = np.r_[self.v, others_v]
        # Team cars use their own speed as desired speed (assumed to hold it).
        all_v0 = np.r_[self.v0, np.maximum(others_v, 0.1)]
        idx = np.arange(n+m)

        def occupants(lane):
            tgt = self.target_lane
            return np.r_[(self.lane == lane) | (tgt == lane), (others_lane == lane) | (others_lane2 == lane)]

        def leader(k, lane, skip):
            mask = occupants(lane) & (all_s > all_s[k]) & (idx != skip) & (idx != k)
            if not np.any(mask):
                return None
            return np.flatnonzero(mask)[np.argmin(all_s[mask])]

        def follower(s, lane, skip):
            mask = occupants(lane) & (all_s < s) & (idx != skip)
            if not np.any(mask):
                return None
            return np.flatnonzero(mask)[np.argmax(all_s[mask])]

        def acc(k, lead):
            if lead is None:
                return self._acc(all_v[k], all_v0[k])
            return self._acc(all_v[k], all_v0[k], all_s[lead]-all_s[k]-length, all_v[lead])

        for i in range(n):
            if self.time+1e-9 < self.next_eval[i]:
                continue
            while self.next_eval[i] <= self.time+1e-9:
                self.next_eval[i] += c.lane_change_interval
            if self.changing[i] or self.time-self.lc_end[i] < c.lane_change_cooldown:   # no back-to-back changes
                continue
            lane, s = self.lane[i], self.s[i]
            a_c = acc(i, leader(i, lane, i))
            fo = follower(s, lane, i)
            best, best_gain = None, c.mobil_threshold
            for new in (lane-1, lane+1):
                if not 0 <= new < lanes:
                    continue
                ln, fn = leader(i, new, i), follower(s, new, i)
                # No car alongside, or within lane_change_gap car lengths bumper to bumper, in the new lane.
                beside = occupants(new) & (np.abs(all_s-s) < length*(1+c.lane_change_gap)) & (idx != i)
                if np.any(beside):
                    continue
                if c.no_walls and self._forms_wall(s, new, skip=i):   # never close the last open lane
                    continue
                if c.no_blockades and self._seals(s, {lane, new}, skip=i):   # never seal the road
                    continue
                at_c = acc(i, ln)
                gain = at_c-a_c
                if fn is not None:
                    lead_n = leader(fn, new, -1)            # currently ahead of the new follower
                    a_n = acc(fn, lead_n)
                    at_n = self._acc(all_v[fn], all_v0[fn], s-all_s[fn]-length, self.v[i])
                    if at_n < -c.mobil_safe_brake:
                        continue
                    gain += c.mobil_politeness*(at_n-a_n)
                if fo is not None:
                    a_o = acc(fo, i)
                    at_o = acc(fo, leader(fo, lane, i))
                    gain += c.mobil_politeness*(at_o-a_o)
                gain += c.keep_right_bias*(1 if new < lane else -1)
                if gain > best_gain:
                    best, best_gain = new, gain
            if best is not None:
                if c.lane_change_prob >= 1 or self.rng.random() < c.lane_change_prob:
                    self.target[i], self.lc_t[i] = best, 0.0
                else:   # decided to stay: the cooldown applies as after a change
                    self.lc_end[i] = self.time

    def step(self, dt, others_s, others_lane, others_v, length, others_d=None, width=0.0):
        """Advance one step. With others_d, a team car whose footprint (width) crosses a lane line also
        occupies the neighbouring lane for MOBIL gap and safety checks (IDM still uses others_lane)."""
        others_s, others_lane, others_v = np.asarray(others_s, float), np.asarray(others_lane, int), np.asarray(others_v, float)
        if self.cfg.lane_changes:
            lane2 = others_lane
            if others_d is not None:
                off = np.asarray(others_d, float)-np.array([self.road.lane_center(l) for l in others_lane], float)
                cross = np.abs(off)+width/2 > self.road.cfg.lane_width/2
                lane2 = np.where(cross, np.clip(others_lane+np.sign(off).astype(int), 0, self.road.cfg.lanes-1), others_lane)
            self._mobil(others_s, others_lane, others_v, length, lane2)
        a = self.idm(others_s, others_lane, others_v, length)
        stop = np.where(a < 0, np.minimum(dt, self.v/np.maximum(-a, 1e-9)), dt)
        self.s = self.s+np.maximum(self.v+0.5*a*stop, 0)*stop
        self.v = np.maximum(self.v+a*dt, 0)
        self.time += dt
        if np.any(self.changing):
            self.lc_t = self.lc_t+dt
            done = self.changing & (self.lc_t >= self.cfg.lane_change_time-1e-9)
            self.lane = np.where(done, self.target, self.lane)
            self.lc_t[done] = np.nan
            self.lc_end[done] = self.time

    def recycle(self, team_s):
        """Respawn cars that fell behind the team at a free spot ahead (skipped if none is free)."""
        rear, front = np.min(team_s), np.max(team_s)
        for i in np.flatnonzero(self.s < rear-self.cfg.recycle_behind):
            s = front+self.cfg.respawn_ahead+self.rng.uniform(0, 50)
            lanes = [l for l in self.rng.permutation(self.road.cfg.lanes) if self._free(s, l, skip=i)]
            if lanes:
                v0 = self.rng.uniform(self.cfg.speed_min, self.cfg.speed_max)
                self.s[i], self.lane[i], self.v[i], self.v0[i] = s, lanes[0], v0, v0
                self.target[i], self.lc_t[i], self.lc_end[i] = lanes[0], np.nan, -np.inf
                self.id[i] = self.next_id; self.next_id += 1

    def predict(self, horizon):
        """Constant-speed prediction of s at the given future times, shape (len(horizon), count)."""
        return self.s[None, :]+np.asarray(horizon)[:, None]*self.v[None, :]
