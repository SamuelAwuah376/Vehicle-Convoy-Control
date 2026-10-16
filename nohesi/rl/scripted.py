"""Hand-written policies: sanity baselines that drive the world without RL."""
import numpy as np
from convoy.controllers import pursuit, follower_acceleration
from ..intentions import encode
from ..world import TEAM, vehicle_ahead, vehicle_behind


def lane_keep_law(state, s, route, params, target_speed, lead=None, max_accel=np.inf):
    """Pure pursuit on a lane centre with convoy gap control; lead is (rear-axle gap, speed) or None."""
    a = 1.2*(target_speed-state[3])
    if lead is not None:
        gap, speed = lead
        a = min(a, follower_acceleration(state, [0, 0, 0, speed], gap-params.length, params))
    return np.array([min(a, max_accel), pursuit(state, s, route, params)])


def lane_keep_control(snap, road, params, i, lane, target_speed, max_accel=np.inf):
    """Lane keeping for teammate i; the lead includes traffic merging into `lane`."""
    from ..prediction import vehicle_ahead_incl_merging
    lead = vehicle_ahead_incl_merging(snap, road, snap.team_s[i], lane, skip=i)
    return lane_keep_law(snap.team[i], snap.team_s[i], road.lane_route(lane), params, target_speed, lead, max_accel)


class LaneKeeping:
    """Each teammate holds its lane at a target speed, following the vehicle ahead in that lane."""
    name = 'lane_keeping'

    def __init__(self, scenario, road, plant):
        self.road, self.params = road, plant.params
        self.target = scenario.team_speed
        self.lanes = [scenario.team_lane]*TEAM

    def __call__(self, snap):
        return np.array([lane_keep_control(snap, self.road, self.params, i, self.lanes[i], self.target)
                         for i in range(TEAM)])


class Overtaker:
    """Action policy for the centralized coordinator: when stuck behind a slower car, ask to move to
    the adjacent lane with clearly more room ahead, or from an outer lane straight to the far lane
    across a clear middle lane when that has the most room; otherwise steer target speed toward
    `desired`. Leans toward slower traffic alongside or just ahead in an adjacent lane for close passes."""
    name = 'overtaker'

    def __init__(self, road, params, desired=32.0, look=60.0, gain=8.0, rear_clear=6.0, lean_window=(-8.0, 25.0),
                 cross_window=(-8.0, 15.0)):
        self.road, self.params = road, params
        self.desired, self.look, self.gain, self.rear_clear = desired, look, gain, rear_clear
        self.lean_window, self.cross_window = lean_window, cross_window
        self.gap_check = None   # (i, lane) -> (gap, reason), bound by the coordinator each decision

    def bind_gap_check(self, check):
        self.gap_check = check

    def __call__(self, snap, intentions):
        actions = np.empty(TEAM, int)
        for i, it in enumerate(intentions):
            speed_cmd = int(np.sign(np.round(self.desired-it.speed)))
            lane_cmd = int(it.lane-it.origin) if it.phase != 'keep' else self.choose_lane(snap, i, it)
            actions[i] = encode(lane_cmd, speed_cmd, self.choose_lean(snap, i, it))
        return actions

    def choose_lean(self, snap, i, it):
        """Lean toward the nearest slower traffic car in an adjacent lane within the window (rear-axle
        offset), unless any vehicle is in that window on the opposite side."""
        lo, hi = self.lean_window
        s, v = snap.team_s[i], snap.team[i, 3]
        lanes = np.r_[self.road.lane_of(snap.traffic_d), self.road.lane_of(np.delete(snap.team_d, i))]
        ds = np.r_[snap.traffic_s, np.delete(snap.team_s, i)]-s
        near = (ds >= lo) & (ds <= hi)
        target = (np.arange(len(ds)) < len(snap.traffic_s)) & near & (np.r_[snap.traffic_v, np.full(TEAM-1, np.inf)] < v)
        best, best_ds = 0, np.inf
        for cmd in (1, -1):
            lane = it.lane+cmd
            if not 0 <= lane < self.road.cfg.lanes or np.any(near & (lanes == it.lane-cmd)):
                continue
            hits = target & (lanes == lane)
            if hits.any() and np.abs(ds[hits]).min() < best_ds:
                best, best_ds = cmd, np.abs(ds[hits]).min()
        return best

    def stuck(self, snap, i, it):
        """Slower traffic (not a teammate) close ahead in the current lane: (gap, speed) or None."""
        lead = vehicle_ahead(snap, self.road, snap.team_s[i], it.lane, skip=i, include_team=False)
        return lead if lead is not None and lead[0] < self.look and lead[1] < it.speed-1 else None

    def accepted(self, snap, i, lane, middle=None):
        """The intention layer's own gap rule when bound (same middle-lane corridor and gap checks); otherwise
        a simple window check of the middle lane for two-lane moves."""
        if self.gap_check is not None:
            return self.gap_check(i, lane)[0] is not None
        return middle is None or self.lane_clear(snap, i, middle)

    def lane_clear(self, snap, i, lane):
        """No other vehicle in `lane` within cross_window (rear-axle offset) of teammate i."""
        lo, hi = self.cross_window
        ds = np.r_[np.delete(snap.team_s, i), snap.traffic_s]-snap.team_s[i]
        lanes = self.road.lane_of(np.r_[np.delete(snap.team_d, i), snap.traffic_d])
        return not np.any((lanes == lane) & (ds >= lo) & (ds <= hi))

    def room(self, snap, i, lane):
        """Room ahead in `lane` (inf if open), or None if the rear is not clear."""
        s = snap.team_s[i]
        behind = vehicle_behind(snap, self.road, s, lane, skip=i)
        if behind is not None and behind[0] <= self.rear_clear:
            return None
        ahead = vehicle_ahead(snap, self.road, s-self.params.length, lane, skip=i)
        return np.inf if ahead is None else ahead[0]-self.params.length

    def choose_lane(self, snap, i, it):
        lead = self.stuck(snap, i, it)
        if lead is None:
            return 0
        lanes = self.road.cfg.lanes
        best, best_room = 0, lead[0]+self.gain
        for cmd in (1, -1):
            if 0 <= it.lane+cmd < lanes and self.accepted(snap, i, it.lane+cmd):
                room = self.room(snap, i, it.lane+cmd)
                if room is not None and room > best_room:
                    best, best_room = cmd, room
        cmd = 2 if it.lane == 0 else -2 if it.lane == lanes-1 else 0
        if cmd and 0 <= it.lane+cmd < lanes and self.accepted(snap, i, it.lane+cmd, middle=it.lane+cmd//2):
            middle = self.room(snap, i, it.lane+cmd//2)
            room = self.room(snap, i, it.lane+cmd)
            if room is not None and room > best_room and (middle is None or room > middle):
                best = cmd
        return best
