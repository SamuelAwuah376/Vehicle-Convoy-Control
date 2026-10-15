"""Hand-written policies: sanity baselines that drive the world without RL or MPC."""
import numpy as np
from convoy.controllers import pursuit, follower_acceleration
from ..world import TEAM, vehicle_ahead


class LaneKeeping:
    """Each teammate holds its lane at a target speed, following the vehicle ahead in that lane."""
    name = 'lane_keeping'

    def __init__(self, scenario, road, plant):
        self.road, self.params = road, plant.params
        self.target = scenario.team_speed
        self.lanes = [scenario.team_lane]*TEAM
        self.routes = [road.lane_route(l) for l in self.lanes]

    def __call__(self, snap):
        controls = np.empty((TEAM, 2))
        for i in range(TEAM):
            state, s = snap.team[i], snap.team_s[i]
            a = 1.2*(self.target-state[3])
            lead = vehicle_ahead(snap, self.road, s, self.lanes[i], skip=i)
            if lead is not None:
                gap, speed = lead
                a = min(a, follower_acceleration(state, [0, 0, 0, speed], gap-self.params.length, self.params))
            controls[i] = a, pursuit(state, s, self.routes[i], self.params)
        return controls
