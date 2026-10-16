import numpy as np


class LaneRoute:
    """Lane centreline with the at(s) interface used by controllers.pursuit."""
    def __init__(self, road, lane):
        self.road, self.d, self.total = road, road.lane_center(lane), road.cfg.length

    def at(self, s):
        s = np.clip(s, 0, self.total)
        return self.road.to_xy(s, np.full_like(np.asarray(s, float), self.d)), self.road.heading(s)


class Road:
    """Straight road along +x; Frenet s = x, d = y. Curvature can replace these maps later."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.width = cfg.lanes*cfg.lane_width

    def lane_center(self, lane):
        return (lane+0.5)*self.cfg.lane_width - self.width/2

    def lane_of(self, d):
        lane = np.floor((np.asarray(d)+self.width/2)/self.cfg.lane_width)
        return np.clip(lane, 0, self.cfg.lanes-1).astype(int)

    def to_frenet(self, xy):
        xy = np.asarray(xy, float)
        return xy[..., 0].copy(), xy[..., 1].copy()

    def to_xy(self, s, d):
        return np.stack(np.broadcast_arrays(np.asarray(s, float), np.asarray(d, float)), axis=-1)

    def heading(self, s):
        return np.zeros_like(np.asarray(s, float))

    def lane_route(self, lane):
        return LaneRoute(self, lane)
