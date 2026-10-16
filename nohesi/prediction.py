"""Traffic prediction shared by the team's planners: constant speed along the road, constant lateral speed until the
target lane centre is reached (lane-changing traffic), constant d otherwise."""
import numpy as np

ENABLED = True   # False: ignore traffic lateral motion (constant-d prediction), for comparisons


def traffic_lanes(snap, road):
    """(current lane, target lane) per traffic car; target equals current when not changing."""
    lane = road.lane_of(snap.traffic_d)
    target = getattr(snap, 'traffic_target_lane', None)
    if not ENABLED or target is None:
        return lane, lane
    return lane, np.asarray(target, int)


def traffic_prediction(snap, road, times):
    """Predicted (s, d, v, psi) of every traffic car at `times`, each shape (len(times), N).

    d moves at the current lateral speed and stops at the target lane centre; psi = atan2(vd, v) while moving
    laterally, 0 once arrived. With zero lateral speed this is the constant-d prediction."""
    t = np.asarray(times, float)[:, None]
    s0, d0, v = snap.traffic_s, snap.traffic_d, snap.traffic_v
    vd = getattr(snap, 'traffic_vd', None)
    vd = np.zeros_like(d0) if vd is None or not ENABLED else np.asarray(vd, float)
    centre = road.lane_center(traffic_lanes(snap, road)[1])
    moving = vd != 0
    arrive = np.where(moving, np.maximum((centre-d0)/np.where(moving, vd, 1.0), 0.0), 0.0)   # 0: away from target
    tt = np.minimum(t, arrive[None, :])
    psi = np.where(t < arrive[None, :], np.arctan2(vd, np.maximum(v, 1e-3))[None, :], 0.0)
    return s0[None, :]+v[None, :]*t, d0[None, :]+vd[None, :]*tt, np.repeat(v[None, :], len(t), axis=0), psi


def vehicle_ahead_incl_merging(snap, road, s, lane, skip, include_team=True):
    """world.vehicle_ahead, also counting traffic changing into `lane`: (CG gap, speed) or None."""
    current, target = traffic_lanes(snap, road)
    s_all = np.r_[snap.team_s, snap.traffic_s]
    in_lane = np.r_[road.lane_of(snap.team_d) == lane, (current == lane) | (target == lane)]
    in_lane[:len(snap.team_s)] &= include_team
    in_lane[skip] = False
    mask = in_lane & (s_all > s)
    if not np.any(mask):
        return None
    j = np.flatnonzero(mask)[np.argmin(s_all[mask])]
    return s_all[j]-s, np.r_[snap.team[:, 3], snap.traffic_v][j]
