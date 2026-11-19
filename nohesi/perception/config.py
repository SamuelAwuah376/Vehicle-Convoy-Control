"""Perception configuration: sensor specs, tracker settings and the Perception switch (design sections 4, 6, 11).

All angles in radians, distances in metres, times in seconds, rates in Hz. Defaults are the design's proposed suite.
"""
from dataclasses import dataclass, field
from math import radians

__all__ = ['Mount', 'LidarSpec', 'RadarSpec', 'CameraSpec', 'TrackerConfig', 'Perception', 'default_lidars',
           'default_radars', 'default_cameras']


@dataclass(frozen=True)
class Mount:
    """Sensor pose on the car. x forward and y left, in metres from the body centre, or with edge 'front' / 'rear'
    from that bumper (x > 0 is further forward either way); z height above the road; yaw the boresight relative to
    the car's heading (0 forward, pi rearward, +pi/2 left). geometry.sensor_pose resolves it for a car's state."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.6
    yaw: float = 0.0
    edge: str = ''      # '', 'front' or 'rear'

    def __post_init__(self):
        if self.edge not in ('', 'front', 'rear'):
            raise ValueError(f'unknown mount edge {self.edge!r}')


@dataclass(frozen=True)
class LidarSpec:
    """Planar scanning lidar (Tier A: 2D ray casting at mount height). Beams sweep the field of view from right
    (-fov/2) to left (+fov/2) once per 1/rate seconds, each fired at its own time (scan skew)."""
    name: str = 'lidar_front'
    mount: Mount = field(default_factory=lambda: Mount(x=-0.05, z=0.6, edge='front'))
    fov: float = radians(120.0)
    range_max: float = 200.0
    rate: float = 10.0
    beam_spacing: float = radians(0.1)
    range_sigma: float = 0.03
    dropout: float = 0.02          # probability a beam returns nothing
    far_dropout: float = 0.5       # extra dropout reached linearly at range_max (reflectivity fall-off)
    layers: int = 1                # points per hit (vertical layers repeat on a flat road with equal-height cars)
    latency: float = 0.05


@dataclass(frozen=True)
class RadarSpec:
    """Detection-level automotive radar. Field of view fov out to near_range, fov_far beyond it, up to range_max."""
    name: str = 'radar_front'
    mount: Mount = field(default_factory=lambda: Mount(x=-0.05, z=0.5, edge='front'))
    fov: float = radians(90.0)
    fov_far: float = radians(18.0)
    near_range: float = 70.0
    range_max: float = 250.0
    rate: float = 20.0
    sigma_range: float = 0.25
    sigma_azimuth: float = radians(0.3)
    sigma_range_rate: float = 0.1
    latency: float = 0.03
    rcs_dbsm: float = 10.0          # car radar cross-section
    pd_ref: float = 0.95            # detection probability of a fully visible car at ref_range
    ref_range: float = 150.0        # SNR falls as range^-4 beyond it
    occluded_pd: float = 0.3        # detection probability factor for a fully hidden car (multipath under/around)
    points_max: int = 3             # reflection points per car (1..points_max)
    false_alarms: float = 2.0       # Poisson mean per scan, uniform over the field of view
    res_range: float = 0.4          # points closer than both resolutions merge
    res_azimuth: float = radians(4.0)


@dataclass(frozen=True)
class CameraSpec:
    """Pinhole camera, detection level in Tier A (projected 2D boxes). Square pixels, principal point at the image
    centre, focal length f = (width_px/2)/tan(hfov/2)."""
    name: str = 'camera_front'
    mount: Mount = field(default_factory=lambda: Mount(x=2.0, z=1.4))
    hfov: float = radians(60.0)
    width_px: int = 1920
    height_px: int = 1080
    rate: float = 20.0
    sigma_px: float = 2.0
    latency: float = 0.08
    min_height_px: float = 15.0
    min_visible: float = 0.3
    pd: float = 0.95
    false_rate: float = 0.1         # false boxes per frame (Poisson)
    pitch_per_accel: float = radians(1.0)/4.9   # body pitch (nose down positive) per m/s^2 of braking (1 deg / 0.5 g)
    pitch_sigma: float = radians(0.1)           # residual pitch noise per frame


def default_lidars():
    return (LidarSpec(),
            LidarSpec(name='lidar_rear', mount=Mount(x=0.05, z=0.6, yaw=radians(180.0), edge='rear'), fov=radians(180.0),
                      range_max=120.0, beam_spacing=radians(0.2)))


def default_radars():
    side = dict(fov=radians(150.0), fov_far=radians(150.0), near_range=80.0, range_max=80.0, sigma_range=0.3,
                sigma_azimuth=radians(1.0), sigma_range_rate=0.15, ref_range=60.0)
    return (RadarSpec(),
            RadarSpec(name='radar_rear', mount=Mount(x=0.05, z=0.5, yaw=radians(180.0), edge='rear'), fov=radians(120.0),
                      fov_far=radians(120.0), near_range=150.0, range_max=150.0, sigma_azimuth=radians(0.5),
                      ref_range=100.0),
            RadarSpec(name='radar_left', mount=Mount(x=0.8, y=0.95, z=1.0, yaw=radians(90.0)), **side),
            RadarSpec(name='radar_right', mount=Mount(x=0.8, y=-0.95, z=1.0, yaw=radians(-90.0)), **side))


def default_cameras():
    side = dict(hfov=radians(100.0), width_px=1280, height_px=960)
    return (CameraSpec(),
            CameraSpec(name='camera_rear', mount=Mount(x=0.3, z=1.2, yaw=radians(180.0), edge='rear'), hfov=radians(90.0),
                       width_px=1280, height_px=960),
            CameraSpec(name='camera_left', mount=Mount(x=0.8, y=0.95, z=1.0, yaw=radians(90.0)), **side),
            CameraSpec(name='camera_right', mount=Mount(x=0.8, y=-0.95, z=1.0, yaw=radians(-90.0)), **side))


@dataclass(frozen=True)
class TrackerConfig:
    """Centralized IMM-UKF tracker (design section 6). Track state x = [s, d, v_s, v_d] of the CG in road coordinates."""
    q_long: float = 2.0             # m/s^2 white acceleration noise along the road
    q_lat: float = 0.5              # m/s^2 across the road
    lane_keep_rate: float = 1.0     # 1/s: lane-keep model pulls d to the lane centre (and v_d to 0) at this rate
    lc_speed: float = 1.0           # m/s lateral speed of the lane-change models
    imm_switch: float = 0.02        # probability per second of switching lane-keep -> each lane change (and back x10)
    gate: float = 0.99              # chi-square gate probability
    confirm_m: int = 3              # confirmed after confirm_m detections within confirm_n update cycles
    confirm_n: int = 5
    delete_exist: float = 0.1
    exist_survive: float = 0.995    # per-second survival probability of an existing car
    init_exist_lidar_dense: float = 0.5   # lidar box with >= dense_points points
    init_exist_lidar: float = 0.3
    init_exist_camera: float = 0.3
    init_exist_radar: float = 0.2
    dense_points: int = 10
    init_speed_sigma_lane: float = 5.0    # speed prior around the lane's tracked median
    init_speed_sigma_own: float = 10.0    # around the car's own speed when the lane has no tracks
    init_vd_sigma: float = 1.0
    radar_speed_min_cos: float = 0.3      # below this |cos(bearing)| a radar birth uses the speed prior
    merge_cycles: int = 5
    history: float = 0.3            # s of filter history kept to replay late measurements
    roi_ahead: float = 300.0        # tracks are kept within this window of the team (m along the road)
    roi_behind: float = 150.0
    imm_init: tuple = (0.9, 0.05, 0.05)   # lane keep, change left (+d), change right (-d)


@dataclass(frozen=True)
class Perception:
    """The perception switch. enabled False: the true snapshot passes through (every result reproduces).
    mode 'sensors': the full pipeline; 'ideal': the pipeline with noise-free, unlimited, occlusion-free sensors (E0);
    'shadow': the pipeline runs and is logged, but the planner and policy still read the true snapshot."""
    enabled: bool = False
    mode: str = 'sensors'                    # 'sensors', 'ideal' or 'shadow'
    lidars: tuple = field(default_factory=default_lidars)
    radars: tuple = field(default_factory=default_radars)
    cameras: tuple = field(default_factory=default_cameras)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    sharing: str = 'team'                    # 'team': one belief from all three cars; 'own': each car its own sensors
    v2v_latency: float = 0.1                 # added to every measurement's delivery to the fusion centre
    preroll: float = 1.0                     # s of synthetic constant-speed history before t = 0
    car_height: float = 1.45                 # every car's body height (camera size cue, 3D boxes)
    seed_tag: int = 0x5E05                   # perception RNG = default_rng([episode seed, seed_tag])

    def __post_init__(self):
        if self.mode not in ('sensors', 'ideal', 'shadow'):
            raise ValueError(f'unknown perception mode {self.mode!r}')
        if self.sharing not in ('team', 'own'):
            raise ValueError(f'unknown perception sharing {self.sharing!r}')
        names = [s.name for s in (*self.lidars, *self.radars, *self.cameras)]
        if len(set(names)) != len(names):
            raise ValueError('sensor names must be unique')
