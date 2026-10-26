"""CARLA world: World's interface (snapshot, step, collisions) with CARLA's physics moving the team cars.

CarlaSession connects once and loads our road (opendrive.xodr) in synchronous mode at the stability layer's rate
(plant.substep_dt, 100 Hz); calling it with (scenario, seed) starts an episode (a CarlaWorld), so it is the `world`
factory of EpisodeRun and NoHesiEnv. Per sim step a CarlaWorld runs dt/tick CARLA ticks; every tick, per team car:
read the state (our frame, at the CG), run the stability layer on the command (as DynamicPlant._step_inner does),
rate-limit the steer, then the low-level controller turns acceleration into throttle or brake (calibrated maps plus
an integral of the acceleration error) and steer into CARLA's [-1, 1] (calibrated wheel angle per unit command).
Traffic keeps our model (same seeds as World): stepped once per sim step from the pre-step snapshot like World,
its CARLA actors moved kinematically (no physics), interpolated over the ticks. Collisions are World's footprint
check, so episodes end exactly as in our simulator; CARLA's collision sensors log physical contacts in `contacts`.
The team cars are CARLA's Tesla Model 3 tuned to the scenario's car (vehicle.tune); the world's scenario carries our
model of that car (the scenario's vehicle with the Calibration's measured geometry, tyre stiffness and inertia), so
the planner's models and the stability layer match what CARLA drives.
"""
from dataclasses import replace
import math
import carla
import numpy as np
from ..world import TEAM, Collision, World
from .control import Longitudinal
from .opendrive import from_carla, to_carla, xodr
from .vehicle import tune

PREROLL = 0.5   # s the team runs up to its start states before an episode
TEAM_COLOR = '230,40,30'   # team cars' paint (traffic: each blueprint's recommended colours)
TRAFFIC_BLUEPRINTS = ('vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe_2020', 'vehicle.dodge.charger_2020',
                      'vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.nissan.micra')


def _reddish(color):
    """A paint ('r,g,b') a viewer could mistake for the team's."""
    r, g, b = (int(c) for c in color.split(','))
    return r > 140 and g < 110 and b < 110


def set_tick(world, tick, settings=None):
    """Synchronous mode with world.tick() advancing `tick` s. This CARLA build (0.9.16 client, UE4) runs more than one
    frame of fixed_delta_seconds per tick (two here), so the frame time is set from the frames a tick measurably
    advances; returns the frames per tick."""
    settings = settings or world.get_settings()
    settings.synchronous_mode = True
    for _ in range(3):   # the first ticks after a map load can be irregular: measure, check, retry
        settings.fixed_delta_seconds = tick
        world.apply_settings(settings)
        for _ in range(3):
            world.tick()
        first = world.tick()
        frames = max(1, round((world.tick()-first)))
        settings.fixed_delta_seconds = tick/frames
        world.apply_settings(settings)
        world.tick()
        start = world.get_snapshot().timestamp.elapsed_seconds
        for _ in range(4):
            world.tick()
        if math.isclose((world.get_snapshot().timestamp.elapsed_seconds-start)/4, tick, rel_tol=1e-3):
            return frames
    raise RuntimeError(f'CARLA tick is not {tick} s')


class CarlaSession:
    """One CARLA connection with our road loaded; session(scenario, seed) -> CarlaWorld for a new episode."""
    def __init__(self, road, calibration, host='localhost', port=2000, tick=0.01, render=True, timeout=60.0,
                 follow=True):
        self.cal, self.tick, self.follow = calibration, tick, follow
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.generate_opendrive_world(xodr(road), carla.OpendriveGenerationParameters(
            vertex_distance=2.0, max_road_length=50.0, wall_height=0.5, additional_width=0.0, smooth_junctions=True,
            enable_mesh_visibility=True))
        settings = self.world.get_settings()
        self.original = (settings.synchronous_mode, settings.no_rendering_mode, settings.fixed_delta_seconds)
        settings.no_rendering_mode = not render
        set_tick(self.world, tick, settings)
        self.blueprints = self.world.get_blueprint_library()
        self.episode = None
        self.recorder = None   # video.Recorder: CarlaWorld calls its start, follow (every tick) and step
        if render:
            self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def __call__(self, scenario, seed):
        if self.episode is not None:
            self.episode.destroy()
        self.episode = CarlaWorld(self, scenario, seed)
        return self.episode

    def close(self):
        if self.episode is not None:
            self.episode.destroy()
            self.episode = None
        settings = self.world.get_settings()
        settings.synchronous_mode, settings.no_rendering_mode, settings.fixed_delta_seconds = self.original
        self.world.apply_settings(settings)


class CarlaWorld(World):
    def __init__(self, session, scenario, seed):
        # Our initial team states and traffic from the same seed; the vehicle is our model of the tuned CARLA car.
        super().__init__(replace(scenario, vehicle=session.cal.apply(scenario.vehicle)), seed)
        self.session, self.cal, self.carla = session, session.cal, session.world
        self.ticks = int(round(scenario.timing.dt/session.tick))
        if not math.isclose(self.ticks*session.tick, scenario.timing.dt):
            raise ValueError('sim step must be a multiple of the CARLA tick')
        self.long = [Longitudinal(self.cal, session.tick) for _ in range(TEAM)]
        self.contacts = []                # (time, team car, other actor type id, impulse N s)
        self.upset = {}                   # team car -> 'gone' or 'upset' (_upset)
        bp = session.blueprints.find(self.cal.blueprint)
        if bp.has_attribute('color'):
            bp.set_attribute('color', TEAM_COLOR)
        self.actors, self.sensors, self.traffic_actors = [], [], []
        for i in range(TEAM):
            x, y, yaw, v = self.team[i]
            start = (x-PREROLL*v*math.cos(yaw), y-PREROLL*v*math.sin(yaw), yaw)
            actor = self.carla.spawn_actor(bp, self._transform(start, self.cal.x_cg, z=0.3))
            actor.apply_physics_control(tune(actor.get_physics_control(), self.scenario.vehicle,
                                             autobox=not self.cal.manual_gear))
            sensor = self.carla.spawn_actor(session.blueprints.find('sensor.other.collision'), carla.Transform(),
                                            attach_to=actor)
            sensor.listen(lambda e, i=i: self.contacts.append((self.time, i, e.other_actor.type_id,
                                                               e.normal_impulse.length())))
            self.actors.append(actor)
            self.sensors.append(sensor)
        rng = np.random.default_rng(seed)
        for k in range(len(self.traffic.s)):
            tbp = session.blueprints.find(TRAFFIC_BLUEPRINTS[rng.integers(len(TRAFFIC_BLUEPRINTS))])
            if tbp.has_attribute('color'):
                colors = [c for c in tbp.get_attribute('color').recommended_values if not _reddish(c)]
                tbp.set_attribute('color', colors[rng.integers(len(colors))])
            actor = self.carla.spawn_actor(tbp, self._traffic_transform(k, z=0.5))   # spawn check clear of the road
            actor.set_simulate_physics(False)
            actor.set_transform(self._traffic_transform(k))
            self.traffic_actors.append(actor)
        self._settle()
        if session.recorder is not None:
            session.recorder.start(self)

    # Frames: our CG state <-> the actor's transform.
    def _transform(self, state, x_offset, z=0.0):
        x, y, yaw = state[0]-x_offset*math.cos(state[2]), state[1]-x_offset*math.sin(state[2]), state[2]
        cx, cy, cyaw = to_carla(x, y, yaw)
        return carla.Transform(carla.Location(float(cx), float(cy), z), carla.Rotation(yaw=float(cyaw)))

    def _traffic_transform(self, k, s=None, d=None, heading=None, z=0.0):
        s = self.traffic.s[k] if s is None else s
        d = self.traffic.d[k] if d is None else d
        heading = self.traffic.heading[k] if heading is None else heading
        xy = self.road.to_xy(s, d)
        return self._transform((xy[0], xy[1], heading), 0.0, z)

    def _read(self, snapshot, i):
        """Car i's [x, y, yaw, vx, vy, r] at the CG, our frame."""
        a = snapshot.find(self.actors[i].id)
        tf, v, w = a.get_transform(), a.get_velocity(), a.get_angular_velocity()
        ox, oy, yaw = from_carla(tf.location.x, tf.location.y, tf.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        vx_w, vy_w, r = v.x, -v.y, -math.radians(w.z)
        vx, vy = c*vx_w+s*vy_w, -s*vx_w+c*vy_w
        vy += r*(self.cal.x_cg-self.cal.x_velocity)
        return np.array([ox+self.cal.x_cg*c, oy+self.cal.x_cg*s, float(yaw), vx, vy, r])

    def _settle(self):
        """Start the team rolling: the cars spawned PREROLL s at their start speed behind their start states settle
        on their suspension at rest, then run at that speed for PREROLL s (CARLA's constant-velocity mode, which
        also spins the wheels up) at the throttle floor, where the engine neither drives nor brakes, and arrive at the
        start; the episode starts from the states CARLA then reports. Traffic waits in
        place meanwhile."""
        for _ in range(40):
            self.carla.tick()
        for i, actor in enumerate(self.actors):
            actor.enable_constant_velocity(carla.Vector3D(float(self.team[i][3]), 0.0, 0.0))
            actor.apply_control(self._control(self.cal.floor_at(self.team[i][3]), 0.0, 0.0))
        for _ in range(int(round(PREROLL/self.session.tick))):
            self.carla.tick()
        for actor in self.actors:
            actor.disable_constant_velocity()
        self._read_all()
        self.snapshot = self._snapshot()

    def _control(self, throttle, brake, steer):
        if self.cal.manual_gear:
            return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer, manual_gear_shift=True, gear=1)
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)

    def _read_all(self):
        snap = self.carla.get_snapshot()
        for i in range(TEAM):
            if snap.find(self.actors[i].id) is None:   # gone (upset): keep its last state
                continue
            x, y, yaw, vx, vy, r = self._read(snap, i)
            self.team[i] = x, y, yaw, math.hypot(vx, vy)
            self.team_lat[i] = vy, r
        return snap

    def step(self, controls):
        """Apply (3, 2) [acceleration, steering] commands over dt; return the new snapshot and applied controls."""
        if self.snapshot.done:
            raise RuntimeError('episode is over')
        prev, dt, p = self.snapshot, self.scenario.timing.dt, self.plant.params
        # Traffic from the pre-step snapshot, as World does; interpolated over the ticks (recycled cars jump).
        s0, d0, h0, id0 = self.traffic.s.copy(), self.traffic.d.copy(), self.traffic.heading.copy(), self.traffic.id.copy()
        self.traffic.step(dt, prev.team_s, self.road.lane_of(prev.team_d), prev.team[:, 3], self.scenario.vehicle.length,
                          prev.team_d, self.scenario.vehicle.width)
        self.traffic.recycle(prev.team_s)
        s1, d1, h1 = self.traffic.s.copy(), self.traffic.d.copy(), self.traffic.heading.copy()
        jumped = self.traffic.id != id0
        applied = np.zeros((TEAM, 2))
        rate = p.steer_rate*self.session.tick
        snap = self.carla.get_snapshot()
        for n in range(self.ticks):
            t = self.time+n*self.session.tick
            batch = []
            for i in range(TEAM):
                state = self._read(snap, i)
                x, y, yaw, vx, vy, r = state
                speed = math.hypot(vx, vy)
                a_cmd, steer_cmd = float(controls[i][0]), float(controls[i][1])
                a_cmd = float(np.clip(min(a_cmd, self.plant.accel_limit(speed)), -p.brake_max, p.accel_max))
                if self.stability is not None:
                    a, target = self.stability.control(i, t, state, self.steer[i], a_cmd, steer_cmd)
                    self.stability.buffer.post(i, t, [x, y, yaw, speed, vy, r], self.steer[i], a, first=n == 0)
                else:
                    a, target = a_cmd, steer_cmd
                a = float(np.clip(min(a, self.plant.accel_limit(speed)), -p.brake_max, p.accel_max))
                self.steer[i] = float(np.clip(np.clip(target, self.steer[i]-rate, self.steer[i]+rate),
                                              -p.steer_max, p.steer_max))
                throttle, brake = self.long[i](a, vx)
                steer = float(np.clip(-self.steer[i]/self.cal.steer_gain_at(vx), -1.0, 1.0))
                batch.append(carla.command.ApplyVehicleControl(self.actors[i].id, self._control(throttle, brake, steer)))
                applied[i] += a/self.ticks, 0.0
            w = (n+1)/self.ticks
            for k, actor in enumerate(self.traffic_actors):
                f = 1.0 if jumped[k] else w
                batch.append(carla.command.ApplyTransform(actor.id, self._traffic_transform(
                    k, s0[k]+f*(s1[k]-s0[k]), d0[k]+f*(d1[k]-d0[k]), h0[k]+f*(h1[k]-h0[k]))))
            self.session.client.apply_batch(batch)
            self.carla.tick()
            snap = self.carla.get_snapshot()
            self.upset.update(self._upset(snap))
            if self.upset:
                break
            if self.session.recorder is not None:
                self.session.recorder.follow(np.array([self._read(snap, i)[:2] for i in range(TEAM)]))
        self._read_all()
        applied[:, 1] = self.steer
        if self.stability is not None:
            for i in range(TEAM):
                self.stability.buffer.post(i, self.time+dt, np.r_[self.team[i], self.team_lat[i]], self.steer[i],
                                           applied[i, 0])
        self.steps += 1
        self.time = self.steps*dt
        self.snapshot = self._snapshot()
        if self.session.follow:
            self._spectate()
        if self.session.recorder is not None:
            self.session.recorder.step(self)
        return self.snapshot, applied

    def _upset(self, snap):
        """Team cars CARLA's physics threw out of our model's reach in a tick: gone (fell out of the world), lifted or
        tipped. CARLA's bodies (2.16 m wide with mirrors) are wider than our 1.90 m footprint, so a pass our check
        allows can touch, and the kinematic traffic car then shoves the team car hard."""
        out = {}
        for i, actor in enumerate(self.actors):
            a = snap.find(actor.id)
            if a is None:
                out[i] = 'gone'
                continue
            tf = a.get_transform()
            if tf.location.z > 0.5 or abs(tf.rotation.roll) > 15.0 or abs(tf.rotation.pitch) > 15.0:
                out[i] = 'upset'
        return out

    def collisions(self, snap):
        """World's footprint check, plus team cars upset in CARLA (kind 'carla')."""
        upset = getattr(self, 'upset', {})
        return super().collisions(snap)+tuple(Collision(i, 'carla', -1) for i in sorted(upset))

    def _spectate(self):
        """Spectator camera behind and above the middle of the team."""
        c = self.team.mean(axis=0)
        x, y, _ = to_carla(c[0]-25.0, c[1], 0.0)
        self.carla.get_spectator().set_transform(carla.Transform(carla.Location(float(x), float(y), 12.0),
                                                                 carla.Rotation(pitch=-20.0)))

    def destroy(self):
        for sensor in self.sensors:
            sensor.stop()
        self.session.client.apply_batch([carla.command.DestroyActor(a.id)
                                         for a in self.sensors+self.actors+self.traffic_actors])
        self.carla.tick()
        self.sensors, self.actors, self.traffic_actors = [], [], []
