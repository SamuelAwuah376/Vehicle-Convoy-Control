"""Render a finished episode (results/<run>/trajectory.npz, from run_carla.py or run_sim.py) in CARLA as a replay: CARLA
only draws it. Every car and the camera are placed for each output frame's time (interpolated from the 0.05 s
states) before the frame is drawn, so the video is smooth whatever CARLA's tick does. (Recording live, a world.tick()
runs one to three frames while rendering, and the camera and the traffic, placed per tick, jerked about one frame in
eight.) Physics is off for every car, so wheels do not turn.

Options:
  camera   'team': the live recorder's rig (behind and above the team, pulling back as it spreads); 'chase': low behind
           the rear team car, looking up the road; 'car<N>': low behind car N (1-3)
  scenery  trees, lamps and signs beside the road every `spacing` m (recycled around the camera) and a sidewalk on each
           side, so the roadside streams past
  fps, shutter  output frame rate; frames rendered per output frame and averaged (2: a half-frame motion blur)
  team_car, traffic_cars  blueprints (default: the calibrated car and CarlaWorld's traffic; 'detailed' traffic: the
           DETAILED_CARS other than the team's). Start the server with -quality-level=Epic for detailed models,
           anti-aliasing and shadows (Low draws coarse ones).
Run with .venv-carla's python and a CARLA server (CarlaUE4.exe -RenderOffScreen):
    python -m nohesi.carla.replay results/<run> [--camera chase] [--start 0 --end 300] [--out file.mp4]
"""
import argparse
import json
import math
import queue
from pathlib import Path
import carla
import imageio
import numpy as np
from PIL import Image, ImageDraw
from ..config import Road as RoadConfig
from ..road import Road
from .control import Calibration
from .opendrive import to_carla, xodr
from .video import MPH, _font, paint_lanes
from .world import TEAM_COLOR, TRAFFIC_BLUEPRINTS, _reddish

# CARLA's detailed (generation 2) cars: the replay draws only, so the team car need not be the calibrated one.
DETAILED_CARS = ('vehicle.dodge.charger_2020', 'vehicle.mercedes.coupe_2020', 'vehicle.lincoln.mkz_2020',
                 'vehicle.mini.cooper_s_2021', 'vehicle.nissan.patrol_2021', 'vehicle.ford.crown')
PROPS = ('aporosatree', 'gardenlamp', 'cypresstree', 'streetsign', 'aporosatree', 'gardenlamp', 'coconutpalm',
         'streetsign01', 'cypresstree', 'gardenlamp', 'aporosatree', 'advertisement')


class Trajectory:
    """States of a saved episode, interpolated at any time (s)."""
    def __init__(self, run):
        run = Path(run)
        z = np.load(run/'trajectory.npz')
        self.metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        self.t, self.team = z['time'], z['team']                   # (T,), (T, 3, 4): x, y, yaw, v at the CG
        self.team_d = z['team_d']
        self.ts, self.td, self.tid = z['traffic_s'], z['traffic_d'], z['traffic_id']
        self.yaw = np.unwrap(self.team[:, :, 2], axis=0)
        score = sum(z[k].sum(axis=1) for k in z.files if k.startswith('score_'))
        self.score = np.r_[0.0, np.cumsum(score)]                    # team score banked at each state
        self.dt = float(self.t[1]-self.t[0])

    def at(self, t):
        """(team (3, 4) [x, y, yaw, v], team_d (3,), traffic_s, traffic_d, traffic_heading (K,), score) at time t."""
        n = int(np.clip((t-self.t[0])//self.dt, 0, len(self.t)-2))
        w = float(np.clip((t-self.t[n])/self.dt, 0.0, 1.0))
        team = (1-w)*self.team[n]+w*self.team[n+1]
        team[:, 2] = (1-w)*self.yaw[n]+w*self.yaw[n+1]
        same = self.tid[n] == self.tid[n+1]                         # a recycled slot jumps: no blend
        ws = np.where(same, w, 0.0)
        s = (1-ws)*self.ts[n]+ws*self.ts[n+1]
        d = (1-ws)*self.td[n]+ws*self.td[n+1]
        heading = np.where(same, np.arctan2(self.td[n+1]-self.td[n], np.maximum(self.ts[n+1]-self.ts[n], 1e-3)), 0.0)
        return team, (1-w)*self.team_d[n]+w*self.team_d[n+1], s, d, heading, (1-w)*self.score[n]+w*self.score[n+1]


class Rig:
    """Camera placement from the team's CG positions (our frame), smoothed with time constant tau (s)."""
    def __init__(self, mode, tau=0.25):
        self.mode, self.tau, self.state = mode, tau, None

    def __call__(self, xy, dt):
        s, d = xy[:, 0], xy[:, 1]
        if self.mode == 'team':   # as video.Recorder.follow: behind the centroid, back and up as the team spreads
            centre, span = float(np.mean(s)), float(np.max(s)-np.min(s))
            target = np.array([centre-float(np.min(s))+9.0+0.12*span, 0.4*float(np.mean(d)), 2.6+0.06*span])
            base, look = centre, None
        else:
            i = int(np.argmin(s)) if self.mode == 'chase' else int(self.mode[3:])-1
            target = np.array([7.5, float(d[i]), 1.9])
            base, look = float(s[i]), 25.0
        a = 1.0 if self.state is None else 1.0-math.exp(-dt/self.tau)
        self.state = target if self.state is None else self.state+a*(target-self.state)
        back, lateral, height = self.state
        x, y, _ = to_carla(base-back, lateral, 0.0)
        pitch = -math.degrees(math.atan2(height-1.0, back if look is None else back+look))
        return carla.Transform(carla.Location(float(x), float(y), float(height)), carla.Rotation(pitch=pitch, yaw=0.0))


class Scenery:
    """Roadside props every `spacing` m on both sides, a pool recycled around the camera's s."""
    def __init__(self, world, road, spacing=22.0, behind=40.0, ahead=360.0, sidewalk=4.0):
        self.spacing, self.behind, self.n = spacing, behind, int(math.ceil((behind+ahead)/spacing))+1
        lib, edge = world.get_blueprint_library(), road.lanes*road.lane_width/2+sidewalk
        self.slots = []   # (actor, side, lateral offset, kind)
        for side in (-1, 1):
            for j in range(self.n):
                kind = PROPS[(j+(3 if side > 0 else 0)) % len(PROPS)]
                far = 9.0 if kind == 'advertisement' else 1.5+0.8*((j*7+side) % 4)
                actor = world.spawn_actor(lib.find(f'static.prop.{kind}'),
                                          carla.Transform(carla.Location(-500.0-10*j, 300.0*side, 0.0)))
                self.slots.append((actor, side, side*(edge+far), kind))

    def commands(self, s_cam):
        """ApplyTransform commands placing every prop slot at its station in the window around s_cam."""
        k0 = int(math.floor((s_cam-self.behind)/self.spacing))
        out = []
        for j, (actor, side, d, kind) in enumerate(self.slots):
            k = k0+((j % self.n)-k0) % self.n                    # the station in the window this slot holds
            s = k*self.spacing+(self.spacing/2 if side > 0 else 0.0)
            x, y, _ = to_carla(s, d, 0.0)
            yaw = -90.0*side if kind == 'advertisement' else 37.0*k % 360   # billboards face the road
            out.append(carla.command.ApplyTransform(actor.id, carla.Transform(carla.Location(float(x), float(y), 0.0),
                                                                                carla.Rotation(yaw=yaw))))
        return out

    def destroy(self, client):
        client.apply_batch([carla.command.DestroyActor(a.id) for a, *_ in self.slots])


def render(run, out=None, camera='chase', start=0.0, end=None, fps=60, shutter=2, scenery=True, width=1920,
           height=1080, fov=None, caption=None, host='localhost', port=2000, calibration='model3_m3', team_car=None,
           traffic_cars=None):
    traj = Trajectory(run)
    m = traj.metrics
    rc = m['scenario']['road']
    length = max(rc['length'], float(traj.team[:, :, 0].max())+600.0)
    road = Road(RoadConfig(lanes=rc['lanes'], lane_width=rc['lane_width'], length=length))
    cal = Calibration.load(calibration)
    end = float(traj.t[-1]) if end is None else min(end, float(traj.t[-1]))
    out = Path(out) if out else Path(run)/f'carla_replay_{camera}.mp4'
    client = carla.Client(host, port)
    client.set_timeout(120.0)
    sidewalk = 4.0 if scenery else 0.0
    world = client.generate_opendrive_world(xodr(road.cfg, sidewalk=sidewalk), carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=50.0, wall_height=0.0 if scenery else 0.5, additional_width=0.0,
        smooth_junctions=True, enable_mesh_visibility=True))
    settings = world.get_settings()
    original = (settings.synchronous_mode, settings.fixed_delta_seconds)
    settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, 1/(fps*shutter), False
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    lib = world.get_blueprint_library()
    actors, sensors = [], []
    try:
        bp = lib.find(team_car or cal.blueprint)
        if bp.has_attribute('color'):
            bp.set_attribute('color', TEAM_COLOR)
        team = [world.spawn_actor(bp, carla.Transform(carla.Location(-200.0-10*i, -200.0, 0.5))) for i in range(3)]
        rng = np.random.default_rng(int(m['seed']))   # the traffic's models and colours as CarlaWorld picks them
        models = tuple(c for c in DETAILED_CARS if c != (team_car or cal.blueprint)) if traffic_cars == 'detailed'             else (traffic_cars or TRAFFIC_BLUEPRINTS)
        traffic = []
        for k in range(traj.ts.shape[1]):
            tbp = lib.find(models[rng.integers(len(models))])
            if tbp.has_attribute('color'):
                colors = [c for c in tbp.get_attribute('color').recommended_values if not _reddish(c)]
                tbp.set_attribute('color', colors[rng.integers(len(colors))])
            traffic.append(world.spawn_actor(tbp, carla.Transform(carla.Location(-200.0-10*k, 200.0, 0.5))))
        actors = team+traffic
        for a in actors:
            a.set_simulate_physics(False)
        props = Scenery(world, road.cfg, sidewalk=sidewalk) if scenery else None
        paint_lanes(world, road.cfg)
        cam_bp = lib.find('sensor.camera.rgb')
        for k, v in (('image_size_x', width), ('image_size_y', height),
                     ('fov', fov or (70.0 if camera == 'team' else 90.0)), ('sensor_tick', 0.0),
                     ('motion_blur_intensity', 0.0), ('bloom_intensity', 0.0), ('lens_flare_intensity', 0.0)):
            cam_bp.set_attribute(k, str(v))
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(0.0, 0.0, 50.0)))
        sensors.append(cam)
        frames = queue.Queue()
        cam.listen(frames.put)
        rig = Rig(camera)
        writer = imageio.get_writer(str(out), fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=8,
                                    output_params=['-crf', '18', '-preset', 'medium'])
        big, small = _font(round(26*height/720)), _font(round(20*height/720))
        caption = caption or f"{m.get('policy', '')}  |  seed {m['seed']}  |  CARLA replay"
        sub = 1/(fps*shutter)
        n_out = int(round((end-start)*fps))
        self_check = []   # |camera x drawn - camera x placed| per rendered frame (0: each frame shows its placement)
        for f in range(n_out):
            acc = None
            for j in range(shutter):
                t = start+f/fps+(j-(shutter-1)/2)*sub
                state, team_d, s, d, heading, score = traj.at(t)
                batch = []
                for i, a in enumerate(team):
                    x, y, yaw = state[i, 0]-cal.x_cg*math.cos(state[i, 2]), state[i, 1]-cal.x_cg*math.sin(state[i, 2]), state[i, 2]
                    cx, cy, cyaw = to_carla(x, y, yaw)
                    batch.append(carla.command.ApplyTransform(a.id, carla.Transform(
                        carla.Location(float(cx), float(cy), 0.0), carla.Rotation(yaw=float(cyaw)))))
                xy = road.to_xy(s, d)
                for k, a in enumerate(traffic):
                    cx, cy, cyaw = to_carla(xy[k, 0], xy[k, 1], heading[k])
                    batch.append(carla.command.ApplyTransform(a.id, carla.Transform(
                        carla.Location(float(cx), float(cy), 0.0), carla.Rotation(yaw=float(cyaw)))))
                tf = rig(state[:, :2], sub if f or j else 0.0)
                batch.append(carla.command.ApplyTransform(cam.id, tf))
                if props is not None:
                    batch += props.commands(float(np.mean(state[:, 0])))
                client.apply_batch_sync(batch, False)
                frame = world.tick()
                image = frames.get(timeout=30.0)
                while image.frame < frame:                          # the frame drawn after this placement
                    image = frames.get(timeout=30.0)
                while not frames.empty():
                    frames.get_nowait()
                self_check.append(abs(image.transform.location.x-tf.location.x))
                rgb = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, 2::-1]
                acc = rgb.astype(np.float32) if acc is None else acc+rgb
            img = Image.fromarray((acc/shutter).astype(np.uint8))
            draw = ImageDraw.Draw(img, 'RGBA')
            k = height/720
            t = start+f/fps
            state, team_d, *_, score = traj.at(t)
            lines = [f'{t:5.1f} s   team score {score:,.0f}'] + \
                    [f'car {i+1}  {state[i, 3]*MPH:5.1f} mph  lane {int(road.lane_of(team_d[i]))+1}' for i in range(3)]
            right = 40*k+max([draw.textlength(caption, font=big)]+[draw.textlength(l, font=small) for l in lines])
            draw.rectangle((12*k, 12*k, right, (56+26*len(lines))*k), fill=(0, 0, 0, 140))
            draw.text((24*k, 20*k), caption, font=big, fill=(255, 255, 255))
            for n, line in enumerate(lines):
                draw.text((24*k, (56+26*n)*k), line, font=small, fill=(235, 235, 235))
            writer.append_data(np.asarray(img))
            if f % (fps*10) == 0:
                print(f'replay {t:6.1f} s of {end:.0f} s', flush=True)
        writer.close()
        err = np.array(self_check)
        print(f'camera drawn where placed: {np.mean(err < 0.01):.1%} of {len(err)} frames (max off {err.max():.2f} m)')
    finally:
        for c in sensors:
            c.stop()
        client.apply_batch([carla.command.DestroyActor(a.id) for a in actors+sensors])
        if scenery and 'props' in locals() and props is not None:
            props.destroy(client)
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = original
        world.apply_settings(settings)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--out')
    p.add_argument('--camera', default='chase')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--end', type=float)
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--shutter', type=int, default=2)
    p.add_argument('--no-scenery', action='store_true')
    p.add_argument('--fov', type=float)
    p.add_argument('--caption')
    p.add_argument('--team-car', help='blueprint, e.g. vehicle.dodge.charger_2020')
    p.add_argument('--traffic-cars', help="'detailed' or comma-separated blueprints")
    a = p.parse_args()
    traffic = a.traffic_cars if a.traffic_cars in (None, 'detailed') else tuple(a.traffic_cars.split(','))
    print(render(a.run, a.out, a.camera, a.start, a.end, a.fps, a.shutter, not a.no_scenery, fov=a.fov, caption=a.caption,
                 team_car=a.team_car, traffic_cars=traffic))


if __name__ == '__main__':
    main()
