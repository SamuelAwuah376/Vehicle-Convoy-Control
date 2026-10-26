"""MP4 of a CARLA episode from a chase camera: an RGB camera rendered at `fps` (sensor_tick = 1/fps, real time), moved
every CARLA tick to sit low behind the team's rear car looking at the team (pulling back and up as the team spreads,
smoothed); frames are written in order with a HUD (caption, the frame's time, team score, each car's speed and lane),
H.264 at constant quality `crf`. Install it as session.recorder: CarlaWorld calls start(world) when an episode
begins, follow(xy) after every tick and step(world) after every sim step.
CARLA's standalone OpenDRIVE mode renders no road markings, so the recorder paints the lane lines and edges as
permanent debug lines (visible to the camera)."""
import math
import queue
import carla
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .opendrive import to_carla

MPH = 2.23694


def _font(size):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except OSError:
        return ImageFont.load_default()


def paint_lanes(world, road, dash=3.0, gap=9.0, z=0.03):
    """Lane lines (dashed) and road edges (solid) of our straight road as permanent white debug lines."""
    width = road.lanes*road.lane_width
    white = carla.Color(150, 150, 150)   # debug lines are unlit: dimmed to read as paint

    def line(x0, x1, d, thickness):
        (ax, ay, _), (bx, by, _) = to_carla(x0, d, 0.0), to_carla(x1, d, 0.0)
        world.debug.draw_line(carla.Location(float(ax), float(ay), z), carla.Location(float(bx), float(by), z),
                              thickness=thickness, color=white, life_time=0.0, persistent_lines=True)
    for d in (-width/2+0.2, width/2-0.2):
        for x0 in np.arange(0.0, road.length, 200.0):
            line(x0, min(x0+200.0, road.length), d, 0.12)
    for k in range(1, road.lanes):
        d = -width/2+k*road.lane_width
        for x0 in np.arange(0.0, road.length, dash+gap):
            line(x0, x0+dash, d, 0.1)


class Recorder:
    def __init__(self, session, path, road, fps=50, width=1920, height=1080, fov=70.0, smooth=0.04, crf=17):
        bp = session.blueprints.find('sensor.camera.rgb')
        for k, v in (('image_size_x', width), ('image_size_y', height), ('fov', fov), ('sensor_tick', 1/fps),
                     ('bloom_intensity', 0.0), ('lens_flare_intensity', 0.0), ('motion_blur_intensity', 0.0)):
            bp.set_attribute(k, str(v))
        self.world = session.world
        self.camera = session.world.spawn_actor(bp, carla.Transform(carla.Location(0.0, 0.0, 60.0)))
        self.frames = queue.Queue()
        self.camera.listen(self.frames.put)
        self.writer = imageio.get_writer(str(path), fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=8,
                                         output_params=['-crf', str(crf), '-preset', 'medium'])
        self.smooth, self.rig = smooth, None
        self.caption, self.score, self.written, self.t0 = '', None, 0, 0.0
        self.scale = height/720
        self.big, self.small = _font(round(26*self.scale)), _font(round(20*self.scale))
        self.hud = []
        paint_lanes(session.world, road)

    def follow(self, xy):
        """Aim from the team's CG positions xy (3, 2), our frame: the camera rig (distance behind the centroid, lateral
        position, height) is smoothed; the centroid itself is not, so the camera keeps pace exactly."""
        s, d = xy[:, 0], xy[:, 1]
        centre, span = float(np.mean(s)), float(np.max(s)-np.min(s))
        target = np.array([centre-float(np.min(s))+9.0+0.12*span, 0.4*float(np.mean(d)), 2.6+0.06*span])
        self.rig = target if self.rig is None else self.rig+self.smooth*(target-self.rig)
        back, lateral, height = self.rig
        x, y, _ = to_carla(centre-back, lateral, 0.0)
        pitch = -math.degrees(math.atan2(height-1.0, back))
        self.camera.set_transform(carla.Transform(carla.Location(float(x), float(y), float(height)),
                                                  carla.Rotation(pitch=pitch, yaw=0.0)))

    def _hud(self, world):
        snap, road = world.snapshot, world.road
        lines = ['' if self.score is None else f'team score {self.score:,.0f}']
        lines += [f'car {i+1}  {snap.team[i, 3]*MPH:5.1f} mph  lane {int(road.lane_of(snap.team_d[i]))+1}'
                  for i in range(len(snap.team))]
        return lines

    def _write(self, image):
        rgb = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, 2::-1]
        frame = Image.fromarray(np.ascontiguousarray(rgb))
        draw = ImageDraw.Draw(frame, 'RGBA')
        k = self.scale
        lines = [f'{image.timestamp-self.t0:5.1f} s   {self.hud[0]}'] + self.hud[1:] if self.hud else []
        right = 40*k+max([draw.textlength(self.caption, font=self.big)]+[draw.textlength(l, font=self.small) for l in lines])
        draw.rectangle((12*k, 12*k, right, (56+26*len(lines))*k), fill=(0, 0, 0, 140))
        draw.text((24*k, 20*k), self.caption, font=self.big, fill=(255, 255, 255))
        for n, line in enumerate(lines):
            draw.text((24*k, (56+26*n)*k), line, font=self.small, fill=(235, 235, 235))
        self.writer.append_data(np.asarray(frame))
        self.written += 1

    def _drain(self, timeout=None):
        while True:
            try:
                self._write(self.frames.get(timeout=timeout) if timeout else self.frames.get_nowait())
            except queue.Empty:
                return

    def start(self, world):
        """A new episode: drop frames rendered before it (settling, pre-roll), restart the clock, aim at its team."""
        while not self.frames.empty():
            self.frames.get_nowait()
        self.t0 = self.world.get_snapshot().timestamp.elapsed_seconds
        self.rig, self.hud = None, self._hud(world)
        self.follow(world.team[:, :2])

    def step(self, world):
        """After a sim step: write the frames rendered so far, then take this step's HUD for the next ones."""
        self._drain()
        self.hud = self._hud(world)

    def close(self, timeout=2.0):
        self.camera.stop()
        self._drain(timeout)
        self.camera.destroy()
        self.writer.close()
