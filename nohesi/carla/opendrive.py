"""Our straight road (config.Road) as an OpenDRIVE 1.4 map for CARLA's standalone mode
(client.generate_opendrive_world), and the conversions between our frame and CARLA's.

Placement: the reference line runs along +x at y = width/2 (the left edge), and the lanes lie to its right (ids -1 ...
-lanes, driven in +s), so the OpenDRIVE frame equals ours: x = s, y = d, lane 0 (rightmost) = id -lanes. CARLA is
left-handed: it flips the OpenDRIVE y axis, so CARLA (x, y) = (s, -d), its yaw (degrees) is -yaw, and its yaw rate
(deg/s, about z up) is -r.
"""
import numpy as np


def xodr(road, name='nohesi_straight', speed_limit=None):
    """OpenDRIVE text for a straight road with road.lanes lanes of road.lane_width, road.length long; dashed lane
    lines, solid edges; speed_limit (m/s) sets every lane's limit, None leaves it unset."""
    width = road.lanes*road.lane_width
    mark = '<roadMark sOffset="0" type="{}" weight="standard" color="standard" width="0.15" laneChange="{}"/>'
    speed = '' if speed_limit is None else f'<speed sOffset="0" max="{speed_limit:.3f}" unit="m/s"/>'
    lanes = []
    for k in range(1, road.lanes+1):
        edge = k == road.lanes
        lanes.append(f'<lane id="{-k}" type="driving" level="false"><link/>'
                     f'<width sOffset="0" a="{road.lane_width:.4f}" b="0" c="0" d="0"/>'
                     + mark.format('solid' if edge else 'broken', 'none' if edge else 'both') + speed + '</lane>')
    return f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="{name}" version="1.0" north="0" south="0" east="0" west="0"/>
  <road name="{name}" length="{road.length:.4f}" id="1" junction="-1">
    <link/>
    <type s="0" type="motorway"/>
    <planView>
      <geometry s="0" x="0" y="{width/2:.4f}" hdg="0" length="{road.length:.4f}"><line/></geometry>
    </planView>
    <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"><link/>{mark.format('solid', 'none')}</lane></center>
        <right>
          {"""
          """.join(lanes)}
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
'''


def to_carla(x, y, yaw):
    """Our (x, y, yaw rad) to CARLA (x, y, yaw deg)."""
    return np.asarray(x, float), -np.asarray(y, float), -np.degrees(yaw)


def from_carla(x, y, yaw_deg):
    """CARLA (x, y, yaw deg) to our (x, y, yaw rad in [-pi, pi))."""
    yaw = -np.radians(yaw_deg)
    return np.asarray(x, float), -np.asarray(y, float), (yaw+np.pi) % (2*np.pi)-np.pi
