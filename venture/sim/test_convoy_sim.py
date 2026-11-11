"""Regression tests for the claims the memo and video make about the haul-convoy simulation.

Run:  python -m pytest -q venture/sim/test_convoy_sim.py   (about 15 s; one simulation shared by all tests)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import convoy_sim as cs  # noqa: E402


@pytest.fixture(scope='module')
def run():
    site, frames, metrics = cs.run(None, quiet=True)
    return site, frames, metrics


def footprint(row):
    x, y, yaw = row[0], row[1], row[2]
    c, s = math.cos(yaw), math.sin(yaw)
    back, front, hw = -cs.REAR_OVERHANG, cs.LENGTH - cs.REAR_OVERHANG, cs.WIDTH / 2
    return np.array([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in ((back, -hw), (front, -hw), (front, hw), (back, hw))])


def edge_points(quad, n=14):
    return np.array([quad[i] + (quad[(i + 1) % 4] - quad[i]) * t for i in range(4) for t in np.linspace(0, 1, n)])


def separation(a, b):
    pa, pb = edge_points(a), edge_points(b)
    return float(np.min(np.linalg.norm(pa[:, None] - pb[None], axis=2)))


def test_route_geometry(run):
    site, _, metrics = run
    assert 8.0 <= metrics['ramp_grade_pct'] <= 10.0
    assert 1.0 / np.abs(site.road_kappa).max() >= 30.0          # haul-truck turning radius


def test_followers_track_the_leaders_line(run):
    _, _, m = run
    for convoy in ('loaded', 'empty'):
        for name, s in m[convoy].items():
            assert s['lat_rms_cm'] < 3.0, (convoy, name, s)
            assert s['lat_max_cm'] < 25.0, (convoy, name, s)
            assert s['gap_min_m'] > 15.0, (convoy, name, s)


def test_light_vehicle_hold(run):
    _, frames, m = run
    det = m['detection']
    assert det['truck'] == 'AUTO 1' and det['path_dist_at_detect'] > 30.0
    assert m['closest_approach_to_light_vehicle_m'] > 12.0
    events = [e for _, e in m['hud_events']]
    assert any('CONVOY HOLD' in e for e in events) and any('RESUME' in e for e in events)
    # while the pickup stands in the lane, AUTO 1 keeps reporting it IN PATH
    in_lane = [f for f in frames if f['lv'][4] == 0.0 and abs(f['lv'][1] - (128.0 - cs.LANE_OFF)) < 3.0 and f['c'][0]['state'] == 'HOLD']
    assert in_lane and all(f['c'][0]['trucks'][1][13] == 2 for f in in_lane)


def test_no_contact_between_any_vehicles(run):
    _, frames, _ = run
    worst_same, worst_cross, worst_lv = 1e9, 1e9, 1e9
    for fr in frames[::3]:
        trucks = [(ci, footprint(r)) for ci, c in enumerate(fr['c']) if c['active'] for r in c['trucks']]
        lv = fr['lv']
        lvq = np.array([[lv[0] - 1.05, lv[1] - 2.8], [lv[0] + 1.05, lv[1] - 2.8], [lv[0] + 1.05, lv[1] + 2.8], [lv[0] - 1.05, lv[1] + 2.8]])
        for i, (ci, a) in enumerate(trucks):
            if np.linalg.norm(a.mean(0) - lvq.mean(0)) < 40:
                worst_lv = min(worst_lv, separation(a, lvq))
            for cj, b in trucks[i + 1:]:
                if np.linalg.norm(a.mean(0) - b.mean(0)) > 30:
                    continue
                d = separation(a, b)
                if ci == cj:
                    worst_same = min(worst_same, d)
                else:
                    worst_cross = min(worst_cross, d)
    assert worst_same > 5.0
    assert worst_cross > 3.0
    assert worst_lv > 8.0


def test_every_loaded_truck_dumps(run):
    _, frames, _ = run
    last = frames[-1]['c'][0]['trucks']
    assert all(r[9] < 0.01 for r in last)          # payload gone
    assert all(r[12] == 4 for r in last)           # DONE


def test_trucks_stay_on_the_haul_road_or_pad(run):
    """No truck leaves the road, the ROM pad or the loading area (regression: the return convoy once drove past the
    end of its route, through the pit wall and up the hillside)."""
    site, frames, _ = run
    x0, x1, y0, y1 = cs.PAD
    road = site.road_xy
    worst = 0.0
    for fr in frames[::15]:
        for c in fr['c']:
            if not c['active']:
                continue
            for r in c['trucks']:
                if x0 - 5 <= r[0] <= x1 + 5 and y0 - 5 <= r[1] <= y1 + 5:
                    continue
                d = float(np.min(np.hypot(road[:, 0] - r[0], road[:, 1] - r[1])))
                worst = max(worst, d)
    assert worst <= cs.ROAD_HALF + 1.0, worst


def test_every_convoy_ends_at_rest(run):
    _, frames, _ = run
    for c in frames[-1]['c']:
        assert all(r[5] < 0.05 for r in c['trucks'])
