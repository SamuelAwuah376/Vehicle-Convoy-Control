import numpy as np
import pytest
from nohesi.config import Road as RoadConfig, Vehicle
from nohesi.road import Road
from nohesi.scoring.events import EventDetector, LaneChange, Measurements
from test_nohesi import snapshot

ROAD = Road(RoadConfig())
DT = 0.05
C = [ROAD.lane_center(l) for l in range(3)]   # -3.6, 0, 3.6


def ramp(a, b, duration):
    """Smooth (cosine) lateral move from a to b."""
    u = np.arange(0, duration, DT)/duration
    return a+(b-a)*(1-np.cos(np.pi*u))/2


def hold(d, duration):
    return np.full(int(round(duration/DT)), d)


def drive(d, v=None, traffic=lambda s: []):
    """Car 0 follows lateral profile d at speed v; teammates sit settled in lane 1 far behind."""
    det = EventDetector(Vehicle(), ROAD)
    v = np.full(len(d), 28.0) if v is None else v
    times, s, changes = DT*np.arange(len(d)), 0.0, []
    for k, (t, dk, vk) in enumerate(zip(times, d, v)):
        snap = snapshot(t, [(s, dk, vk), (s-60, C[1], 28.), (s-80, C[1], 28.)], traffic(s))
        changes += det.update(snap, DT).lane_changes
        s += vk*DT
    return times, changes


def test_single_change_times():
    d = np.concatenate((hold(C[1], 1), ramp(C[1], C[2], 3), hold(C[2], 2)))
    times, ch = drive(d)
    assert len(ch) == 1
    e = ch[0]
    assert (e.vehicle, e.from_lane, e.to_lane, e.size) == (0, 1, 2, 1)
    assert e.start_time == pytest.approx(times[np.argmax(np.abs(d-C[1]) > 0.9+1e-9)-1])
    assert e.time == pytest.approx(times[np.argmax(np.abs(d-C[2]) <= 0.9+1e-9)]+0.5)
    assert e.min_traffic_ds == np.inf


def test_fast_double_is_one_event():
    d = np.concatenate((hold(C[0], 1), ramp(C[0], C[2], 2), hold(C[2], 2)))
    assert 0 < (np.abs(d-C[1]) <= 0.9+1e-9).sum()*DT < 0.5
    _, ch = drive(d)
    assert [(e.from_lane, e.to_lane, e.size) for e in ch] == [(0, 2, 2)]


def test_pause_in_middle_gives_two_singles():
    d = np.concatenate((hold(C[0], 1), ramp(C[0], C[1], 2), hold(C[1], 1), ramp(C[1], C[2], 2), hold(C[2], 1)))
    _, ch = drive(d)
    assert [(e.from_lane, e.to_lane, e.size) for e in ch] == [(0, 1, 1), (1, 2, 1)]
    assert ch[1].start_time > ch[0].time


@pytest.mark.parametrize('peak', [0.6, 1.7, 2.4])
def test_leaning_never_emits(peak):
    d = np.concatenate((hold(C[1], 1), ramp(C[1], peak, 1), hold(peak, 2), ramp(peak, C[1], 1), hold(C[1], 1)))
    assert drive(d)[1] == []


def test_min_traffic_ds_destination_lane():
    d = np.concatenate((hold(C[1], 1), ramp(C[1], C[2], 3), hold(C[2], 2)))
    # A car 12 m ahead in the destination lane counts; one 3 m away in uninvolved lane 0 does not.
    _, ch = drive(d, traffic=lambda s: [(1, s+12, C[2], 28.), (2, s+3, C[0], 28.)])
    assert len(ch) == 1 and ch[0].min_traffic_ds == pytest.approx(12.0)


def test_min_speed_over_transition():
    d = np.concatenate((hold(C[1], 1), ramp(C[1], C[2], 3), hold(C[2], 2)))
    v = np.full(len(d), 30.0)
    v[:5] = 20.0                         # before the move: ignored
    mid = 20+30                          # mid-transition dip
    v[mid-3:mid+3] = 26.0
    v[-5:] = 21.0                        # after settling: ignored
    _, ch = drive(d, v)
    assert len(ch) == 1 and ch[0].min_speed == pytest.approx(26.0)


def test_measurements_default_lane_changes():
    m = Measurements(0.1, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), (), ())
    assert m.lane_changes == ()
    assert isinstance(LaneChange(1., 0, 0.5, 1, 2, 1, np.inf, 25.), LaneChange)
