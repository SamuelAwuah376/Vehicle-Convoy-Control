from nohesi.config import Road as RoadConfig, Vehicle
from nohesi.dynamics import KinematicPlant
from nohesi.evaluation.metrics import double_lane_changes
from nohesi.intentions import ACTIONS, Intention, Record
from nohesi.road import Road
from nohesi.rl.scripted import Overtaker
from tests.test_nohesi import snapshot

TEAM = [(400, -3.6, 30), (100, -3.6, 30), (-300, -3.6, 30)]   # car 1 in lane 0 at s=100
SLOW = (1, 120, -3.6, 20)                                      # slow car 20 m ahead in lane 0


def lane_cmd(traffic):
    policy = Overtaker(Road(RoadConfig()), KinematicPlant(Vehicle(), 0.05).params)
    return ACTIONS[policy(snapshot(0.0, TEAM, traffic), [Intention(0, 30.0) for _ in range(3)])[1]][0]


def test_crosses_clear_middle_to_open_far_lane():
    assert lane_cmd([SLOW, (2, 135, 0.0, 20)]) == 2


def test_prefers_single_change_when_middle_is_best():
    assert lane_cmd([SLOW, (2, 250, 0.0, 30), (3, 150, 3.6, 20)]) == 1


def test_no_double_change_with_middle_occupied_alongside():
    assert lane_cmd([SLOW, (2, 102, 0.0, 30)]) != 2


def test_double_lane_change_metrics():
    r = [Record(0, 0, 'accepted', lane=2), Record(1, 0, 'completed', lane=2), Record(2, 1, 'accepted', lane=1),
         Record(3, 0, 'accepted', lane=1), Record(4, 0, 'cancelled', lane=2), Record(5, 1, 'completed', lane=1),
         Record(6, 0, 'accepted', lane=0), Record(7, 0, 'rejected', 'gap', 1), Record(8, 0, 'completed', lane=0)]
    # A cancelled record's lane is the lane the car keeps (here back in lane 2).
    assert double_lane_changes(r, 0) == {'accepted': 2, 'completed': 2}
    # A stalled two-lane move that leaves the car in the middle lane: the next change from there is single.
    stall = [Record(0, 0, 'accepted', lane=2), Record(3, 0, 'cancelled', 'stalled', lane=1),
             Record(5, 0, 'accepted', lane=2), Record(6, 0, 'completed', lane=2)]
    assert double_lane_changes(stall, 0) == {'accepted': 1, 'completed': 0}
