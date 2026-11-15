"""Named yard scenarios (the YardEnvConfig.scenario key)."""
from dataclasses import replace
from .config import Sensors, YardScenario


def yard1():
    """The default staging yard: 100 x 70 m pad, 1-3 followers, 2-6 rock piles, 0-3 parked machines, 0-2 windrows,
    a 60 m dust-derated leader lidar missing 15 % of the low obstacles, and 30 % of driver picks free clicks."""
    return YardScenario()


def yard1_clear():
    """yard1 with a perfect scan (no missed obstacles): isolates coordination from perception surprises."""
    return replace(YardScenario(), version='yard1_clear', sensors=Sensors(miss_prob=0.0))


YARD_SCENARIOS = {'yard1': yard1, 'yard1_clear': yard1_clear}
