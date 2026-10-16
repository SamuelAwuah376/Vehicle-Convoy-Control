import numpy as np
import pytest
from nohesi.planning.base import Target
from tests.test_nohesi_m2 import CFG, P, drive

KEY = ('team', 2)
CLEAR = P.length+CFG.margin_long
SLOT = CLEAR+CFG.side_zone   # changer's MPC keeps a laterally clear car "beside" within this


@pytest.mark.parametrize('offset, sign', [(1.0, -1), (-1.0, 1)])   # teammate level-ahead: drop back; level-behind: pull ahead
def test_yield_opens_slot_for_teammate_changing_in(offset, sign):
    # Teammate one lane to the right at the ego's speed, nearly level; plain MPC stays alongside.
    cars = [(offset, -3.6, 28.0)]
    gaps = {}
    for yield_to in ((), (KEY,)):
        out = drive(Target(d=0.0, speed=28.0, yield_to=yield_to), cars, seconds=4.0, keys=(KEY,))
        gaps[yield_to] = sign*(out['state'][-1, 0]-out['cars'][-1, 0])
        assert np.all(out['state'][:, 1] > -3.6+P.width+CFG.margin_lat-0.05)   # hard side bound unchanged
    assert gaps[()] < CLEAR
    assert gaps[(KEY,)] >= SLOT
