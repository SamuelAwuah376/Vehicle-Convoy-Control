from dataclasses import dataclass

VERSIONS = ('e4', 'e5')   # env versions this code runs: e5 = e4 plus overtake actions (overtakes=True)


@dataclass(frozen=True)
class EnvConfig:
    """NoHesiEnv settings. Bump version whenever the observation, mask or reward definition changes."""
    version: str = 'e4'
    scenario: str = 'v4'            # SCENARIOS key
    score: str = 'v4'               # SCORES key
    vehicle_model: str = 'kinematic'
    mpc: str = 'linear'
    reward_scale: float = 100.0     # game points per unit reward
    aux_rejected: float = -0.1      # per rejected lane request (gap rule or MPC)
    aux_fallback: float = -0.2      # per shifted / lane_keep fallback event
    aux_conflict: float = -0.5      # per unresolved plan conflict
    aux_emergency: float = -0.5     # per emergency (yield-brake) fallback
    collision_extra: float = 4.0    # training-only: extra multiple of the game's collision penalty (4 -> 5x in total)
    # Training-only fixed reward (reward units, x reward_scale game points) when an episode ends in a collision, on top
    # of the game's terms: the game's crash cost scales with the banked points, so an early crash is nearly free, and
    # under a spread penalty ending the episode can even pay; the MPC's vehicle clearance is a soft constraint.
    crash_penalty: float = 0.0
    mask_gap_rule: bool = True      # mask lane changes the intention layer's gap rule would reject right now
    traffic_k: int = 8              # nearest traffic cars observed per car
    ahead: float = 250.0            # traffic observation window along s (m)
    behind: float = 80.0
    # e5: overtake actions (intentions.OVERTAKES: pass the car ahead left/right at a passing margin), their masks and
    # observations; an overtake abandoned before the pass costs aux_overtake_abort (ending in the passing lane does not).
    overtakes: bool = False
    aux_overtake_abort: float = -0.1
    # Training-only reward per second (x reward_scale game points) for each team car stuck behind traffic for longer
    # than stuck_grace: the nearest traffic car ahead in its lane is within stuck_window (CG) and the car is less than
    # stuck_closing faster than it. Teammates in between do not shield it (a queue behind a slow car is stuck).
    aux_stuck: float = 0.0
    stuck_window: float = 100.0
    stuck_closing: float = 2.0
    stuck_grace: float = 2.0

    def __post_init__(self):
        if self.overtakes != (self.version == 'e5'):
            raise ValueError("overtakes=True is env version 'e5' (and e5 has overtakes)")
