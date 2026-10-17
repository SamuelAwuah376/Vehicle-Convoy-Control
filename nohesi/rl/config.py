from dataclasses import dataclass


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
    mask_gap_rule: bool = True      # mask lane changes the intention layer's gap rule would reject right now
    traffic_k: int = 8              # nearest traffic cars observed per car
    ahead: float = 250.0            # traffic observation window along s (m)
    behind: float = 80.0
