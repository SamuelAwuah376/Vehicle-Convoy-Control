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
    # With stuck_escape_only, a stuck car is charged only while it has a way out at the decision (a missed escape, see
    # NoHesiEnv.escapes): an overtake would start, or the mask allows a lane change into a lane whose nearest traffic car
    # ahead within stuck_window is at least stuck_closing faster than the one in its own lane (or has none).
    stuck_escape_only: bool = False
    # With stuck_pass_lanes, a lane change into a lane that is not faster is a way out too when the car can get past its
    # lead traffic car through it (NoHesiEnv._passable: the overtake corridor test at the team's top speed); taking it
    # exempts the car, staying is charged.
    stuck_pass_lanes: bool = False
    # Training-only reward per m/s of speed braked away, summed over the team (x reward_scale game points): a car braking
    # at b m/s^2 for dt seconds costs aux_brake*b*dt (applied acceleration; emergency and safety braking count too).
    aux_brake: float = 0.0
    # Training-only reward (x reward_scale game points) per lean change while a car holds its lane (keep, no overtake,
    # a lane command of 0): the lean is needed for close passes (< pass_threshold side clearance), flipping it is not.
    aux_lean_change: float = 0.0
    # Training-only reward per lane change the policy drops while pending (intention record 'aborted'; the planner's
    # cancels are not charged).
    aux_lc_abort: float = 0.0
    # False holds the observation's episode clock (time / duration) at 0: decisions from what the car sees, not the time
    # left. The input stays in place, so a warm start keeps its shape.
    obs_clock: bool = True
    # Policy decision rate: Planner.decision_interval (s; 0.5 = 2 Hz) and the target speed change per speed action
    # (Planner.speed_step, m/s; 2.0 at 2 Hz is 4 m/s per second of commands).
    decision_interval: float = 0.5
    speed_step: float = 2.0
    # Other Planner settings for this env, as (name, value) pairs applied over the defaults (e.g. (('retry_cooldown',
    # 1.0), ('w_lateral', 10.0))). () keeps every default.
    planner_overrides: tuple = ()
    # Follow mode (e5 only; Planner.follow, intentions.FOLLOW): action FOLLOW (the slot of overtake right while slowing,
    # which then never starts) follows the teammate directly ahead in the lane; masked unless the car follows or
    # IntentionManager.follow_check would start one; while it follows it is the only action (no decisions of its own
    # until a break-off ends the follow). Two observations per car are appended (following, follow possible now), so a
    # warm start from an env without them keeps the other inputs' places.
    follow: bool = False

    def __post_init__(self):
        if self.overtakes != (self.version == 'e5'):
            raise ValueError("overtakes=True is env version 'e5' (and e5 has overtakes)")
        if self.follow and not self.overtakes:
            raise ValueError('follow needs the e5 action space (its FOLLOW slot is an overtake slot)')
