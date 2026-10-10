from dataclasses import dataclass, asdict
import numpy as np
from .config import Config
from .route import Route
from .dynamics import step
from .controllers import pursuit, leader_acceleration, follower_acceleration


@dataclass
class Result:
    time: np.ndarray
    states: np.ndarray
    progress: np.ndarray
    lateral: np.ndarray
    controls: np.ndarray
    gaps: np.ndarray
    cfg: Config
    scenario: str
    controller: str
    mpc_failures: int = 0

    def metrics(self):
        cfg = self.cfg
        desired = cfg.standstill_gap + cfg.headway*self.states[:, 1:, 3]
        settled = self.time >= 10
        return {
            'scenario': self.scenario, 'controller': self.controller,
            'max_abs_cross_track_m': float(np.max(np.abs(self.lateral))),
            'rms_cross_track_m': float(np.sqrt(np.mean(self.lateral**2))),
            'min_bumper_gap_m': float(np.min(self.gaps)),
            'gap_error_rms_after_10s_m': float(np.sqrt(np.mean((self.gaps[settled]-desired[settled])**2))) if np.any(settled) else None,
            'final_speed_mps': self.states[-1, :, 3].tolist(),
            'final_progress_m': self.progress[-1].tolist(),
            'leader_distance_to_goal_m': float(Route().total-4-self.progress[-1, 0]),
            'max_acceleration_mps2': float(np.max(self.controls[:, :, 0])),
            'min_acceleration_mps2': float(np.min(self.controls[:, :, 0])),
            'max_steering_rad': float(np.max(np.abs(self.controls[:, :, 1]))),
            'mpc_failures': self.mpc_failures, 'config': asdict(cfg),
        }


def simulate(cfg=None, scenario='nominal', controller='baseline'):
    cfg = cfg or Config()
    if scenario not in ('nominal', 'braking', 'offset'):
        raise ValueError('unknown scenario')
    if controller not in ('baseline', 'mpc'):
        raise ValueError('unknown controller')
    route = Route()
    spacing = cfg.length+cfg.standstill_gap+cfg.headway*cfg.cruise
    starts = 5+spacing*np.array([2, 1, 0])
    if starts[0] >= route.total:
        raise ValueError('initial convoy does not fit route')
    states = np.empty((3, 4))
    for i, s in enumerate(starts):
        xy, yaw = route.at(s)
        if scenario == 'offset':
            xy = xy + (0.5 if i != 1 else -0.5)*np.array([-np.sin(yaw), np.cos(yaw)])
            yaw += 0.05
        states[i] = [*xy, yaw, cfg.cruise]
    steering = np.zeros(3)
    mpc = None
    if controller == 'mpc':
        from .mpc import SteeringMPC
        mpc = SteeringMPC(cfg)
    times = np.arange(int(round(cfg.duration/cfg.dt))+1)*cfg.dt
    histories, progresses, laterals, actions, gaps = [], [], [], [], []
    for k, t in enumerate(times):
        projections = np.array([route.project(st[:2]) for st in states])
        s, lateral = projections.T
        gap = s[:-1]-s[1:]-cfg.length
        histories.append(states.copy()); progresses.append(s); laterals.append(lateral); gaps.append(gap)
        if k == len(times)-1:
            break
        commands = []
        for i in range(3):
            a = (leader_acceleration(states[i], s[i], t, route, cfg, scenario) if i == 0
                 else follower_acceleration(states[i], states[i-1], gap[i-1], cfg))
            delta = pursuit(states[i], s[i], route, cfg)
            if mpc is not None and i == 0 and states[i, 3] > 0.3:
                delta = mpc.command(states[i], s[i], route, steering[i], delta)
            commands.append((a, delta))
        # All commands use the same state snapshot (no sequential-update bias).
        applied = []
        for i, (a, delta) in enumerate(commands):
            states[i], a, steering[i] = step(states[i], a, delta, steering[i], cfg)
            applied.append((a, steering[i]))
        actions.append(applied)
    return Result(times, np.array(histories), np.array(progresses), np.array(laterals),
                  np.array(actions), np.array(gaps), cfg, scenario, controller,
                  mpc.failures if mpc else 0)
