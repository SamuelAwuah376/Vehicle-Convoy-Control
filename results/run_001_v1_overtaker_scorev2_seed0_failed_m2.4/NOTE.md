# Failed run: scenario v1, overtaker, score v2, seed 0, planner m2.4 (pre-fix)

Reproduced with the pre-fix gap rule, no stale-lock check and the pre-fix MPC constraint selection patched back in;
trajectory verified identical to the episode captured before the fixes. Planner m2.4 = 0.25 m hard margins, 0.3 s
traffic headway, 0.1 m check margin; overtaker gain 8, rear_clear 6.

- QP solves {'solved': 3553, 'infeasible': 46, 'timeout': 1}; fallbacks {'shifted': 21, 'lane_keep': 26}
- min teammate clearance 0.16 m (below the 0.25 m hard margin); no collision; team score 16533

Timeline: at 29.0 s car 1 merged lane 2 -> 1 about 8 m ahead of the faster, accelerating car 0 (gap rear = car 0).
29.0-33.5 s: car 1 pinned on the lane line by the rear-gap "stay beside" rule; car 0's constraint type flickered between
beside/ahead/behind along the horizon (|dd| exactly at the lateral threshold) -> 40 infeasible solves; car 0 on shifted /
lane-keep fallbacks passes car 1 side by side. 36.0-36.5 s: car 1's lock still named car 0 as rear gap car after car 0
had passed; the beside constraint switched sides -> 6 infeasible solves.

Fixes: MPC constraint tolerance + no flips between horizon steps + stale gap cars as ordinary obstacles; gap rule
rear-closing-speed and currently-alongside checks; stale-lock re-lock/cancel. Planner config fields added after this run
appear in metrics.json with current defaults but were not used by it (rear_closing_time, lateral_tolerance).
