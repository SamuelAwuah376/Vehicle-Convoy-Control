# Validation record

Verified using Python 3.12 on Windows with NumPy 2.2.6, Matplotlib 3.10.3, Pillow 12.3.0, CasADi 3.7.2 and pytest 8.4.1.

**12 tests passed**, including four 90-second baseline acceptance cases, a 12-second MPC offset recovery test, and an injected solver failure checking the fallback command and failure count. See `results/tests.xml` for machine-readable test results.

The five full-route runs below are 90 seconds each. `braking_dt01` uses 0.1 s integration; the others use 0.05 s. All vehicles ended below 0.00002 m/s, and absolute leader goal errors were below 1e-9 m. This high final position precision reflects deterministic feedback convergence in simulation, not real-world accuracy.

| Run | Max cross-track, all vehicles (m) | Leader cross-track RMS (m) | Min gap (m) | Gap-error RMS after 10 s (m) | Lane violation (m) | Collision pair samples | MPC fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | 0.0319 | 0.0163 | 5.0000 | 0.2512 | 0.0000 | 0 | 0 |
| braking | 0.0315 | 0.0156 | 5.0000 | 0.3631 | 0.0000 | 0 | 0 |
| offset | 0.5870 | 0.0673 | 5.0000 | 0.2512 | 0.0000 | 0 | 0 |
| braking_dt01 | 0.0291 | 0.0145 | 5.0000 | 0.3648 | 0.0000 | 0 | 0 |
| mpc | 0.0319 | 0.0003 | 5.0000 | 0.2512 | 0.0000 | 0 | 0 |

Every run's complete inputs and metrics are in its `metrics.json`; raw histories are in CSV and NPZ alongside the plot. `braking/convoy.gif` animates the baseline braking case.

The MPC run uses leader steering MPC and baseline followers. Its lower leader tracking RMS does not demonstrate general superiority over pure pursuit or robustness to uncertainty. All cases use exact states and known dynamics. Gap RMS includes the stopping period; inspect the time histories for transient behavior.

During development, the original stopping law crossed its intended goal and the truncated MPC endpoint reference produced excessive steering. The final position feedback stops the leader with a 4 m route-end buffer, and the final full-route runs shown here have no sampled lane violations. An initial Windows native-solver/pytest capture issue was resolved by isolating the MPC test in a subprocess; the final complete suite passes.

Collision checks use oriented rectangles at sampled times. Lane checks project all four rectangle corners onto the gently curved route. These are regression measurements, not continuous-time or formal safety certificates. No uncertainty-aware or RL capabilities are implemented.
