# Controller / plant sweep (re-run of the milestone 2, part 3 comparison)

Run on the project machine (Windows, native acados build) with `experiments/nmpc_sweep.py` at commit
`ec3f1d6` (clean `nohesi/`), planner `m3.3`, on 2026-10-19. Scripted overtaker on scenarios v1 (10 cars) and
v4 (20 lane-changing cars, blind merges), seeds 0–4, score v3 (the score the original milestone 2 numbers used).
Episodes are deterministic: re-running one gives the same score to the decimal.

- `episodes.csv`: one row per episode (30).
- `summary.csv` / `summary.json`: per scenario and case (5 episodes each); `summary.json` also records the setup.

| Scenario | Case | Crashed | Team score mean (min–max) | Close passes | Lane changes | Fallbacks | Plan error d, mean (max) | Min clearance traffic / teammates |
|---|---|---|---|---|---|---|---|---|
| v1 | linear MPC + kinematic sim | 0 / 5 | 16,757 (9,634–27,913) | 37.2 | 22.0 | 0.0 | 4.86 mm (22.8) | 0.243 / 1.784 m |
| v1 | linear MPC + dynamic sim | **5 / 5** (7–45 s) | 3,748 | 9.2 | 1.0 | 5.0 | 18.5 mm (617) | 0.177 / 38.5 m |
| v1 | nonlinear MPC + dynamic sim | 0 / 5 | 13,522 (8,658–17,066) | 33.8 | 24.4 | 2.8 | **0.19 mm** (12.5) | 0.242 / 1.108 m |
| v4 | linear MPC + kinematic sim | 0 / 5 | 16,998 (8,308–25,892) | 35.8 | 21.2 | 2.4 | 4.84 mm (47.3) | 0.243 / 0.118 m |
| v4 | linear MPC + dynamic sim | **5 / 5** (11–41 s) | 5,267 | 12.4 | 1.6 | 12.6 | 24.7 mm (600) | 0.000 / 0.148 m |
| v4 | nonlinear MPC + dynamic sim | 0 / 5 | 10,135 (3,304–14,952) | 27.0 | 26.6 | 7.6 | **0.31 mm** (49.7) | 0.233 / 0.234 m |

Plan error = distance between where the plan put the car one MPC step (0.1 s) ahead and where it actually was.

**Solve times.** Measured one episode at a time (v4, seed 2), nothing else running:
linear MPC (OSQP) 2.38 ms mean, 5.97 ms p95; nonlinear MPC (acados SQP-RTI, 1 iteration) **6.88 ms mean, 11.6 ms p95**.
The sweep's own per-episode solve times (`solve_ms_*` in the CSVs: about 9.5–10.5 ms for the NMPC) ran 10 episodes
in parallel and are inflated by CPU contention; use the single-episode figures for timing claims.

**Changes from the earlier README claims** (measured with planner m2.x before milestone 3):
- Unchanged: linear MPC + dynamic sim crashes in all 10 runs; nonlinear MPC + dynamic sim has 0 crashes; the
  nonlinear MPC's plan error is about 0.2–0.3 mm against about 5 mm for the linear MPC on its kinematic sim;
  linear MPC + kinematic sim scores about 16,800 on both scenarios.
- Changed: the nonlinear MPC scores 13,500 on v1 but **10,100 on v4** (was about 13,000–13,500 on both). v4 now has
  blind-merging traffic and the milestone 3 fallbacks.
- Corrected: nonlinear MPC solves take **about 7 ms mean (12 ms p95)**, not 3–6 ms.
- Not 0: `solve_failures_total` in `summary.csv` counts solves that did not return `solved`
  (nonlinear MPC: 14 on v1, 37 on v4, of about 18,000 solves each); fallbacks covered them.
