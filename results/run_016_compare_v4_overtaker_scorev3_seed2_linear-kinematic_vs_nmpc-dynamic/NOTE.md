# Comparison: v4_overtaker_scorev3_seed2

Same scenario, seed and settings; only the controller and vehicle physics differ.
Top panel of `compare.gif`: linear MPC + kinematic sim (stored as `run_015_v4_overtaker_scorev3_seed2`). Bottom: nonlinear MPC + dynamic sim (same episode as `run_014_v4_overtaker_scorev3_seed2_dynamic_nmpc`, re-run deterministically).
`compare.png` overlays team score, speeds and lane positions (solid = first, dashed = second; colours = cars 0/1/2).

| Case | Team score | End | Collisions | Close passes | Lane changes | Fallbacks | Plan error d (mean / max) | Solve | Min clearance traffic / teammates |
|---|---|---|---|---|---|---|---|---|---|
| linear MPC + kinematic sim | 24639 | time (120 s) | 0 | 48 | 25 | 6 | 5.4 / 34.1 mm | 2.2 ms | 0.25 / 1.29 m |
| nonlinear MPC + dynamic sim | 24653 | time (120 s) | 0 | 52 | 28 | 2 | 0.3 / 4.9 mm | 14.6 ms | 0.81 / 0.24 m |
