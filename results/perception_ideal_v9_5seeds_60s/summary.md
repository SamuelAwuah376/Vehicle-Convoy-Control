# Perception ideal evaluation: v9, 5 seeds x 60 s

2026-11-19T10:03:14. Policy overtaker (Planner mpc 'nonlinear', overrides {'retry_cooldown': 1.0, 'w_lateral': 10.0}), plant dynamic, score v10; seeds 1000-1004. Mode 'ideal': the planner drives on the belief (experiment E0).
Window -80 to +250 m of the team centre, confirmed tracks, every belief time t >= 0 (6005 frames).

## Driving

Team score mean 21,245, crashed episodes 0/5.
Against perception off on 5 seeds: 0/5 byte-identical; score difference mean -809 (min -5,209, max 1,136); crashed episodes 0 vs 0 off.

| seed | score | off | difference | collisions | off |
|---|---|---|---|---|---|
| 1000 | 38,407 | 37,271 | 1,136 | 0 | 0 |
| 1001 | 38,583 | 38,595 | -12 | 0 | 0 |
| 1002 | 6,062 | 6,041 | 21 | 0 | 0 |
| 1003 | 7,070 | 12,278 | -5,209 | 0 | 0 |
| 1004 | 16,102 | 16,084 | 18 | 0 | 0 |

## GOSPA (p 2, c 5 m, alpha 2), per frame

| GOSPA (m) | localization (m^2) | missed (m^2) | false (m^2) | RMS position error of matches (m) | cars | tracks | missed | false | coverage |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 20.0 | 20.0 | 0.00 | 0.00 | 1.0000 |

By offset from the team centre (m):

| from | to | cars | coverage | tracks | false fraction |
|---|---|---|---|---|---|
| -80 | -40 | 14232 | 1.0000 | 14232 | 0.0000 |
| -40 | -10 | 11889 | 1.0000 | 11889 | 0.0000 |
| -10 | 10 | 7017 | 1.0000 | 7017 | 0.0000 |
| 10 | 40 | 13501 | 1.0000 | 13501 | 0.0000 |
| 40 | 100 | 22200 | 1.0000 | 22200 | 0.0000 |
| 100 | 175 | 26133 | 1.0000 | 26133 | 0.0000 |
| 175 | 250 | 25348 | 1.0000 | 25348 | 0.0000 |

## NEES of the matched tracks

| | samples | mean | 95 % bounds of the mean | median (chi2 median) | samples inside their 95 % interval | time steps inside (Bar-Shalom) | > 99.9 % quantile |
|---|---|---|---|---|---|---|---|
| state (4 dof) | 120320 | 0.01 | 3.984-4.016 | 0.00 (3.36) | 0.004 | 0.000 (100 pairs per step, below 1.00, above 0.00) | 0.0000 |
| position (2 dof) | 120320 | 0.00 | 1.989-2.011 | 0.00 (1.39) | 0.008 | 0.000 (100 pairs per step, below 1.00, above 0.00) | 0.0000 |

State NEES by offset (m), by time since last seen (s) and by lane change (true |v_d| > 0.3 m/s):

| bin | samples | mean | median | inside |
|---|---|---|---|---|
| offset -80 to -40 | 14232 | 0.009 | 0.000 | 0.005 |
| offset -40 to -10 | 11889 | 0.008 | 0.000 | 0.003 |
| offset -10 to 10 | 7017 | 0.007 | 0.000 | 0.003 |
| offset 10 to 40 | 13501 | 0.006 | 0.000 | 0.004 |
| offset 40 to 100 | 22200 | 0.008 | 0.000 | 0.002 |
| offset 100 to 175 | 26133 | 0.009 | 0.000 | 0.007 |
| offset 175 to 250 | 25348 | 0.009 | 0.000 | 0.004 |
| age 0 to 0.3 | 120320 | 0.008 | 0.000 | 0.004 |
| age 0.3 to 1.0 | 0 |  |  |  |
| age 1 to inf | 0 |  |  |  |
| keeping lane | 120247 | 0.008 | 0.000 | 0.004 |
| changing lane | 73 | 0.072 | 0.069 | 0.000 |

Per state component: normalized error mean and spread (1 when consistent), error mean and RMS:

| | z mean | z std | error mean | error RMS |
|---|---|---|---|---|
| s | 0.008 | 0.041 | 0.000 | 0.000 |
| d | -0.000 | 0.003 | -0.000 | 0.000 |
| v_s | 0.017 | 0.088 | 0.006 | 0.032 |
| v_d | 0.000 | 0.006 | 0.000 | 0.000 |

## NIS (tracker records, gate-truncated chi-square)

| group | records | mean | mean dof | expected | 95 % bounds | ratio | inside (0.95) | PIT (10 bins, flat = 0.1) |
|---|---|---|---|---|---|---|---|---|
| all | 177200 | 0.018 | 3.00 | 2.894 | 2.884-2.904 | 0.006 | 0.020 | 0.99 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 |
| kind:lidar_box | 177200 | 0.018 | 3.00 | 2.894 | 2.884-2.904 | 0.006 | 0.020 | 0.99 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 |
| sensor:lidar_ideal | 177200 | 0.018 | 3.00 | 2.894 | 2.884-2.904 | 0.006 | 0.020 | 0.99 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 |

## Identity

0 identity switches (0 takeovers, the old track still confirmed; 0 rebirths, the old one gone) and 0 track swaps over 100 tracked car-minutes (0.0000 switches per car-minute); 1.000 track ids per matched car.

## Detection probability against range (after processing)

Opportunities: traffic cars with their body centre in the field of view and nominal range; visible: at least 0.3 of the facing outline in sight.

| sensor | all | visible | 0-10 | 10-25 | 25-50 | 50-75 | 75-100 | 100-150 | 150-200 | 200-250 | false per scan | teammate per scan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar_ideal | 1.000 | 1.000 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 0.094 | 0.0000 |

Range cells: all / visible.

## Time

| stage | ms per call |
|---|---|
| sim_lidar | 0.00 |
| sim_radar | 0.00 |
| sim_camera | 0.00 |
| proc_lidar | 0.00 |
| proc_radar | 0.00 |
| proc_camera | 0.00 |
| ideal | 0.37 |
| tracker | 6.32 |
| snapshot | 0.10 |
| total | 6.79 |

Planner 14.5 ms per step; whole step 31.0 ms (world, planner, perception); episode wall time mean 37 s, max 38 s with 10 workers; evaluation 2.6 s per episode; peak worker memory 203 MB.
