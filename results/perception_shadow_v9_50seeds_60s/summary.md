# Perception shadow evaluation: v9, 50 seeds x 60 s

2026-11-19T09:49:26. Policy overtaker (Planner mpc 'nonlinear', overrides {'retry_cooldown': 1.0, 'w_lateral': 10.0}), plant dynamic, score v10; seeds 1000-1049. Mode 'shadow': the planner drives on the truth, the belief is evaluated.
Window -80 to +250 m of the team centre, confirmed tracks, every belief time t >= 0 (60050 frames).

## Driving

Team score mean 23,117, crashed episodes 0/50.
Against perception off on 2 seeds: 2/2 byte-identical; score difference mean 0 (min 0, max 0); crashed episodes 0 vs 0 off.

| seed | score | off | difference | collisions | off |
|---|---|---|---|---|---|
| 1000 | 37,271 | 37,271 | 0 | 0 | 0 |
| 1001 | 38,595 | 38,595 | 0 | 0 | 0 |

## GOSPA (p 2, c 5 m, alpha 2), per frame

| GOSPA (m) | localization (m^2) | missed (m^2) | false (m^2) | RMS position error of matches (m) | cars | tracks | missed | false | coverage |
|---|---|---|---|---|---|---|---|---|---|
| 2.19 | 0.49 | 0.23 | 7.46 | 0.155 | 20.4 | 20.9 | 0.02 | 0.60 | 0.9991 |

By offset from the team centre (m):

| from | to | cars | coverage | tracks | false fraction |
|---|---|---|---|---|---|
| -80 | -40 | 147528 | 0.9998 | 149400 | 0.0128 |
| -40 | -10 | 111739 | 1.0000 | 114013 | 0.0201 |
| -10 | 10 | 74740 | 1.0000 | 76405 | 0.0216 |
| 10 | 40 | 120452 | 0.9999 | 121975 | 0.0128 |
| 40 | 100 | 228739 | 1.0000 | 231027 | 0.0101 |
| 100 | 175 | 273936 | 0.9995 | 285526 | 0.0413 |
| 175 | 250 | 265205 | 0.9965 | 278726 | 0.0513 |

False confirmed tracks (35856 track-frames): duplicates (a car within c, matched to another track) 0.805, misplaced (c to 3c from a car) 0.177, ghosts (beyond 3c) 0.018; 0.859 from ids never matched in their episode (2463 ids, false for a median 0.20 s, 90 % 1.60 s, max 11.7 s); mean existence 0.960 (matched tracks 0.998).

## NEES of the matched tracks

| | samples | mean | 95 % bounds of the mean | median (chi2 median) | samples inside their 95 % interval | time steps inside (Bar-Shalom) | > 99.9 % quantile |
|---|---|---|---|---|---|---|---|
| state (4 dof) | 1221216 | 1.54 | 3.995-4.005 | 0.45 (3.36) | 0.465 | 0.000 (1017 pairs per step, below 1.00, above 0.00) | 0.0079 |
| position (2 dof) | 1221216 | 1.00 | 1.996-2.004 | 0.25 (1.39) | 0.769 | 0.005 (1017 pairs per step, below 0.99, above 0.00) | 0.0069 |

State NEES by offset (m), by time since last seen (s) and by lane change (true |v_d| > 0.3 m/s):

| bin | samples | mean | median | inside |
|---|---|---|---|---|
| offset -80 to -40 | 147492 | 0.514 | 0.172 | 0.229 |
| offset -40 to -10 | 111739 | 0.349 | 0.133 | 0.163 |
| offset -10 to 10 | 74740 | 0.399 | 0.133 | 0.185 |
| offset 10 to 40 | 120444 | 0.730 | 0.177 | 0.248 |
| offset 40 to 100 | 228736 | 1.357 | 0.400 | 0.443 |
| offset 100 to 175 | 273795 | 2.332 | 0.989 | 0.645 |
| offset 175 to 250 | 264270 | 2.666 | 1.261 | 0.732 |
| age 0 to 0.3 | 1125529 | 1.488 | 0.406 | 0.445 |
| age 0.3 to 1.0 | 95543 | 2.193 | 1.024 | 0.695 |
| age 1 to inf | 144 | 8.317 | 0.375 | 0.319 |
| keeping lane | 1219655 | 1.524 | 0.446 | 0.465 |
| changing lane | 1561 | 17.842 | 10.918 | 0.500 |

Per state component: normalized error mean and spread (1 when consistent), error mean and RMS:

| | z mean | z std | error mean | error RMS |
|---|---|---|---|---|
| s | 0.159 | 0.439 | 0.023 | 0.075 |
| d | 0.038 | 0.882 | 0.005 | 0.136 |
| v_s | 0.048 | 0.259 | 0.045 | 0.255 |
| v_d | -0.005 | 0.274 | -0.002 | 0.085 |

## NIS (tracker records, gate-truncated chi-square)

| group | records | mean | mean dof | expected | 95 % bounds | ratio | inside (0.95) | PIT (10 bins, flat = 0.1) |
|---|---|---|---|---|---|---|---|---|
| all | 3850101 | 2.530 | 3.26 | 3.155 | 3.153-3.158 | 0.802 | 0.907 | 0.21 0.13 0.11 0.10 0.09 0.08 0.07 0.07 0.07 0.07 |
| kind:camera_box | 1094264 | 3.607 | 3.93 | 3.813 | 3.808-3.818 | 0.946 | 0.950 | 0.11 0.11 0.11 0.10 0.10 0.10 0.10 0.09 0.09 0.08 |
| kind:lidar_box | 695039 | 1.132 | 3.00 | 2.894 | 2.889-2.899 | 0.391 | 0.805 | 0.45 0.18 0.11 0.07 0.05 0.04 0.03 0.02 0.02 0.02 |
| kind:radar | 2060798 | 2.429 | 3.00 | 2.894 | 2.891-2.897 | 0.839 | 0.919 | 0.18 0.13 0.12 0.10 0.09 0.08 0.08 0.07 0.07 0.08 |
| sensor:camera_front | 582746 | 3.592 | 3.97 | 3.849 | 3.842-3.856 | 0.933 | 0.950 | 0.12 0.11 0.11 0.10 0.10 0.10 0.10 0.09 0.09 0.08 |
| sensor:camera_left | 28129 | 2.530 | 3.18 | 3.069 | 3.042-3.095 | 0.825 | 0.946 | 0.15 0.13 0.12 0.11 0.10 0.10 0.09 0.08 0.07 0.06 |
| sensor:camera_rear | 463454 | 3.742 | 3.96 | 3.848 | 3.840-3.855 | 0.972 | 0.951 | 0.11 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.09 |
| sensor:camera_right | 19935 | 2.460 | 3.10 | 2.998 | 2.967-3.029 | 0.821 | 0.946 | 0.15 0.13 0.11 0.12 0.10 0.09 0.09 0.08 0.07 0.06 |
| sensor:lidar_front | 331379 | 1.192 | 3.00 | 2.894 | 2.887-2.902 | 0.412 | 0.809 | 0.44 0.18 0.11 0.08 0.06 0.04 0.03 0.02 0.02 0.02 |
| sensor:lidar_rear | 363660 | 1.077 | 3.00 | 2.894 | 2.887-2.901 | 0.372 | 0.801 | 0.46 0.19 0.12 0.07 0.05 0.04 0.03 0.02 0.01 0.01 |
| sensor:radar_front | 1034524 | 2.374 | 3.00 | 2.894 | 2.890-2.898 | 0.820 | 0.914 | 0.19 0.13 0.12 0.10 0.09 0.08 0.07 0.07 0.07 0.08 |
| sensor:radar_left | 90221 | 2.642 | 3.00 | 2.894 | 2.880-2.909 | 0.913 | 0.937 | 0.13 0.12 0.11 0.11 0.10 0.09 0.09 0.08 0.08 0.09 |
| sensor:radar_rear | 869396 | 2.455 | 3.00 | 2.894 | 2.889-2.899 | 0.848 | 0.921 | 0.17 0.13 0.11 0.10 0.09 0.08 0.08 0.07 0.07 0.08 |
| sensor:radar_right | 66657 | 2.658 | 3.00 | 2.894 | 2.877-2.911 | 0.918 | 0.938 | 0.13 0.12 0.11 0.11 0.10 0.09 0.09 0.08 0.08 0.09 |

## Identity

210 identity switches (105 takeovers, the old track still confirmed; 105 rebirths, the old one gone) and 13 track swaps over 1018 tracked car-minutes (0.2064 switches per car-minute); 1.064 track ids per matched car.

## Detection probability against range (after processing)

Opportunities: traffic cars with their body centre in the field of view and nominal range; visible: at least 0.3 of the facing outline in sight.

| sensor | all | visible | 0-10 | 10-25 | 25-50 | 50-75 | 75-100 | 100-150 | 150-200 | 200-250 | false per scan | teammate per scan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| camera_front | 0.332 | 0.898 | 0.82/0.82 | 0.74/0.94 | 0.55/0.94 | 0.34/0.93 | 0.21/0.90 | 0.13/0.81 | 0.09/0.74 |  | 0.096 | 0.0017 |
| camera_left | 0.614 | 0.641 | 0.61/0.64 | 0.29/0.31 |  |  |  |  |  |  | 0.100 | 0.0000 |
| camera_rear | 0.718 | 0.912 | 0.72/0.73 | 0.85/0.94 | 0.70/0.94 | 0.60/0.94 |  |  |  |  | 0.096 | 0.0016 |
| camera_right | 0.635 | 0.648 | 0.63/0.65 | 0.41/0.50 |  |  |  |  |  |  | 0.099 | 0.0000 |
| lidar_front | 0.289 | 0.852 | 1.00/1.00 | 0.80/1.00 | 0.59/0.99 | 0.36/0.97 | 0.21/0.85 | 0.08/0.51 | 0.02/0.14 |  | 0.023 | 0.0004 |
| lidar_rear | 0.538 | 0.853 | 0.99/1.00 | 0.92/1.00 | 0.75/0.99 | 0.55/0.92 | 0.28/0.63 | 0.10/0.26 |  |  | 0.012 | 0.0000 |
| radar_front | 0.401 | 0.758 | 0.83/0.84 | 0.77/0.90 | 0.61/0.86 | 0.46/0.77 | 0.38/0.70 | 0.34/0.63 | 0.30/0.53 | 0.22/0.37 | 0.733 | 0.0068 |
| radar_left | 0.703 | 0.746 | 0.73/0.74 | 0.67/0.78 | 0.28/0.39 | 0.30/0.43 | 0.13/0.21 |  |  |  | 0.608 | 0.0020 |
| radar_rear | 0.555 | 0.778 | 0.73/0.74 | 0.87/0.93 | 0.75/0.91 | 0.60/0.81 | 0.48/0.72 | 0.36/0.55 |  |  | 0.781 | 0.0095 |
| radar_right | 0.712 | 0.749 | 0.74/0.74 | 0.68/0.77 | 0.30/0.48 | 0.30/0.51 | 0.21/0.25 |  |  |  | 0.605 | 0.0015 |

Range cells: all / visible.

## Time

| stage | ms per call |
|---|---|
| sim_lidar | 11.70 |
| sim_radar | 5.22 |
| sim_camera | 10.61 |
| proc_lidar | 40.44 |
| proc_radar | 4.32 |
| proc_camera | 13.77 |
| ideal | 0.00 |
| tracker | 21.85 |
| snapshot | 0.00 |
| total | 107.91 |

Planner 28.2 ms per step; whole step 193.0 ms (world, planner, perception); episode wall time mean 232 s, max 248 s with 16 workers; evaluation 47.7 s per episode; peak worker memory 278 MB.
