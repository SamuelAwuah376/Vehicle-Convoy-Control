# Yard policy evaluation: yard_ppo_001 / final

- model: `runs_rl/yard_ppo_001/final_model.zip` (400384 timesteps, sha256 7b64db5aa5f3)
- VecNormalize: `runs_rl/yard_ppo_001/final_vecnormalize.pkl`
- env: `nohesi.yard.env` scenario yard1, version ye1, followers scenario default (config `runs_rl/yard_ppo_001/config.json`)
- seeds: 1020-1069 (50); ppo deterministic with action masks; scripted: the prioritized baseline (scripted_actions)
- 2026-11-15, commit d3c08b06d7b9b83df25f317a7149abcf033d2532, 4 workers

| | ppo | scripted |
|---|---:|---:|
| episodes | 50 | 50 |
| success rate | 0.96 | 1.00 |
| collision rate | 0.00 | 0.00 |
| timeout rate | 0.02 | 0.00 |
| stalled rate | 0.02 | 0.00 |
| parked fraction | 0.98 | 1.00 |
| mean duration, all episodes (s) | 78.8 | 73.8 |
| mean duration, successes (s) | 72.6 | 73.8 |
| mean duration, 1 follower: all / successes (s) | 45.5 / 45.5 (n 13) | 45.5 / 45.5 (n 13) |
| mean duration, 2 followers: all / successes (s) | 69.7 / 69.7 (n 19) | 68.6 / 68.6 (n 19) |
| mean duration, 3 followers: all / successes (s) | 112.5 / 98.2 (n 18) | 99.7 / 99.7 (n 18) |
| safety stops / episode | 0.00 | 0.00 |
| replans / episode | 0.14 | 0.16 |
| back-off / episode (m) | 0.0 | 0.0 |

## Paired: ppo - scripted duration on the seeds both solved (negative: ppo faster)

| | |
|---|---|
| seeds both solved | 48 |
| mean ± std (s) | +0.6 ± 3.5 (95 % CI of the mean ±1.0) |
| median (s) | +0.0 |
| ppo faster / slower / equal | 14 / 9 / 25 |
| solved by ppo only | - |
| solved by scripted only | 1025, 1062 |
| solved by neither | - |

## By follower count

| followers | seeds | ppo success | scripted success | both solved | mean Δ (s) | ppo faster / slower |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 13 | 1.00 | 1.00 | 13 | -0.1 | 1 / 0 |
| 2 | 19 | 1.00 | 1.00 | 19 | +1.2 | 7 / 4 |
| 3 | 18 | 0.89 | 1.00 | 16 | +0.4 | 6 / 5 |

## Rendered cases

| case | seed | followers | ppo | scripted | Δ (s) | files |
|---|---:|---:|---|---|---:|---|
| ppo_wins_most | 1069 | 2 | parked, 2/2 parked, 73.0 s | parked, 2/2 parked, 81.1 s | -8.1 | [ppo_wins_most_seed1069_ppo.png](ppo_wins_most_seed1069_ppo.png) [ppo_wins_most_seed1069_ppo.gif](ppo_wins_most_seed1069_ppo.gif) [ppo_wins_most_seed1069_scripted.png](ppo_wins_most_seed1069_scripted.png) [ppo_wins_most_seed1069_scripted.gif](ppo_wins_most_seed1069_scripted.gif) |
| ppo_loses_most | 1059 | 2 | parked, 2/2 parked, 86.2 s | parked, 2/2 parked, 68.6 s | +17.6 | [ppo_loses_most_seed1059_ppo.png](ppo_loses_most_seed1059_ppo.png) [ppo_loses_most_seed1059_ppo.gif](ppo_loses_most_seed1059_ppo.gif) [ppo_loses_most_seed1059_scripted.png](ppo_loses_most_seed1059_scripted.png) [ppo_loses_most_seed1059_scripted.gif](ppo_loses_most_seed1059_scripted.gif) |
| ppo_fails | 1025 | 3 | stalled, 2/3 parked, 153.7 s | parked, 3/3 parked, 132.3 s | - | [ppo_fails_seed1025_ppo.png](ppo_fails_seed1025_ppo.png) [ppo_fails_seed1025_ppo.gif](ppo_fails_seed1025_ppo.gif) [ppo_fails_seed1025_scripted.png](ppo_fails_seed1025_scripted.png) [ppo_fails_seed1025_scripted.gif](ppo_fails_seed1025_scripted.gif) |

