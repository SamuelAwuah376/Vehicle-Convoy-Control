# Leadhaul: venture study, investor memo and promo video

**Leadhaul** is a working name. It is a study of whether autonomous convoy trucks can move goods on repetitive routes, and what an investable company built on that idea would look like.

## The recommendation, in brief

- **Don't build highway convoys.** Fuel-only platooning (Peloton) and human-led highway convoys (Locomation, which raised $105M) both failed. Driverless highway trucking is a capital race of $100M+ a year with almost no revenue yet.
- **Pivot to private haul roads** at mines, quarries and heavy-civil sites. Autonomy there is already proven and paid for, but still covers only a few percent of the fleet.
- **The wedge:** one operator drives the lead truck, and up to three retrofitted trucks follow it driverless.
- **The value is capacity first.** A site that can staff only 3 of 4 trucks recovers about 240 kt/yr per convoy, roughly $1.2M of margin at an assumed $5/t, for a fee of about $0.3M/yr.
- **Cost savings are second.** Measured per ton, including truck ownership, a convoy saves 4–17% depending on wages. The break-even convoy efficiency is 0.77–0.89.
- **We sell only where the math works:** long private hauls with two loading points, where convoying costs little throughput. On short, loader-paced quarry loops convoys lose money, so we stay out of those.
- **The path forward:** grow from convoys into a path-follow mode, then into driverless leaders.
- **The raise:** $2.5M pre-seed on a post-money SAFE with an $18M cap. It funds 18 months to a paid driverless-follower pilot, with explicit gates and kill criteria.
- **Stress-tested:** the model and memo were revised after an adversarial investor-style review. That review added truck ownership, nuisance stops, break-even efficiency, consistent billable hours, mid-range kit costs, a plan timeline that matches the gates, and a realistic budget.

## What's here

| Path | What |
|---|---|
| `memo/Leadhaul_preseed_memo.pdf` | **The 2-page investor memo** (built from `memo/memo.html` with `node memo/build_pdf.js`) |
| `video/promo.mp4` | **The 92 s promo video** (1080p30, original synthesized soundtrack) |
| `model/unit_economics.py` | Customer ROI by site type, per-follower unit economics, market sizing, 5-year plan, pre-seed budget. Writes `model_output.md` and `model_output.json` |
| `sim/convoy_sim.py` | The leader-follower haul convoy simulation behind the video and the memo's technical numbers |
| `sim/test_convoy_sim.py` | Regression tests for the sim claims made in the memo and video: tracking error, gaps, the hold and stop margin, no contact between any vehicles, every truck dumps (`python -m pytest -q venture/sim/test_convoy_sim.py`) |
| `memo/build_web.py` | Builds a responsive, light/dark web version of the memo with the video embedded, for sharing as a page |
| `video/` | Three.js renderer (`main.js`, `story.js`, `index.html`), frame capture (`render.js`) and soundtrack (`soundtrack.py`) |
| `research/` | Research notes with a source and confidence tag on every figure (`01`–`04`), plus `sources.md` mapping each memo claim to its source |

## The simulation (`sim/convoy_sim.py`)

**Site:** a terraced pit (three 10 m benches), a wall ramp at 8.8%, an ex-pit haul road with a light-vehicle crossing, and a ROM pad beside a crusher plant.

**Trucks** are loaded 60 t-class rigid haul trucks (100 t gross, 520 kW):
- kinematic bicycle with steering lag and rate limits;
- drive/brake lag;
- power- and grade-limited acceleration;
- 0.3 g service braking.

**Lead truck:** a human driver model with natural lateral wander and a reaction delay.

**Followers:**
- track the leader's front-axle path from V2V breadcrumbs (20 Hz, 40–60 ms latency, 2% loss) with a Stanley controller;
- keep spacing with CACC (20 m + 3.0 s gap);
- each one checks its own swept corridor ahead and can hold the whole convoy;
- spot and paddock-dump autonomously.

**Results** (in `video/data/metrics.json` after a run):
- path error 1.5–1.7 cm RMS for every follower, 14 cm max;
- a pickup running the crossing is detected 44 m ahead along the path;
- the convoy holds, and the nearest follower stops 18 m clear;
- the operator resumes and the convoy re-forms.

The sim is pure Python (numpy and scipy). It is separate from the `nohesi` package, whose multi-car MPC and RL team-driving stack is the founder's prior work cited in the memo.

The video is rendered with Three.js in headless Chromium rather than CARLA. The CARLA server (Unreal Engine) needs a GPU and about 20 GB, neither of which this environment has. The video is labelled throughout as a simulation, not field footage.

## Reproduce

```bash
pip install numpy scipy pillow matplotlib
python venture/sim/convoy_sim.py --out venture/video/data      # ~10 s: scene.json, terrain_*.bin, traj.json, metrics.json
python venture/model/unit_economics.py                         # model tables
cd venture/video && npm install && python soundtrack.py
node render.js --frames 450,1150,1500                          # stills -> preview/
node render.js                                                 # all 2,760 frames -> frames/, then promo.mp4 (~2 h on 4 CPU cores, SwiftShader)
cd ../memo && node build_pdf.js                                # memo PDF (needs Playwright's Chromium)
```

## Before this goes to investors

The open items are listed at the end of `research/sources.md`:
- verify every link again;
- replace the estimated fleet counts with AEM or Off-Highway Research data;
- measure how many sites match the ideal customer (MSHA mine data plus imagery);
- get hardware quotes;
- check SBIR status;
- run a trademark search on the name.
