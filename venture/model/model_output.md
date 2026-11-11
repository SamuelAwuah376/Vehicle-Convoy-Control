# Leadhaul model output

## Customer ROI per convoy (1 operator + 3 followers)

| case | eff | $/follower-h | h/yr | manual $/yr | convoy $/yr | saving $/yr | saving % | t per operator-h |
|---|---|---|---|---|---|---|---|---|
| ICP: 5 km private haul, 40 t ADT/road-class truck, remote site | 1.0 | 24.0 | 4160 | 2,041,728 | 1,639,872 | 401,856 | 19.7 | 80.0 -> 320.0 |
| Mid: 3 km haul, 60 t truck, US median wage | 0.92 | 24.0 | 4160 | 2,546,416 | 2,435,091 | 111,325 | 4.4 | 165.0 -> 607.2 |
| Avoid: short quarry loop, 60 t truck, single loader | 0.8 | 24.0 | 4160 | 2,546,416 | 2,800,355 | -253,939 | -10.0 | 235.7 -> 754.3 |
| ICP, convoy efficiency 0.9 | 0.9 | 24.0 | 4160 | 2,041,728 | 1,822,080 | 219,648 | 10.8 | 80.0 -> 288.0 |
| ICP, convoy efficiency 0.95 | 0.95 | 24.0 | 4160 | 2,041,728 | 1,726,181 | 315,547 | 15.5 | 80.0 -> 304.0 |
| ICP, price $18/follower-h | 1.0 | 18.0 | 4160 | 2,041,728 | 1,564,992 | 476,736 | 23.3 | 80.0 -> 320.0 |
| ICP, price $30/follower-h | 1.0 | 30.0 | 4160 | 2,041,728 | 1,714,752 | 326,976 | 16.0 | 80.0 -> 320.0 |
| ICP, single shift | 1.0 | 24.0 | 2080 | 1,020,864 | 819,936 | 200,928 | 19.7 | 80.0 -> 320.0 |
| ICP at the US median wage ($28.19/h) | 1.0 | 24.0 | 4160 | 1,880,816 | 1,599,644 | 281,172 | 14.9 | 80.0 -> 320.0 |
| ICP with road-class truck (opex $40/h) | 1.0 | 24.0 | 4160 | 1,542,528 | 1,159,392 | 383,136 | 24.8 | 80.0 -> 320.0 |

Manual vs convoy cost per ton (ICP): $1.53 -> $1.23; Leadhaul revenue per convoy: $299,520/yr; price per ton equivalent: $0.30

## Leadhaul economics per follower

| stage | revenue/yr | COGS/yr | GM % | upfront capex | capex payback (months) |
|---|---|---|---|---|---|
| pilot | 90,435 | 31,400 | 65.3 | 72,000 | 11.8 |
| scale | 108,522 | 19,000 | 82.5 | 40,000 | 4.9 |

## Market (estimates)

- US aggregates beachhead ARR potential: $65M-$130M
- Global off-highway rigid + ADT fleet (est.): 160,000 trucks -> TAM $8.6B/yr
- Fleet counts are DERIVED estimates (USGS tonnage / truck productivity; OHR sales x life). Validate with AEM/OHR data.

## Plan

| year | sites | followers | $/h | exit ARR |
|---|---|---|---|---|
| 2027 | 1 | 3 | 16.0 | $0.14M |
| 2028 | 3 | 12 | 20.0 | $0.72M |
| 2029 | 9 | 40 | 23.0 | $2.76M |
| 2030 | 22 | 110 | 24.0 | $7.92M |
| 2031 | 45 | 260 | 24.0 | $18.72M |

## Pre-seed budget (18 months)

- Team: founder + 3 robotics/controls engineers + field-ops lead (18 mo, loaded): $1,150,000
- Vehicles: 2 used 40-60 t trucks/ADTs (or leased) + 3 scaled UTVs with by-wire: $260,000
- Autonomy kits: sensors, compute, V2V, by-wire integration (3 sets): $180,000
- Safety: HARA, UL 4600-structured case, third-party pre-audit; insurance; test site: $170,000
- Legal/IP (Stanford OTL check), travel, customer discovery: $90,000
- Contingency: $150,000
- **Total: $2,000,000**
