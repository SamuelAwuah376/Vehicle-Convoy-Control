# Leadhaul model output

## Customer ROI per convoy (1 operator + 3 followers, 4,160 h/yr, cost incl. truck ownership)

| case | operator $/h | convoy eff. | $/t manual -> convoy | saving | $/yr | break-even eff. | t per operator-h | labour-short: extra t/yr, value at $5/t |
|---|---|---|---|---|---|---|---|---|
| ICP, remote-site wage $35/h | 52.7 | 0.933 | 2.096 -> 1.923 | 8.3% | 215,821 | 0.856 | 80 -> 299 | 244,053 t, $1,220,267 |
| ICP, US median wage $28.19/h (BLS) | 43.0 | 0.933 | 1.975 -> 1.89 | 4.3% | 105,864 | 0.893 | 80 -> 299 | 244,053 t, $1,220,267 |
| ICP, FIFO/union labour ($80/h loaded) | 80.0 | 0.933 | 2.438 -> 2.014 | 17.4% | 526,240 | 0.771 | 80 -> 299 | 244,053 t, $1,220,267 |
| ICP with road-class trucks, $35/h | 52.7 | 0.933 | 2.354 -> 2.029 | 13.8% | 252,567 | 0.804 | 50 -> 187 | 152,533 t, $762,667 |
| ICP, single bin loadout, $35/h | 52.7 | 0.891 | 2.096 -> 2.014 | 3.9% | 97,435 | 0.856 | 80 -> 285 | 187,578 t, $937,891 |
| ICP in path-follow mode, $35/h | 52.7 | 0.98 | 2.096 -> 1.831 | 12.7% | 346,045 | 0.856 | 80 -> 314 | 306,176 t, $1,530,880 |
| Avoid: 12 min quarry loop, one loader, 60 t trucks | 43.0 | 0.56 | 0.775 -> 1.264 | -63.1% | -1,253,587 | 0.914 | 275 -> 616 | -869,440 t, $-4,347,200 |

Leadhaul fee per convoy: $299,520/yr (2 shifts) = $0.321/t on follower tons.

## Leadhaul economics per follower (3,000 billable h/yr)

| stage | $/h | revenue/yr | COGS/yr | GM % | capex | capex payback (mo) |
|---|---|---|---|---|---|---|
| pilot | 20.0 | 60,000 | 38,750 | 35.4 | 87,000 | 24.3 |
| scale | 24.0 | 72,000 | 24,000 | 66.7 | 48,000 | 9.6 |

## Market (estimates)

- Global off-highway rigid + ADT fleet: 160,000; TAM $8.6B/yr; ICP-qualified SAM $3.5B/yr
- US aggregates ICP: 900-1,800 followers = $65-130M ARR

## Plan

| year | sites | followers | $/h | exit ARR |
|---|---|---|---|---|
| 2027 | 0 | 0 | 0.0 | $0.00M |
| 2028 | 2 | 6 | 20.0 | $0.36M |
| 2029 | 6 | 24 | 22.0 | $1.58M |
| 2030 | 15 | 70 | 24.0 | $5.04M |
| 2031 | 35 | 180 | 24.0 | $12.96M |

2031 followers = 10-20% of the US aggregates ICP fleet, so the plan needs mining and heavy civil from 2029.
Equity needed to 2031 (est.): $40M.

## Pre-seed budget (18 months)

- Team: founder, 3 robotics/controls engineers, field-ops/mining lead from month 3 (loaded, 18 mo): $1,270,000
- By-wire engineering partner and OEM/J1939 access (steering, brakes, hoist): $190,000
- Vehicles: 2 used 40 t ADTs or leases + 3 by-wire UTVs, transport, site costs: $300,000
- Autonomy kits, 3 sets (sensors, compute, V2V, actuation): $225,000
- Safety: HARA, UL 4600-structured case, third-party pre-audit, insurance, test site: $250,000
- Legal/IP (Stanford OTL check), customer discovery, travel: $100,000
- Contingency: $165,000
- **Total: $2,500,000**
