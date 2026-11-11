# 03 — Investor & Fundraising Research: Retrofit Leader-Follower Convoy Autonomy (Pre-seed / Seed)

Prepared 2026-11-11. Company concept: retrofit kit so one human-driven lead vehicle leads N driverless followers on repetitive private-site routes (quarries, mines, industrial yards), using an MPC + RL multi-vehicle stack validated in CARLA, with dual-use optionality (Army leader-follower / autonomous convoy resupply).

## How this was researched, and its limits (read first)

- **Method:** WebSearch only. WebFetch could not reach `carta.com` or `techcrunch.com` (DNS blocked by the session network allowlist), so I could not open Carta's primary pages. Many Carta figures below come from Carta page snippets returned by search, or from secondary sites that cite Carta. Each is labelled.
- **Search budget ran out:** the shared 200-search-per-turn limit was used up before Sections 5 (non-dilutive) and 6 (Stanford) could be searched. Items in those sections that come from my training data (cutoff June 2026) and were not re-checked here are marked **UNVERIFIED (training knowledge)**. Before relying on them, check sbir.gov, nsf.gov/seedfund, startx.com and the other primary sites.
- **Convention:** each claim is followed by its source URL and the date of the data or publication. "UNVERIFIED" means I could not confirm it from a primary source in this session.

---

## 1. Pre-seed and seed benchmarks (2025–2026)

### 1a. Pre-seed (mostly SAFEs)

| Metric | Value | Source / date |
|---|---|---|
| Median post-money SAFE cap, rounds < $250K | **$10M** | Finro citing Carta Q2 2026, https://www.finrofca.com/news/safe-valuation-caps-2026 (Q2 2026 data, published 2026) — secondary |
| Median cap, $250K–$499K | **$12M** | same (Q2 2026) — secondary |
| Median cap, $500K–$999K | **$12.5M** | same (Q2 2026) — secondary |
| Median cap, $1M–$2.4M | **$18M** | same (Q2 2026) — secondary |
| Median cap, ≥ $2.5M | **$35M**, up 40% YoY; 75th percentile near **$60M**; 90th percentile up to **$100M** | Carta, "In the pre-seed market, valuation caps on new SAFEs keep getting bigger," https://carta.com/data/safe-valuation-caps-q2-2026/ and https://carta.com/data/state-of-pre-seed-q2-2026/ (Q2 2026). Carta says the upper tail is "largely tied to investor enthusiasm for AI." |
| Post-money share of SAFEs | **91%** of SAFEs on Carta in Q2 2026 were post-money | Carta State of Pre-Seed Q2 2026, https://carta.com/data/state-of-pre-seed-q2-2026/ (Q2 2026) |
| Full-year 2025 caps | about **$10M** for $250K–$1M rounds; about **$15M** for $1M–$2.5M rounds | Carta State of Pre-Seed: 2025 in Review, https://carta.com/data/state-of-pre-seed-2025/ (FY2025, via search snippet) |
| Q2 2025 caps | < $250K: $7.5M (up from $6.5M the prior quarter); $250K–$500K: $10M | Carta State of Pre-Seed Q2 2025, https://carta.com/data/state-of-pre-seed-q2-2025/ (Q2 2025) |
| Q3 2025 trend | Caps flat for most round sizes over the previous 12 months | Carta State of Pre-Seed Q3 2025, https://carta.com/data/state-of-pre-seed-q3-2025/ (Q3 2025) |
| SAFE share of pre-seed | About 85% of 2025 pre-seed rounds used SAFEs | Secondary blog https://myunicornclub.substack.com/p/carta-state-of-preseed-2025-explained (2026) — **UNVERIFIED** |
| Pre-seed dilution | One secondary guide claims about 19–20% median dilution for $1–2.4M SAFE rounds and 5–6% under $250K | https://startupa.ge/blog/startup-fundraising-guide-pre-seed-series-a (2026) — **UNVERIFIED**. This conflicts with simple arithmetic ($1.5M / $18M post ≈ 8%), so it probably measures cumulative pre-seed dilution. Do not use it for pricing. |

### 1b. Seed (mostly priced)

| Metric | Value | Source / date |
|---|---|---|
| Carta median seed, Q2 2026 | **$4.5M raised at $23.9M post-money**, median dilution **19.4%** | Carta Q2 2026 fundraising benchmarks, cited on https://carta.com/data/linkedin-vc-fundraising-benchmarks-2026/ and https://www.equidam.com/ranges-of-negotiation-at-different-stages-of-a-startup/ (Q2 2026) |
| Carta median seed, six months to July 2026 | **$4.1M at $24.3M post**, **18%** dilution | Waveup citing Carta, https://waveup.com/blog/seed-round-benchmarks/ (July 2026) — secondary; Carta's author page also shows "$24.3M median in the last 6 months," https://carta.com/author/peter-walker/ |
| Carta median seed, Q1 2026 | $4.1M (up from $3.2M the prior quarter) at $24M post (up from $18M a year earlier) | https://valueaddvc.com/blog/startup-funding-rounds-in-2025-whats-normal-at-pre-seed-seed-a-and-b (2026) — secondary |
| Carta median seed pre-money, Q3 2025 | **$16M** (record; +14% YoY); FY2024 median $14.8M | Carta State of Private Markets Q3 2025, https://carta.com/data/state-of-private-markets-q3-2025/ (Q3 2025) |
| PitchBook-NVCA median seed deal | **$3.0M** (2026 YTD to Q3); seed is the only stage where the median shrank this year ($3.4M → $3.0M) | Q3 2026 PitchBook-NVCA Venture Monitor, https://nvca.org/wp-content/uploads/2026/10/Q3-2026-PitchBook-NVCA-Venture-Monitor.pdf (Oct 2026); summary at https://scalable.news/p/seed-got-smaller |
| PitchBook-NVCA median seed pre-money | **$18.3M** (2026 through Q3) | https://startupappraisal.com/MarketIntelligence, transcribed from p. 11 of the Q3 2026 Venture Monitor — **secondary; not checked against the PDF** |
| Peter Walker (Carta) commentary | Median seed company on Carta has **4 employees** besides the founders. A small cohort of AI-infrastructure seeds is priced at $160–200M post, which skews the headline figures. | PMF Show podcast, https://www.pmf.show/episodes/ep-37-q1-2026-w-carta-what-you-need-to-raise-a-series-a-peter-walker-head-of-insights- (May 2026) |
| Founder ownership | Median founding team holds **about 56%** fully diluted at seed; median employee pool **12.1%** at seed; about 36% after Series A (secondary) | Carta Founder Ownership Report 2026, https://carta.com/data/founder-ownership-2026/ (2026; covers rounds 2021–2025) |

### 1c. Hardware / robotics / deep tech specifically

- Carta publishes a hardware seed benchmark page (https://carta.com/data/explore/round-benchmarking/hardware-seed/), but its figures were not in search snippets and the page could not be fetched. **No verified hardware-specific median.**
- Robotics seed rounds "cluster between $3M and $8M, deeptech-leaning rounds stretching to $12M." Source: Causo Hub, https://hub.causo.ai/guides/raising-seed-robotics-startup-2026 (2026). Vendor blog, **UNVERIFIED**.
- Robotics seed median "$3.52M." Source: https://www.roboticscenter.ai/state-of-robotics-2026 (2026). **UNVERIFIED**.
- Robotics seed valuations "$8–20M post-money, $10–14M sweet spot." Source: https://www.22astronauts.com/p/safe-notes-and-cap-tables-for-robotics (2026). Practitioner estimate, **UNVERIFIED**.
- Bessemer: robotics has "historically required significant capital, and commercialization and innovation cycles in deep tech are longer in comparison to SaaS markets." Source: https://www.bvp.com/atlas/intelligent-robotics-the-new-era-of-physical-ai (Bessemer, 2025/2026).
- **Interpretation:** the Carta headline ($24M post at seed) is driven by software and AI. Hardware-heavy robotics at pre-seed and seed usually prices at or below the all-sector median unless the team is exceptional (for example Bedrock, led by the former head of Waymo trucking, or Humble). The current "physical AI" hype lifts the ceiling for autonomy-software-led companies.

---

## 2. Comparable rounds — industrial / off-road / logistics autonomy and physical AI (2024–2026)

### 2a. Seed and pre-seed comps (most relevant for pricing)

| Company (what) | Round | Amount | Date | Lead(s) | Source |
|---|---|---|---|---|---|
| **Overland AI** (off-road defense autonomy; one operator managing several vehicles) | Seed | $10M | 2024-05-20 | Point72 Ventures (with Shasta, Ascend, Pioneer Square Labs, Voyager, Cubit) | https://www.overland.ai/news/overland-ai-secures-10m-in-seed-funding-to-accelerate-development-of-autonomous-vehicle-software-for-high-risk-missions |
| **Scout AI** (defense ground and air autonomy foundation model) | Seed | $15M | 2025-04 (one later article says Jan 2025) | Align Ventures, Booz Allen Ventures (+ Draper Associates, Decisive Point) | https://www.prnewswire.com/news-releases/scout-ai-emerges-from-stealth-with-15m-seed-round-lands-2-dod-contracts-and-unveils-fury--robotic-foundation-model-for-defense-302429902.html |
| **Bedrock Robotics** (excavator retrofit autonomy) | Seed + Series A (combined) | $80M | 2025-07-16 | Eclipse (seed), 8VC (A); also NVentures, Samsara Ventures, Two Sigma, Valor | https://www.prnewswire.com/news-releases/bedrock-robotics-emerges-from-stealth-with-80m-in-funding-for-autonomous-construction-technology-302506159.html |
| **Genesis AI** (robot foundation model) | Seed | $105M | 2025-07 | Eclipse, Khosla (co-leads) | https://www.roboticscenter.ai/institutions/investors/khosla-ventures (secondary) |
| **Gravis Robotics** (ETH Zurich excavator retrofit) | Pre-A | $23M | 2025-11 | IQ Capital, Zacua Ventures | https://www.gravisrobotics.com/series-a and https://thenextweb.com/news/gravis-robotics-200m-series-a-softbank-autonomous-construction |
| **Crewline AI** (autonomous roller retrofit, about 1 hour install) | Seed | $7.1M | 2026-04 | Initialized Capital, Nebular | https://www.therobotreport.com/crewline-secures-7-1m-to-automate-constructions-most-repetitive-task/ |
| **Humble Robotics** (cabless autonomous electric hauler, dock-to-dock) | Seed | $24M | 2026-04-21 | Eclipse (+ Energy Impact Partners) | https://www.prnewswire.com/news-releases/humble-emerges-from-stealth-to-bring-autonomous-electric-haulers-to-real-world-freight-302748014.html |
| **Origin Robotics** (interior-wall finishing) | Seed + A | $20M combined | 2026 (Nov coverage) | Accel (seed), Zacua (A) | https://thenextweb.com/news/zacua-ventures-135m-fund-ii-construction-tech (2026-11) |
| **Polymath Robotics** (plug-and-play industrial vehicle autonomy software) | Seed / strategic | about $500K total (aggregators disagree) | 2022–2024 | Y Combinator; Samsara Ventures (strategic); TRAC | https://tracxn.com/d/companies/polymath-robotics/__7CFxRvjseXcxtJmvQixJROcHQsyhZqk_9bxTg73fFF8/funding-and-investors; https://www.polymathrobotics.com/blog/polymath-robotics-announces-investment-from-samsara-ventures — **amounts UNVERIFIED** |
| **Autonomous mining haul seed** | — | — | — | No verified 2025–2026 seed round for a mining or quarry haulage autonomy startup was found. Tracxn notes one new "autonomous mining vehicles" startup founded in 2026, unnamed. | https://tracxn.com/d/trending-business-models/startups-in-autonomous-mining-vehicles/__-YPGEQg2Q0AWMolNukrFL-8YCVFmI6Mlqvzi-L4f8RQ/companies (Nov 2026) |
| **Leader-follower convoy seed** | — | — | — | No current leader-follower convoy startup seed round was found. The main precedent, **Locomation**, shut down (below). | https://www.freightwaves.com/news/locomation-convoy-plan-doubles-down-on-autonomous-trucking-challenge |

### 2b. Series A and later comps (valuation trajectory and investor map)

| Company | Round / event | Amount / valuation | Date | Lead(s) | Source |
|---|---|---|---|---|---|
| Overland AI | Series A | $32M | 2025-01-13 | 8VC (+ Point72, Overmatch, Shasta, Ascend, OUP, Caprock) | https://www.globenewswire.com/news-release/2025/01/13/3008705/0/en/Overland-AI-Raises-32-Million-in-Series-A-Funding-to-Advance-Ground-Autonomy-for-U-S-Defense-Forces.html |
| Overland AI | Growth | $100M (includes $20M TriplePoint venture debt) | 2026-02 | 8VC (equity) | https://theaiinsider.tech/2026/02/06/overland-ai-raises-100m-to-scale-autonomous-ground-systems-for-defense-applications/ |
| Overland AI | Contract | Up to $18.6M two-year Army/DIU prototype OT; DARPA RACER Phase II | 2024 | — | https://www.geekwire.com/2025/seattle-startup-overland-ai-lands-32m-to-help-u-s-military-with-off-road-autonomous-driving-tech/ |
| Bedrock Robotics | Series B | $270M at $1.75B | 2026-02-04 | CapitalG, Valor Atreides AI Fund | https://www.constructiondive.com/news/bedrock-robotics-raise-ai-automation-funding/811982/ |
| Bedrock Robotics | Series C | **$600M at $4.1B post**; about $1B raised in total; paid fully autonomous work on live sites from mid-Aug 2026 | 2026-10-26 | CapitalG, Emergence (+ Bessemer, USIT, Temasek) | https://www.globenewswire.com/news-release/2026/10/26/3388197/0/en/bedrock-robotics-raises-600-million-series-c-to-scale-autonomous-construction-equipment-across-u-s-sites.html; https://www.bloomberg.com/news/articles/2026-10-26/autonomous-construction-firm-bedrock-hits-4-1-billion-valuation |
| Gravis Robotics | Series A | $200M; reported $1B valuation (disputed) | 2026-08-17 | SoftBank (sole) | https://siliconangle.com/2026/08/17/gravis-robotics-gets-200m-funding-softbank-retrofit-excavators-self-driving-ai-systems/ |
| Scout AI | Series A | $100M; $11M in DoD/DARPA/AAL contracts | 2026-04-29 | Align Ventures, Draper Associates | https://techcrunch.com/2026/04/29/coby-adcocks-scout-ai-raises-100-million-to-train-models-for-war-we-visited-its-bootcamp/ |
| Humble Robotics | Series A | $51M (about $75M total) | 2026 (exact date **UNVERIFIED**) | Obvious Ventures | https://www.freightwaves.com/news/humble-robotics-51m-series-a-cabless-hauler |
| Forterra | Series B | $75M | 2024-09 | — | https://www.forterra.com/posts/forterra-raises-75-million-series-b |
| Forterra | Series C | $238M (about $50M of it debt), valuation > $1B | 2025-11-12 | Moore Strategic Ventures | https://www.govconwire.com/articles/forterra-raises-238m-series-c-fund-autonomy; https://www.bloomberg.com/news/articles/2025-11-12/autonomous-military-vehicle-startup-notches-over-1-billion-valuation-in-new-deal |
| Forterra | Contracts | Army ATV-S prototyping (with Carnegie Robotics); $114M Army breaching prime; Army GEARS Phase II; Kalmar driverless terminal tractors | 2025 | — | https://breakingdefense.com/2025/02/army-picks-carnegie-forterra-for-autonomous-logistics-truck-prototyping/; https://www.forbes.com/sites/richardbishop1/2025/11/25/in-a-new-era-forterra-pursues-both-defense-and-commercial-launches/ |
| Teleo (supervised autonomy retrofit for construction and mining) | Series A + extensions | $12M (2022) + $16.2M (2024); $29.8M total | 2022-06 / 2024-04 / 2024-11 | UP.Partners | https://www.therobotreport.com/teleo-raises-16-2-million-to-scale-supervised-autonomy-for-heavy-equipment/ |
| Pronto (quarry and mine haulage autonomy) | Commercial | Heidelberg Materials: 100+ trucks over 3 years, exclusive in North America | 2025-02-24 | — | https://www.prnewswire.com/news-releases/pronto-announces-global-agreement-to-deploy-100-autonomous-haul-trucks-302382511.html |
| Pronto | Channel | Komatsu Smart Quarry Autonomous (retrofit or new, via Komatsu dealers) | 2025-08-07 | — | https://www.komatsu.com/en-us/newsroom/2025/komatsu-north-america-and-pronto-launch-autonomy-solution-for-quarry-size-trucks |
| Pronto | M&A | Acquired SafeAI (about 12 engineers; price "in the millions") | 2025-07-15 | — | https://techcrunch.com/2025/07/15/av-startup-pronto-ai-acquires-off-road-autonomous-vehicle-rival-safeai/ |
| Pronto | M&A | Acquired by Atoms (Travis Kalanick), becoming "Atoms Mining" | 2026 | — | https://bricks-bytes.com/funding-ma/ubers-pronto-atoms-autonomous-machinery/ |
| Mariana Minerals (AI-run mining; uses Pronto at Copper One) | Series B | $310M | 2026 (date **UNVERIFIED**) | Khosla | https://www.tectonicdefense.com/mariana-minerals-raises-310m-series-b/; https://techcrunch.com/2026/04/09/mariana-minerals-taps-pronto-to-help-automate-a-copper-mine/ |
| Intramotev (autonomous battery railcars at mines) | Series A | $14.4M; earlier $200K Michigan grant for a mine-site deployment | 2024-08 | Flybridge, Alpaca VC (+ Decisive Point, Cathcart Rail) | https://www.prnewswire.com/news-releases/intramotev-raises-14-4-million-to-retrofit-a-new-era-of-growth-for-rail-industry-302223983.html; https://www.mining.com/missouri-start-up-gets-200000-grant-to-accelerate-us-autonomous-railcar-technology/ |
| Outrider (yard autonomy) | Series D | $62M; about $250M equity in total; TÜV SÜD certification Jul 2025 | 2024-10-24 | Koch Disruptive Technologies, NEA (+ 8VC, ARK, NVentures, Prologis) | https://www.outrider.ai/press-releases/outrider-closes-62-million-series-d-financing-to-drive-growth-of-yard-automation/; https://www.forbes.com/sites/richardbishop1/2025/07/10/outrider-powers-scaling-of-yard-autonomy-with-tv-sd-certification/ |
| ISEE (yard trucks) | Series B | $40M | 2022-11-16 | Founders Fund | https://www.businesswire.com/news/home/20221116005996/en/ |
| ISEE | Partnership | Holman (an investor) will lease, retrofit and maintain ISEE trucks in North America | 2026-10 | — | https://www.freightwaves.com/news/isee-holman-partnership-autonomous-yard-trucks |
| Fernride (teleop/autonomous terminal tractors, now defense) | Series A extension | €18M (€75M total); TÜV SÜD certified; driverless | 2025-09-04 | Helantic | https://tech.eu/2025/09/04/fernride-bags-eur18m-series-a-extension-as-it-expands-into-defence/ |
| Fernride | M&A | Acquired by Quantum Systems | 2025-12-17 per Tracxn (**UNVERIFIED**) | — | https://tracxn.com/d/companies/fernride/__M0zOJ4gTNJ-NFlRh3r2oS2qejJnISdm3AqihEtv2lt0 |
| Gatik (middle-mile box trucks) | Series D | $200M; about $500M total; $600M+ contracted revenue; valuation undisclosed (Forbes: > $800M, **UNVERIFIED**) | 2026-08-25 | Qatar Investment Authority, Koch Disruptive Technologies | https://techcrunch.com/2026/08/25/self-driving-truck-startup-gatik-raises-200m-following-pepsico-deal/ |
| Waabi | Series C | $750M (+ up to $250M from Uber on milestones); about $3B pre reported (**UNVERIFIED**) | 2026-01-28 | Khosla, G2 Venture Partners | https://www.cnbc.com/2026/01/28/autonomous-startup-waabi-raises-750-million-to-expand-into-robotaxis.html; https://betakit.com/waabi-lands-750-million-usd-series-c-partners-with-uber-on-robotaxis/ |
| Einride | SPAC (Legato III) | Announced at $1.8B (Nov 2025); repriced and closed at **$1.35B pre-money**; $113.3M PIPE; listed as ENRD 2026-06-10 | 2026-06 | — | https://www.einride.tech/press/einride-a-leading-digital-electric-and-autonomous-freight-technology-company-to-go-public-via-a-business-combination-with-legato-merger-corp-iii-at-a-1-8-billion-valuation; https://electrek.co/2026/03/06/spac-attack-einride-bets-1-35b-that-their-spac-will-be-different/ |
| Kodiak (KDK) | SPAC (Ares II) | About $2.5B valuation; > $212.5M ($145M PIPE); heavy redemptions; −13% intraday on day 1 | Closed 2025-09-24 | — | https://www.globenewswire.com/news-release/2025/09/24/3155836/0/en/Kodiak-Completes-Business-Combination-with-Ares-Acquisition-Corporation-II-to-Accelerate-Journey-as-a-Leading-AI-Powered-Autonomous-Vehicle-Technology-Company.html; https://www.truckingdive.com/news/kodiak-robotics-spac-vote-publicly-traded-nasdaq/760986/ |
| Built Robotics | Pivot / grant | Pivoted to solar piling (2023); 3-year Blattner deal (2025-09-09, per Teardown); ARENA grant up to A$13.3M for a Brisbane plant (2026-10) | 2023–2026 | — | https://www.teardown.ai/companies/built-robotics (secondary); https://arena.gov.au/news/robotics-and-software-funding-to-drive-down-solar-costs/ |
| Carbon Robotics | Series D | $70M; $157M total | 2024-10-21 | BOND | https://www.businesswire.com/news/home/20241021330997/en/ |
| Monarch Tractor | Collapse | Raised ≥ $220M ($133M of it in 2024); warned of shutdown Nov 2025; ABC; assets to Caterpillar Apr 2026; dealer suits alleged tractors "unable to operate autonomously" | 2025-11 → 2026-04 | — | https://techcrunch.com/2025/11/19/monarch-tractor-preps-for-layoffs-and-warns-employees-it-may-shut-down/; https://techcrunch.com/2026/04/15/monarch-tractors-collapse-ends-in-with-an-acquisition-by-caterpillar/ |
| Bluewhite (tractor retrofit kit, RaaS) | Series C | $39M at $200–300M post (TechCrunch); $89M total; New Holland partnership (Jun 2024) | 2024-01-23 | Insight Partners | https://techcrunch.com/2024/01/23/bluewhite-rakes-in-39m-for-robots-as-a-service-that-can-be-retrofitted-to-drive-any-tractor/ |
| Agtonomy | Series A / B | $22.5M A (late 2023, Momenta) + $10M (Oct 2024, Autotech) = $32.8M; $18M Series B (2025-10-28, DBL Partners — **single source + Tracxn**) | 2023–2025 | see above | https://www.vcaonline.com/news/2024101609/; https://tracxn.com/d/companies/agtonomy/__hzXeKPczfyoXD-WDpupt1wEiXWoYRy_HPykYFlhETXE/funding-and-investors |
| Scythe Robotics | Series B | $42M; pay-as-you-mow pricing | 2023-01 | Energy Impact Partners | https://www.prnewswire.com/news-releases/scythe-robotics-secures-42m-series-b-financing-to-accelerate-production-of-zero-emissions-autonomous-mower-301728617.html |
| Physical Intelligence | Series B / talks | $600M at $5.6B (2025-11, CapitalG); in talks for ≥ $1B at about $20B (2026-10-27) | 2025–2026 | CapitalG; Lux and Avenir in talks | https://www.bloomberg.com/news/articles/2025-11-20/robotics-startup-physical-intelligence-valued-at-5-6-billion-in-new-funding; https://www.bloomberg.com/news/articles/2026-10-27/robotics-lab-physical-intelligence-in-talks-for-a-20-billion-valuation |
| Skild AI | Series C | $1.4B at > $14B | 2026-01-14 | SoftBank (+ NVentures, Macquarie, Bezos Expeditions) | https://www.businesswire.com/news/home/20260114335623/en/Skild-AI-Raises-$1.4B-Now-Valued-Over-$14B |
| Applied Intuition | Series F | $600M + tender at $15B; acquired EpiSci; defense products Axion/Acuity; earlier bought Embark for $71M | 2025-06-17 | BlackRock, Kleiner Perkins | https://www.appliedintuition.com/press-releases/series-f |

### 2c. Cautionary precedents that apply directly to a convoy pitch

- **Locomation** ran a human-driven lead truck with an autonomous follower on public highways. It raised **$105M** and wound down in early 2023. Source: https://www.freightwaves.com/news/locomation-convoy-plan-doubles-down-on-autonomous-trucking-challenge (2022/2023). One analysis argues convoying "might be easier 95% of the time, but in the remaining 5% it still degenerates to requiring full autonomy." Source: https://itcanthink.substack.com/p/whatever-happened-to-the-self-driving (Chris Paxton). **Investors will raise this.** The answer is the operating domain: private sites, low speeds, fixed routes, no public-road traffic, and site rules the operator controls.
- **Embark:** the CEO said "capital markets have turned their backs on pre-revenue companies." It was sold to Applied Intuition for $71M in cash. Source: https://www.robotics247.com/article/embark_trucks_shuts_down_amidst_uncertainty_in_autonomous_trucking_market (2023).
- **TuSimple** left US trucking after losing Navistar, and is now CreateAI. Source: https://www.freightwaves.com/news/the-leaders-followers-and-also-rans-in-autonomous-trucking.
- **Army leader-follower:** the Army ended its internal leader-follower truck kit development in June 2023 and turned to commercial solutions through the GEARS prototyping competition. Source: https://breakingdefense.com/2023/06/army-closing-down-leader-follower-robotic-truck-development-eyeing-commercial-solutions/ (2023-06). This is a signal that the Army wants commercial providers. Robotic Research received $53.1M for the final 60-month Expedient Leader Follower sprint (same source).
- **Monarch Tractor** (above) shows the risk of building your own vehicle and making autonomy claims customers then dispute.

---

## 3. VC firms active in industrial autonomy / logistics robotics / dual-use (pre-seed–seed focus)

| Firm | Evidence (portfolio and thesis) | Stage / check | Source / date | Fit |
|---|---|---|---|---|
| **Eclipse** | Led the Bedrock seed (2025), Humble's $24M seed (Apr 2026) and co-led Genesis AI's $105M seed. Closed **$1.3B** on 2026-04-07: Fund VI **$720M** early stage plus Early Growth III $591M, positioned as "physical AI." Its Bedrock post cites reshoring manufacturing, **mining** and logistics as the reason heavy-machine autonomy matters. | Seed to growth; also incubates | https://fundmomentum.vc/blog/eclipse-ventures-1-3b-fund-vi-early-growth-iii-physical-ai-2026 (2026-04); https://eclipse.capital/blog/accelerating-industrial-innovation/ | Very high |
| **8VC** | Led Overland's Series A (Jan 2025) and its Feb 2026 equity; led Bedrock's Series A; invested in Outrider. Logistics thesis: autonomy for "specific, high-value functions" is more promising than broad AVs, which it calls "crowded and extremely long-term." Says Overland "completes the trifecta" of ground, air and sea autonomy with Saronic and Anduril. | Seed to growth | https://www.8vc.com/theses/logistics; https://www.8vc.com/resources/announcing-our-investment-in-overland-ai | Very high (thesis matches private-site convoys plus defense) |
| **Point72 Ventures** | Led Overland's $10M seed (May 2024). Defense-tech partner Chris Morales: self-driving built for streets and enclosed lots will struggle off-road. | Seed | https://p72.vc/defense-tech/our-investment-in-overland-ai/; https://techcrunch.com/2024/05/18/vcs-and-the-military-are-fueling-self-driving-startups-that-dont-need-roads/ (2024-05) | High (dual-use seed lead). No "Hyperscale" link was confirmed. |
| **a16z American Dynamism** | Dec 2025 "Big Ideas": an "AI-native, software-first industrial base" including **mining** and "fleets of autonomous machines." Aug 2025 "Electro-Industrial Stack." The AD50 list includes Rune Technologies (military logistics) and Locus. | Seed to growth | https://a16z.com/newsletter/big-ideas-2026-part-2/; https://a16z.com/the-electro-industrial-stack-will-move-the-world/; https://a16z.com/american-dynamism-50-2025/ | Medium–high (dual-use story) |
| **Lux Capital** | "Bodies, not just brains" essay (Jul 2025); Physical Intelligence, Formic (RaaS); **Fund IX $1.5B** (Jan 2026). | Seed to growth | https://www.luxcapital.com/news/the-age-of-embodied-intelligence-why-the-future-of-robotics-is-morphology-agnostic | Medium |
| **Khosla Ventures** | Co-led Waabi's $750M C (Jan 2026), co-led Genesis AI's seed, led Mariana Minerals (AI mining). Usually writes large checks. | Seed (large) to growth | https://www.roboticscenter.ai/institutions/investors/khosla-ventures (secondary) | Medium |
| **Bessemer** | "Bessemer Predicts: Robotics and physical AI" (2026-04-16) calls this a "GPT-2.5 moment," with a wide gap between lab and field. Invested in Bedrock's Series C (Oct 2026). Robotics contact: robotics@bvp.com. | Seed to growth | https://www.bvp.com/atlas/bessemer-predicts-robotics-and-physical-ai | Medium (stronger at A) |
| **DCVC** | Deep Tech Opportunities Report 2025 (Jun 2025), themed on an "American industrial renaissance." Includes the essay **"Robots should be backward-compatible with the real world"**, which is the retrofit thesis. Portfolio: Agility, Slip Robotics. | Seed–A | https://www.dcvc.com/news-insights/dcvc-dtor-2025-robots-should-be-backward-compatible-with-the-real-world/; https://www.dcvc.com/reports/deep-tech-opportunities-report-2025/ | High (thesis match) |
| **Construct Capital** | **$300M Fund III** (2025-03-31), $2–10M seed–A checks, about two dozen companies, "foundational industries": logistics, manufacturing (Veho, Telegraph rail, Hadrian, Chef Robotics). | Seed–A lead | https://www.businesswire.com/news/home/20250331964620/en/ | High |
| **Root Ventures** | Deep-tech seed fund that "leads initial funding for technical founders," $3–5M (its own site per search). Fund IV reportedly $190M (secondary). Dusty Robotics; co-led Volund's $12M seed. | Pre-seed / seed lead | https://root.vc/; https://venturecapitalarchive.com/venture-funds/root-ventures-root-vc | High |
| **Schematic Ventures** | Seed fund for "technology that makes & moves the world." Pre-seed $500K–$1M, seed $1–2M, US/Canada only. Outrider, PlusOne Robotics, Reshape Automation. Fund II May 2025. | Pre-seed / seed | https://www.schematicventures.com/; https://www.drypowder.live/funds/schematic-ventures | High |
| **Trucks VC** | **$70M Fund 3** (Jan 2025): about 30 seed deals, $500K–$2M checks, targets about 10% ownership, leads or co-leads. Gatik; Bear Flag Robotics (sold to Deere). | Pre-seed / seed lead | https://techcrunch.com/2025/01/06/trucks-vc-launches-70m-fund-to-find-the-next-great-batch-of-transportation-startups/ | High |
| **Zacua Ventures** | **$135M Fund II** (2026-11-10, 35% oversubscribed), targeting construction robotics, AI machinery, **mining**, energy. Gravis (pre-A), Origin Robotics (led the A). Co-authored the Construction Robotics Report 2026 (construction robotics raised **$1.36B in Q1–Q3 2025**, per PitchBook). | Pre-seed–A | https://thenextweb.com/news/zacua-ventures-135m-fund-ii-construction-tech (2026-11) | High |
| **Shield Capital** | Focus areas: AI, **autonomy**, cyber, space. Fund I $186M. Led Seasats' $10M round (Feb 2025); co-led UForce's seed. | Seed–A, about $1–15M | https://shieldcap.com/news; https://f4.fund/firms/shield-capital | Medium–high (dual-use) |
| **Align Ventures / Booz Allen Ventures / Draper Associates / Decisive Point** | Scout AI seed and A; Decisive Point also in Intramotev's A. | Seed–A | see Scout AI and Intramotev rows | Medium–high (defense) |
| **Samsara Ventures** (strategic) | Bedrock seed/A; Polymath. Samsara's telematics covers industrial fleets. | Strategic co-investor | https://www.polymathrobotics.com/blog/polymath-robotics-announces-investment-from-samsara-ventures | Medium–high (distribution) |
| **UP.Partners** | Led Teleo's Series A and both extensions. | Seed–A | https://www.prnewswire.com/news-releases/teleo-raises-16-2-million-in-series-a-extension-funds-led-by-uppartners-302313414.html | Medium |
| **Initialized Capital** | Co-led Crewline AI's $7.1M seed (Apr 2026). | Seed | https://www.therobotreport.com/crewline-secures-7-1m-to-automate-constructions-most-repetitive-task/ | Medium |
| **Energy Impact Partners / Obvious Ventures** | EIP: Scythe B, Humble seed. Obvious: led Humble's A. | Seed–A | see rows above | Medium |
| **Y Combinator** | Fall 2026 RFS opens with "AI is moving into the physical world" and asks for field-robot and fleet coordination platforms. S26 reportedly had **24 robotics startups (about 10% of the batch)** (secondary). Polymath is a YC alum. | Pre-seed | https://www.ycombinator.com/rfs; https://www.insidedeeptech.com/inside-y-combinators-2026-robotics-startups/ | Medium (fast pre-seed). Current YC standard deal terms **UNVERIFIED**. |
| Congruent, Fifty Years, Haystack, Lerer Hippeau, Fusion Fund | **Not researched** (search budget exhausted). No portfolio evidence gathered in this session. | — | — | UNVERIFIED |

---

## 4. Investor red flags, and what de-risks a robotics pre-seed or seed

### Red flags (with sources)
1. **Pilot purgatory / "innovation theater."** An IEEE Systems Council investor panel (moderator Andra Keay, Silicon Valley Robotics; Michael Harries, Robotics Hub Fund) called pilots that never convert the "quiet graveyard of hardware startups." The panel advised retiring risks in order: regulatory, then technical, then customer. Source: https://ieeesystemscouncil.org/post/blog/hard-tech-venture-capital-and-systems-engineering (2025/2026).
2. **Free pilots** "fail to show willingness to pay." Investors want a paid pilot or LOI signed by the person who owns the cost. Sources: https://www.22astronauts.com/p/investor-q-and-a-robotics-and-ai; https://valu.vc/robotics-investment/ (2025–2026, investor blogs).
3. **Contracts under 12 months** "read as a glorified pilot." Source: https://hub.causo.ai/guides/raising-seed-robotics-startup-2026 (2026, vendor blog, **UNVERIFIED**).
4. **Hardware capex and owning manufacturing.** Generalist seed VCs want "a software-style revenue model (RaaS) wrapped around the hardware, not a pure capex story." Advice: "skip owning factories early." Sources: Causo (above); https://roboticscfo.substack.com/p/the-robots-didnt-fail-the-capital. Monarch Tractor's collapse after its manufacturing partner left is the cautionary example (TechCrunch 2025-11-19).
5. **AV-trucking fatigue.** Embark: "capital markets have turned their backs on pre-revenue companies" (2023). Locomation's convoy model failed after $105M. 8VC calls broad AVs "crowded and extremely long-term." Kodiak pivoted toward constrained environments. Sources: Section 2c; https://www.8vc.com/theses/logistics; https://itcanthink.substack.com/p/the-state-of-autonomous-trucks-in.
6. **Lab-to-field gap.** Bessemer's "GPT-2.5 moment": capabilities are real but the gap between lab and field is wide (2026-04-16). A **sim-only (CARLA) stack is the biggest diligence gap**; investors will ask for real-vehicle data. Point72's Morales: on-road and enclosed-lot autonomy "will struggle" off-road (2024-05).
7. **Failure causes.** A dataset of 25 robotics startup autopsies: competition 32%, product/tech failure 28%, ran out of cash 28%. Source: https://www.loot-drop.io/deep-dive/robotics (methodology unclear, **UNVERIFIED**).
8. **Enterprise procurement timelines** stall pilot-to-fleet rollouts. Source: https://www.dealmaker.tech/content/robotics-venture-funding-navigating-capital-raising-in-2025 (2025). Also in Robotics CFO.

### De-risking signals (with precedents)
- **Paid pilot → multi-month, multi-vehicle contract.** "Move at least one paid pilot into a multi-robot, multi-month commercial deployment with a signed RaaS contract." Seed robotics "typically needs at least 1–2 pilot deployments or very strong LOIs." Sources: Causo; 22astronauts (2026).
- **Retrofit plus a partner that carries the capex.** ISEE–Holman: the investor-partner leases, retrofits and maintains the trucks (Oct 2026). Pronto–Komatsu: dealer-channel retrofit (Aug 2025). Bluewhite: retrofit kit sold as RaaS. Crewline: about 1-hour install. Bluewhite's founder says a tractor can be made autonomous in about 14 hours (https://agfundernews.com/meet-the-founder-bluewhites-ben-alfi-on-mobility-labor-challenges-and-how-to-make-a-tractor-autonomous-in-14-hours).
- **Anchor enterprise or exclusive agreements.** Pronto–Heidelberg Materials: 100+ trucks, exclusive in North America (Feb 2025). This is the template for an aggregates-major design partner.
- **Independent safety certification.** Outrider (TÜV SÜD, Jul 2025) and Fernride (TÜV SÜD, driverless) both used it to unlock driverless operation and orders.
- **Paid driverless work on live sites.** Bedrock (from Aug 2026) preceded its $4.1B round.
- **Contracted revenue.** Gatik: $600M+ contracted, 85,000 driverless orders (Aug 2026).
- **Government contracts as non-dilutive validation.** Overland's $18.6M OT came before its A. Scout AI had $11M of contracts before its $100M A.
- **Unit economics rules of thumb:** BOM payback under 18 months; gross margin per robot above 50% at scale; a path to 40–60% gross margin. Sources: Causo; 22astronauts. Single-source rules of thumb, **UNVERIFIED**.

---

## 5. Non-dilutive funding paths

> Searches for this section hit the budget limit. Items marked "training knowledge" were not re-verified on 2026-11-11.

### 5a. Defense (most relevant to leader-follower)
- **Army demand signal (VERIFIED).** In June 2023 the Army ended its in-house leader-follower truck kit effort and turned to commercial solutions via the GEARS prototyping competition. Source: https://breakingdefense.com/2023/06/army-closing-down-leader-follower-robotic-truck-development-eyeing-commercial-solutions/. Forterra won GEARS Phase II (Forbes, 2025-11-25).
- **ATV-S (Autonomous Transport Vehicle System) (VERIFIED).** The Army and DIU picked **Carnegie Robotics and Forterra** for prototyping in Feb 2025, with a mid-FY2026 decision point that could narrow to one vendor and fielding "as soon as 2027 if budgets allow." Funding amounts were not disclosed. Source: https://breakingdefense.com/2025/02/army-picks-carnegie-forterra-for-autonomous-logistics-truck-prototyping/; also https://www.fw-mag.com/shownews/433/. Oshkosh received an $89M order for "autonomy-ready" PLS A2 trucks: https://armyrecognition.com/news/army-news/2025/us-army-orders-autonomous-oshkosh-pls-a2-trucks-to-secure-frontline-convoys. **Implication:** a pre-seed company will not win the prime contract. The realistic route is subcontracting or teaming (for example MPC/RL multi-vehicle coordination modules), or entering the next CSO.
- **DIU CSO → prototype OT (VERIFIED precedent).** Overland AI received a two-year, up to $18.6M Army/DIU prototype OT. Source: https://www.overland.ai/news/overland-ai-secures-10m-in-seed-funding-to-accelerate-development-of-autonomous-vehicle-software-for-high-risk-missions (2024-05).
- **DoD SBIR/STTR (Army SBIR, AFWERX Open Topic, Army xTech prize competitions):** **UNVERIFIED (training knowledge).** DoD Phase I awards are typically about $50K–$250K over 6–12 months and Phase II about $0.75M–$2M. Direct-to-Phase-II and AFWERX STRATFI/TACFI matching exist. Army xTech runs cash-prize pitch competitions that can feed into SBIR topics. Check current topics on dodsbirsttr.mil and xtech.army.mil.

### 5b. SBIR/STTR authorization status: **UNVERIFIED**
- Training knowledge: the program was authorized through **2025-09-30** by the SBIR and STTR Extension Act of 2022. Authority **lapsed on 2025-10-01** while Congress disputed reform terms (Sen. Ernst's INNOVATE Act reforms versus a clean extension) during the Oct–Nov 2025 government shutdown. During a lapse, agencies generally cannot issue new solicitations or awards.
- **I could not confirm whether, when, or on what terms the program was reauthorized in 2026.** Check https://www.sbir.gov before planning around SBIR revenue.

### 5c. NSF SBIR/STTR ("America's Seed Fund"): **UNVERIFIED**
- Training knowledge: Phase I up to about **$305,000** (6–12 months); Phase II up to about **$1.25M** (24 months); a Project Pitch is required before a full proposal. NSF accepts robotics and autonomy under its topic areas. Program changes and pauses in 2025–2026 tied to the authorization lapse and NSF restructuring were not checked. Check https://seedfund.nsf.gov.

### 5d. State and foreign grants (VERIFIED precedents)
- Michigan's Office of Future Mobility and Electrification gave Intramotev a **$200,000 grant** to deploy three TugVolt railcars at a mine in the Upper Peninsula. Source: https://www.mining.com/missouri-start-up-gets-200000-grant-to-accelerate-us-autonomous-railcar-technology/ (2024). State mobility offices can fund mine and quarry pilots.
- ARENA (Australia) granted Built Robotics Australia up to **A$13.3M** (some outlets say A$14.5M) for a Brisbane manufacturing hub. Source: https://arena.gov.au/news/robotics-and-software-funding-to-drive-down-solar-costs/ (2026-10). Shows that foreign governments fund automation manufacturing for mining and energy.

### 5e. DOE, MSHA, NIOSH: **UNVERIFIED, low priority**
- DOE Vehicle Technologies Office has historically funded energy-efficient mobility, including platooning fuel savings. Current 2025–2026 funding opportunities were not checked.
- MSHA Brookwood-Sago grants fund mine-safety *training*, not technology development. NIOSH's Mining Program funds research mostly through universities. Training knowledge (not verified) is that 2025 HHS workforce cuts hit NIOSH, including mining research, with partial reversals. Treat NIOSH and MSHA as **partnership and credibility** channels (safety validation, a test mine) rather than founder funding. Check grants.gov.

---

## 6. Stanford ecosystem: **UNVERIFIED (training knowledge)**

> Search budget was exhausted before this section. Everything below is from training knowledge (cutoff June 2026) and should be confirmed.

- **StartX:** nonprofit accelerator for Stanford-affiliated founders (students, alumni, faculty, staff). It historically **takes no equity**. The StartX Fund, with Stanford Management Company, has historically co-invested in StartX companies. Current terms are unverified.
- **Pear VC (PearX):** pre-seed and seed firm with deep Stanford ties, running PearX and on-campus programs. Check sizes and current robotics focus are unverified.
- **Stanford Angels & Entrepreneurs:** alumni angel network useful for filling out a pre-seed round.
- **Student and early programs:** Cardinal Ventures (student accelerator), BASES, Stanford Venture Studio (GSB), and Stanford Technology Ventures Program (Eng).
- **Office of Technology Licensing (OTL):** if any of the MPC/RL convoy work was done using Stanford resources or under a Stanford appointment, clarify IP ownership or a license *before* the pre-seed closes. Investors will check.
- **Defense on-ramps:** Hacking for Defense (H4D, created at Stanford by Steve Blank and others) and the Gordian Knot Center for National Security Innovation. Both connect teams to DoD problem sponsors and are useful for framing Army leader-follower demand.
- **Relevant labs (technical advisors, recruiting, credibility):**
  - Autonomous Systems Lab (ASL, Prof. Marco Pavone): multi-agent planning, AV safety.
  - Dynamic Design Lab (Prof. Chris Gerdes): vehicle dynamics, control at the limits; Gerdes was formerly NHTSA's Chief Innovation Officer.
  - Stanford Intelligent Systems Lab (SISL, Prof. Mykel Kochenderfer): safety validation of autonomous systems, directly useful for a safety case.
  - Center for Automotive Research at Stanford (CARS): industry affiliates program with OEM and supplier access.
  - IRIS Lab (Prof. Chelsea Finn, also a co-founder of Physical Intelligence): robot learning.

---

## 7. Market tailwinds investors cite (2025–2026)

| Claim | Figure | Source / date |
|---|---|---|
| Robotics VC, 2025 full year vs 2026 | **$15B** in 2025; **$18.8B** in 2026 by June, already a record | Crunchbase, https://news.crunchbase.com/robotics/startup-venture-funding-surges-2026-data/ (2026-06). A narrower Crunchbase cut cited elsewhere gives $13.8B for 2025. |
| Robotics VC, Q3 2026 | **$9.9B** vs $4.3B in Q3 2025; Q1–Q3 2026 up about 185% YoY (excludes defense) | Crunchbase, https://news.crunchbase.com/venture/q3-2026-us-china-robotics-startup-funding-jumps-physical-ai-push/ (2026-10) |
| Physical AI VC, H1 2026 | **$47.4B across 521 deals** (robotics, AVs, aerospace, drones, industrial automation, sensors). Waymo's $16B round is about one-third of it. | Crunchbase, https://news.crunchbase.com/venture/physical-ai-funding-startups-robotics-aerospace-h1-2026/ (2026-07) |
| Robotics VC, one month | ≥ **$4.35B across 93 rounds** in July 2025 | The Robot Report, https://www.therobotreport.com/robotics-investments-top-43b-in-july-2025/ (2025-08) |
| Construction robotics VC | **$1.36B** in Q1–Q3 2025 (PitchBook) | Zacua / Hilti Ventures / 94 Ventures Construction Robotics Report 2026, via https://thenextweb.com/news/zacua-ventures-135m-fund-ii-construction-tech (2026-11) |
| US share of robotics VC 2025 | $4.9B of $9.4B global (52%), a US record | https://www.roboticscenter.ai/robotics-market-united-states — **UNVERIFIED** (conflicts with Crunchbase's totals) |
| New dedicated capital | Eclipse $1.3B (2026-04); Lux Fund IX $1.5B (2026-01); Construct Fund III $300M (2025-03); Zacua Fund II $135M (2026-11); Trucks VC Fund 3 $70M (2025-01) | sources in Section 3 |
| Mega-round signal | Bedrock $600M at $4.1B (2026-10); Gravis $200M A (2026-08); Gatik $200M (2026-08); Skild $1.4B at > $14B (2026-01); Waabi $750M (2026-01); Physical Intelligence in talks at about $20B (2026-10) | Section 2 |
| Defense autonomy appetite | Forterra > $1B valuation (2025-11); Overland $100M (2026-02); Scout AI $100M A (2026-04) | Section 2 |
| IIJA (infrastructure demand for aggregates) | About **$1.2T** total, including about $550B in new federal spending, over FY2022–2026 | Statute, Nov 2021. **Training knowledge, not re-verified.** Note the FY2026 end of IIJA authorization: surface transportation reauthorization status in late 2026 is **UNVERIFIED**. |
| Labor shortage narrative | Eclipse cites reshoring of manufacturing, mining and logistics (Bedrock post). Bluewhite and Agtonomy cite farm labor. **No verified 2025–2026 operator-shortage statistic for quarries or mines was gathered.** | https://eclipse.capital/blog/accelerating-industrial-innovation/; https://agfundernews.com/meet-the-founder-bluewhites-ben-alfi-on-mobility-labor-challenges-and-how-to-make-a-tractor-autonomous-in-14-hours — **UNVERIFIED for numbers** |

---

## 8. Recommendations (synthesis; the reasoning is mine, built on the data above)

### Pre-seed (raise now)
- **Size: $1.5M–$2.5M** on a **post-money SAFE** (91% of Q2 2026 SAFEs were post-money).
- **Cap: $12M–$18M post.** Carta Q2 2026 medians are $12.5M for $500K–$1M rounds and $18M for $1M–$2.4M rounds. A sim-only, pre-customer hardware-retrofit company should expect the middle of that band. Push toward **$18M–$25M** only if the team adds a field-robotics or mining-ops co-founder and has signed LOIs or a paid pilot. The $35M median for ≥ $2.5M SAFEs is AI-inflated and is not a realistic anchor for a pre-revenue retrofit company.
- **Dilution: about 10–15%.**
- **Runway:** 18–24 months covering 3–5 engineers plus one field-ops lead, retrofit of 3 vehicles (1 lead + 2 followers) at a design-partner site, a safety case, and SBIR/DIU proposals.
- **Alternatives:** YC or PearX, then a $1.5–2M follow-on.

### Seed (in about 12–18 months)
- **Size: $5M–$8M priced.**
- **Valuation: $22M–$35M post.** Carta's median is $4.1–4.5M at $23.9–24.3M post with 18–19.4% dilution. PitchBook's median is $3.0M on an $18.3M pre. Autonomy comps: Crewline $7.1M (Apr 2026), Overland $10M (May 2024), Scout $15M (Apr 2025). Humble's $24M with Eclipse is an outlier set by its team.
- **Dilution: about 18–22%.**
- Combine with non-dilutive DoD or NSF awards to stretch runway.

### Milestones to show
- **To close the pre-seed:**
  - Real-vehicle demo of the follower controller, even on UTVs or golf carts, with logged data. Moving from CARLA to real vehicles is the first gap to close.
  - 1–3 LOIs from named quarry, aggregates, mine or yard operators, signed by whoever owns haulage cost.
  - A complete team, with a field-ops or heavy-equipment co-founder.
  - A clear wedge, such as pit-to-crusher haul with 1 driver and N trucks.
  - Unit-economics model: kit BOM, RaaS price per truck-hour versus loaded operator cost.
  - Clean IP from Stanford OTL.
  - A clear answer to "why is this not Locomation?": private site, low speed, fixed routes, site-controlled rules.
- **To close the seed:**
  - At least one **paid** pilot converted into a multi-month, multi-vehicle contract (12 months or longer).
  - Hundreds of autonomous follower hours with an intervention-rate trend.
  - Retrofit install time measured in hours.
  - A safety case, ideally with a third-party assessment on the path that Outrider and Fernride took with TÜV SÜD.
  - A second site or customer, or a channel partner (an OEM or dealer like Komatsu–Pronto, or a lessor like Holman–ISEE).
  - At least one DoD award or active DIU/Army engagement (SBIR Phase I/II, xTech, a CSO, or ATV-S teaming).
- **Series A preview:**
  - Driverless followers with no safety operator.
  - Multiple sites.
  - About $1–3M ARR in RaaS.
  - Gross margin trending toward 50% or more.
