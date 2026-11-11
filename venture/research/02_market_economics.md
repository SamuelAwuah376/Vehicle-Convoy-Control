# 02 — Market Sizing & Unit Economics: Leader-Follower Haul Convoys (Private Sites)

Research date: 2026-11-11. Prepared for venture due diligence.

## How to read this file (confidence legend)

- **[PRIMARY]**: figure read directly from a primary document in this session (USGS PDF text extracted locally) or quoted from a primary-source page in search results (USGS, BLS, MSHA, OEM newsroom or earnings call, ATRI, AGC).
- **[SECONDARY]**: from a trade press or analyst summary that cites a primary source. Usually reliable, but I did not open the primary document.
- **[LOW-TRUST]**: market-research aggregator, vendor blog, or job-board data. Directional only.
- **UNVERIFIED**: not confirmed from any source in this session. These figures come from the analyst's background knowledge (cutoff mid-2026) or are my own arithmetic. Re-check before they go in a deck.
- **DERIVED**: my arithmetic on sourced inputs. The inputs are cited and the method is shown.

**Session limits (affects completeness):**
- The sandbox network blocks `pubs.usgs.gov`, `www.bls.gov`, `blsmon1.bls.gov`, `www.usgs.gov` and `mining.cbmo.net`. As a result:
  - USGS MCS chapter PDFs could not be opened. MCS figures come from search-result quotes of those PDFs.
  - The USGS quarterly Mineral Industry Surveys (MIS) PDFs were reachable on the USGS S3 host. They were downloaded and parsed locally, so those figures are [PRIMARY].
  - BLS OEWS industry-specific tables (NAICS 212300, SOC 47-5022) could not be opened.
- The shared web-search budget for this turn ran out partway through the research. These topics got little or no live searching:
  - Topic 7: hardware prices.
  - Topic 8: analyst forecasts.
  - Topic 9: platooning studies.
  - Topic 6, partly: drayage and ATA figures.
- Items in those sections are labelled UNVERIFIED. A follow-up session with `pubs.usgs.gov` and `bls.gov` allowed, and a fresh search budget, should close these gaps.

---

## 1. US (and global) aggregates

### 1a. USGS quarterly MIS (latest revision; supersedes MCS for 2025) [PRIMARY]
Source: USGS Mineral Industry Surveys, *Crushed Stone and Sand and Gravel in the Second Quarter 2026*, published September 2026. Units are **metric tons**.
URL: https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/mis-2026q2-conagg.pdf

| Item (US, calendar 2025) | Quantity | Value | Implied $/t (DERIVED) |
|---|---|---|---|
| Crushed stone | **1,480 Mt** | **$26.9 B** | $18.2/t |
| Construction sand & gravel | **923 Mt** | **$13.1 B** | $14.2/t |
| Total construction aggregates | **2,400 Mt** | **$40.0 B** | $16.7/t |

- The quarterly split of 2025 crushed stone (Mt) was Q1 284, Q2 387, Q3 443, Q4 367. Production is strongly seasonal, with Q3 about 1.56x Q1. This matters for fleet utilization assumptions.
- Q2 2026: crushed stone was 413 Mt (+6.8% YoY); construction sand and gravel was 261 Mt (+4.0%); total aggregates were 674 Mt (+5.7%). For the first half of 2026, crushed stone was up 6.4%, sand and gravel 5.0% and total aggregates 5.9%. 33 of 41 estimated states grew.
- Dollar values are estimated from the 2024 Minerals Yearbook, so they are not 2025 transaction prices.
- Prior years from USGS MIS Q3 2024, 2023 annual [PRIMARY]: crushed stone 1,555 Mt and $24.1 B; construction sand and gravel 967 Mt and $11.4 B; total 2,522 Mt and $35.5 B. Tonnage was roughly flat to down from 2023 to 2025, while value rose about 13%, reflecting price-led growth.
  - URL: https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/mis-2025q3-conagg.pdf. The file is named "2025q3" but contains 2023 annual and 2024 Q1–Q3 data.

### 1b. USGS Mineral Commodity Summaries [SECONDARY: MCS text quoted via search; PDFs blocked]

| Edition (data year) | Crushed stone | Construction sand & gravel |
|---|---|---|
| **MCS 2026 (2025)** | ~1.5 billion t, **$27 B** (one USGS summary gives $26.8 B); **~1,400 companies, 3,500 quarries**, 180+ sales/distribution yards, 50 states. 70% limestone/dolomite, 14% granite, 6% traprock; ~72% used as construction aggregate. Top-5 states (TX, PA, FL, MO, OH…) produce ~56%. | **~870 Mt, $12.6 B; ~3,400 companies, 6,500 pits**, 50 states. TX, CA and AZ produce ~54%. Uses: 42% PCC aggregate, 20% road base, 12% fill, 9% asphalt aggregate. |
| **MCS 2025 (2024)** | ~1.5 billion t, **$26 B**; 1,400 companies; 3,500 quarries; 180+ yards. Crushed stone was 24% of the value of all US nonfuel mineral production. | **~890 Mt, ~$12 B**; 3,400 companies; 6,500 pits. |

MCS URLs:
- https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-stone-crushed.pdf
- https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-sand-gravel.pdf
- https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-stone-crushed.pdf
- https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-sand-gravel.pdf

Note: the 870 Mt (MCS, January 2026) and 923 Mt (MIS, September 2026) sand and gravel figures differ because MIS revisions supersede MCS. Use the MIS figure for 2025.

- **Operations count (DERIVED):** about 3,500 quarries plus 6,500 pits gives roughly **10,000 aggregate operations** run by about 4,800 company-entries. The two company lists overlap, since many firms run both.
- **Average output per site (DERIVED):**
  - Quarry: 1.48 Gt ÷ 3,500 ≈ **420 kt/yr**.
  - Pit: 923 Mt ÷ 6,500 ≈ **140 kt/yr**.
  - These averages are skewed. A few large quarries of 3–10 Mt/yr carry most of the tonnage, and most sites are small.
- **Price per ton:** a pricing blog quotes a USGS 2025 national average of $18.50/metric ton ($16.78/short ton) for crushed stone at the quarry [LOW-TRUST, source: https://americanrocksupply.com/blog/how-much-does-gravel-cost]. That is consistent with the $18.2/t derived above.

### 1c. US industrial sand / frac sand [SECONDARY via MCS 2026 quote]
Source: https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-sand-industrial.pdf
- 2025 US apparent consumption of industrial sand and gravel was **~120 Mt**, down 6% from 2024.
- **~81% of tonnage** was frac sand plus well-packing and cementing sand; 7% went to glass.
- 131 companies ran 207 operations in 38 states. Leading states, in order: TX, WI, OK, LA.
- Exports were ~8 Mt (+4%); imports ~0.21 Mt (−22%).
- Oversupply led to lower prices and idled operations. Texas began taxing frac sand on 2025-07-01.
- **DERIVED:** frac sand ≈ 0.81 × (120 + 8 − 0.2) ≈ **~100 Mt/yr US (2025)**. This is approximate because the 81% share is applied to tonnage.

### 1d. Global aggregates [SECONDARY / mixed]
- GAIN (Global Aggregates Information Network) estimates about **40 billion t/yr**, roughly 5 t per person [GAIN site, https://www.gain.ie/, undated]. A later best estimate is **~50 billion t/yr** (6.5 t per person), with Asia producing about two-thirds [Aggregates Business, J. O'Brien, 2019, https://aggbusiness.com/feature/global-aggregates-growth-examined-gain-convenor-jim-obrien]. GAIN projects 55 Gt by 2030 [Quarry magazine, ~2020, https://www.quarrymagazine.com/global-aggregates-industry-under-the-spotlight/].
- A peer-reviewed estimate puts 2018 global sand and gravel at **33.7 ± 1.3 Gt** [Resources, Conservation & Recycling 2025, https://www.sciencedirect.com/science/article/abs/pii/S0921344925003386].
- On these figures the US produces about 2.4 Gt, roughly 5–6% of world aggregates.

---

## 2. Haul truck populations and autonomous fleet counts

### 2a. Installed base [SECONDARY / LOW-TRUST]
- **Parker Bay Company**, a mining equipment database (subscription only; pages undated):
  - ~**66,000 active haul trucks ≥90 t payload** worldwide, rear- and bottom-dump.
  - ~86,000 including inactive units.
  - The full mobile-equipment database lists 139,700 units in 124 countries.
  - The largest machines work at about 1,700–1,760 mines.
  - Source: https://parkerbaymining.com/ and https://parkerbaymining.com/mining-equipment/mining-trucks
- Parker Bay index: **2025 mining-truck deliveries fell 9% in units and 11% in value versus 2024**, and surface-equipment unit shipments fell 6% [via https://parkerbaymining.com/industry-information/surface-mining-equipment-index.htm, 2025–26].
- **Off-Highway Research:** worldwide sales of rigid plus articulated dump trucks peaked at **>16,000 units in 2023**, a 15-year high [Power Progress, ~2024, https://www.powerprogress.com/news/dump-truck-developments/8038993.article] [SECONDARY].
- **ADT only:** ~9,950 units produced globally in 2024 [Global Info Research, https://www.marketresearch.com/GlobalInfoResearch-v4117/Global-Articulated-Dump-Trucks-Manufacturers-45408542/] [LOW-TRUST].
- **Mining rigid trucks:** ~11,200 units delivered in 2024, up from 9,750 in 2023; China >2,900 [aggregator sites] [LOW-TRUST, and inconsistent with OHR's 16k combined total].
- **US-specific installed base** of rigid trucks and ADTs: **NOT FOUND** in this session. AEM and OHR hold this data behind paywalls.

### 2b. Bottom-up estimate of the US quarry haul-truck fleet (DERIVED, UNVERIFIED assumptions)
- **Assumptions:**
  - 775-class truck with ~54 t payload.
  - ~5 loaded cycles per hour on a 0.5–1.5 km quarry haul.
  - ~2,500 productive hours per year (mostly single shift plus overtime).
  - That gives ~0.68 Mt per truck per year.
- **Cross-check:** Luck Stone's Bull Run quarry hauled ~2 Mt in the first year with a small fleet of autonomous Cat 777s, described as "four trucks" in a Caterpillar anecdote. That implies ~0.5 Mt per truck per year. See §2c.
- **Calculation:** 1.48 Gt ÷ 0.5–0.68 Mt ≈ **2,200–3,000 truck-equivalents for crushed stone face-to-crusher haul**.
- **Additions:** overburden and stripping, plus sand-and-gravel pits that use trucks or ADTs instead of loaders and conveyors. These plausibly put the US aggregates haul fleet at **~3,000–6,000 rigid trucks and ADTs**.
- **Implication:** the US quarry haul fleet is a small, concentrated market. Most quarries run 2–6 haul trucks, so a convoy of one leader plus N followers needs the large quarries (≥1.5–2 Mt/yr). **Treat the whole of §2b as UNVERIFIED** until AEM or dealer population data is obtained.

### 2c. Autonomous haul trucks deployed [PRIMARY where OEM; SECONDARY for GlobalData]

| Source | Count | Date | URL |
|---|---|---|---|
| **GlobalData Mining Intelligence Center**: all autonomous plus autonomous-ready haul trucks on surface mines worldwide | **3,832** (China 2,090, up from 562 in 2024; Australia 1,024, up from 927; Canada 344; Chile 208). Cat, Komatsu, Tonly and LGMG supply 88%. | Jul 2025 | https://www.mining-technology.com/analyst-comment/china-global-leader-autonomous-surface-mining-trucks/ |
| GlobalData, *Development of Autonomous Trucks in the Global Mining Sector, 2026* | **5,684** (China 3,843). OEM ranking: Tonly #1, LGMG #2, Komatsu #3, Cat #4. | Jul 2026 | https://www.globaldata.com/store/report/development-of-autonomous-trucks-in-mining-market-analysis/ |
| **Komatsu FrontRunner** | 900+ commissioned | Oct 2025 | https://www.komatsu.com/en-us/technology/smart-mining/loading-and-haulage/autonomous-haulage-system |
| Komatsu newsroom | **1,000th ultra-class AHS truck** (930E-5AT at Nevada Gold Mines); >11.5 Bt hauled cumulatively; >500 autonomous 930Es | 2026-04-22 | https://www.komatsu.com/en-us/newsroom/2026/komatsu-becomes-first-oem-to-commission-1000-ultra-class-autonomous-haul-trucks |
| Komatsu (2024) | 700+ trucks at 23 sites in 5 countries; 7.5 Bt hauled | 2024 | https://www.komatsu.com/en-us/newsroom/2024/komatsu-achieves-major-autonomous-milestones |
| **Caterpillar** (Q4 2025 earnings call) | **827** autonomous trucks at end-2025, up from 690 at end-2024 (+20%) | 2026-01-29 | https://s25.q4cdn.com/358376879/files/doc_financials/2025/q4/updated/4Q-2025-Caterpillar-Inc-Earnings-Conference-Call_Transcript.pdf |
| Caterpillar Investor Day | Aims to roughly triple the fleet by 2030 (>2,000); Vale Carajás to have ~90 autonomous trucks by 2028 | 2025-11-04 | https://im-mining.com/2025/11/07/caterpillar-sets-out-to-hit-over-2000-autonomous-mining-trucks-by-2030/ |
| China, vendor claim (EACON) | ">4,000 autonomous mining trucks in China by end-2025" [LOW-TRUST] | 2026-04 | https://www.eaconmining.com.au/content/details126_493.html |
| Hitachi, Liebherr, Epiroc | **NOT FOUND** this session (search budget exhausted) | n/a | n/a |

- **Global autonomous growth (DERIVED):** +48% from July 2025 to July 2026. Growth is mostly in China (+84%), where Chinese OEMs such as Tonly and LGMG retrofit or build lower-cost wide-body trucks.

**Direct competitive signal, Cat in quarries [PRIMARY: Caterpillar press]:**
- Cat 777 autonomous haulage went live at Luck Stone's Bull Run Quarry (Chantilly, VA) in **November 2024**. It was the first autonomous deployment in the aggregates industry.
  - Source: https://www.caterpillar.com/en/news/corporate-press-releases/h/autonomous-truck-luck-stone.html
- The site passed **1 Mt hauled by July 2025** and **>2 Mt in the first year**.
  - Source: https://www.cat.com/en_US/news/machine-press-releases/caterpillar-and-luck-stone-celebrate-one-million-tons-hauled-autonomously-at-bull-run-quarry.html
- In September 2026 the program expanded to Luck Stone's **Boscobel and Bealeton** quarries. These are the first deployments of Cat AHS on the **775** platform, which Cat describes as the key quarry size.
  - Sources: https://www.metaltechnews.com/story/2026/09/16/mining-tech/caterpillar-expands-va-quarry-autonomy/2953.html and https://roboticsandautomationnews.com/2026/11/10/luck-stone-builds-on-autonomous-hauling-success-with-caterpillar/106901/
- Total hauled reached **>3.5 Mt** by November 2026.
- Caterpillar cites labor as the driver: "if one of four trucks is down for lack of an operator, that cuts a fourth of production."

**Conclusion:** the incumbent OEM is actively moving downmarket into 40–90 t quarry trucks. That is the founder's beachhead.

---

## 3. Haul cost economics

### 3a. Haulage share of mine operating cost [SECONDARY / LOW-TRUST]
- Truck haulage is commonly cited at **~50% of open-pit operating cost**. One source puts waste haulage at "up to 50%" [MDPI Applied Sciences 2025, https://doi.org/10.3390/app152413148]. An AusIMM APCOM 2011 truck-dispatch paper says "material transportation represents 50 per cent of the operating costs for an open pit mine" [https://www.ausimm.com/publications/conference-proceedings/35th-apcom-symposium-2011/a-practical-approach-to-truck-dispatch-for-open-pit-mines/].
- An open-pit copper case study ranks haulage third, behind milling and crushing [MDPI Appl. Sci. 15(3):1602, 2025, https://www.mdpi.com/2076-3417/15/3/1602].
- Fuel can be up to **22% of truck cost** [ScienceDirect, https://www.sciencedirect.com/science/article/abs/pii/S2213138820314053].
- **Caveat:** these are metal-mine figures. A quarry has no mill, so hauling is a large share of quarry opex, but the gate value of aggregate is small (~$14–18/t).

### 3b. Truck hourly cost [SECONDARY]
- **Costmine (InfoMine) Equipment Cost Service:**
  - A **100-ton haul truck cost US$175/hr to run in 2022**, up from US$108/hr in 2015.
  - The truck's purchase price was about US$1.6 M and barely changed over that period.
  - Workforce costs are **~43% of surface-mine operating costs**, and oil-based products (fuel, explosives, tires) are just under half.
  - Sources: Costmine via North Shore News / Fluid Life, June 2023, https://www.nsnews.com/resources-agriculture/labour-costs-will-soon-beat-oil-as-mines-biggest-expense-new-data-indicate-7105938 and https://www.fluidlife.com/blog-mine-expenses-labor-overtaking-oil-biggest-expense/
  - UNVERIFIED: to my knowledge Costmine's "operating cost" excludes operator wages and ownership (depreciation and interest). Confirm before relying on it.
- A 150–240 t rigid truck runs at more than $250/hr owning and operating cost. Tires cost $20–45k each, and a set for an ultra-class truck can reach ~$600k [heavyvehicleinspection.com, https://heavyvehicleinspection.com/blog/post/off-highway-truck-cost-per-ton] [LOW-TRUST].
- A per-truck percentage breakdown (operator, fuel, tires, maintenance, depreciation) from a primary source was **NOT FOUND**.

**Illustrative 100 t truck stack, 2022–25 US dollars (DERIVED, UNVERIFIED):**

| Line item | $/hr | Basis |
|---|---|---|
| Operating cost | 175 | Costmine 2022: fuel, tires, parts, repair labor, lube |
| Ownership | ~55–70 | ~$1.6 M over a ~30,000 h life, plus interest |
| Operator | ~39–40 | §4 median wage × 1.42 benefit load |
| **Total** | **~$270–285/hr** | |

- On this stack the **operator is ~14%** of truck cost per hour.
- For a smaller 40–60 t quarry truck or an ADT, the total is plausibly $130–200/hr while the operator cost is the same, so the **operator share is ~20–30%**.
- Trade press claims labor is "nearly 40%" of traditional hauling opex [Skillings 2026, https://www.skillings.net/autonomous-haul-trucks-cost-savings-fleet-optimization-and-2026-outlook/] [LOW-TRUST].
- **Key implication:** the value of leader-follower depends as much on recovering production lost when operators are absent, and on extra shifts, as on wage savings.

### 3c. Haul cost per ton (DERIVED, UNVERIFIED)
- A 100 t truck carries ~91 t per load × ~4 cycles/hr ≈ 364 t/hr. At $270–285/hr that is **~$0.75–0.80/t haul-only for a ~1-mile cycle**, and **~$0.10–0.11/t of it is operator labor**.
- Adding loading cost, quarry **load-and-haul is plausibly $1.0–2.0/t, roughly 6–12% of the $16–18/t gate value**. No primary benchmark was found.
- For context:
  - Aggregate road freight is $0.10–0.20/ton-mile [Steel Wheel Logistics blog, https://steelwheellogistics.com/blog/aggregate-construction-materials-rail] [LOW-TRUST].
  - NSSGA reportedly puts transport at 50–70% of delivered aggregate cost [same source; UNVERIFIED against NSSGA].
  - Aggregate operators' margins are "$0.50–2/ton" [LOW-TRUST].

### 3d. Autonomy benefit claims [PRIMARY: vendor claims]
- Komatsu reports an average **40% improvement in tire and brake life** and a **13% reduction in maintenance** for trucks operating within the design envelope [https://www.komatsu.com/en-us/technology/smart-mining/loading-and-haulage/autonomous-haulage-system].
- "Up to 15% cost savings" is attributed to Komatsu by an academic review [MDPI 2025, https://www.mdpi.com/2673-6489/6/3/61] [SECONDARY].
- Trade press cites a **15–30% reduction in load-and-haul cost** and **15–20% productivity gains** from mature AHS [Skillings 2026] [LOW-TRUST].
- Ultra-class autonomous trucks cost **US$5.5–9 M each**, with a **36–48 month payback** [Skillings] [LOW-TRUST].

### 3e. Typical quarry haul cycles [LOW-TRUST / DERIVED]
- Loaded trucks run at 15–30 km/h on 8–15% grades and at 30–45 km/h empty. Loading takes 3–5 min (5–8 passes) and dumping 1–2 min [https://onlineminingexam.graphy.com/blog/truck-haulage-surface-mines-selection-cycle-time-output-dgms] [LOW-TRUST].
- A 1-mile (1.6 km) haul therefore gives a **cycle of roughly 10–18 min** (DERIVED), or about 3–6 cycles per hour.
- A worked fleet-sizing example (90 t trucks): spot 0.5 + queue 1.5 + load 3 + haul 9 + dump 1 + return 6.5 = **21.5 min** [SAIMM J. v79n07, https://www.saimm.co.za/Journal/v079n07p185.pdf].
- The haul and return legs are about 65–80% of cycle time [vendor claim, LOW-TRUST].
- **Convoy relevance:** at 10–30 km/h a leader-follower convoy must also handle single-lane loading and dumping, with followers queuing at the loader. Convoy productivity loss from queuing is a key ROI risk to model.

---

## 4. Labor

### 4a. BLS OEWS wages [SECONDARY: BLS text quoted via search; bls.gov blocked]
- **Heavy and tractor-trailer truck drivers (SOC 53-3032), all industries:**
  - **May 2025 median $58,640/yr** (≈ **$28.19/hr** DERIVED at 2,080 h). Bottom 10% earn under $40,140 and top 10% over $79,380 [BLS OOH, https://www.bls.gov/ooh/transportation-and-material-moving/heavy-and-tractor-trailer-truck-drivers.htm].
  - **May 2024 median $57,440/yr (≈ $27.62/hr)**, with bottom 10% under $38,640 and top 10% over $78,800 [BLS OOH, May 2024 edition].
  - May 2024 mean $58,240/yr ($28.00/hr) [TruckRadar citing OEWS, https://truckradar.ai/driver-salary] [SECONDARY].
- **Excavating & loading machine and dragline operators, surface mining (SOC 47-5022, formerly 53-7032):** May 2025 median **$57,430/yr (≈ $27.61/hr)** [WageDex citing OEWS May 2025, https://wagedex.com/card/excavating-and-loading-machine-and-dragline-operators-surface-mining] [SECONDARY]. Mean and May 2024 values: **NOT RETRIEVED**.
- **Industry-specific wages for NAICS 212300 (nonmetallic mineral mining and quarrying), truck drivers and loader operators:** **NOT RETRIEVED** because the BLS page was blocked. Fetch https://www.bls.gov/oes/current/naics4_212300.htm in a follow-up.
- **Mine shuttle car operators (SOC 53-7111):** **NOT RETRIEVED**. This occupation is underground coal and has low relevance.
- **Job-board quarry truck driver pay:** average **$27.09/hr** with a range of $18–40/hr (May 2026) [ZipRecruiter, https://www.ziprecruiter.com/Jobs/Quarry-Truck-Driver] [LOW-TRUST].

### 4b. Benefit load (BLS ECEC) [PRIMARY: BLS release quoted]
- **June 2025, private industry:** total compensation **$45.65/hr worked**, of which wages are $32.07 (70.2%) and benefits $13.58 (29.8%). The **load factor is 1.42×** (DERIVED).
  - Sources: https://www.bls.gov/news.release/archives/ecec_09122025.htm and https://www.bls.gov/opub/ted/2025/employee-benefits-cost-employers-13-58-per-hour-for-private-industry-workers-in-june-2025.htm (release 2025-09-12).
- **June 2026, private industry:** wages $32.82 and benefits $14.07, a **1.43× load** (DERIVED).
- A goods-producing or transport-specific ECEC breakdown was **NOT RETRIEVED**. Goods-producing benefit shares are typically somewhat higher than the private average (UNVERIFIED).
- **Fully loaded operator cost (DERIVED):** $28.19 × 1.42 ≈ **$40/hr**, or **~$83k per 2,080-h shift-year**.
  - Covering one seat 24/7 takes ~4.2 FTE, about **$350k/yr**.
  - Two shifts take ~2 FTE plus relief, about **$170–190k/yr**.
  - These figures exclude overtime, recruiting and training (MSHA Part 46/48 new-miner training).

### 4c. Shortage and turnover evidence
- **AGC 2025 Workforce Survey (national):** **77%** of firms with openings had difficulty filling **crane and heavy-equipment operator** positions (n = 508 firms). Overall, **~92%** of firms had difficulty filling craft positions, and **45%** reported project delays from shortages [PRIMARY, https://www.agc.org/sites/default/files/users/user21902/2025_Workforce_Survey_National_FINAL_m.pdf, published ~Aug 2025]. AGC covers construction, not aggregates producers.
- **SME:** **more than half of the US mining workforce, ~221,000 workers, will retire by 2029.** Mining needs ~11,000–13,000 hires per year [SME Technical Briefing, https://www.smenet.org/What-We-Do/Technical-Briefings/Workforce-Trends-in-the-US-Mining-Industry] [SECONDARY].
- **Caterpillar and Luck Stone** say labor shortage drove the quarry autonomy program (see §2c).
- **Aggregates-specific survey** (NSSGA or Pit & Quarry) with a quantified shortage: **NOT FOUND**. **Haul-truck operator turnover rate: NOT FOUND.**
- **Equipment Insider 2025 Operator Workforce Study** (847 contractors, 1,423 operators): nearly a quarter of operators are within 10–15 years of retirement [https://www.equipmentinsiderhq.com/posts/2025-10-28-operator-workforce-study-2025/] [LOW-TRUST, AI-assisted publication].

### 4d. Australian FIFO context [LOW-TRUST: job ads and guides]
- Pilbara haul truck operators earn **A$50–58/hr** at BHP, Rio Tinto and FMG sites, and ads offer **A$56–63+/hr**. Annual base is typically **A$110k–150k**, and allowances, accommodation and flights can add 50–70% to package value [https://gofifo.com.au/fifo-salary-guide/; https://www.glassdoor.com.au/Salaries/haul-truck-operator-salary-SRCH_KO0,19.htm (national average A$118.5k, A$57/hr, n = 24)].
- Fully loaded FIFO cost per seat is roughly 2× US quarry levels (DERIVED, UNVERIFIED FX ~0.65 USD/AUD). This is why AHS scaled first in Western Australia.

---

## 5. Safety (MSHA)

### 5a. Powered haulage share of fatalities

| Year | Total mining fatalities | Powered haulage | Share | Source |
|---|---|---|---|---|
| 2017 | n/a | n/a | **50%** | EHS Today citing MSHA, https://www.ehstoday.com/safety/article/21919622/msha-promotes-powered-haulage-safety [SECONDARY] |
| Jan–Apr 2018 | n/a | n/a | **57%** | same |
| 2014–~2020 (cumulative) | n/a | n/a | **34%** | MSHA/DOL Agency Priority Goal FY2021, https://trumpadministration.archives.performance.gov/labor/FY2021_january_Worker_Safety_Reduce_Miner_Injuries.pdf [PRIMARY-ish] |
| 2018–2023 by year | **NOT RETRIEVED** (search budget) | n/a | n/a | n/a |
| 2024 | **28** | **12** | **43%** | Pit & Quarry, Jan 2025, https://www.pitandquarry.com/mine-fatalities-near-all-time-low-as-2024-approaches-finish/ [SECONDARY]. NIOSH/PMC: 7 of 18 M/NM (38.9%) and 5 of 10 coal (50%), https://pmc.ncbi.nlm.nih.gov/articles/PMC11790053/ |
| **2025** | **33** | **13** | **39%** | **MSHA Powered Haulage page**, https://www.msha.gov/safety-and-health/safety-and-health-initiatives/powered-haulage-safety [PRIMARY]. Powered haulage also caused 426 of 4,792 non-fatal injuries (9%). |
| 2026 YTD (to Oct 5) | 26 | 7 | 27% | MSHA fatality alert, Oct 5, 2026 (front-end loader struck a miner), https://www.msha.gov/data-reports/fatality-reports/2026/october-5-2026-fatality/fatality-alert [PRIMARY] |
| 1983–mid-2025 (cumulative) | n/a | 933 | **30%** (largest category) | miningincidents.org analysis of MSHA 7000-1 data, https://miningincidents.org/research/powered-haulage-has-killed-7-times-as-many-miners-as-gas-or-dust-explosions-since-1983 [SECONDARY] |

- MSHA's "powered haulage" category covers mobile equipment, conveyors, and anything else under power that hauls people or materials. Not every powered-haulage death is a haul truck.
- **Usable claim:** powered haulage is the **#1 cause of US mining deaths, at roughly 30–50% of fatalities in most years from 2017 to 2025.**

### 5b. Safety Program for Surface Mobile Equipment rule (RIN 1219-AB91) [PRIMARY / SECONDARY]
- **Timeline:** the final rule was published **2023-12-20**, took effect **2024-01-19**, and has required **compliance since 2024-07-17**.
  - Sources: Federal Register, https://www.federalregister.gov/documents/2023/12/20/2023-27640/safety-program-for-surface-mobile-equipment and MSHA, https://www.msha.gov/final-rule-safety-program-surface-mobile-equipment
- **Requirement:** a written safety program for surface mobile equipment, excluding belt conveyors, at **all surface mines, including those with ≤5 miners**. Programs must cover hazard identification and evaluation, maintenance, training, and *consideration of* technologies such as collision warning and avoidance.
- **Enforcement:** MSHA issued **863 citations under the rule between July 2024 and April 2026** [Ogletree, July 2026, https://ogletree.com/insights-resources/blog-posts/trends-in-msha-surface-mobile-equipment-compliance/] [SECONDARY].
- **Status:** no delay or rescission was found in this session. Re-check the Federal Register and RIN 1219-AB91 for 2025–26 deregulatory actions.
- **Relevance:** the rule gives buyers a compliance reason to adopt collision-avoidance and autonomy technology, even though it does not mandate it.

---

## 6. Other repetitive-route markets

### 6a. Permian frac sand
- US frac sand is roughly **100 Mt/yr** (2025, DERIVED from USGS, §1c).
- Estimates of the Permian share conflict:
  - 25–35 Mt/yr, a blog citing EIA and Rystad.
  - 70–90 Mt/yr, the same blog.
  - About 40–50% of a 65–75 Mt US market.
  - All from https://sisuenergyllc.com/frac-sand-hauling-permian-basin-owner-operator-guide-2026/ [LOW-TRUST].
  - **UNVERIFIED.** Commonly cited industry estimates for the Permian are about 50–60 Mt/yr (2023–25), but this is not confirmed.
- **In-basin sand is now ~50–70% of Permian supply** [AOGR, https://www.aogr.com/magazine/frac-facts/permian-driving-frac-sand-supply-shift] [SECONDARY].
- **Last-mile carrier pay:**
  - <50 mi: $15–22/t.
  - 50–150 mi: $20–30/t.
  - >150 mi: $25–40+/t.
  - Loads are about 25 t (120,000 loads per 3 Mt/yr mine) [Sisu Energy blog, 2026] [LOW-TRUST].
  - An older IHS estimate put delivered Texas brown sand at about $78/t (~2018) [LOW-TRUST, dated].
- **Atlas Energy Dune Express:**
  - A 42-mile conveyor that cost about $400 M, with capacity of about **13 Mt/yr** [Houston Chronicle, https://www.chron.com/news/houston-texas/article/dune-express-west-texas-oil-20153692.php].
  - **Kodiak driverless trucks** haul sand on private lease roads.
  - On the Q3 2026 call, Atlas guided to **~45 Kodiak trucks by year-end 2026 and 100 by mid-2027**. Public-road operation is expected around mid-2027 [Motley Fool transcript, 2026-11-03, https://www.fool.com/earnings/call-transcripts/2026/11/03/atlas-energy-aesi-q3-2026-earnings-call-transcript/] [SECONDARY].
  - **This is a direct, funded competitor for private-road repetitive hauling.**

### 6b. Yard trucks / terminal tractors [SECONDARY / LOW-TRUST]
- The US yard tractor stock is about **62,000 units** (attributed to CALSTART). About 2.8% of it is zero-emission. 519 new ZE units were deployed in 2025.
- About **5,000 new terminal tractors are produced in a normal year**, and perhaps 3,000 in 2026 (NACFE consultant).
- Orange EV expects about 1,000 North American deliveries in 2026.
- Sources: CALSTART, 2026-10-09, https://calstart.org/the-argument-left-in-the-yard-is-financial-act-news/; NACFE Terminal Tractor Report, May 2024, https://nacfe.org/wp-content/uploads/2024/05/Terminal-Tractor-Report-FINAL_compressed.pdf
- The CALSTART-attributed figures are internally inconsistent: 519 units described as "3%" would imply a base of about 17k. **Verify.**
- Analyst terminal-tractor market sizes range from $0.8 B to $14.7 B [LOW-TRUST, ignore].
- Autonomous yard players include Outrider, plus OEM programs (UNVERIFIED). Their pricing was **NOT FOUND**.

### 6c. Port drayage
- **NOT RETRIEVED** (search budget exhausted).
- **UNVERIFIED:**
  - The Port Drayage Truck Registry at San Pedro Bay (POLA/POLB) lists on the order of 15,000–20,000+ trucks.
  - US drayage drivers number on the order of tens of thousands.
  - Do not use these until verified.

### 6d. Truck driver shortage and cost per mile
- **ATRI, *An Analysis of the Operational Costs of Trucking: 2026 Update* (2025 data, published July 2026):**
  - Total cost was **$2.336/mile**, a record and up 3.4%.
  - **Driver wages were $0.818/mi**, up from $0.798.
  - **Driver benefits were $0.210/mi**, up from $0.197.
  - **Driver total is $1.028/mi, 44% of cost** (DERIVED).
  - The 2024-data report gave driver compensation of $0.995/mi, 43.8% of cost.
  - Sources: https://truckingresearch.org/2026/07/new-atri-report-details-accelerating-costs-and-low-profitability-despite-cuts/ and https://www.ttnews.com/articles/atri-truck-costs-2025 [PRIMARY/SECONDARY]
- **ATA shortage estimate:**
  - About **60,000 drivers (2023)**. ATA reportedly stopped publishing an annual figure after that [secondary, https://boostpoint.com/industries-trucking/truck-driver-shortage/] [SECONDARY].
  - UNVERIFIED from memory: ATA's earlier figures were about 80,000 (2021), projected to 160,000 by 2030.
- **ATRI Top Industry Issues 2025:** "driver shortage" fell to **#12**, out of the top 10 for the first time in 21 years [CCJ, https://www.ccjdigital.com/regulations/safety-compliance/article/15770320/trucking-has-a-quality-driver-shortage-not-quantity].
  - The headline "shortage" narrative is weaker in for-hire trucking than in mining and heavy equipment.
- **IRU global:** about 2.9 M unfilled driver positions across surveyed markets [DC Velocity, https://www.dcvelocity.com/supply-chain/other-services/labor/iru-report-finds-global-truck-driver-shortage] [SECONDARY].

---

## 7. Autonomy hardware costs, 2025–2026 (**UNVERIFIED**: search budget exhausted before this topic; background knowledge only)

| Component | Indicative price | Notes |
|---|---|---|
| Hesai ATX (automotive long-range lidar) | ~US$200 target at OEM volume (announced Apr 2024) | Hesai's 2024 blended ASP ≈ RMB 2.08 B revenue ÷ ~502k units ≈ RMB ~4,100 (~US$570). UNVERIFIED |
| Hesai AT128 | ~US$400–500 OEM ASP (2023–24) | UNVERIFIED |
| RoboSense MX / M-series | "<US$200" (MX, 2024 announcement) | UNVERIFIED |
| Ouster OS0/OS1/OS2 (industrial/robotics) | list ~US$6k–20k+, by channel count and range | Industrial-grade with IP69K options. Better fit for ruggedized off-highway use than automotive units. UNVERIFIED |
| Luminar Iris | ~US$500–1,000 OEM target | **Company reportedly filed Chapter 11 in Dec 2025.** Supplier risk. UNVERIFIED |
| 4D imaging radar | ~US$100–300 automotive volume; ~US$1–5k ruggedized industrial | UNVERIFIED |
| RTK GNSS | module (u-blox ZED-F9P class) ~US$200–300; industrial dual-antenna machine-control receivers ~US$3k–15k; RTK correction service ~US$0.5–1.5k/yr | UNVERIFIED |
| NVIDIA Jetson AGX Orin 64GB | dev kit US$1,999; module ~US$1.6–1.8k | UNVERIFIED |
| NVIDIA Jetson AGX Thor | dev kit **US$3,499**; T5000 module ~US$2,999 at 1k qty (launched Aug 2025) | UNVERIFIED |
| Heavy-equipment drive-by-wire retrofit | **No public price found** | OEM "autonomy-ready" trucks (Cat 777/775, Komatsu 930E-AT) bundle this. Aftermarket vendors (Pronto, which reportedly acquired SafeAI in 2025; ASI/Epiroc; Teleo) do not publish prices. UNVERIFIED |
| Vendor RaaS / per-hour / per-ton pricing | **No public price found** | Kodiak sells "driver-as-a-service" per truck to Atlas; terms not retrieved. UNVERIFIED |

**Illustrative follower-kit BOM (DERIVED from UNVERIFIED inputs):**

| Item | Cost |
|---|---|
| 2–3 lidar (ruggedized) | $3k–30k |
| 4–6 radar | $1k–15k |
| Dual RTK GNSS | $3k–10k |
| Compute (Thor-class, rugged enclosure) | $5k–10k |
| V2V radio / mesh | $2k–8k |
| DBW actuation and integration for a mechanical-drive 40–100 t truck | $20k–60k |
| **Total** | **~$35k–130k hardware per follower** |

The leader kit is lighter because it needs no DBW.

---

## 8. Market forecasts (**LOW-TRUST**; little retrieved)
- **No analyst AHS market-size figure was verified this session.** Aggregator reports exist, for example Dataintelo's *Autonomous Haulage System Market* (https://dataintelo.com/report/autonomous-haulage-system-market) and Research&Markets/GlobalData's *Development of Autonomous Trucks in the Global Mining Sector 2025/2026*. Their headline numbers were not captured. UNVERIFIED, from background: AHS market figures typically circulate at about US$1–3 B for 2023–25 with 6–15% CAGRs. **Do not cite them.**
- **Better proxies (sourced):**
  - Installed autonomous trucks grew from **3,832 (Jul 2025) to 5,684 (Jul 2026)**, +48% YoY [GlobalData].
  - Cat targets **>2,000 by 2030** [Cat Investor Day 2025-11-04].
  - Komatsu passed **1,000 in April 2026**.
  - Against about 66,000 active ≥90 t trucks, **autonomy penetration is ~6–9% of the large-truck base** (DERIVED).
  - Penetration in the <90 t quarry and ADT segment is near zero apart from Cat/Luck Stone, Pronto and Chinese players.
- Articulated dump truck market value for 2025 is cited at **$4.3–7.8 B** depending on the publisher [LOW-TRUST aggregators, §2a].

---

## 9. Platooning fuel savings (**UNVERIFIED**: search budget exhausted; recall of the published studies, citations given for follow-up)

| Study | Setup | Lead | Follower | Team | Notes |
|---|---|---|---|---|---|
| **NREL / Peloton** (Lammert et al., SAE 2014-01-2438) | 2 Class 8 trucks, 65,000 lb, 55–70 mph, gaps 20–75 ft | up to ~5.3% | up to ~9.7% | ~3.7–6.4% | Savings rose with speed; follower savings were not monotonic at the shortest gaps. UNVERIFIED |
| **Auburn Univ. / FHWA EAR** (Bevly et al. 2015, Phase One) | 2 trucks, ~65 mph, 30–75 ft | ~4–5% | ~10% | ~6–7% | At very short gaps, the follower's engine cooling-fan demand eroded savings. UNVERIFIED |
| **NREL / Volvo / PATH CACC** (McAuliffe et al., SAE 2018-01-1181) | 3 trucks, up to 65 mph, time gaps ~0.1–1.5 s | low single digits | middle and trailing ~8–14% | ~5–8% | UNVERIFIED |
| **California PATH** (Browand et al., 2004) | 2 trucks, 55 mph, 3–10 m | ~5% | ~10% | n/a | UNVERIFIED |

**Low-speed falloff (DERIVED, physics):**
- Aerodynamic drag scales with v². Relative to 105 km/h (65 mph), drag is about 15% at 40 km/h and about 6% at 25 km/h.
- At highway speed, aero is about half of a Class 8 truck's tractive energy. In a quarry at 10–35 km/h on 8–15% grades, grade and rolling resistance dominate.
- **Expected platooning fuel benefit in quarry or mine hauling is therefore ≈0–1%.** It is negligible, and **the ROI case must rest on labor and productivity, not fuel.**
- Fuel savings only matter for highway-speed convoys such as long private haul roads or future on-road use.

---

## 10. Inputs for the bottom-up TAM and per-truck ROI

| Parameter | Value | Confidence |
|---|---|---|
| US aggregates tonnage (2025) | 2.40 Gt ($40.0 B): crushed stone 1.48 Gt ($26.9 B), sand & gravel 0.92 Gt ($13.1 B) | PRIMARY (USGS MIS, Sep 2026) |
| US aggregate operations | ~3,500 quarries (1,400 cos) + ~6,500 pits (3,400 cos) | SECONDARY (MCS 2026) |
| Average quarry output | ~420 kt/yr | DERIVED |
| US quarry haul trucks and ADTs | ~3,000–6,000 | DERIVED / UNVERIFIED |
| Global active ≥90 t haul trucks | ~66,000 | SECONDARY (Parker Bay) |
| Global autonomous haul trucks | 5,684 (Jul 2026), +48% YoY | SECONDARY (GlobalData) |
| Cat / Komatsu autonomous | 827 (end-2025) / 1,000 (Apr 2026) | PRIMARY |
| Operator wage (US) | ~$27.6–28.2/hr median (HTT driver 2024/25; surface-mining loader operator 2025) | SECONDARY (BLS via OOH/aggregator) |
| Benefit load | 1.42–1.43× | PRIMARY (BLS ECEC) |
| Fully loaded operator cost | ~$40/hr; ~$83k per shift-seat-year; ~$350k per 24/7 seat | DERIVED |
| Truck operating cost (100 t) | $175/hr ex-operator, ex-ownership (2022) | SECONDARY (Costmine) |
| Operator share of truck $/hr | ~14% (100 t) to ~20–30% (40–60 t, ADT) | DERIVED / UNVERIFIED |
| Haul cost per ton (quarry) | ~$0.5–0.8/t haul; ~$1–2/t load-and-haul | DERIVED / UNVERIFIED |
| AHS side benefits (vendor) | +40% tire/brake life, −13% maintenance (Komatsu) | PRIMARY (vendor claim) |
| Powered haulage share of deaths | 39% (2025), 43% (2024) | PRIMARY / SECONDARY |
| Trucking driver cost | $1.028/mi of $2.336/mi (44%), 2025 | PRIMARY/SECONDARY (ATRI) |
| Platooning fuel at quarry speeds | ≈0–1% | DERIVED |
| Follower kit hardware | ~$35k–130k | UNVERIFIED |

**Per-follower ROI sketch (DERIVED, UNVERIFIED assumptions):**
- One follower removes one operator seat. That is worth about **$83k/yr on a single 2,080-h shift** and **$165–190k/yr on two shifts**, before productivity effects.
- Add about 5–15% maintenance and tire savings on a $130–200/hr truck at 2,500 h, worth roughly $15–75k/yr, if vendor AHS claims transfer to a convoy. That transfer is unproven.
- Subtract convoy productivity loss: followers wait in the loader queue and cannot be dispatched independently.
- On this sketch, a $50–100k follower kit pays back in about 0.5–1.5 years on one shift.
- **Main risks:**
  - Small fleets per site (2–6 trucks).
  - Cat's AHS move onto the 775 platform at quarries.
  - Kodiak at Atlas on private haul roads.
  - Chinese low-cost AHS.
