# 01 — Competitive Landscape & Cautionary Tales
**Autonomous convoy / leader-follower trucking on repetitive industrial routes**
Research date: 2026-11-11. Prepared for pre-seed/seed venture diligence.

## How to read this file
- Every factual claim has a source URL and a publication date in brackets: `[source, date]`.
- **UNVERIFIED** means I could not confirm the claim from a source this session. That covers background knowledge, single secondary sources, and conflicting figures.
- **Method limits.** Most facts come from web-search result summaries of the cited pages. I did not read most primary pages in full. WebFetch could not reach these hosts from this session: caterpillar.com, volvoautonomoussolutions.com, mineralresources.com.au, nationaldefensemagazine.org and insidedefense.com. The shared web-search budget also ran out before I could check sections 4–5 more deeply. Gaps are listed in §6.

---

## 0. Executive takeaways (decision-relevant)

1. **Driverless trucks already make money on private industrial roads, and they don't use a human leader.**
   - Kodiak's trucks for Atlas Energy run unsupervised on Permian lease roads: 37 trucks as of Nov 2026, more than 450,000 tons of sand hauled by 31 Mar 2026, and a 100-truck order due in 1H27 [Kodiak/GlobeNewswire, 2026-11-03].
   - Volvo hauls all of Brønnøy Kalk's production driverless and charges per tonne [IM, 2026-03-23].
   - The private-road, repetitive-route thesis is validated. The specific wedge of "human-led leader-follower" is a transitional product unless it is much cheaper or faster to deploy.
2. **Platooning that only saves fuel failed on economics.**
   - Best-case savings were about 4–7% for the whole team, rising to 14% only with aero trailers [NREL FY15; Caltrans/PATH 2018].
   - In Daimler's real-traffic data, ideal platooning happened only about 20% of the time. Daimler said there was "no business case" [Supply Chain Dive, 2019-01].
   - Peloton (about $78M raised by 2017) was liquidated in summer 2021.
3. **A human-led convoy on public highways also died.**
   - Locomation had a 1,120-unit commitment from Wilson Logistics and raised $105M.
   - It could not raise more money and sold substantially all its assets in 2023 [FreightWaves 2023-02; Jones Day 2023-04].
   - The lesson: a follower truck on a public highway still needs near-L4 safety validation, so the capital needs are close to full autonomy while only half the driver labor is saved.
4. **Mining autonomous haulage (AHS) is mature and dominated by OEMs.**
   - Komatsu has commissioned 1,000 trucks [2026-04-22].
   - Caterpillar had 690 trucks at end-2024 and targets more than 2,000 by 2030 [IM 2025-11-07].
   - GlobalData counts 5,684 autonomous or autonomy-ready haul trucks worldwide (July 2026), 3,843 of them in China (secondary relay, UNVERIFIED against the original report).
5. **Quarries and aggregates are the contested middle tier right now.**
   - Cat is at Luck Stone: more than 3.5M tons hauled, and expanding to Cat 775 fleets [2026-09].
   - Pronto (now owned by Kalanick's Atoms) is at Heidelberg Materials, with about 30 autonomous vehicles planned for 2026 [2026-04-30].
   - Komatsu has launched Smart Quarry Autonomous.
   - Applied Intuition, sensmore and Epiroc are working with Heidelberg [IM 2026-05-01].
6. **There is a lot of capital in full autonomy.**
   - Bedrock Robotics: $600M at a $4.1B valuation [TechCrunch 2026-10-26].
   - Waabi: $750M plus up to $250M from Uber [2026-01-28].
   - ASI: $225M from SoftBank [2026-09-24].
   - Gatik: $200M Series D [2026-08-25].
   - Forterra: $238M Series C [2025-11].
   - Overland AI: $100M [2026-02-03].
   - A pre-seed company cannot win a head-on fight on full autonomy.
7. **Revenue is still tiny, even for public leaders.**
   - Aurora: about $3M revenue and a $234M net loss in Q3 2026; FY26 revenue guidance cut to about $9M [2026-10-28].
   - Kodiak: $2.97M Q3 revenue [2026-11-05].
   - Cyngn: $249K revenue in 1H26.
8. **The usual exit for small off-road autonomy companies is an acqui-hire or tuck-in.**
   - SafeAI went to Pronto for "in the millions" [2025-07-15].
   - Teleo went to Havoc [2026-03-11].
   - ASI Mining went to Epiroc [2024-07].
   - Embark went to Applied Intuition for $71M, down from a $5.2B SPAC valuation [2023].
9. **The defense leader-follower program was ended and restarted on a commercial path, and a winner is already picked.**
   - The Army closed ExLF in 2023.
   - The successor, ATV-S, moved to commercial kits through DIU.
   - Carnegie Robotics reportedly won the phase-3 downselect [InsideDefense, 2026-04-29].
   - The FY27 request for the leader-follower line is only about $22M [InsideDefense, 2026-04].
10. **There are three working pricing models to copy.**
    - Per tonne delivered, as transport-as-a-service (Volvo/Brønnøy).
    - Recurring Driver-as-a-Service on customer-owned trucks (Kodiak).
    - Subscription bundles (Outrider).
    - Bot Auto claims its humanless cost is $1.89/mile against an ATRI industry figure of $2.26/mile [2026-04].

---

## 1. On-highway autonomous trucking

### Aurora Innovation (Nasdaq: AUR)
- **What and where:** The Aurora Driver for Class 8 trucks on public interstates. It runs driverless commercially in Texas: Dallas–Houston from spring 2025, plus Fort Worth–El Paso.
- **Fleet and miles (Q3 2026):**
  - More than 20 driverless trucks built.
  - Upfitting with Roush toward annualized capacity of more than 1,000 trucks.
  - Production about 6 weeks behind; target of 20 trucks/week by early December 2026.
  - A 200-truck driverless fleet expected by end-February 2027.
  - More than 500,000 commercial driverless miles by September 2026, with "100% on-time" delivery.
  - Target to exit 2027 with more than 1,000 driverless trucks.
  - Sources: [Aurora Q3 2026 results, https://ir.aurora.tech/news-events/press-releases/detail/155/aurora-announces-third-quarter-2026-results, 2026-10-28]; [Q3 2026 Shareholder Letter, https://ir.aurora.tech/_assets/_b43b1c20802e72db7dcf51238ac7febe/aurora/db/956/10062/shareholder_letter/26Q3_Shareholder_Letter.pdf, 2026-10-28]
- **Financials:**
  - Q3 2026 revenue about $3M; net loss $234M (vs $201M in Q3 2025).
  - FY2026 revenue guidance cut to about $9M.
  - Cash and investments above $1B.
  - Sources: [same, 2026-10-28]; [StockTitan, https://www.stocktitan.net/news/AUR/aurora-announces-third-quarter-2026-5s88elx1t96b.html, 2026-10-28]
- **Customer:** Charger Logistics committed to 500 vehicles [GuruFocus call highlights, https://www.gurufocus.com/news/9150178/..., ~2026-10-29]. The headline mislabels the quarter as "Q3 2027"; treat the detail as UNVERIFIED against a primary source.
- **OEM channel:** The Volvo VNL Autonomous runs the Aurora Driver.
  - It has hauled freight for DHL since 4 Dec 2024, with safety drivers.
  - Volvo plans to go driver-out in Texas in Q1 2027.
  - Sources: [Transport Topics, https://www.ttnews.com/articles/volvo-autonomous-dhl, 2024-12]; [FreightWaves, https://www.freightwaves.com/news/volvo-driverless-trucks-q1-2027, 2026]
- **Funding:** Public since its 2021 SPAC (Reinvent III). UNVERIFIED this session.
- **Status:** Alive, public (AUR).

### Kodiak AI (Nasdaq: KDK) — closest analog to the founder's idea
- **Listing:**
  - Its SPAC merger with Ares Acquisition Corp II closed 24 Sep 2025; KDK/KDKRW began trading 25 Sep 2025.
  - Gross proceeds above $212.5M, including a $145M PIPE and about $62.9M from the trust.
  - Implied valuation about $2.5B.
  - Sources: [GlobeNewswire, https://www.globenewswire.com/news-release/2025/09/24/3155836/0/en/..., 2025-09-24]; [Nasdaq Trader ECA2025-527, https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2025-527, 2025-09]; [SEC 8-K, https://www.sec.gov/Archives/edgar/data/1853138/000119312525225299/d933659d8k.htm, 2025-09]
- **Atlas Energy deployment (private Permian lease roads, frac sand):**
  - 28 driverless trucks on 15 routes as of 31 Mar 2026, carrying about 7,000 loads and more than 450,000 tons of sand.
  - More than 40,000 cumulative driverless hours by 30 Jun 2026.
  - A second simultaneous load-out point added on Atlas's Dune Express conveyor, about 90 minutes from the first.
  - A record 176 loads in one day on 20 Jul 2026.
  - Sources: [Atlas/Kodiak IR, https://investors.kodiak.ai/news-releases/news-release-details/atlas-energy-solutions-expands-ai-enabled-digital-transformation, 2026-07-31]; [https://ir.atlas.energy/news-events/press-releases/detail/74/..., 2026-07-31]
  - The first two Western Star 49X trucks were delivered in September 2026. They are the first of 65 Western Stars that complete Atlas's initial 100-truck order, due 1H 2027.
  - The fleet is now 37 trucks, which Kodiak calls the "world's largest fleet of its kind". Atlas owns and operates the trucks.
  - Public-road operation is expected by early 2027.
  - Sources: [GlobeNewswire, https://www.globenewswire.com/news-release/2026/11/03/3394865/0/en/..., 2026-11-03]; [Truck News, https://www.trucknews.com/technology/kodiak-delivers-first-two-driverless-western-star-49x-trucks-to-atlas-energy-solutions/1003224592/, 2026-11]
  - Note: the July release says the 100 trucks arrive "mid-2027"; the November release says "1H 2027".
- **Q3 2026 results (reported 5 Nov 2026):**
  - Revenue $2.97M (vs $0.77M), mostly recurring Driver-as-a-Service plus a one-time $0.5M of defense revenue.
  - Operating loss $50.4M. Net income $27.4M, but only because of a $78.2M non-cash gain on warrant fair value.
  - Free cash flow –$38M; cash and securities $113.6M.
  - Binding agreement with DTL Transport for 75 customer-owned unsupervised trucks.
  - "Autonomy Readiness Measure" at 99%; targeting a year-end 2026 unsupervised long-haul launch with IKEA Supply.
  - Q4 guide: unsupervised deployments in the mid-40s. FY26 free cash flow guide –$153M to –$155M.
  - Targets: 200 trucks by end-2027, 5,000 by 2029, free-cash-flow breakeven in 2029.
  - Sources: [GlobeNewswire, https://www.globenewswire.com/news-release/2026/11/05/3397075/0/en/..., 2026-11-05]; [TradingView, https://www.tradingview.com/news/tradingview:8f9942844f79b:0-kodiak-ai-q3-revenue-rises-286-to-3-0-million-net-income-reaches-27-4-million/, 2026-11-05]
- **Business model:** Driver-as-a-Service on customer-owned trucks. The per-truck price is UNVERIFIED.
- **Defense:**
  - $49.9M DoD/DIU contract to develop automated Army ground vehicles [CNBC, https://www.cnbc.com/2022/12/06/kodiak-wins-50m-deal-to-develop-driverless-army-vehicles.html, 2022-12-06].
  - "Leonidas" autonomous ground vehicle with General Dynamics Land Systems [Army Technology, https://www.army-technology.com/news/kodiaq-general-dynamics-military-vehicles/, 2025/26, exact date n/a].
- **Cash risk:** $113.6M of cash against a –$153M annual free-cash-flow burn implies a raise within roughly 12 months. This is my inference.

### Waabi (private)
- **Latest round:** $750M Series C, co-led by Khosla Ventures and G2 Venture Partners, announced 28 Jan 2026.
  - Uber committed up to $250M more, tied to milestones, for a total of up to $1B.
  - Other investors: Uber, NVentures, Volvo Group VC and Porsche SE.
  - Valuation undisclosed; the Globe and Mail reported a $3B target.
  - Plan for at least 25,000 Waabi-powered robotaxis exclusively on Uber.
  - Volvo trucks running Waabi still carry safety drivers in Texas. The driverless target slipped from end-2025.
  - Sources: [CNBC, https://www.cnbc.com/2026/01/28/autonomous-startup-waabi-raises-750-million-to-expand-into-robotaxis.html, 2026-01-28]; [TechCrunch, https://techcrunch.com/2026/01/28/waabi-raises-1b-and-expands-into-robotaxis-with-uber/, 2026-01-28]
- **Earlier rounds:** $83.5M Series A (2021) and $200M Series B (2024). UNVERIFIED this session.
- **Status:** Alive. No driver-out commercial trucking as of July 2026 [New Market Pitch, https://newmarketpitch.com/blogs/news/autonomous-vehicle-who-is-ahead-trucks, 2026-07].

### PlusAI (private, SPAC failed)
- **SPAC:** $1.2B merger with Churchill Capital Corp IX announced 5 Jun 2025, with planned Nasdaq ticker "PLS".
  - The shareholder meeting was pushed from 11 Feb to 15 Apr 2026.
  - The merger was terminated by mutual consent on 20 Apr 2026, "due to market conditions".
  - Sources: [SEC 8-K, https://www.sec.gov/Archives/edgar/data/2006291/000119312526165149/d148615d8k.htm, 2026-04-20]; [Business Wire, https://www.businesswire.com/news/home/20260421921727/en/PlusAI-and-Churchill-Capital-Corp-IX-Terminate-Business-Combination, 2026-04-21]
- **Product:** SuperDrive factory-built autonomous trucks, targeting a 2027 commercial launch [PlusAI release, 2025-12]. OEM partners (Traton/International, IVECO, Hyundai) are UNVERIFIED this session.
- **Note:** PlusAI did not acquire Locomation; I found no evidence linking them.

### Torc Robotics (Daimler Truck subsidiary)
- **Ownership and funding:** Daimler Truck has held a majority since 2019. Funded by the parent; no venture rounds [Trucking Dive, https://www.truckingdive.com/news/torc-robotics-timeline-late-2026-nick-elder/720331/, 2024].
- **Timeline:** Aims for no safety driver on public roads before end-2026, first commercial freight in 2027, scaling from 2028. Daimler builds the redundant Cascadia chassis [FreightWaves, https://www.freightwaves.com/news/daimler-truck-torc-autonomous-system, 2026-09].
- **Footprint:** Wound down its Albuquerque and Stuttgart sites and moved to Dallas–Fort Worth and Ann Arbor. First lane is Dallas–Laredo [Transport Topics, https://www.ttnews.com/articles/torc-robotics-plans-offices, 2024-12].
- **2026 layoffs:** None found (searched once).

### Bot Auto (private, Houston)
- **First humanless run:** A commercial load on 29 Apr 2026, 230 miles from Houston to Hutchins (Dallas), with no safety driver, remote operator or observer.
  - Claimed humanless cost of $1.89/mile vs ATRI's industry figure of $2.26/mile.
  - Partner: Ryan Transportation.
  - About $40M raised in total, 80 employees, 12 tractors.
  - Sources: [FreightWaves, https://www.freightwaves.com/news/bot-auto-first-humanless-commercial-truckload, 2026-04/05]; [Bot Auto press, https://bot.auto/press/houston-we-don-t-have-a-problem-bot-auto-delivers-america%E2%80%99s-first-fully-humanless-commercial-truckload, 2026-04]
- **Earlier round:** $20M in October 2024 (Brightway Future Capital, Cherubic) [Dallas Innovates, https://dallasinnovates.com/houstons-bot-auto-partners-with-ryan-on-humanless-freight-runs-between-dallas-and-houston/, 2026]. One Bot Auto article calls it a seed round; the label varies.
- **Status:** Alive and fundraising [Bot Auto, https://bot.auto/news/houston-based-bot-auto-seeks-funding-after-driverless-validation-run-on-i-10, 2025/26].

### Gatik (private) — repetitive-route, middle-mile box trucks
- **Series D:** $200M on 25 Aug 2026, led by Qatar Investment Authority and Koch Disruptive Technologies; about $500M raised in total.
  - "Dozens" of driverless trucks, targeting more than 100 by end-2026.
  - Fully driverless orders grew from about 60,000 (start of 2026) to 85,000 (August 2026).
  - Operates in TX, AR, AZ, NE and Ontario.
  - Its largest public partnership is PepsiCo/Frito-Lay, with 41 driverless box trucks.
  - Isuzu invested $30M in 2024; a jointly developed redundant chassis is planned for mass production from 2027.
  - Sources: [TechCrunch, https://techcrunch.com/2026/08/25/self-driving-truck-startup-gatik-raises-200m-following-pepsico-deal/, 2026-08-25]; [FleetOwner, https://www.fleetowner.com/technology/news/55401124/gatik-lands-200m-to-expand-driverless-middle-mile-truck-rollout, 2026-08]
- **Freight-only:** Running freight-only (no one on board) since mid-2025 [Business Wire, https://www.businesswire.com/news/home/20260127742958/en/..., 2026-01-27].
- **"$600M contracted revenue":** Secondary source only, UNVERIFIED [valueaddvc.com, 2026].
- **Relevance:** Short, fixed, repeated routes are exactly where driverless commercial scale arrived first.

### Einride (Nasdaq: ENRD)
- **SPAC with Legato Merger Corp III:**
  - Announced 12 Nov 2025 at $1.8B, later revised to a $1.35B pre-money equity value.
  - $113M PIPE on 26 Feb 2026.
  - F-4 effective 14 May 2026; shareholder vote 4 Jun 2026.
  - Sources: [CNBC, https://www.cnbc.com/2025/11/12/autonomous-ev-trucker-einride-to-go-public-in-spac-at-1point8-billion-value.html, 2025-11-12]; [Einride press, https://www.einride.tech/press/einride-and-legato-merger-corp.-iii-announce-public-filing-of-registration-statement-on-form-f-4, 2026]; [Automotive World, https://www.automotiveworld.com/news/einride-to-trade-on-nasdaq-after-spac-merger-approval/, 2026-06]
- **Trading:** Nasdaq debut around 10 Jun 2026, up 70–90% intraday. Single source (CryptoBriefing), UNVERIFIED.
- **Model:** Electric and autonomous freight capacity, plus licensing of the Einride Driver and its "Saga" fleet software.

### Stack AV (private, SoftBank)
- **Funding:** More than $1B committed by SoftBank, per Bloomberg [Fortune, https://fortune.com/2023/09/07/softbank-backs-ai-startup-stack-av-as-it-tries-to-automate-trucking/, 2023-09-07].
- **Product:** Unveiled its first truck, with PACCAR/Peterbilt, in Las Vegas in May 2026 [Yahoo Autos, https://autos.yahoo.com/ev-and-future-tech/articles/stack-av-unveils-autonomous-truck-161957156.html, 2026-05].
- **Status:** No driver-out commercial service as of July 2026 [New Market Pitch, 2026-07].

---

## 2. Platooning / convoy failures — cautionary tales

### Peloton Technology (platooning; liquidated 2021)
- **Product:** Level 1 driver-assistive truck platooning: a V2V link, with a human driver in every truck.
- **Funding:** $60M Series B in April 2017, led by Omnitracs; $78M raised in total by then.
  - Investors included Intel Capital, UPS Strategic Enterprise Fund, Volvo Group, DENSO, BP Ventures and Lockheed Martin.
  - Sources: [Peloton, https://peloton-tech.com/seriesb/, 2017-04]; [Fortune, https://fortune.com/2017/04/06/peloton-trucking-funding-round/, 2017-04-06]
- **Pivot:** In July 2019 it launched "Automated Following", a Level 4 system with no driver in the follower truck [Forbes/R. Bishop, https://www.forbes.com/sites/richardbishop1/2020/05/02/..., 2020-05-02].
- **End:** Ceased operations and entered liquidation in summer 2021 [Wikipedia, https://en.wikipedia.org/wiki/Peloton_Technology, accessed 2026-11]. No primary post-mortem found. Asset buyer UNVERIFIED.
- **Why it died (inferred from evidence):**
  - Fuel savings of a few percent did not pay for hardware, integration and driver behavior change while the driver was still paid.
  - Its anchor OEM partner pulled back (Daimler, January 2019; see below).
  - The L4 pivot put it in the same capital race as fully driverless companies.

### Locomation (human-led "Autonomous Relay Convoy", ARC; assets sold 2023)
- **Product:** A human-driven lead truck with an autonomous follower whose driver rests off-duty; the two trucks swap roles. On public highways.
- **Claimed economics:** 33% lower operating cost per mile and 8% lower fuel cost [FreightWaves, https://www.freightwaves.com/news/locomation-convoy-plan-doubles-down-on-autonomous-trucking-challenge, 2021].
- **Commitments:**
  - Wilson Logistics committed to 1,120 ARC units [GlobeNewswire, https://www.globenewswire.com/news-release/2022/03/10/2401272/0/en/..., 2022-03-10].
  - Other contracts: PGT Trucking, Christenson Transportation, and an 8-year deal with Stevens Transport in August 2022.
- **Funding:** $105M raised since 2018, including $15M in October 2022 (PitchBook) [FreightWaves, https://www.freightwaves.com/news/reports-autonomous-platooning-startup-locomation-closing-its-doors, 2023-02].
- **End:**
  - It failed to raise more capital. Most non-engineering staff were laid off in February 2023; the CEO denied the company was closing [same; Pittsburgh Post-Gazette, https://www.post-gazette.com/business/tech-news/2023/02/23/..., 2023-02-23].
  - Substantially all assets (vehicles, software, IP) were then sold [Jones Day, https://www.jonesday.com/en/practices/experience/2023/04/locomation-sells-ai-autonomous-trucking-solutions-business, 2023-04]. Buyer and price not disclosed.
  - I found no link to Tyson and no December 2022 shutdown.
- **Lesson for the founder:** This is the closest prior attempt at "human-led leader-follower". It had real fleet demand (1,120 units) and still died.
  - On public highways, the follower needs safety validation close to driverless.
  - Hours-of-service relief was uncertain.
  - Revenue only came at scale. When the capital markets closed (2022–23), the company ran out of runway.

### Embark Trucks (SPAC; sold for parts 2023)
- **Rise and fall:** Its 2021 SPAC valued it at about $5.2B. In March 2023 it laid off about 70% of staff. Q3 2022 cash was about $190M.
- **Sale:** Applied Intuition bought it for about $71M ($2.88/share, all cash), announced 25 May 2023 and closed 2 Aug 2023. Its test fleet (about 24 trucks) was retired.
- Sources: [TechCrunch, https://techcrunch.com/2023/05/25/applied-intuition-to-buy-autonomous-trucking-spac-embark-for-71m/, 2023-05-25]; [Applied Intuition, https://www.appliedintuition.com/news/embark, 2023-08]

### TuSimple (exited US trucking; now CreateAI)
- **US exit:** Laid off about 75% of its US workforce and exited US operations [TechCrunch, https://techcrunch.com/2023/12/05/another-blow-for-self-driving-trucks-as-former-industry-leader-abandons-the-u-s/, 2023-12-05].
- **Delisting:** Announced a Nasdaq delisting and SEC deregistration on 17 Jan 2024 [FreightWaves, https://www.freightwaves.com/news/autonomous-truck-developer-tusimple-going-private, 2024-01].
- **Pivot:** Became CreateAI (generative AI for animation and games) [CNBC, https://www.cnbc.com/2024/12/19/china-self-driving-truck-company-tusimple-pivots-to-genai-for-games.html, 2024-12-19].
- **Settlements:**
  - $189M securities-fraud settlement (CNBC, single source).
  - $6M CFIUS-related settlement [TechCrunch, https://techcrunch.com/2025/05/27/report-tusimple-sent-sensitive-self-driving-data-to-china-after-us-national-security-agreement/, 2025-05-27].

### Starsky Robotics (shut down March 2020)
- **Model:** Highway autonomy plus remote teleoperation for the first and last mile. The goal was one remote driver supervising 18–20 trucks. It ran an unmanned highway test in 2019.
- **End:** A $20M Series B collapsed in November 2019, followed by furloughs, a sale attempt, and a shutdown in March 2020.
- **Founder's diagnosis:** AI progress was slower than investors expected. Investors disliked owning the operator business, and safety spend was not rewarded.
- Sources: [Seltz-Axmacher, "The End of Starsky Robotics", https://medium.com/starsky-robotics-blog/the-end-of-starsky-robotics-acb8a6a8a5f5, 2020-03]; [FreightWaves, https://www.freightwaves.com/news/starsky-robotics-ends-remote-truck-experiment-shuts-down-operations, 2020-03]; [Crunchbase News, https://news.crunchbase.com/venture/self-driving-truck-startup-starsky-robotics-shuts-down-after-series-b-falls-through/, 2020-03]
- **Postscript:** Seltz-Axmacher went on to Polymath Robotics, an industrial-autonomy API company (LinkedIn listing; see §3).

### Published evidence on platooning fuel savings

| Study | Setup | Result | Source |
|---|---|---|---|
| NREL / Peloton track test | 2 platooned Class 8 SmartWay tractors plus a control truck | Lead truck up to 5.3%; trailing truck up to 9.7% (2.8–9.7% overall; trailing-truck savings dropped when the engine fan ran at short gaps); team 3.7–6.4%; best at 55 mph, 30 ft gap, 65,000 lb | NREL TP-5400-62348, https://docs.nrel.gov/docs/fy15osti/62348.pdf, FY2015; fact sheet https://docs.nrel.gov/docs/fy15osti/64133.pdf |
| Caltrans / California PATH 3-truck CACC (cooperative adaptive cruise control) | 65 mph, standard vs aero trailers | Net platoon savings 5.2–7.8% (standard trailers); up to 14.2% with aero trailers at a 17.4 m gap. Per truck: lead ≈0%, 2nd 6–7%, 3rd 9–11% | ITS JPO 2018-B01246, https://www.itskrs.its.dot.gov/2018-b01246, 2018; ITS JPO 2019-B01393, https://www.itskrs.its.dot.gov/2019-b01393, 2019; Caltrans CA18-2623, https://dot.ca.gov/-/media/dot-media/programs/research-innovation-system-information/documents/final-reports/ca18-2623-finalreport-a11y.pdf |
| NREL telematics (Volvo data) | About 57,000 trucks, 210M miles over 2 weeks | 63% of miles at platoonable speeds; 55.7% platoonable once partner availability is included | SAE 2018-01-1083 / NREL, https://docs.nrel.gov/docs/fy18osti/70869.pdf, 2018; https://www.itskrs.its.dot.gov/2018-b01313 |
| NREL Fleet DNA | Class 7/8 duty cycles | About 65% of miles platoonable, giving about 4% total truck fuel reduction (an upper bound) | NREL, https://research-hub.nrel.gov/en/publications/potentials-for-platooning-in-us-highway-freight-transport-preprin/, ~2017 |
| Daimler Trucks (CES 2019) | Real-traffic US tests with new aero trucks | "No business case for customers driving platoons with new, highly aerodynamic trucks." Reported figures conflict: about 4% at 15 m in ideal conditions with ideal platooning only ~20% of the time (Commercial Motor), vs. an expected 7–11% for the middle truck but ≤2% achieved (The Truck Expert). Treat both as UNVERIFIED | Supply Chain Dive, https://www.supplychaindive.com/news/Daimler-platooning-automated-truck-CES/545524/, 2019-01; Commercial Motor, https://www.commercialmotor.com/product-news/article/daimler-pull-plug-truck-platooning, 2019-01; The Truck Expert, https://www.thetruckexpert.co.uk/platooning-jitters-following-daimler-u-turn/, 2019-01 |

**Why fuel-only platooning failed (synthesis):**
1. The realized team savings were only about 4–7%. They shrink further with modern aero tractors and with the extra fuel spent re-forming a platoon after cut-ins.
2. Fuel is only part of cost per mile, while driver labor is the largest line item, and Level 1 platooning leaves labor unchanged. The fuel share of cost per mile is UNVERIFIED this session; use ATRI's annual "Operational Costs of Trucking" report.
3. Finding a partner truck limits use to about 55–65% of miles, even in theory.
4. OEMs exited (Daimler, 2019), which removed the factory-fit channel.
5. Only removing the follower driver changes the economics. That requires L4, so the business becomes an autonomy company (Peloton's 2019 pivot, Locomation).

---

## 3. Off-road / industrial autonomy

### Pronto (now part of Atoms / Travis Kalanick)
- **SafeAI acquisition:** Closed 15 Jul 2025. Terms undisclosed ("in the millions"); about 12 SafeAI engineers and its IP moved over.
  - The deal pairs Pronto's camera-only stack with SafeAI's multi-sensor, ASIL-D safety framework, so it can sell tiered products from small quarries to large mines.
  - Sources: [Business Wire, https://www.businesswire.com/news/home/20250715506479/en/Pronto-Acquires-SafeAI-Expanding-Leadership-in-Off-Road-Autonomy, 2025-07-15]; [TechCrunch, https://techcrunch.com/2025/07/15/av-startup-pronto-ai-acquires-off-road-autonomous-vehicle-rival-safeai/, 2025-07-15]; [IM, https://im-mining.com/2025/07/15/oem-agnostic-consolidation-move-as-pronto-completes-safeai-acquisition/, 2025-07-15]
- **Products:**
  - Runs on Komatsu HD605/HD785 trucks [IM, 2025-07-15].
  - "AHS Editions", a tiered autonomous haulage portfolio, launched 5 Feb 2026 [Business Wire, https://www.businesswire.com/news/home/20260205818079/en/..., 2026-02-05].
- **Heidelberg Materials:**
  - At the Lake Bridgeport quarry (TX), a mixed fleet has hauled more than 2M tons: Cat 775G plus Komatsu HD605-8 and HD605-10 trucks [Pronto, https://pronto.ai/pronto-announces-industry-milestone-north-americas-first-fully-autonomous-mixed-fleet-quarry-surpasses-two-million-tons-hauled/, 2025/26].
  - Expanding to Mitchell (IN) and Servtex (TX), with about 30 autonomous vehicles in the 2026 expansion [Business Wire, https://www.businesswire.com/news/home/20260430207186/en/..., 2026-04-30].
- **Other deals:**
  - Mariana Minerals, Copper One mine in Utah [TechCrunch, https://techcrunch.com/2026/04/09/mariana-minerals-taps-pronto-to-help-automate-a-copper-mine/, 2026-04-09].
  - MoU with Hitachi Construction Machinery [Highways Today, https://highways.today/2026/07/17/hitachi-pronto-mining/, 2026-07-17].
- **Ownership:** Kalanick's Atoms acquired Pronto, which became the core of "Atoms Mining". Kalanick was already its largest investor [Bricks & Bytes, https://bricks-bytes.com/funding-ma/ubers-pronto-atoms-autonomous-machinery/, 2026, exact date n/a].
- **Funding:** Undisclosed. UNVERIFIED.

### SafeAI
- **Model:** Retrofit autonomy for mining and construction (multi-sensor).
- **Fate:** Acquired by Pronto in July 2025 (above). Prior funding UNVERIFIED.

### Caterpillar — MineStar Command (mining plus aggregates)
- **Fleet:**
  - 690 autonomous trucks running Command for hauling at end-2024.
  - Target to triple that, to more than 2,000, by 2030.
  - Sources: [IM, https://im-mining.com/2025/11/07/caterpillar-sets-out-to-hit-over-2000-autonomous-mining-trucks-by-2030/, 2025-11-07]
  - A 2026 Cat release says autonomous trucks have hauled more than 13B tonnes over more than 455M km with no reported injuries [Caterpillar, https://www.caterpillar.com/en/news/caterpillarNews/2026/scaling-autonomy-system.html, 2026; page not fetchable, figure from the search summary].
- **Luck Stone, Bull Run quarry (Chantilly, VA):**
  - The first aggregates deployment, using Cat 777 trucks, live since November 2024.
  - 1M tons by July 2025 [IM, https://im-mining.com/2025/07/15/caterpillar-and-luck-stone-celebrate-1-million-tons-hauled-autonomously/, 2025-07-15]; more than 3.5M tons by September 2026.
  - Next phase: two Cat 775 fleets, at Boscobel and Bealeton. This is the first autonomous Cat 775 deployment, a key quarry truck size.
  - Sources: [Caterpillar IR, https://investors.caterpillar.com/news/news-details/2026/Luck-Stone-Builds-on-Autonomous-Hauling-Success-with-Caterpillar/default.aspx, 2026-09-14/15]; [Global Mining Review, https://www.globalminingreview.com/mining/16092026/luck-stone-builds-on-autonomous-hauling-success-with-caterpillar/, 2026-09-16]
  - Customer rationale, quoted in the release: if one of four trucks is idle for lack of an operator, a quarter of the day's production is lost.
- **Fortescue:** Cat renewed its MineStar Command agreement across 3 Western Australian sites in March 2026 [Canadian Mining Journal, https://www.canadianminingjournal.com/news/fortescue-extends-autonomous-mining-truck-agreement-with-caterpillar/, 2026-03].
- **Vale:** Autonomous fleet expansion deal in Pará [IM, https://im-mining.com/2025/12/08/vale-confirms-autonomous-truck-fleet-expansion-deal-with-caterpillar/, 2025-12-08].

### Komatsu — FrontRunner AHS
- **Scale:** Commissioned its 1,000th autonomous ultra-class haul truck, a 930E-5AT (290 t payload) at Barrick's Nevada Gold Mines. It calls itself the first OEM to reach 1,000.
  - More than 11.5B t moved since FrontRunner launched commercially in 2008.
  - More than 500 of the trucks are 930Es.
  - Sources: [Komatsu, https://www.komatsu.com/en-us/newsroom/2026/komatsu-becomes-first-oem-to-commission-1000-ultra-class-autonomous-haul-trucks, 2026-04-22]
- **Quarries:** "Smart Quarry Autonomous", a quarry-specific product [Komatsu, https://www.komatsu.com/en-us/newsroom/2026/smart-quarry-autonomous-finalist-for-industry-award--expands-quarry-specific-digital-offerings, 2026]. Details UNVERIFIED.

### Volvo Autonomous Solutions (V.A.S.)
- **Brønnøy Kalk (Norway limestone):**
  - Contract signed November 2018 as a "total transport service" paid **per tonne delivered**. The rate is undisclosed [Volvo Trucks, https://www.volvotrucks.com/en-en/news-stories/press-releases/2018/nov/pressrelease-181120.html, 2018-11-20].
  - Route: about 5 km from mine to crusher, with tunnels and steep grades.
  - Autonomous hauling began in 2023. 1M tonnes by May 2025 with 7 FH trucks [Volvo Group, https://www.volvogroup.com/en/news-and-media/news/2025/may/volvo-achieves-milestone-with-over-1-million-tonnes-hauled-autonomously-at-bronnoy-kalk.html, 2025-05].
  - 9 autonomous FH trucks with trailers now haul all production; Volvo aims for 7 per shift. It will not share productivity versus manual haulage [IM, https://im-mining.com/2026/03/23/volvo-fh-autonomous-truck-fleet-now-hauling-all-production-for-bronnoy-kalk/, 2026-03-23].
  - 3M tonnes hauled autonomously [PR Newswire/Morningstar, https://www.morningstar.com/news/pr-newswire/20260924hs56019/three-million-tonnes-hauled-autonomously-by-volvo, 2026-09-24].
- **On-highway (Autona/freight):**
  - VNL Autonomous with the Aurora Driver, with safety drivers. Lanes: Dallas–Houston, Fort Worth–El Paso and Dallas–OKC.
  - Partners: DHL and Uber Freight; DSV added in May 2026.
  - About 20 trucks; driver-out planned for Q1 2027; "hundreds" by late 2027.
  - Sources: [Volvo Group, https://www.volvogroup.com/en/news-and-media/news/2026/may/volvo-autonomous-solutions-and-dsv-announce-autonomous-freight-operations.html, 2026-05]; [FreightWaves, https://www.freightwaves.com/news/volvo-driverless-trucks-q1-2027, 2026]
- **Why it matters:** Volvo proves that customers accept per-tonne transport-as-a-service pricing for autonomous haulage on short private routes.

### ASI — Autonomous Solutions Inc. (Utah)
- **SoftBank:** Invested $225M, and the two formed a joint venture for autonomous construction (mixed fleets of haul trucks, dozers, loaders and compactors) on ASI's OEM-agnostic Mobius platform.
  - Sources: [ASI, https://asirobots.com/asi-and-softbank-group-accelerate-autonomous-construction-with-225-million-investment, 2026-09-24]; [Utah Business, https://www.utahbusiness.com/press-releases/2026/09/28/asi-raises-225-million-softbank-accelerate-autonomous-construction-technology/, 2026-09-28]
- **Ag deployment:** At U.S. Sugar, 4 John Deere 8R tractors and one 9R were retrofitted and are supervised from a command station [same].
- **ASI Mining divestiture:** Epiroc bought the remaining 66% (it already held 34% from 2018). Announced 29 May 2024, closed July 2024; ASI Mining had about SEK 300M revenue in 2023.
  - Sources: [Epiroc, https://www.epirocgroup.com/en/media/corporate-press-releases/2024/20240529-epiroc-to-acquire-remaining-share-of-autonomous-solutions-provider-asi-mining, 2024-05-29]; [https://www.epirocgroup.com/en/media/corporate-press-releases/2024/20240703-..., 2024-07-03]
- **Leader-follower / "follow-me":** ASI's capability here is UNVERIFIED this session; my searches did not surface it.
- **Threat level: high.** ASI is OEM-agnostic, works across ag, construction and logistics, and now has $225M plus SoftBank.

### Teleo (acquired by Havoc)
- **Product:** Supervised autonomy and teleoperation: one operator runs several heavy machines, as a brand-agnostic retrofit.
- **Funding:** About $28–30M raised, including a $16.2M Series A extension led by UP.Partners [The Robot Report, https://www.therobotreport.com/teleo-raises-16-2-million-to-scale-supervised-autonomy-for-heavy-equipment/, 2024].
- **Acquisition:** Havoc (HavocAI) bought Teleo and Mavrik [PR Newswire, https://www.prnewswire.com/news-releases/havoc-advances-all-domain-collaborative-autonomy-through-acquisitions-of-mavrik-and-teleo-302711100.html, 2026-03-11].
  - Havoc's CEO wants to apply the tech to "large truck convoys, Humvees, even M1 tanks" [Tectonic, https://www.tectonicdefense.com/exclusive-havocais-all-domain-shopping-spree/, 2026-03].
- **Status:** Shown with a Hitachi ZW310 wheel loader at CONEXPO 2026 [Hitachi CM, https://www.hitachicm.com/us/en/news/2026/teleo-to-demonstrate-a-supervised-autonomy-workforce-shortage-so/, 2026-03].

### Bedrock Robotics (autonomous excavators)
- **Series B:** $270M at a $1.75B valuation, co-led by CapitalG and Valor Atreides AI Fund; more than $350M raised in total [PR Newswire, https://www.prnewswire.com/news-releases/bedrock-robotics-raises-270-million-in-series-b-funding-to-accelerate-the-future-of-autonomous-construction-302679014.html, 2026-02-04].
- **Series C:** $600M at a $4.1B valuation, co-led by CapitalG and Emergence [TechCrunch, https://techcrunch.com/2026/10/26/this-startups-self-driving-excavators-help-dig-up-600m-in-new-funding/, 2026-10-26]. Single source.
- **Commercial sites:** A Nevada water-treatment plant, Champion Site Prep in Texas, and a 1.2M-cubic-yard Zachry project.

### Built Robotics
- **History:** Pivoted from general earthmoving to autonomous solar pile driving (the RPD 35, March 2023) after a $64M Series C in 2022. About 25–30% headcount cuts in FY23 (secondary source, UNVERIFIED) [Teardown, https://www.teardown.ai/companies/built-robotics, 2025/26].
- **Commercial:** Blattner (Quanta) is deploying "dozens" of units [ENR, https://www.enr.com/articles/61329-..., 2025-09].
- **Grant:** Up to AUD 14.5M from ARENA for a Brisbane manufacturing hub to build 12 piling systems [pv magazine, https://www.pv-magazine.com/2026/10/13/arena-pumps-32-3-million-into-solar-robotics-manufacturing-projects/, 2026-10-13].
- **Status:** Alive. Narrowed to a niche.

### Polymath Robotics
- **Product:** Industrial-vehicle autonomy delivered as an API ("tell the vehicle what to do over REST").
- **Funding:** Aggregators show about $500K seed from Y Combinator in 2022 and conflicting investor counts [Tracxn, https://tracxn.com/d/companies/polymath-robotics/..., accessed 2026-11]. No priced round confirmed; UNVERIFIED.
- **Founder:** Linked to Stefan Seltz-Axmacher (ex-Starsky) per LinkedIn.
- **Relevance:** A small direct competitor for "autonomy kit for private industrial sites".

### Outrider (autonomous yard trucks)
- **Funding:** $62M Series D led by Koch Disruptive Technologies and NEA, with NVIDIA's NVentures and Prologis Ventures; more than $250M of equity raised in total [Outrider, https://www.outrider.ai/press-releases/outrider-closes-62-million-series-d-financing-to-drive-growth-of-yard-automation/, 2024-10]; [Forbes, 2024-10-24].
- **Safety:** TÜV SÜD certification [Forbes, https://www.forbes.com/sites/richardbishop1/2025/07/10/outrider-powers-scaling-of-yard-autonomy-with-tv-sd-certification/, 2025-07-10].
- **Rollout:** Driverless deployments with select customers began in 2H 2025. Enterprise support launched for a 2026 commercial rollout [PR Newswire, https://www.prnewswire.com/news-releases/outrider-launches-first-enterprise-class-support-services-for-driverless-yard-operations-302619518.html, 2025-11].
- **Model:** Subscription bundling the autonomy stack, cloud software, trailer tracking and 24/7 support. Says its customers run more than 20% of North American yard trucks.

### Fernride (terminal tractors, teleoperation-assisted)
- **Certification:** First TÜV SÜD certification of an autonomous terminal tractor under the EU Machinery Directive.
- **Deployment:** Moving to driverless operation at HHLA TK Estonia, with 3 tractors [Forbes, https://www.forbes.com/sites/richardbishop1/2025/07/10/fernride-launches-driverless-operations-at-hhla-tk-estonia-seaport/, 2025-07-10].
- **Funding:** Series A extended to €75M with an €18M extension led by Helantic, while expanding into defense [Tech.eu, https://tech.eu/2025/09/04/fernride-bags-eur18m-series-a-extension-as-it-expands-into-defence/, 2025-09-04]. No 2026 round found.

### ISEE (yard trucks)
- **Holman partnership (investor):** Holman leases, retrofits and maintains ISEE-equipped trucks; factory-built units come via TICO from 2027 [FreightWaves, https://www.freightwaves.com/news/isee-holman-partnership-autonomous-yard-trucks, 2026-10-02].
- **Funding:** No 2026 round found. UNVERIFIED.
- **Competitor note:** Venti has announced more than 130 trucks across 8 sites by 2027 at a US intermodal rail yard (same article).

### Oxa (UK; pivoted from on-road to industrial)
- **Series D first close:** $103M, anchored by a $50M commitment from the UK National Wealth Fund, with NVentures, IP Group, Hostplus and bp Ventures. More than $250M raised in total.
- **Customers:** DHL, Vantec and bp. Can convert a port truck in under a day.
- **Write-downs:** IP Group and Ocado had written down their holdings.
- Sources: [Tech.eu, https://tech.eu/2026/03/04/uk-self-driving-startup-oxa-raises-103m-to-scale-industrial-deployments/, 2026-03-04]; [Automotive World, https://www.automotiveworld.com/news/oxas-pivot-away-from-on-road-avs-could-be-paying-off/, 2026]

### Cyngn (Nasdaq: CYN) — cautionary tale for an industrial-autonomy microcap
- **Revenue:**
  - Q1 2026: $105K. Q2 2026: $144.5K; 1H26 $249K.
  - FY2025: $219K (vs $368K in 2024).
  - All from DriveMod Tugger subscriptions.
- **Loss and balance sheet:** Q2 net loss $6.4M; working capital $41.3M.
- **Reverse splits:** 1:100 in July 2024 and 1:150 in February 2025. On 3 Nov 2026 shareholders authorized another split of 1:5 to 1:150.
- Sources: [PR Newswire Q2, https://www.prnewswire.com/news-releases/cyngn-reports-2026-2nd-quarter-financial-results-302850061.html, 2026-08]; [Cyngn, https://www.cyngn.com/pr/cyngn-announces-reverse-stock-split, 2025-02-13]

### Forterra (defense plus commercial)
- **Funding:** $238M Series C (equity plus debt) at a valuation above $1B, led by Moore Strategic Ventures [Bloomberg, https://www.bloomberg.com/news/articles/2025-11-12/..., 2025-11-12]; [Forterra, https://www.forterra.com/posts/powering-the-future-of-the-battlefield-with-a-238m-series-c, 2025-11].
  - More than $500M raised in total [TechCrunch, https://techcrunch.com/2026/07/07/the-first-american-autonomous-ground-vehicles-are-fighting-in-ukraine/, 2026-07-07]. Tracxn says $263M; the figures conflict.
- **Defense:** More than 100 autonomous ATVs in Ukraine [TechCrunch, 2026-07-07]. A $114M Army autonomous-breaching contract (date n/a, UNVERIFIED). Was an ATV-S phase-2 vendor.
- **Commercial:** AutoTT terminal tractor with Kalmar, 2026 [Forbes, https://www.forbes.com/sites/richardbishop1/2025/11/25/in-a-new-era-forterra-pursues-both-defense-and-commercial-launches/, 2025-11-25].

### Overland AI (defense off-road autonomy)
- **Funding:** $100M led by 8VC, including $20M of TriplePoint venture debt; its prior raise was $42M [GlobeNewswire, https://www.globenewswire.com/news-release/2026/02/03/3231434/0/en/Overland-AI-Raises-100-Million-to-Scale-Ground-Autonomy-with-U-S-Armed-Forces.html, 2026-02-03].
- **Customers:** Works with the 82nd Airborne, 1st Cavalry, the 2nd Marine Logistics Group and other units.
- **Contract:** A $20M Engineer Autonomous Breaching contract, September 2026 (press listing, UNVERIFIED).

### Mineral Resources (MinRes) — Onslow Iron autonomous road trains (in-house, private road)
- **Plan:** 120 fully autonomous road trains (an older article says 150), each carrying about 330 t of ore.
  - The private haul road runs about 150 km from Ken's Bore to the Port of Ashburton and tunnels under the North West Coastal Highway and Onslow Road, so the trucks never meet public traffic.
  - Hexagon converts about 10 trucks a month.
  - Trucks are trialled with safety drivers before going driverless.
  - Sources: [MinRes, https://www.mineralresources.com.au/our-business/iron-ore/onslow-iron-project/autonomous-road-trains/, n/a]; [Australian Mining, https://www.australianmining.com.au/first-autonomous-road-trains-land-at-onslow-iron/, 2024]
- **Haul-road sale:** 49% of the haul road sold to Morgan Stanley Infrastructure Partners for about A$1.2B net cash after tax [MinRes, date n/a, UNVERIFIED date].
- **Current driverless truck count:** UNVERIFIED. No 2026 primary figure found.
- **Relevance:** A private, very long, repetitive road with huge volume, which is the ideal environment for convoys. The miner did it in-house with a systems integrator (Hexagon) rather than a startup.

### Global AHS scale (mining)
- **GlobalData:** 5,684 autonomous haul trucks, including autonomy-ready units, as of July 2026 (up from 3,832 in July 2025); 3,843 are in China [relayed via NewsBreak/Farmonaut, https://www.newsbreak.com/news/4895510296898-autonomous-mining-technology-2026-5-684-trucks-and-the-cost-curve, 2026]. Secondary relay, UNVERIFIED against GlobalData.
- **By operator:**
  - Rio Tinto: more than 400 trucks [Canadian Mining Journal, 2026]; another source says more than 360 across 17 mines [RTS Labs, 2026]; Mining Technology listed 305 [https://www.mining-technology.com/analyst-comment/china-global-leader-autonomous-surface-mining-trucks/, 2025-08]. Sources conflict.
  - BHP: about 360 (single analysis, UNVERIFIED).
  - Fortescue: more than 175 at its Chichester Hub, with a claimed 30% productivity gain (2026 article, secondary). Its Liebherr deal covers 360 battery-electric autonomous trucks over 2025–2030.
- **Parker Bay "2,000 trucks" figure:** Not found. UNVERIFIED.

---

## 4. Defense leader-follower convoys

### Expedient Leader-Follower (ExLF), US Army
- **2018 awards:**
  - TARDEC gave Oshkosh Defense $49M to integrate autonomy kits on PLS trucks: an initial 70 kits, with an option for up to 150 [Oshkosh, https://oshkoshdefense.com/u-s-army-awards-oshkosh-defense-49-million-to-bring-autonomous-technology-to-the-battlefield/, 2018].
  - Robotic Research got a 3-year, $49.7M contract for the autonomy kits [Military.com, https://www.military.com/defensetech/2018/06/12/army-spends-50-million-build-autonomous-supply-convoys.html, 2018-06-12].
  - The Army wanted 70 self-driving supply trucks by 2020 [Breaking Defense, https://breakingdefense.com/2018/08/army-wants-70-self-driving-supply-trucks-by-2020/, 2018-08].
- **2023 closure:**
  - The Army closed ExLF. It paid Oshkosh and Robotic Research to return the vehicles to non-robotic configuration; Robotic Research received $53.1M for a final 60-month sprint.
  - Robotic driving kits moved to a commercial acquisition path [Breaking Defense, https://breakingdefense.com/2023/06/army-closing-down-leader-follower-robotic-truck-development-eyeing-commercial-solutions/, 2023-06].
  - The technology did not go into production.
- **Pratt Miller:** No link to ExLF found. UNVERIFIED.

### Autonomous Transport Vehicle System (ATV-S): the successor
- **GEARS (DIU commercial retrofit-kit competition):** Missions are convoy operations, waypoint navigation and teleoperation. Awards went to Robotic Research, Neya and Carnegie Robotics [C4ISRNet, https://www.c4isrnet.com/unmanned/2023/12/19/gears-of-war-us-army-picks-3-companies-to-advance-autonomous-resupply/, 2023-12-19].
- **Phase 2:** Carnegie Robotics and Forterra were selected for prototyping. A mid-FY26 decision could narrow to one vendor; fielding was possible "as soon as 2027, if budgets allow" [Breaking Defense, https://breakingdefense.com/2025/02/army-picks-carnegie-forterra-for-autonomous-logistics-truck-prototyping/, 2025-02]. Budget delays and a continuing resolution threatened the schedule [National Defense Magazine, https://www.nationaldefensemagazine.org/articles/2025/2/25/just-in-budget-delays-could-affect-fielding-of-armys-autonomous-resupply-vehicle, 2025-02-25].
- **FY26 funding:** $8.3M under the PLS Extended Service Program to outfit HEMTT PLS A2 trucks with By-Wire Active Safety (BWAS), a prerequisite for ATV-S [Inside Unmanned Systems, https://insideunmannedsystems.com/report-what-unmanned-systems-is-americas-military-buying-in-2026/, 2025-11].
- **April 2026:** The Army called ATV-S the "centerpiece of our strategy" for wheeled-vehicle autonomy; its mission sets are convoy support and waypoint navigation [National Defense Magazine, https://www.nationaldefensemagazine.org/articles/2026/4/24/army-ponders-autonomys-place-in-wheeled-vehicle-fleet, 2026-04-24].
- **Phase-3 downselect:** Carnegie Robotics won, with the first unit to be issued by March 2028 [InsideDefense, https://insidedefense.com/insider/carnegie-robotics-wins-atv-s-downselect, 2026-04-29]. Paywalled, read via search summary. Award value undisclosed.
- **FY27 request:** About $22M for the leader-follower line inside the Family of Heavy Tactical Vehicles program element. The first unit was planned for 2Q FY27 [InsideDefense, 2026-04].
  - This conflicts with "by March 2028" above.
  - Defense Daily headline: "Army Aims To Begin Fielding First Autonomous Logistics Trucks In FY '27" [https://www.defensedaily.com/army-aims-to-begin-fielding-first-autonomous-logistics-trucks-in-fy-27/army/, 2026].
- **Reorganization:** The Army reshuffled its autonomy office as robotic priorities shifted [InsideDefense, https://insidedefense.com/share/227908, 2026, details UNVERIFIED].
- **Takeaway:** The defense convoy kit niche is small (about $22M a year in FY27) and has a selected incumbent (Carnegie Robotics). A pre-seed startup could only join through a subcontract or a small R&D award (SBIR/OTA), not the program of record.

### Other DoD ground-autonomy and logistics efforts (partial)
- Kodiak's $49.9M DIU contract (2022) and its Leonidas AGV with General Dynamics Land Systems (above).
- Forterra's breaching and autonomy contracts; Overland AI's breaching contract; Havoc + Teleo's convoy ambitions (above).
- JTAARS aerial cargo contract to SURVICE Engineering, May 2026 (search summary, UNVERIFIED).
- Not researched in depth (search budget exhausted): RCV, S-MET increment 2, Marine Corps autonomous logistics, and Robotic Research's current corporate status (RRAI). All UNVERIFIED.

---

## 5. Human-led leader-follower on PRIVATE industrial, mining or ag roads: white space?

**What the evidence shows is occupied:**
- **Fully driverless on private haul roads, already commercial:** Kodiak/Atlas (Class 8 on lease roads), Volvo (Brønnøy, per tonne), MinRes/Hexagon (road trains on a private road), Cat and Komatsu AHS (mines and quarries), Pronto (mixed-fleet quarries, with a cheap camera-only tier). All skip the human leader.
- **Supervised autonomy and teleoperation as the labor multiplier** (one operator, many machines): Teleo/Havoc, Fernride, ASI Mobius command station, Starsky's earlier model.
- **Human-led leader-follower on public highways:** Locomation and Peloton tried it, and both died.
- **Defense leader-follower:** ExLF closed; ATV-S downselected Carnegie Robotics.

**Not found in my searches, so possible white space (UNVERIFIED that it is empty):**
- I found no dedicated, funded startup selling **human-led leader-follower convoys of road-class trucks** (Class 6–8 dump trucks, tractor-trailers, road trains) on **private industrial routes**. Examples: aggregates/quarry-to-plant roads, frac sand, port-to-yard, steel and paper mills, timber haul roads, sugar or ag haul roads.
  - The one partial hint is promotional Gallatin AI material ("When the Convoy Drives Itself", https://www.gallatin.ai/news/autonomous-convoy, undated), which looks defense-oriented and unverified.
- **Known ag adjacencies I could not verify this session** (search budget exhausted), so treat them as likely competitors to check:
  - Raven/CNH OMNiDRIVE (a driverless grain cart following a human-driven combine): human-led leader-follower in ag. UNVERIFIED.
  - Kinze autonomous grain cart, AgXeed, Sabanto, Monarch. UNVERIFIED.
  - ASI's legacy "follow-me" capability. UNVERIFIED.

**Why the gap may exist (a warning, not only an opportunity):**
- Where the route is private and repetitive, the market has gone straight to full driverless (Kodiak, Volvo, MinRes, Cat, Komatsu, Pronto). The leader-follower wedge has to show it is meaningfully cheaper or faster to certify and deploy. Possible advantages:
  - The lead driver supplies perception and judgment, so no remote-ops center is needed.
  - Lower sensor cost.
  - Faster site commissioning.
  - Works on mixed public/private segments where full driverless is not yet allowed.
- Small sites (2–10 trucks) are the segment OEM AHS under-serves. But Pronto's tiered "AHS Editions" and its camera-only product target exactly the small quarry.

---

## 6. Gaps and items to verify next (search budget exhausted; WebFetch blocked for several hosts)
- Kodiak's per-truck Driver-as-a-Service pricing and Atlas's unit economics (check the KDK 10-Q/10-K and the Atlas AESI 10-Q).
- Pronto/Atoms deal date and terms; Pronto's total funding; SafeAI's prior funding.
- MinRes's current count of driverless road trains (ASX quarterly) and the date of the Morgan Stanley haul-road deal.
- Value of Carnegie Robotics' ATV-S award; the reconciled first-unit date (2Q FY27 vs March 2028); FY26 enacted funding for the leader-follower line.
- Raven OMNiDRIVE, Kinze and other ag leader-follower products and their pricing.
- ASI leader-follower capability; Polymath funding; ISEE funding.
- Buyer of Locomation's assets; buyer of Peloton's assets, if any.
- ATRI 2025/2026 cost-per-mile breakdown (fuel share vs driver wages) to size platooning fuel economics against labor.
- Aurora total funding, and Waabi's earlier rounds, confirmed from primary sources.
