# 04 — Regulation, Standards & Technical State of the Art
## Leader-follower haul-truck convoys (private sites to driverless; dual-use military)

Prepared 2026-11-11 for technical/regulatory due diligence.

### How to read this file
- Each claim has a source URL and a source date. "acc." means accessed 2026-11-11 through a web-search result snippet.
- **[SEC]** marks a secondary source: a vendor, law-firm or trade blog, or a snippet summary.
- **[UNVERIFIED-PK]** marks prior knowledge that I could not check against a source in this session. Treat it as a hypothesis to verify.
- **[UNVERIFIED]** marks a claim that was searched for but not confirmed, or where sources conflict.

### Research limits in this session
1. **WebFetch could not open most primary sources.** These hosts did not resolve in the sandbox: eur-lex.europa.eu, congress.gov, ogletree.com, pitandquarry.com, hklaw.com and azcapitoltimes.com. For those, I relied on search-engine snippets of the documents.
2. **The shared WebSearch budget ran out partway through.** The limit is 200 calls per turn across all agents. Sections 5 (by-wire vendors), 6 (safety statistics) and 7 (simulation tools) therefore rest mostly on **[UNVERIFIED-PK]** and need a follow-up pass.

---

## 1. Standards

### 1.1 ISO 17757 — Earth-moving machinery and mining: autonomous and semi-autonomous machine system (ASAMS) safety
- **Latest edition.** ISO 17757:2019 (2nd edition) is the current version. It replaced ISO 17757:2017, which is withdrawn.
  - https://www.iso.org/standard/76126.html (2019)
  - https://knowledge.bsigroup.com/products/earth-moving-machinery-and-mining-autonomous-and-semi-autonomous-machine-system-safety (BSI, 31 Aug 2019)
  - https://www.iso.org/standard/60473.html (2017 edition, withdrawn)
- **No 3rd edition or DIS found** as of 2026-11 [UNVERIFIED]. Check the ISO/TC 127/SC 2 work programme.
- **Scope.** It covers safety requirements for autonomous and semi-autonomous machines and their systems across the whole life cycle: machine, infrastructure, hardware and software. It applies to the earth-moving machinery defined in ISO 6165, and to surface and underground mobile mining machines.
  - https://www.iso.org/standard/76126.html (2019)
  - https://cdn.standards.iteh.ai/samples/76126/01f23b9df1bd4b9c9633dbe61ac694bd/ISO-17757-2019.pdf (2019 preview)
- **Exclusions.** It does not cover remote control (that is ISO 15817). It also excludes function-specific automated features unless they are used as part of an ASAMS. Source: same ISO page (2019).
- **Companion standards.** General machine safety comes from ISO 20474 (earth-moving) and ISO 19296 (underground mining) (ISO 17757 preview, 2019).
- **Requirements visible in the preview and snippets** (https://cdn.standards.iteh.ai/samples/76126/...ISO-17757-2019.pdf, 2019):
  - Safety-related control parts must meet functional-safety requirements under ISO 13849, ISO 19014, IEC 62061 or IEC 61508.
  - Every autonomous machine must be stoppable from a safe remote distance, and any remote supervisor system needs its own all-stop.
  - The ASAMS must go to a safe state when it detects loss of brake stored energy.
  - EMC must meet ISO 13766-1/-2.
  - Fire suppression is required if the risk assessment calls for it; the risk assessment also decides whether activation is automatic or remote.
  - Purchase the full text before relying on clause-level detail.
- **Known gap.** A 2020 VTT review says these standards are written mainly for machine manufacturers and do not address worksite-level responsibilities. https://doi.org/10.1515/eng-2020-0074 (2020)
- **Implication for the founder.** In a leader-follower convoy, each driverless follower is an "autonomous machine" under ISO 17757. The manned leader plus the V2V/supervisor link form part of the ASAMS infrastructure. ISO 17757 has no specific leader-follower clauses [UNVERIFIED-PK], so the safety case must argue the leader-dependence explicitly.

### 1.2 ISO 23725:2024 — Autonomous system (AHS) to fleet management system (FMS) interoperability
- **Edition.** First edition, published August 2024. https://www.iso.org/standard/76769.html (2024)
- **Scope.** It defines the interfaces for:
  - dispatching haul-truck tasks;
  - real-time communication and message definitions;
  - sharing the mine map;
  - production monitoring.
  It applies to surface mining. https://cdn.standards.iteh.ai/samples/76769/751f0d4094bd4da08e808c4dccd08368/ISO-23725-2024.pdf (2024)
- **Purpose.** To make AHS–FMS integration reproducible and avoid custom work at every site (same source).
- **Out of scope:**
  - authentication, authorization and cybersecurity;
  - safe execution of tasks, which is left to agreements between the FMS and AHS suppliers (same source).
- **Commentary.** The standard defines only the boundary between vendors, not the internals of the autonomy stack. https://www.openautonomy.com/article/interface-control-documents-autonomous-mining (date n/a) [SEC]
- **Implication.** Aligning the convoy-manager API with ISO 23725 would let the product plug into incumbent FMSs (Cat MineStar, Komatsu, Hexagon, Wenco and others) [UNVERIFIED-PK that these vendors implement it].

### 1.3 ISO 21815 — Collision warning and avoidance (CWAS/CAS) for earth-moving machinery
- **Part 1:2022 (general requirements).** https://cdn.standards.iteh.ai/samples/77302/f1d344788b2f410c84afc741608a7839/ISO-21815-1-2022.pdf (2022). The DIS ballot opened 2020-03-23.
- **ISO/TS 21815-2:2021 (on-board J1939 communication interface).** It defines a J1939 interface through which a separate collision-avoidance device can intervene in machine motion. https://www.iso.org/standard/77303.html (2021-07)
  - This is directly relevant to retrofits: it is a standardized way for an add-on device to command slow-down or stop on machines that implement it.
- **Part 3:2023 (risk area and risk level for forward/reverse motion).** Type-C standard. https://www.iso.org/standard/77266.html (2023)

### 1.4 ISO 3691-4:2023 — Driverless industrial trucks and their systems
- **Scope.** Safety requirements and verification for driverless industrial trucks and their systems: control system, guidance means and power system. It also applies to trucks with operator-enabled automatic modes, rider capability, manual modes or maintenance mode.
- **Exclusions.** Rail or mechanically guided trucks, remote-controlled trucks, power sources, towed trailers and elevating-operator trucks. Annex A covers preparation of the operating zone.
- **Sources:**
  - https://www.iso.org/standard/83545.html (2023)
  - https://blog.ansi.org/ansi/iso-3691-4-2023-driverless-industrial-trucks/ (n/a)
- **Revision under way.** ISO/DIS 3691-4 will replace the 2023 edition. https://www.iso.org/standard/88615.html (acc.)
- **EU status.** The EN version is harmonized under the Machinery Directive. The 2023/1230 Regulation replaces the Directive from January 2027. https://www.fabrico.io/blog/iso-3691-4-driverless-industrial-trucks/ [SEC]
- **Implication.** Haul trucks and ADTs are not "industrial trucks" (forklifts, tuggers, AMRs) [UNVERIFIED-PK]. ISO 3691-4 is therefore relevant only for industrial-campus tugger or yard-tractor variants. Its Annex A zone-preparation approach is a useful model for site preparation.

### 1.5 Functional safety: ISO 25119, ISO 19014 and ISO 13849
- **ISO 25119** (agricultural and forestry tractors and machinery, safety-related control parts).
  - Part 1:2018 + Amd 1:2020 set general principles. Application-specific AgPL comes from type-C standards or the manufacturer. https://www.iso.org/standard/69025.html (2018)
  - There are five AgPLs (a–e), plus QM for functions that are not safety-related. AgPL is derived through a HARA based on severity, exposure and avoidance. https://cdn.standards.iteh.ai/samples/78306/.../ISO-25119-2-2019.pdf (2019)
  - A revision is in progress (ISO/CD 25119-1.2). https://www.iso.org/standard/93007.html (acc.)
  - EN ISO 25119-1:2023 exists. https://standards.iteh.ai/catalog/standards/cen/f370b529-c4c4-4f82-bcc2-cb742d315d62/en-iso-25119-1-2023 (2023)
  - ISO 25119 applies only to ag/forestry machines. For haul trucks and ADTs, the correct sector standard is **ISO 19014** (earth-moving).
- **ISO 19014** (earth-moving).
  - It is published in parts and uses "MPL" (machine performance level). An industry white paper refers to "MPLa" (achieved) and two methods for determining performance. https://www.cmeig.com.au/wp-content/uploads/CMEIG-EMESRT-ICMM-White-Paper-and-Guiding-Principles-for-Functional-Safety-on-Earthmoving-Machinery-Ver.-0.5-March-2020.pdf (CMEIG/EMESRT/ICMM, Mar 2020)
  - "MPLr" means *required* MPL [UNVERIFIED-PK; inferred by analogy with AgPLr].
  - Part breakdown [UNVERIFIED-PK]: Part 1 is methodology and MPLr; Part 2 is hardware design; Part 3 is environmental; Part 4 is software.
- **ISO 13849-1** (generic machinery, PL a–e). ISO 17757 accepts it as an alternative (ISO 17757 preview, 2019). The latest edition is ISO 13849-1:2023 [UNVERIFIED-PK].
- **Related standards (all [UNVERIFIED-PK]):**
  - ISO 18497 series (2024) on autonomous agricultural machine safety.
  - ISO 26262 and ISO 21448 (SOTIF), which apply to road vehicles and matter only for any public-road product.
  - ISO/PAS 8800:2024 (AI in road vehicles).

### 1.6 UL 4600 — Evaluation of autonomous products (safety-case standard)
- **Current edition.** Edition 3, published 17 March 2023. It adds requirements specific to autonomous trucking.
  - https://webstore.ansi.org/standards/ul/ul4600ed2023 (2023-03-17)
  - https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking (2023)
- **Scope** (https://users.ece.cmu.edu/~koopman/pubs/Koopman2023_UL4600_Safety_Case.pdf, Koopman 2023; Koopman 2023 lecture https://users.ece.cmu.edu/~koopman/lectures/2023-06-19-UL4600_update_Koopman.pdf):
  - The safety case covers infrastructure, services and support processes.
  - It covers individual vehicles that are members of a team, and teams as a whole. This directly fits convoys.
  - It assumes no human supervision for the driving task; driver handoff is out of scope.
  - Foreseeable misuse and physical attacks are in scope.
  - It sets no fixed mileage requirement. Evidence must combine analysis, simulation, closed-course testing and (if relevant) road testing.
  - Its prompt lists are non-exhaustive, so the safety case is "necessary but possibly not sufficient."
- **Off-road.** UL markets UL 4600 training for off-road and fleet vehicles. https://www.ul.com/services/achieve-confidence-autonomous-vehicle-safety-ul-4600 (n/a) [SEC]. Whether the standard's scope clause explicitly covers off-road vehicles is [UNVERIFIED].
- **Implication.** UL 4600 is the best available *structure* for the safety case. Use it alongside ISO 17757 compliance and ISO 19014/13849 functional safety.
- **Leader-follower caveat.** In phase 1 the human leader is part of the system, which UL 4600 treats as out of scope. Handle the leader through human-factors analysis inside the safety case.

### 1.7 EU Machinery Regulation (EU) 2023/1230
- **Application date.** It applies from **20 January 2027**, repealing Directive 2006/42/EC. The OJ L 165 text (29 Jun 2023) said 14 January; a corrigendum (OJ L 169, 4 Jul 2023) changed it to 20 January.
  - https://physical-ai-safety.com/blog/eu-machinery-regulation-2027-primer (n/a) [SEC]
  - https://osha.europa.eu/en/legislation/directive/regulation-20231230eu-machinery (EU-OSHA, acc.)
- **Structure.**
  - The essential health and safety requirements moved to Annex III.
  - Annex I lists high-risk categories. Part A requires mandatory third-party (notified-body) assessment.
  - Annex I Part A item 5 covers "safety components with fully or partially self-evolving behaviour using machine learning approaches ensuring safety functions."
  - https://sopx.io/insights/eu-machinery-regulation-2023-1230/ (n/a) [SEC]
  - https://literature.rockwellautomation.com/idc/groups/literature/documents/sp/oem-sp123_-en-p.pdf (Rockwell guide) [SEC]
- **Autonomous mobile machinery.** Requirements are in Annex III §3, cited by one checklist as §3.6.3.3 [SEC, sopx.io]. Cybersecurity is in §1.1.9 (protection against corruption) and control-system reliability against "reasonably foreseeable malicious attempts" is in §1.2.1. https://inkog.io/labs/eu-machinery-regulation-ai-agents [SEC]
- **Verbatim Annex III text was not retrieved** (eur-lex unreachable). From prior knowledge [UNVERIFIED-PK]:
  - Annex III §1.1.1 defines "autonomous mobile machinery" as mobile machinery with an autonomous mode in which all essential safety functions are ensured in its travel and working zone without permanent operator interaction.
  - A §3.2.4 "supervisor function" must let a remote supervisor receive information, stop the machine, start it safely and bring it to a safe state.
  - The control system must keep the machine within its defined travel and working zone.
- **Further prior-knowledge points [UNVERIFIED-PK]:**
  - A party who **substantially modifies** machinery (such as retrofitting autonomy) is treated as the **manufacturer** and must do the full conformity assessment (Art. 18 / definition of substantial modification).
  - The EU AI Act (Reg. 2024/1689) treats AI safety components of Machinery-Regulation products as high-risk, with obligations from August 2027. Delays were proposed in 2025 "digital omnibus" talks.
  - Implication: a **frozen** (non-self-evolving) RL or ML policy may avoid Annex I Part A item 5. If RL performs a safety function, expect notified-body involvement and AI Act obligations.

### 1.8 Australia
- **WA Code of Practice "Safe mobile autonomous mining in Western Australia".**
  - Launched in 2015 as a claimed world first, after an 18-month working-group effort. It was created because autonomous systems were not specifically covered by the Mines Safety and Inspection Act 1994.
    - https://www.australianmining.com.au/wa-launch-world-first-autonomous-mining-safety-codes/ (2015)
    - https://www.dmp.wa.gov.au/News/DMP-code-to-guide-global-safety-20574.aspx (2015)
  - The current WorkSafe WA page is dated **15 Feb 2025**. https://www.worksafe.wa.gov.au/publications/safe-mobile-autonomous-mining-western-australia-code-practice (2025-02-15); PDF: https://www.worksafe.wa.gov.au/system/files/migrated/sites/default/files/atoms/files/221188_cp_msms_0.pdf
  - Its aims include setting safety outcomes for approval of the project management plan.
  - It continues in effect under the WHS Act 2020 transitional arrangements. "Must" denotes a legal requirement and "should" a recommendation. https://www.dmp.wa.gov.au/Safety/Codes-of-practice-16145.aspx (acc.)
  - What changed in 2025 is [UNVERIFIED].
- **GMG guideline.** The Global Mining Guidelines Group based its international guideline on the WA code (DMP news, 2015). The GMG "Guideline for the Implementation of Autonomous Systems in Mining" (2019) and its functional-safety guideline are [UNVERIFIED-PK].
- **NSW and Queensland** guidance for autonomous plant is [UNVERIFIED].

---

## 2. US regulators

### 2.1 MSHA (mines and quarries, including stone, sand and gravel)
- **No autonomy-specific MSHA rule, approval process or RFI from 2025–2026 was found** [UNVERIFIED; searched].
  - Press reports from 2023 said MSHA would probably not consider its regulations applicable unless there is a specific hazard to miners near autonomous vehicles. They also said operators would not need a separate Mine ID per autonomous unit. https://ogletree.com/insights-resources/blog-posts/msha-weighs-in-on-the-future-of-autonomous-mining-equipment/ (2023) [SEC]
- **NIOSH Automation and Emerging Technologies Partnership.** A forum on mobile-equipment autonomy (full and supervised) and collision avoidance, with MSHA invited. https://www.cdc.gov/niosh/mining/partnerships/automation.html (Feb 2025)
- **Safety Program for Surface Mobile Equipment final rule.**
  - Published 20 Dec 2023, effective 19 Jan 2024, compliance required from **17 Jul 2024**.
    - https://www.federalregister.gov/documents/2023/12/20/2023-27640/safety-program-for-surface-mobile-equipment (2023-12-20)
    - https://www.dol.gov/newsroom/releases/msha/msha20231219 (2023-12-19)
  - Operators and independent contractors using surface mobile equipment must keep a written safety program. MSHA reviews it during inspections; **no advance approval is required**. https://www.msha.gov/final-rule-safety-program-surface-mobile-equipment (acc.)
  - Rule FAQ: https://www.msha.gov/sites/default/files/Regulations/FREQUENTLY-ASKED-QUESTIONS-2024-07-15.pdf (2024-07-15)
- **Status of the rule, 2025–2026.**
  - It was **not** among the 18 deregulatory proposals MSHA published in July 2025. Those targeted blacksmith shops, surface drilling, aerial tramways, trolleys and hazard communication.
    - https://ogletree.com/insights-resources/blog-posts/what-mshas-deregulatory-push-means-for-surface-mines/ (2025) [SEC]
    - https://www.jacksonlewis.com/insights/what-mshas-18-proposed-rules-signal-about-future-mine-safety (2025) [SEC]
  - A Federal Register notice of **27 Jul 2026** proposes extending the rule's information collection (comments due 25 Sep 2026), so the rule is in force. https://www.federalregister.gov/documents/2026/07/27/2026-15141/proposed-extension-of-information-collection-safety-program-for-surface-mobile-equipment (2026-07-27)
  - MSHA issued 863 citations under the rule between July 2024 and April 2026, and citations are trending down. https://ogletree.com/insights-resources/blog-posts/trends-in-msha-surface-mobile-equipment-compliance/ (Jul 2026) [SEC]
- **Unverified related points:**
  - Whether autonomous trucks fall under the program is inferred from the broad definition of "mobile equipment" [UNVERIFIED].
  - CFR sections 30 CFR 56.23000–56.23004 and their Part 57/77 analogues [UNVERIFIED-PK].
  - The program reportedly must address hazard identification, maintenance, training and **evaluation of safety technologies** such as collision warning and avoidance [UNVERIFIED-PK; MSHA safety-topic page: https://www.msha.gov/safety-and-health/safety-and-health-materials/safety-topics/safety-topic-mobile-equipment-surface-mines].
- **Existing MSHA standards that still apply to driverless trucks** [UNVERIFIED-PK]:
  - 30 CFR 56.9100 (traffic control)
  - 56.9101 (speed and control)
  - 56.14101 (brakes able to stop and hold a typical load on the maximum grade)
  - 56.9300 (berms)
  - 56.14100 (pre-shift inspection, which is a procedural problem when there is no driver)
  - 56.14132 (backup alarms)
  - An autonomy vendor working on site would likely need an MSHA independent-contractor ID and Part 46/48 training.
- **US precedent: Freeport-McMoRan Bagdad (Arizona).**
  - 33 Cat 793 trucks are being converted with Cat MineStar Command. Announced Sept 2023, on a 3-year plan: two years of retrofit, infrastructure and training, then operation. https://im-mining.com/2023/09/15/freeport-says-bagdad-to-have-first-fully-autonomous-mining-truck-fleet-in-us/ (2023-09-15)
  - The first trucks ran in an **isolation zone** separate from manned operations. https://www.fcx.com/freeport-features/010825 (2025-01-08)
  - The full fleet of 33 has run autonomously since August 2025, with 7 manned trucks in a separate area during transition and a "zero job loss" goal. https://www.fcx.com/freeport-features/100125 (2025-10-01)
  - Whether the last manned truck was parked by end of 2025 is [UNVERIFIED].
  - **The isolation-zone-first approach is the de facto MSHA-era playbook.**
- **USW/Asarco (Arizona).** Autonomous trucks are in service; see §3.3.

### 2.2 OSHA (non-mine private industrial campuses)
- **Jurisdiction.** MSHA covers mines and quarries; OSHA covers other workplaces under the 1979 MSHA–OSHA interagency agreement [UNVERIFIED-PK].
- **No AV-specific OSHA standard.**
  - 29 CFR 1910.178 (powered industrial trucks) assumes a human operator but covers much of the same ground. https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178 ; https://palletenterprise.com/safety-check-forklifts-and-other-automated-guided-vehicles/ [SEC]
  - Hazards not covered by a standard fall under the **General Duty Clause** (5(a)(1)). https://www.osha.gov/fom/chapter-4 (OSHA FOM, acc.)
  - **ANSI/ITSDF B56.5** is the US consensus standard for driverless industrial vehicles [SEC]. A report that OSHA proposed incorporating B56.5 is [UNVERIFIED].
- **Implication.** On a campus, follow recognized consensus standards: ISO 3691-4 or B56.5 for tugger and yard vehicles, and ISO 17757 for haul trucks. Any public-road crossing triggers state vehicle code and state AV law.

### 2.3 FMCSA, NHTSA and DOT (public roads only)
- **Aurora warning-device exemption.** FMCSA granted a **5-year exemption (7 Oct 2026 – 7 Oct 2031)** allowing cab-mounted warning beacons in place of warning triangles for Level 4 ADS trucks.
  - It covers 49 CFR 392.22(b), 393.25(e) and 393.95(f).
  - Other carriers can use it if they notify FMCSA in writing.
  - Longer combination vehicles are limited to two 28-ft trailers.
  - Sources: https://www.fmcsa.dot.gov/sites/fmcsa.dot.gov/files/2026-10/Aurora%20Operations%20Exemption_10-07-2026.pdf (2026-10-07); https://www.federalregister.gov/documents/2026/10/09/2026-20734/parts-and-accessories-necessary-for-safe-operation-application-for-exemption-from-aurora-operations (91 FR 64736, 2026-10-09)
- **History of the exemption.**
  - Denied December 2024.
  - A waiver ran 10 Apr – 9 Jul 2026. https://www.fmcsa.dot.gov/sites/fmcsa.dot.gov/files/2026-04/Cover%20Letter%20to%20Aurora%20Operations%20Inc.%20-%20Waiver%20of%20Warning%20Device%20Requirements%20-%20April%208%202026.pdf (2026-04-08); https://www.federalregister.gov/documents/2026/04/15/2026-07288/... (2026-04-15)
  - Aurora reported 34 trucks, more than 500k miles and no stopped-vehicle collisions while using beacons (Oct 2025 – Jan 2026).
  - OOIDA opposed the exemption; AUVSI, TechNet and Volvo supported it (FMCSA notice, 2026-10).
- **No comprehensive FMCSA ADS rule found.** Earlier steps were the 2019 ANPRM and a 2023 supplemental ANPRM [UNVERIFIED-PK].
- **NHTSA.** The April 2025 "AV Framework" and the expanded exemption program [UNVERIFIED-PK]. Retrofitted vehicles that keep manual controls generally stay FMVSS-compliant [UNVERIFIED-PK].

### 2.4 State law (public roads)
- **Platooning** (following-too-closely exemptions):
  - 19 states by April 2019. https://www.multistate.us/insider/2019/4/11/states-open-highways-to-truck-platooning-technology (2019-04-11)
  - "Nearly two dozen" by 2019. https://cei.org/studies/authorizing-automated-vehicle-platooning-2019-edition/ (2019)
  - 20 states plus 4 pending. https://simonlawpc.com/trucking-accident/autonomous-truck-platooning-legal-in-your-state/ (Nov 2023) [SEC]
  - **No authoritative 2025–2026 count was found** [UNVERIFIED]. Check the NCSL AV legislation database.
  - Many statutes still require a CDL driver in each truck [SEC].
- **Kentucky.** HB 252 (2025) proposed requiring a CDL holder in each platoon vehicle, reversing a 2024 lead-only law. Its final outcome is [UNVERIFIED].
  - https://apps.legislature.ky.gov/record/25RS/hb252.html (2025)
  - KRS 281.764 was amended by 2024 Ky. Acts ch. 176, effective 1 Aug 2026. https://apps.legislature.ky.gov/law/statutes/statute.aspx?id=55850 (acc.)
- **Driverless operation, count by state.** Estimates range from about 20 to 26 states that allow it under conditions; the count depends on how "driverless" is defined.
  - "At least 26 states." https://elevationprovinggrounds.com/article/where-are-autonomous-vehicles-legal/ (Sept 2026 per snippet) [SEC]
  - "21 states." [SEC; aggregate]
  - "5 states" (outdated). https://thetexaslawdog.com/faqs/which-states-allow-driverless-trucks/ [SEC]
  - **Treat as UNVERIFIED** until checked against NCSL.
- **Texas.** A requirement for authorization of commercial driverless operation became enforceable on 28 May 2026 [SEC; aggregate snippet, UNVERIFIED].
- **California.** OAL approved the DMV heavy-duty (≥10,001 lb) AV regulations on **28 Apr 2026**.
  - Phased path: safety-driver testing, then driverless testing, then deployment.
  - A safety case covering 14 areas is required.
  - Reported mileage: 500k heavy-duty miles per phase [SEC; verify].
  - Police "notice of AV noncompliance" from 1 Jul 2026.
  - Sources: https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/california-autonomous-vehicle-regulations/autonomous-vehicle-milestones/ (2026-04-28); https://www.dmv.ca.gov/portal/news-and-media/new-autonomous-vehicle-regulations-strengthen-oversight-and-enforcement-authorize-trucks-and-transit/ (2026-04); https://environmentalhealthsafetybrief.sidley.com/2026/05/08/california-finalizes-a-new-regulatory-regime-for-testing-and-deploying-autonomous-vehicles/ (2026-05-08)
  - SB 1246 (commercial-AV incident response) is pending. https://www.act-news.com/news/california-av-bill-advances-as-heavy-duty-testing-rules-take-effect/ (2026)

### 2.5 Federal AV legislation (119th Congress)
- **SELF DRIVE Act of 2026 (H.R. 7390).** The House E&C Commerce, Manufacturing and Trade subcommittee forwarded it on a **12–11 party-line vote**. Democrats objected to preemption of state law.
  - https://energycommerce.house.gov/posts/cmt-subcommittee-forwards-motor-vehicle-safety-and-automotive-leadership-bills-to-full-committee (2026)
  - https://legis1.com/news/house-markup-advances-auto-safety-bills-av-law-splits-panel (2026)
  - https://environmentalhealthsafetybrief.sidley.com/2026/01/08/members-of-congress-propose-a-new-bill-to-regulate-autonomous-vehicles/ (2026-01-08)
  - No full-committee or floor action was found [UNVERIFIED].
- **BUILD America 250 Act (H.R. 8870).** A five-year, $580B surface-transportation reauthorization. House T&I approved it **62–2 on 22 May 2026**.
  - Subtitle E, "Safe Integration of Autonomous Commercial Vehicles," creates a federal framework for ADS commercial motor vehicles and brings them into federal preemption review.
  - https://transportation.house.gov/news/documentsingle.aspx?DocumentID=410054 (2026-05-22)
  - A Holland & Knight headline says "House Passes…", but its body describes only committee action. https://www.hklaw.com/en/insights/publications/2026/06/house-passes-autonomous-vehicles-framework (2026-06) [SEC]
  - **No House floor vote or Senate action was confirmed** [UNVERIFIED]. What happened at the 30 Sep 2026 expiry of current surface-transportation law is also [UNVERIFIED].
- **Senate.** The Autonomous Vehicle Acceleration Act of 2025 (S. 1798) was introduced. https://www.congress.gov/bill/119th-congress/senate-bill/1798/text (2025). Senate Commerce reportedly is drafting broader AV legislation [SEC].
- **Bottom line.** As of 2026-11 there is **no enacted federal AV statute** [SEC + no contrary evidence found].

---

## 3. Labor and union constraints

### 3.1 ILA (US East and Gulf Coast ports)
- **Contract.** A six-year ILA–USMX Master Contract running **1 Oct 2024 – 30 Sep 2030**, ratified in February 2025 with about 99% approval. https://www.iladistrict.com/ila-ratifies-six-year-master-contract-with-nearly-99-approval-record-wage-increases-automation-protections-until-2030/ (2025-02)
- **Automation terms** (https://www.freightwaves.com/news/ila-usmx-reach-agreement-avoiding-strike-at-us-ports, 2025-01; https://www.logisticsmgmt.com/article/ila_membership_ratifies_new_master_contract_with_the_usmx, 2025-02):
  - No fully automated terminals and no fully automated equipment during the term.
  - Semi-automated equipment only after the parties agree on workforce protections and staffing, with jobs tied to each piece of equipment.
  - A New Technology Committee of 7 members per side.
  - The full text was not released publicly.
- **Implication.** Driverless followers at East/Gulf ports are effectively **blocked until at least September 2030**. A manned-lead, driverless-follower setup is "semi-automated" at best and needs union agreement.

### 3.2 ILWU (US West Coast)
- **Automation rights.** Employers have had the right to automate since 2008, under **Section 15** of the Pacific Coast Longshore Contract Document. Coast Arbitrator rulings have upheld it.
  - https://www.pmanet.org/fact-sheets/technology-automation/ (acc.)
  - https://www.pmanet.org/wp-content/uploads/2025/03/Technology-Automation-Fact-Sheet-5.pdf (2025-03)
- **Current contract:** 2022–2028. https://ilwu13.com/wp-content/uploads/simple-file-list/Contracts/Pacific_Coast_Longshore_Contract_Document_2022-2028.pdf. The expiry of 1 Jul 2028 is [UNVERIFIED-PK].
- **Conflicting claim.** One analysis says the CBA expired 1 Jul 2026 and that a tentative six-year deal was reached. https://www.als-int.com/insights/posts/west-coast-port-labor-automation-dispute-2026/ [SEC]. This **conflicts with the 2022–2028 term** and is [UNVERIFIED].
- **Implication.** Automation is contractually possible on the West Coast but politically contested. Ports are not a good beachhead market.

### 3.3 Teamsters and state human-operator bills
- **California:**
  - **AB 316 (2023)** would have required a human in AVs over 10,000 lb. Vetoed by Newsom as "unnecessary." https://www.kqed.org/news/11962285/newsom-vetoes-bill-that-would-mandate-humans-in-autonomous-trucks-could-cost-state-a-quarter-million-jobs (2023)
  - **AB 2286 (2024)** was a re-run that passed the legislature and was **vetoed** in 2024. https://www.ttnews.com/articles/california-self-driving (2024); https://www.planetizen.com/news/2024/10/132026-california-governor-vetoes-autonomous-truck-ban-approves-new-guidelines-avs (2024-10)
  - **AB 33 (2025)** would require a human safety operator in AVs delivering commercial goods.
    - Passed the Assembly 57–7–15 (May/June 2025). https://aguiar-curry.asmdc.org/press-releases/20250605-california-assembly-passes-ab-33-bipartisan-support-safe-and-responsible (2025-06-05)
    - Ordered to the **Senate inactive file on 9 Sep 2025** as a two-year bill. https://legiscan.com/CA/bill/AB33/2025 ; https://ww2.arb.ca.gov/2025-assembly-bill-33-aguiar-curry-cecilia-autonomous-vehicles-2-year-0 (2025–26)
    - Still on the Senate floor as of December 2025. https://techcrunch.com/2025/12/04/californias-ban-on-self-driving-trucks-could-soon-be-over/ (2025-12-04)
    - **Final 2026 outcome: UNVERIFIED.**
- **Colorado.** HB25-1122 would have required a CDL holder aboard autonomous commercial vehicles. It passed the House 55–9 and the Senate 27–8, and was **vetoed by Gov. Polis on 29 May 2025**.
  - https://www.leg.colorado.gov/bills/hb25-1122
  - https://coloradonewsline.com/2025/06/09/polis-vetoes-11-bills/ (2025-06-09)
  - https://www.safetyandhealthmagazine.com/articles/26915-colorado-governor-vetoes-bill-banning-driverless-trucks (2025)
- **Other states.** Human-operator bills reportedly failed in Delaware, Indiana, Maryland and Texas [SEC aggregate; UNVERIFIED].
- **Relevance.** These bills target **public roads**. None found restricts autonomy on private mine or industrial sites.

### 3.4 Mining unions
- **USW at Asarco (Arizona).**
  - Bargaining barred Asarco from using autonomous haulage to cut union jobs. 72 displaced drivers were moved into other roles, including **autonomous-truck escorts**, plus maintenance training.
  - A safety chair reported that the trucks "are constantly stopped."
  - https://usw.org/news/usw-members-focus-on-jobs-safety-as-asarco-rolls-out-autonomous-trucks/ (2026-03)
  - **Nuisance stops are a real operational and political KPI.**
- **CFMEU / Mining and Energy Union (Queensland).** It opposed BHP's autonomous truck rollouts (for example Goonyella Riverside, 86 trucks) as job cuts and challenged their safety impacts. https://meu.org.au/beware-job-cuts-in-disguise-as-driverless-trucks-roll-in/ (2020-07-08) [union statement]
- **Freeport's "zero job loss" pledge** at Bagdad: https://www.fcx.com/freeport-features/100125 (2025-10-01).
- **Implication.** Leader-follower keeps one driver per convoy. Position it as **redeployment**: leader drivers, escorts and technicians. Productivity wins are preferable to headcount cuts in union sites.

---

## 4. Leader-follower technical state of the art

### 4.1 US Army ExLF and ATV-S
- **Architecture.**
  - Robotic Research supplied the autonomy software (AutoDrive). Oshkosh supplied the **drive-by-wire / active-safety kit** on M1075 PLS trucks. https://www.oshkoshcorp.com/blog/2022/1-11-22-autonomous-technology-feature-story (2022-01-11)
  - Oshkosh's 2018 contract was about $49M for an initial 70 kits, with an option up to 150. https://oshkoshdefense.com/u-s-army-awards-oshkosh-defense-49-million-to-bring-autonomous-technology-to-the-battlefield/ (2018); https://breakingdefense.com/2018/08/army-wants-70-self-driving-supply-trucks-by-2020/ (2018-08)
- **Sensors and software.**
  - Lidar, radar, cameras and GPS with ML (Oshkosh, 2022).
  - Built on the government-owned **Robotics Technology Kernel (RTK)**, **ROS-M** and MOSA. Robotic Research said its stack was MOSA-compliant so the government could add its own modules. https://www.nationaldefensemagazine.org/articles/2022/1/21/army-sees-progress-with-leader-follower-vehicle-technology (2022-01-21)
- **Behaviors** (National Defense, 2022; Oshkosh, 2022):
  - Follow directly behind a lead truck.
  - Convoy **assembly and disassembly** on command.
  - **Retrotraverse**, including reversing with trailers.
  - Teleoperation from another vehicle in the convoy.
  - Optionally manned at any time.
- **Fielding and feedback.** The 41st Transportation Company had about 60 leader-follower PLS trucks. Soldier feedback fed **90-day agile software sprints** (National Defense, 2022).
- **Outcome.**
  - In June 2023 the Army **wound down in-house ExLF development**. Close-out contracts returned the fleet to non-robotic trucks, and the Army pivoted to commercial kits through DIU's **GEARS** prototype effort and the **ATV-S** program. https://breakingdefense.com/2023/06/army-closing-down-leader-follower-robotic-truck-development-eyeing-commercial-solutions/ (2023-06)
  - In 2023, GEARS selected Carnegie Robotics, Forterra and Neya Systems. https://www.diu.mil/latest/accelerating-autonomous-vehicle-technology-for-the-dod (acc.)
  - In February 2025, **Carnegie Robotics and Forterra** were picked for ATV-S prototyping, with a **mid-FY2026 decision point**. https://breakingdefense.com/2025/02/army-picks-carnegie-forterra-for-autonomous-logistics-truck-prototyping/ (2025-02)
  - Forterra holds a GEARS Phase II award. https://www.forbes.com/sites/richardbishop1/2025/11/25/in-a-new-era-forterra-pursues-both-defense-and-commercial-launches/ (2025-11-25)
  - The result of the mid-FY2026 decision is [UNVERIFIED].
  - That Forterra is the rebranded Robotic Research autonomy unit is [UNVERIFIED-PK].
- **Lessons for the founder:**
  1. The Army wants **commercial, OEM-agnostic retrofit kits** with open interfaces (MOSA, RTK, ROS-M compatibility).
  2. Iterating directly with users on a fast cadence was valued.
  3. Incumbents with an ExLF pedigree (Forterra, Carnegie Robotics) are in the ATV-S lane. A pre-seed entrant would realistically enter through SBIR, xTech or subcontracting.
  - A detailed public ExLF lessons-learned report was not found. Check DTIC and GVSC.
  - An NPS thesis exists on applying ExLF. https://calhoun.nps.edu/server/api/core/bitstreams/288f732c-8d0b-45b8-8026-5ccee2a3c78b/content (n/a)

### 4.2 Academic work: breadcrumb / delayed-path tracking and string stability
- **Leader-path following with a waypoint queue** (essentially breadcrumbs). Real-time system tested on UGVs on-road and off-road. Zhao, Yao et al., "Design of leader's path following system for multi-vehicle autonomous convoy." https://www.semanticscholar.org/paper/Design-of-leader's-path-following-system-for-convoy-Zhao-Yao/eb881b0deb17e3fbc9b044cbecf599c8c79bcf5b (n/a)
- **Teach-and-repeat extended to convoys** (UTIAS MASc, 2013). VT&R makes followers track the leader's path but does **not** solve spacing; a separate longitudinal controller is needed. It cites Swaroop & Hedrick (1996) for string stability. https://asrl.utias.utoronto.ca/~tdb/bib/kulani_masc13.pdf (2013)
- **String stability with delayed leader-state broadcast.** "Leader tracking in homogeneous vehicle platoons with broadcast delays." https://www.sciencedirect.com/science/article/abs/pii/S0005109813004676 (Automatica, 2013)
- **CACC with 802.11p** (Lei, van Eenennaam, Wolterink, Karagiannis, Ploeg 2012). Co-simulation in SUMO, Simulink and OMNeT++, with a 0.7 s headway and 10 Hz beacons. With about 10 vehicles, communication delay stays minimal; high packet loss in 802.11p comes with extra contention delay. https://link.springer.com/article/10.1186/1687-1499-2012-116 (2012)
- **Graceful degradation** (Ploeg et al., ITSC 2013). On communication loss, the minimum string-stable headway rose from about 0.25 s to more than 3 s (snippet). A Smith predictor can compensate for delay. https://vandewouw.dc.tue.nl/ITSC2013_Ploeg.pdf (2013)
- **Public-road leader-follower precedent:** driverless-follower truck-mounted attenuators in work zones. A Missouri study found an **average gap error of −3.21 ft** in Kansas City mobile work zones. https://www.itskrs.its.dot.gov/2023-b01787 (USDOT ITS JPO, 2023). That the vendor is Kratos/Royal Truck is [UNVERIFIED-PK].
- **Known pitfalls [UNVERIFIED-PK, standard textbook]:**
  - Direct pursuit of the leader's current position "cuts corners." Breadcrumb tracking removes this, but its errors grow with GNSS bias between vehicles.
  - Predecessor-only topologies are string-unstable at short time gaps. Adding leader information (predecessor-leader-follower) helps.
  - Haul trucks vary widely (empty vs loaded mass about 2–2.5×, and slow brake and retarder actuators). This argues for **time gaps of several seconds** and constant-distance-plus-time policies, not tight highway-style CACC.
- **Industry precedents [UNVERIFIED-PK]:**
  - Peloton "Automated Following" (about 2019; company wound down about 2021).
  - Locomation "Autonomous Relay Convoy" (shut down about 2022–2023).
  - EU ENSEMBLE multi-brand platooning project (2018–2022).
  - Japanese expressway platooning trials with unmanned followers (about 2021).

### 4.3 V2V communications and latency
- **3GPP TS 22.186 (eV2X service requirements) includes platooning:**
  - V2V within a platoon of up to 20 UEs (R.5.2-002);
  - relative longitudinal position accuracy under 0.5 m (R.5.2-003);
  - sub-second gaps.
  - Latest version V19.0.0 (2025-10). https://www.etsi.org/deliver/etsi_ts/122100_122199/122186/19.00.00_60/ts_122186v190000p.pdf (2025-10)
  - The exact platooning latency and reliability rows were not retrieved. The figures 10 ms at 99.99% for high automation, and 25 ms at 90% for lower degrees, are **[UNVERIFIED-PK]**.
- **US 5.9 GHz.** The FCC reallocated the band in 2020 (lower 45 MHz to unlicensed, upper 30 MHz to C-V2X) and adopted C-V2X rules in November 2024 [UNVERIFIED-PK]. On private sites, private networks are what matter.
- **Rajant Kinetic Mesh.**
  - Claimed latency under 0.4 ms per hop.
  - "Certified" by Epiroc, Hitachi, Komatsu and Sandvik.
  - Used by Whitehaven's Maules Creek Hitachi AHS after other systems underperformed.
  - Claimed V2V capability.
  - Source: https://www.metaltechnews.com/story/2023/06/28/mining-tech/rajant-offers-connectivity-for-every-mine/1360.html (2023-06-28) [vendor claim]
- **Private LTE/5G.**
  - Komatsu was the first to run AHS over dedicated private LTE. https://onestore.nokia.com/asset/i/206541/ (Nokia white paper, n/a) [SEC]
  - A Thiess/Caterpillar project in 2023 had 21 autonomous trucks, 3 drills and private LTE. https://www.xtonomy.ai/wp-content/uploads/2023/09/IM-September-23-Automation-Infiltration.pdf (IM, 2023-09) [SEC]
  - Vendor targets: uplink under 10 ms; typical LTE 30–50 ms. https://inhandgo.com/en-ad/blogs/solutions/mining-and-quarry-connectivity-5g-routers-for-open-pit-operations [vendor; UNVERIFIED]
- **Design note [analysis].** Spacing safety must **not depend on the network**. Each follower needs an onboard range and range-rate sensor (radar) to its predecessor and a comm-loss watchdog that triggers a minimal-risk stop. Use V2V for feed-forward (leader braking intent, breadcrumbs) and for performance, not as the only safety path.

### 4.4 GNSS/RTK in open pits
- **Failure modes.**
  - Pit walls behave like urban canyons; horizon satellites are blocked.
  - Trucks can halt on GNSS degradation with no physical fault.
  - Source: https://www.openautonomy.com/article/gnss-blind-spots-positioning-resilience-autonomous-mining (n/a) [SEC]
- **Academic evidence.** NLOS signals and directional multipath degrade BDS RTK ambiguity resolution in open pits; elevation and azimuth masking improves ratio-test stability. https://doi.org/10.3390/s26113347 (Sensors, 2026)
- **False fixes.** Multipath can produce false integer fixes with errors of several inches while the receiver still reports a fix. NTRIP latency and packet loss also hurt near quarry high walls. https://thedronepilotbrief.com/gnss-rtk-base-station-vs-ntrip-benchmark/ (n/a) [SEC, non-mining]
- **Mitigation.** Tightly coupled multi-band, multi-constellation GNSS/INS. https://www.advancednavigation.com/tech-articles/why-cost-effective-all-band-gnss-ins-is-key-to-fleet-management-efficiency/ [vendor]. **No independent outage statistics were found** [UNVERIFIED].
- **Further mitigations [UNVERIFIED-PK]:**
  - Locata ground-based positioning, used at Newmont Boddington.
  - Lidar or radar map-matching.
  - UWB or radar relative ranging between convoy members, which makes **relative** position (the quantity that matters for following) less sensitive to common-mode GNSS error.

### 4.5 Dust and weather: lidar vs radar
- **Lidar and cameras.** Dust particles are comparable in size to optical wavelengths, causing scattering, attenuation and loss of range. Dust buildup on sensors adds to this, and dehazing only partly recovers image quality. https://doi.org/10.3390/s26134258 (Sensors review, 2026)
- **mmWave radar** is much less affected (same source). But near load and dump zones, radar can still produce **false proximity alerts that trigger protective slowdowns**. https://globalroadtechnology.com/dust-control-for-ahs-haul-road-performance/ (n/a) [vendor]
- **4D imaging radar** is highly susceptible to multipath from metallic structures and uneven terrain. Validation in dust, fog and rain was *not* done in the study. https://doi.org/10.3390/s26144615 (Sensors, 2026)
- **Sensor stack overview:** https://www.appliedintuition.com/blog/autonomous-haul-truck-sensor-stack (n/a) [vendor]
- **No quantitative dust-chamber benchmarks were found** [UNVERIFIED].

### 4.6 Teach-and-repeat navigation
- **Visual Teach & Repeat** (Furgale & Barfoot 2010). Stereo-only navigation over a manifold of overlapping submaps, without GPS. Field-tested in an urban setting and a High Arctic planetary analog; one related source reports a repeat path of more than 32 km. https://onlinelibrary.wiley.com/doi/10.1002/rob.20342 (J. Field Robotics, 2010)
- **VT&R 2.0** long-term field test. https://link.springer.com/chapter/10.1007/978-3-319-67361-5_27 (FSR, 2018)
- **VT&R3** is open source with lidar support. https://utiasasrl.github.io/ (acc.). Lidar support is [UNVERIFIED-PK].
- **Multi-experience localization** ("Bridging the Appearance Gap," IROS 2016): title only [UNVERIFIED detail].
- **Relevance.** The human leader's drive *is* the "teach" pass. VT&R-style local localization gives a GNSS-independent fallback for followers on repeated routes.

### 4.7 Haul-road speeds and stopping distances
- **Grades.** Most haul roads run 6–10%. https://www.miningdoc.tech/question/what-maximum-longitudinal-gradients-are-acceptable-for-different-classes-of-haul-trucks/ (n/a) [SEC]; USBM manual https://minemountain.in/assets/uploads/attachment_83941686987907.pdf
- **Braking-distance guidance.** About **200 ft (61 m)** is a recommended maximum braking distance, based on more than 1,000 stopping tests at British Columbia mines. Exact attribution is UNVERIFIED (Tannant & Regensburg 2001 or USBM). https://www.researchgate.net/publication/277759950_Guidelines_for_Mine_Haul_Road_Design (2001)
- **Test example.** A Cat 785 service-brake test stopped in **67 m from about 60 km/h on a 9% downgrade**, per SAE J1473 (same source, 2001).
- **ISO 3450-based figures.** About **114 m at 50 km/h** and **73 m at 40 km/h** on a 10% downgrade. https://www.coursehero.com/file/255287574/W4-L1-Haul-Roadpdf/ [weak SEC; UNVERIFIED]
- **Typical speeds [UNVERIFIED-PK]:**
  - Loaded uphill on 8–10%: about 10–15 km/h.
  - Loaded on the flat: about 30–50 km/h.
  - Empty: up to about 50–60 km/h.
  - Site speed limits are often 40–60 km/h.
  - No OEM AHS speed specification was retrieved.
- **Design note [analysis].** At 30 km/h, a 4 s time gap is about 33 m. That is *less than* the loaded downhill stopping distance. Either the follower relies on V2V braking intent (a safety-critical communication link) or the gap must exceed the stopping distance plus reaction margin (about 60–120 m at 40–50 km/h downhill). **The gap policy has to be part of the safety case.**

---

## 5. Retrofit drive-by-wire feasibility
*(Mostly [UNVERIFIED-PK]: the search budget ran out before this section.)*
- **Confirmed:**
  - ISO/TS 21815-2:2021 defines a **J1939** interface through which an external collision-avoidance device can command machine motion intervention. https://www.iso.org/standard/77303.html (2021-07)
  - Oshkosh's ExLF by-wire kit is the military precedent (§4.1).
  - The Freeport Bagdad retrofit of 33 Cat 793 trucks with Cat Command took about **2 years** of retrofit, infrastructure and training before full operation (§2.1).
- **OEM closed ecosystems [UNVERIFIED-PK]:**
  - Komatsu FrontRunner runs on 830E/930E/980E. Komatsu commissioned its **1,000th** ultra-class AHS truck in **April 2026**; the 930E accounts for more than 500 units. https://www.komatsu.com/en-us/newsroom/2026/komatsu-becomes-first-oem-to-commission-1000-ultra-class-autonomous-haul-trucks (2026-04) [confirmed]
  - Cat Command for hauling runs on 793/797/798. https://dirtmatch.com/blog/autonomous-semi-autonomous-haul-trucks-guide-2026 [SEC]
  - Cat has reportedly retrofitted non-Cat trucks [UNVERIFIED-PK].
  - Komatsu and Toyota partnered on autonomous light vehicles [UNVERIFIED-PK].
- **Retrofit and by-wire suppliers to check (all [UNVERIFIED-PK]):**
  - **ASI** (Autonomous Solutions Inc.; Mobius). Its mining unit was reportedly acquired by **Epiroc** (about 2022).
  - **Danfoss Power Solutions**: PLUS+1 controllers, electro-hydraulic steering valves, and the reported "DAVIS" autonomy integration platform.
  - **New Eagle**: Raptor controls.
  - **Dataspeed**: mostly on-road by-wire kits.
  - **Teleo**: supervised autonomy for loaders and dozers.
  - **Pronto** (quarry haul-truck retrofit AHS; reportedly acquired **SafeAI** about 2025). This would be the most direct competitor.
  - **Volvo Autonomous Solutions**: Brønnøy Kalk quarry.
  - **Bell**: ADT autonomy trials.
  - **Hexagon**: collision avoidance and autonomy.
- **Feasibility heuristics [analysis / UNVERIFIED-PK]:**
  - Modern ADTs and mid-size rigid trucks usually have J1939 engine and transmission control and electro-hydraulic or pilot-operated steering and brakes. That makes throttle, gear and retarder by-wire relatively easy.
  - **Steering and service brakes are the hard part.** They need redundant actuation, an independent safety controller (target MPL/PL d is likely, to be confirmed by HARA), and a fail-safe spring-applied park or emergency brake path.
  - The dump body and hoist also need actuation and interlocks.
  - Expect OEM warranty and liability friction. Reported vendor install times ("a day" to "a week") are **UNVERIFIED**. Plan on 2–6 weeks per first-of-type truck.

## 6. Safety evidence for autonomous haulage
- **Confirmed:**
  - Komatsu reports an **average 40% improvement in tyre and brake life** and a **13% cut in overall maintenance** for FrontRunner trucks operating within their design envelope. https://www.komatsu.com/en-us/technology/smart-mining/loading-and-haulage/autonomous-haulage-system (acc.) [OEM claim]; https://www.australianmining.com.au/an-autonomous-frontrunner/ [SEC]
  - Komatsu reached 1,000 trucks in April 2026 (above).
  - USW reports frequent unexpected stops at Asarco (2026-03, §3.4).
  - A review of AHS safety and efficiency exists: https://www.mdpi.com/2673-6489/6/3/61 (MDPI, n/a). Its contents were not reviewed.
- **[UNVERIFIED-PK] (verify before quoting):**
  - Komatsu and Cat "zero lost-time injuries attributable to AHS" style claims, and cumulative tonnage claims in the billions of tonnes.
  - Rio Tinto: autonomous trucks about 15% cheaper to run and about 700 more operating hours per truck per year (around 2018).
  - Fortescue: about 30% productivity gain from AHS at Solomon (around 2017).
  - **Incident:** Fortescue Christmas Creek, February 2019. Two autonomous trucks collided after a network or communication issue, with a WA regulator investigation reported.
  - WA regulator safety bulletins on autonomous haulage.
- **Gap.** I found no independent (non-OEM) peer-reviewed injury-rate comparison [UNVERIFIED].

## 7. Sim-to-real tools for off-road autonomy
*(Mostly [UNVERIFIED-PK]: the search budget ran out before this section.)*
- **CARLA:**
  - Built on Unreal Engine (0.9.x on UE4; a UE5-based 0.10.x line is reported).
  - Designed for urban driving; vehicle dynamics are simplified (PhysX/Chaos).
  - No deformable terrain; few off-road maps; no heavy articulated-truck models out of the box.
  - Strengths: sensors, scenarios (ScenarioRunner) and the ROS bridge.
- **Project Chrono** (UW–Madison / Parma):
  - Open-source multiphysics.
  - **Chrono::Vehicle** for wheeled and tracked vehicles, including articulated ones.
  - SCM and granular/CRM deformable terrain.
  - **Chrono::Sensor** (GPU lidar, camera, GNSS, IMU) and a ROS 2 interface.
  - Used in US Army GVSC work.
  - **Best choice for haul-truck dynamics.**
- **MAVS** (Mississippi State CAVS): physics-based off-road sensor simulation, including lidar in dust and rain. License terms are unverified.
- **NVIDIA Isaac Sim / Isaac Lab:**
  - Omniverse- and PhysX 5-based, with strong RTX sensor simulation and RL tooling.
  - Not designed for heavy vehicles on deformable terrain.
  - NVIDIA Cosmos world models exist for data generation.
- **AirSim:** Microsoft archived it in 2022. The commercial successor is Project AirSim; community forks include Colosseum. **Avoid for new work.**
- **Others:** BeamNG.tech (soft-body vehicle dynamics), Applied Intuition and Duality Falcon (commercial), Ansys AVxcelerate, IPG TruckMaker.
- **A credible pre-seed validation pipeline [analysis]:**
  1. **Co-simulation:** Chrono for truck dynamics (mass, articulation, grade, friction, retarder and brake lag) coupled to CARLA or Chrono::Sensor for perception. Add a network emulator (ns-3 or tc/netem) for latency, loss and jitter.
  2. **Scenario library:** derived from the HARA and ISO 17757 hazards. Include:
     - communication loss;
     - GNSS degradation;
     - pedestrian or light-vehicle intrusion between convoy members;
     - leader error;
     - slippery downgrade;
     - dust;
     - leader emergency stop.
  3. **SIL and HIL:** run the real controller binaries on target ECUs over J1939, with fault injection.
  4. **Scaled and small-vehicle convoys** (UTVs or golf carts with commercial by-wire) on a test track, to validate the breadcrumb, spacing and watchdog logic against simulation.
  5. **Full-size retrofit in an isolation zone** with safety drivers, then escorts, then driverless.
  6. **Metrics:**
     - cross-track error;
     - gap error;
     - string-stability gain;
     - time to minimal-risk condition (MRC) and stopping distance;
     - nuisance stops per hour;
     - interventions per 100 km.
  7. **Calibration:** system identification of real trucks to calibrate the Chrono models. Report the **sim-vs-real residuals** as part of the safety case evidence (UL 4600 requires simulation validity arguments [UNVERIFIED-PK]).

---

## 8. Synthesis

### 8.1 Regulatory path: private-site leader-follower vs public-road platooning

| Dimension | Private mine/quarry (US) | Private industrial campus (US) | Public-road platooning, driverless followers (US) | EU sites from 20 Jan 2027 |
|---|---|---|---|---|
| Primary regulator | MSHA | OSHA (General Duty Clause) | States + FMCSA + NHTSA | Machinery Reg. 2023/1230 (+ AI Act) |
| Pre-approval? | None found. Written surface-mobile-equipment safety program reviewed at inspection | None | State permits (CA: phased + safety case); FMCSA exemptions (e.g., warning devices) | CE conformity. Retrofitter likely becomes "manufacturer"; ML safety functions may need a notified body |
| Key standards | ISO 17757, 21815, 19014, 23725; WA Code / GMG as best practice | ISO 3691-4 / ANSI B56.5 (tuggers); ISO 17757 (haul) | UL 4600, ISO 26262/21448, state law | EN ISO 17757, harmonized standards, Annex III |
| Labor | USW/CFMEU bargaining, no statutory bar | Site-specific | Teamsters bills (vetoed in CA ×2, CO ×1; CA AB 33 pending/unknown) | Works councils (UNVERIFIED) |
| Time to first revenue | ~12–18 months (isolation zone) | ~12 months | 4+ years | ~24+ months |

### 8.2 Hardest technical risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Gap policy vs stopping distance.** Loaded downhill stops of 60–120 m make tight, communication-dependent spacing safety-critical. | Gap ≥ predecessor stopping distance + reaction margin unless V2V is certified. Add onboard radar ranging and a comm-loss watchdog that triggers MRC. Use speed and grade-scheduled gaps. |
| 2 | **Follower-own perception.** The road ahead can change after the leader passes (light vehicles, people, rocks). Dust degrades lidar. | Radar-primary obstacle detection with lidar and camera fusion. Integrate ISO 21815 collision avoidance. Share leader perception over V2V. Slow down when dust is detected. Track nuisance-stop KPI. |
| 3 | **Localization integrity near high walls.** | GNSS/INS with integrity monitoring. Relative ranging between vehicles. Teach-and-repeat map matching on the leader's breadcrumb. Halt on loss of integrity. |
| 4 | **ML/RL in safety functions.** Certification, EU notified-body exposure and the AI Act. | **Simplex architecture.** MPC/RL in the performance channel; a verified, simple controller plus an independent safety PLC/ECU (target MPL/PL d per HARA) in the safety channel. Freeze policies; no online learning. |
| 5 | **By-wire steering and brakes on retrofitted trucks.** | Choose a target truck with J1939 powertrain and electro-hydraulic steering. Redundant actuators and a spring-applied fail-safe brake. Partner with a by-wire supplier. Plan 2–6 weeks for each first-of-type install. |
| 6 | **Heterogeneous dynamics.** Empty vs loaded mass, wet roads after water carts, retarder lag. | Online mass and friction estimation. Conservative MPC constraints. Chrono-calibrated models. |
| 7 | **Leader human factors.** Followers replicate leader errors, such as running near a berm edge. | Map-corridor constraint on breadcrumbs, leader HMI, and an independent follower plausibility check. |
| 8 | **Competition.** OEM AHS (Komatsu at 1,000 trucks), retrofit AHS (Pronto/SafeAI, UNVERIFIED), and defense incumbents (Forterra, Carnegie Robotics). | Target the **mid-size quarry and ADT segment** that OEM AHS does not serve. Use the manned-lead model as a lower-cost, union-friendlier on-ramp. |

### 8.3 Month-by-month technical plan (M1 = December 2026)

**M1 — Concept and safety framing**
- ConOps and ODD: 1 lead + 2 followers, ≤25–30 km/h, one fixed haul loop, an isolation zone, daylight and dry conditions.
- HARA (ISO 12100, ISO 17757, ISO 19014-1 MPLr).
- UL 4600-structured safety-case skeleton.
- Sign a design-partner quarry.

**M2 — Simulation and test infrastructure**
- Chrono–CARLA co-simulation with a haul-truck model and a network emulator.
- Breadcrumb tracker and spacing controller baseline (MPC), plus watchdog and MRC logic.
- Metrics dashboard.

**M3 — Hardware in the loop**
- J1939 HIL bench with a safety controller.
- Scaled or UTV convoy rigs with RTK/INS and radar.
- Comms trial units: mesh vs private LTE (CBRS).

**M4 — Closed-track convoys**
- Two-vehicle UTV convoy on a track at ≤25 km/h.
- Fault-injection campaign: comms loss, GNSS loss, pop-up obstacle.

**M5 — Safety architecture freeze**
- Simplex design, independent e-stop and supervisor all-stop (per ISO 17757).
- Cybersecurity baseline.
- Choose an ISO 23725-aligned API.

**M6 — Gate 1**
- Three-vehicle track demo with a V&V report and sim-vs-real residuals.
- Select the target truck model and by-wire partner.
- Raise seed.

**M7–M8 — Retrofit design and site survey**
- Retrofit design (steering, brakes, throttle, transmission, retarder, hoist), FMEA and MPL allocation.
- Site survey: GNSS sky-view, RF and dust.
- Procure or lease 2 trucks.

**M9 — Truck 1 by-wire**
- By-wire install on truck 1.
- Brake and steering verification (ISO 3450 / SAE J1473-style tests with by-wire actuation).

**M10 — Truck 2 and site network**
- Truck 2 by-wire.
- Follower perception stack.
- Site network deployment.

**M11 — Full-size convoy, empty**
- Full-size leader-follower in the isolation zone, empty trucks, ≤15 km/h, safety driver in each follower.

**M12 — Gate 2**
- Loaded runs, grades and night operation; about 200 supervised hours.
- Independent safety-assessor pre-audit (TÜV, UL or exida).

**M13–M14 — Scale the loop**
- Add a third truck.
- Run the full loop with light-vehicle interaction rules.
- Integrate ISO 21815 collision avoidance.
- FMS interface pilot.

**M15 — First driverless follower**
- Remove the safety driver from follower 1, with escort and remote e-stop.
- Update the operator's MSHA written safety program.
- Brief the MSHA district office.

**M16 — Production-like operation**
- Production-like shifts.
- Log incidents and near-misses.
- Measure productivity against a manned baseline.

**M17 — Reliability growth**
- Reliability growth with a target to cut nuisance stops.
- Dust and rain campaigns.
- Dual-use data package for SBIR/xTech.

**M18 — Gate 3**
- 1 manned lead + 2 driverless followers in production for several weeks.
- Safety case v1 with an ISO 17757 compliance matrix.
- Convert to a paid pilot.
- EU Machinery Regulation gap analysis.

### 8.4 Highest-priority follow-ups
1. Get the EUR-Lex Annex III verbatim text for autonomous mobile machinery and the supervisor function, plus Art. 18 on substantial modification.
2. Get the NCSL 2026 counts of states allowing platooning and driverless trucks.
3. Check the 2026 outcomes of CA AB 33 and H.R. 8870.
4. Confirm the competitor landscape: Pronto/SafeAI, Epiroc/ASI, Teleo, Danfoss DAVIS.
5. Find primary safety statistics and incident reports (WA regulator, Fortescue 2019).
6. Get the 3GPP TS 22.186 platooning latency table.
7. Check the CARLA 0.10 and MAVS license and status.
