"""Leadhaul unit economics, customer ROI, market sizing and a 5-year plan.

Every input is listed in ASSUMPTIONS with its source or its status (SOURCED / DERIVED / ASSUMPTION / TARGET).
Sources are in venture/research/sources.md. Run:  python venture/model/unit_economics.py  (writes model_output.json/.md)
"""
from __future__ import annotations

import json
from pathlib import Path

A = {
    # --- labour (SOURCED: BLS OEWS May 2025 median, heavy & tractor-trailer drivers $58,640/yr; BLS ECEC Jun 2025 1.42x)
    'wage_per_hr': 28.19,
    'benefit_load': 1.42,
    'overhead_per_hr': 3.0,            # ASSUMPTION: recruiting, MSHA Part 46/48 training, turnover, overtime premium
    # --- site and haul (ASSUMPTION unless noted; the ICP: >= 2 km one-way private haul, >= 4 trucks, 2 shifts)
    'hours_per_truck_yr': 4160,        # 2 x 8 h shifts x 260 d
    'payload_t': 55.0,                 # 60 t-class rigid truck (Cat 775 class) or large ADT
    'cycle_min': 20.0,                 # load 3 + haul 3 km loaded + dump 1.2 + return + spot/queue (DERIVED from §3e cycle ranges)
    'truck_opex_per_hr': 110.0,        # fuel, tyres, parts, repair, lube for a 60 t-class truck; Costmine 2022 puts a 100 t truck at $175/h
    'convoy_efficiency': 0.92,         # convoy throughput vs independent dispatch on long hauls with batch-friendly load/dump (to validate)
    'maint_tyre_saving': 0.05,         # share of truck opex saved by smoother automated driving on followers (Komatsu claims more; discounted)
    # --- Leadhaul offer (TARGET)
    'price_per_follower_hr': 24.0,     # ~55% of the loaded cost of the operator seat it replaces
    'followers_per_convoy': 3,
    # --- Leadhaul cost per follower (ASSUMPTION; research BOM range $35-130k, pilot vs scale)
    'kit_cost_pilot': 60_000.0, 'kit_cost_scale': 32_000.0,
    'install_pilot': 8_000.0, 'install_scale': 4_000.0,
    'kit_life_yr': 5.0,
    'support_pilot': 14_000.0, 'support_scale': 8_000.0,      # field service, spares, remote monitoring per follower-year
    'cloud_insurance': 3_000.0,
    'lead_kit': 12_000.0,                                     # leader: GNSS, V2V radio, cab tablet, driver monitoring
    # --- market (SOURCED counts; DERIVED fleet sizes)
    'us_aggregates_gt': 2.40,          # USGS MIS 2026-Q2: 2025 US construction aggregates 2.40 Gt, $40.0B
    'us_quarries': 3500, 'us_pits': 6500,
    'us_quarry_trucks_low': 3000, 'us_quarry_trucks_high': 6000,   # DERIVED (research §2b), unverified
    'global_haul_trucks_90t': 66_000,  # Parker Bay active >= 90 t
    'global_rigid_adt_sales_yr': 16_000,   # Off-Highway Research 2023 peak (rigid + ADT)
    'fleet_life_yr': 10,
    'icp_share': 0.4,                  # share of trucks on routes that fit the ICP (long haul, >= 4 trucks)
}


# Site archetypes. The convoy only removes operator hours, so its value scales with (operator cost / truck opex) and
# with how little throughput convoying costs (travel-dominated cycles with batch-friendly loading lose least).
SITES = {
    'icp': dict(label='ICP: 5 km private haul, 40 t ADT/road-class truck, remote site', wage=35.0, opex=70.0, payload=40.0, cycle=30.0, eff=1.00,
                note='travel ~75% of cycle; relief operator covers breaks/shift change (+5% utilisation) offsets convoy queuing (-5%)'),
    'icp_median': dict(label='ICP at the US median wage ($28.19/h)', wage=28.19, opex=70.0, payload=40.0, cycle=30.0, eff=1.00, note=''),
    'mid': dict(label='Mid: 3 km haul, 60 t truck, US median wage', wage=28.19, opex=110.0, payload=55.0, cycle=20.0, eff=0.92, note=''),
    'avoid': dict(label='Avoid: short quarry loop, 60 t truck, single loader', wage=28.19, opex=110.0, payload=55.0, cycle=14.0, eff=0.80,
                  note='loader-paced cycle: a tight convoy idles the loader'),
}


def customer_roi(a=A, eff=None, price=None, hours=None, opex=None, label='base', site=None):
    s = SITES[site] if site else None
    eff = (s['eff'] if s else a['convoy_efficiency']) if eff is None else eff
    price = a['price_per_follower_hr'] if price is None else price
    hours = a['hours_per_truck_yr'] if hours is None else hours
    opex = (s['opex'] if s else a['truck_opex_per_hr']) if opex is None else opex
    wage = s['wage'] if s else a['wage_per_hr']
    payload = s['payload'] if s else a['payload_t']
    cycle = s['cycle'] if s else a['cycle_min']
    label = s['label'] if s and label == 'base' else label
    n = a['followers_per_convoy'] + 1
    op_cost = wage * a['benefit_load'] + a['overhead_per_hr']
    tph = payload * 60.0 / cycle                                      # t/h per truck, independent dispatch
    tons = n * tph * hours                                            # tons the 4 trucks move manually per year
    manual = n * hours * (op_cost + opex)
    # convoy: same tons need hours/eff truck-hours; operator seats only on the leader
    th = hours / eff
    convoy = th * op_cost + n * th * opex * 1.0 - (n - 1) * th * opex * a['maint_tyre_saving'] + (n - 1) * th * price
    return {
        'label': label, 'convoy_efficiency': eff, 'price_per_follower_hr': price, 'hours_per_truck_yr': hours,
        'operator_cost_per_hr': round(op_cost, 2), 'tons_per_yr': round(tons), 'manual_cost': round(manual), 'convoy_cost': round(convoy),
        'saving_per_yr': round(manual - convoy), 'saving_pct_of_haul_cost': round(100 * (manual - convoy) / manual, 1),
        'manual_cost_per_t': round(manual / tons, 3), 'convoy_cost_per_t': round(convoy / tons, 3),
        'operator_seats_per_day_manual': n * 2, 'operator_seats_per_day_convoy': 2,
        'leadhaul_revenue_per_convoy': round((n - 1) * th * price),
        'price_per_ton_equiv': round(price / (tph * eff), 3),
        'tons_per_operator_hr_manual': round(tph, 1), 'tons_per_operator_hr_convoy': round(n * tph * eff, 1),
    }


def follower_economics(a=A, scale=False, price=None, hours=None):
    price = a['price_per_follower_hr'] if price is None else price
    hours = (a['hours_per_truck_yr'] / a['convoy_efficiency']) if hours is None else hours
    rev = price * hours
    kit = a['kit_cost_scale'] if scale else a['kit_cost_pilot']
    inst = a['install_scale'] if scale else a['install_pilot']
    sup = a['support_scale'] if scale else a['support_pilot']
    lead_alloc = a['lead_kit'] / a['followers_per_convoy']
    dep = (kit + inst + lead_alloc) / a['kit_life_yr']
    cogs = dep + sup + a['cloud_insurance']
    return {'scale': 'scale' if scale else 'pilot', 'revenue_per_yr': round(rev), 'cogs_per_yr': round(cogs),
            'gross_margin_pct': round(100 * (rev - cogs) / rev, 1), 'upfront_capex': round(kit + inst + lead_alloc),
            'capex_payback_months': round(12 * (kit + inst + lead_alloc) / (rev - sup - a['cloud_insurance']), 1)}


def market(a=A):
    rev_per_follower = a['price_per_follower_hr'] * 3000          # blended: not every site runs two shifts
    f_share = a['followers_per_convoy'] / (a['followers_per_convoy'] + 1)
    us_lo = a['us_quarry_trucks_low'] * a['icp_share'] * f_share * rev_per_follower
    us_hi = a['us_quarry_trucks_high'] * a['icp_share'] * f_share * rev_per_follower
    global_base = a['global_rigid_adt_sales_yr'] * a['fleet_life_yr']
    tam = global_base * f_share * rev_per_follower
    return {'rev_per_follower_blended': rev_per_follower,
            'us_aggregates_beachhead_arr_range': [round(us_lo), round(us_hi)],
            'global_offhighway_fleet_est': global_base, 'global_tam_arr': round(tam),
            'note': 'Fleet counts are DERIVED estimates (USGS tonnage / truck productivity; OHR sales x life). Validate with AEM/OHR data.'}


def plan(a=A):
    # followers in service at year end, sites, blended price ramp (pilot discounts early)
    years = [2027, 2028, 2029, 2030, 2031]
    followers = [3, 12, 40, 110, 260]
    sites = [1, 3, 9, 22, 45]
    price = [16.0, 20.0, 23.0, 24.0, 24.0]
    hours = 3000
    out = []
    for y, f, s, p in zip(years, followers, sites, price):
        arr = f * p * hours
        out.append({'year': y, 'followers': f, 'sites': s, 'price_per_hr': p, 'exit_arr': round(arr)})
    return {'years': out, 'raises': [
        {'round': 'pre-seed', 'when': 'Q1 2027', 'amount': 2.0e6, 'instrument': 'post-money SAFE', 'cap': 16e6},
        {'round': 'seed', 'when': 'H2 2028', 'amount': 7.0e6, 'trigger': 'paid pilot -> 12-month contract, 2nd site, safety case v1'},
        {'round': 'Series A', 'when': '2029-30', 'amount': 25e6, 'trigger': '>= $3M ARR, 40+ followers, 9+ sites'}]}


def preseed_budget():
    items = [('Team: founder + 3 robotics/controls engineers + field-ops lead (18 mo, loaded)', 1_150_000),
             ('Vehicles: 2 used 40-60 t trucks/ADTs (or leased) + 3 scaled UTVs with by-wire', 260_000),
             ('Autonomy kits: sensors, compute, V2V, by-wire integration (3 sets)', 180_000),
             ('Safety: HARA, UL 4600-structured case, third-party pre-audit; insurance; test site', 170_000),
             ('Legal/IP (Stanford OTL check), travel, customer discovery', 90_000),
             ('Contingency', 150_000)]
    return {'items': items, 'total': sum(v for _, v in items)}


def main():
    base = customer_roi(site='icp')
    sens = [customer_roi(site='mid'), customer_roi(site='avoid')] + \
           [customer_roi(site='icp', eff=e, label=f'ICP, convoy efficiency {e}') for e in (0.90, 0.95)] + \
           [customer_roi(site='icp', price=p, label=f'ICP, price ${p:.0f}/follower-h') for p in (18.0, 30.0)] + \
           [customer_roi(site='icp', hours=2080, label='ICP, single shift')] + \
           [customer_roi(site='icp_median', label='base')] + \
           [customer_roi(site='icp', opex=40.0, label='ICP with road-class truck (opex $40/h)')]
    fe = [follower_economics(scale=False, price=20.0), follower_economics(scale=True)]
    res = {'assumptions': A, 'sites': SITES, 'customer_roi_base': base, 'sensitivity': sens, 'follower_economics': fe,
           'market': market(), 'plan': plan(), 'preseed_budget': preseed_budget()}
    here = Path(__file__).parent
    (here / 'model_output.json').write_text(json.dumps(res, indent=1))
    lines = ['# Leadhaul model output', '', '## Customer ROI per convoy (1 operator + 3 followers)', '',
             '| case | eff | $/follower-h | h/yr | manual $/yr | convoy $/yr | saving $/yr | saving % | t per operator-h |', '|---|---|---|---|---|---|---|---|---|']
    for r in [base] + sens:
        lines.append(f"| {r['label']} | {r['convoy_efficiency']} | {r['price_per_follower_hr']} | {r['hours_per_truck_yr']} | {r['manual_cost']:,} | {r['convoy_cost']:,} | {r['saving_per_yr']:,} | {r['saving_pct_of_haul_cost']} | {r['tons_per_operator_hr_manual']} -> {r['tons_per_operator_hr_convoy']} |")
    lines += ['', 'Manual vs convoy cost per ton (ICP): $%.2f -> $%.2f; Leadhaul revenue per convoy: $%s/yr; price per ton equivalent: $%.2f' % (base['manual_cost_per_t'], base['convoy_cost_per_t'], f"{base['leadhaul_revenue_per_convoy']:,}", base['price_per_ton_equiv'])]
    lines += ['', '## Leadhaul economics per follower', '', '| stage | revenue/yr | COGS/yr | GM % | upfront capex | capex payback (months) |', '|---|---|---|---|---|---|']
    for f in fe:
        lines.append(f"| {f['scale']} | {f['revenue_per_yr']:,} | {f['cogs_per_yr']:,} | {f['gross_margin_pct']} | {f['upfront_capex']:,} | {f['capex_payback_months']} |")
    m = res['market']
    lines += ['', '## Market (estimates)', '', f"- US aggregates beachhead ARR potential: ${m['us_aggregates_beachhead_arr_range'][0]/1e6:.0f}M-${m['us_aggregates_beachhead_arr_range'][1]/1e6:.0f}M",
              f"- Global off-highway rigid + ADT fleet (est.): {m['global_offhighway_fleet_est']:,} trucks -> TAM ${m['global_tam_arr']/1e9:.1f}B/yr", f"- {m['note']}",
              '', '## Plan', '', '| year | sites | followers | $/h | exit ARR |', '|---|---|---|---|---|']
    for y in res['plan']['years']:
        lines.append(f"| {y['year']} | {y['sites']} | {y['followers']} | {y['price_per_hr']} | ${y['exit_arr']/1e6:.2f}M |")
    b = res['preseed_budget']
    lines += ['', '## Pre-seed budget (18 months)', ''] + [f'- {k}: ${v:,}' for k, v in b['items']] + [f"- **Total: ${b['total']:,}**"]
    (here / 'model_output.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
