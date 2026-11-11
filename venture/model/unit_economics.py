"""Leadhaul unit economics, customer ROI, market sizing and a 5-year plan.

Every input is listed in A or SITES with its status: SOURCED (see venture/research/sources.md), DERIVED, ASSUMPTION or
TARGET. Run:  python venture/model/unit_economics.py  -> model_output.md / model_output.json

Customer ROI is computed per convoy (1 operator + 3 followers) over a fixed number of operating hours. Convoying costs
throughput (trucks wait for each other at single-point loading; nuisance stops), so the convoy moves fewer tons in the
same hours; we compare cost per ton including truck ownership, and report the break-even convoy efficiency.
"""
from __future__ import annotations

import json
from pathlib import Path

A = {
    'benefit_load': 1.42,              # SOURCED: BLS ECEC, private industry, Jun 2025 ($45.65 total / $32.07 wages)
    'overhead_per_hr': 3.0,            # ASSUMPTION: recruiting, MSHA Part 46/48 training, turnover, overtime premium
    'hours_per_yr': 4160,              # ASSUMPTION: 2 x 8 h shifts x 260 days (customer ROI)
    'billable_hours_per_follower': 3000,   # ASSUMPTION: blended across 1- and 2-shift sites (all Leadhaul revenue)
    'followers_per_convoy': 3,
    'maint_tyre_saving': 0.05,         # ASSUMPTION: share of follower running cost saved by smoother automated driving
    'nuisance_loss': 0.02,             # ASSUMPTION: throughput lost to false/precautionary stops
    'price_per_follower_hr': 24.0,     # TARGET (billed per ton at the equivalent rate)
    'margin_per_t': 5.0,               # ASSUMPTION: producer contribution margin per ton (~28% of the ~$18/t gate price, USGS-derived)
    # Leadhaul cost per follower (ASSUMPTION; research BOM range $35-130k, unverified)
    'kit_pilot': 75_000.0, 'kit_scale': 40_000.0, 'install_pilot': 8_000.0, 'install_scale': 4_000.0,
    'lead_kit': 12_000.0, 'kit_life_yr': 4.0, 'support_pilot': 14_000.0, 'support_scale': 9_000.0, 'cloud_insurance': 3_000.0,
    # market
    'us_quarry_trucks_low': 3000, 'us_quarry_trucks_high': 6000,   # DERIVED (research 02 §2b), unverified
    'global_rigid_adt_sales_yr': 16_000, 'fleet_life_yr': 10,      # SOURCED (Off-Highway Research, 2023) / ASSUMPTION
    'icp_share': 0.4,                                              # ASSUMPTION: trucks on routes that fit the ICP
}

# wage = hourly wage ($/h); opex = truck running cost ex-operator ($/h); own = ownership ($/h); payload t; cycle min;
# t_load = minutes per truck at the loading point; loaders = loading points; mode 'tight' or 'path' (staggered follow)
SITES = {
    'icp_remote': dict(label='ICP, remote-site wage $35/h', wage=35.0, opex=70.0, own=45.0, payload=40.0, cycle=30.0, t_load=1.0, loaders=2, mode='tight'),
    'icp_median': dict(label='ICP, US median wage $28.19/h (BLS)', wage=28.19, opex=70.0, own=45.0, payload=40.0, cycle=30.0, t_load=1.0, loaders=2, mode='tight'),
    'icp_fifo': dict(label='ICP, FIFO/union labour ($80/h loaded)', wage=(80.0 - 3.0) / 1.42, opex=70.0, own=45.0, payload=40.0, cycle=30.0, t_load=1.0, loaders=2, mode='tight'),
    'icp_road': dict(label='ICP with road-class trucks, $35/h', wage=35.0, opex=40.0, own=25.0, payload=25.0, cycle=30.0, t_load=1.0, loaders=2, mode='tight'),
    'icp_single': dict(label='ICP, single bin loadout, $35/h', wage=35.0, opex=70.0, own=45.0, payload=40.0, cycle=30.0, t_load=1.0, loaders=1, mode='tight'),
    'icp_path': dict(label='ICP in path-follow mode, $35/h', wage=35.0, opex=70.0, own=45.0, payload=40.0, cycle=30.0, t_load=1.0, loaders=1, mode='path'),
    'avoid': dict(label='Avoid: 12 min quarry loop, one loader, 60 t trucks', wage=28.19, opex=110.0, own=60.0, payload=55.0, cycle=12.0, t_load=3.0, loaders=1, mode='tight'),
}


def convoy_efficiency(s, a=A):
    """Tight convoy: followers wait for each other at the loading point(s); path-follow: no batching loss."""
    n = a['followers_per_convoy'] + 1
    batch = 0.0 if s['mode'] == 'path' else (n - 1) * s['t_load'] / s['loaders']
    return s['cycle'] / (s['cycle'] + batch) * (1 - a['nuisance_loss'])


def customer_roi(key, a=A, price=None):
    s = SITES[key]
    price = a['price_per_follower_hr'] if price is None else price
    n, H = a['followers_per_convoy'] + 1, a['hours_per_yr']
    op = s['wage'] * a['benefit_load'] + a['overhead_per_hr']
    eff = convoy_efficiency(s, a)
    tph = s['payload'] * 60.0 / s['cycle']
    manual_cost, manual_t = n * H * (op + s['opex'] + s['own']), n * tph * H
    convoy_cost = H * (op + n * (s['opex'] + s['own']) - (n - 1) * s['opex'] * a['maint_tyre_saving'] + (n - 1) * price)
    convoy_t = eff * manual_t
    cpt_m, cpt_c = manual_cost / manual_t, convoy_cost / convoy_t
    fee = (n - 1) * H * price
    # labour-short case: the site can staff only 3 of the 4 trucks on this route
    extra_t = convoy_t - (n - 1) / n * manual_t
    return {'label': s['label'], 'operator_cost_per_hr': round(op, 1), 'convoy_efficiency': round(eff, 3),
            'manual_cost_per_t': round(cpt_m, 3), 'convoy_cost_per_t': round(cpt_c, 3),
            'cost_per_t_saving_pct': round(100 * (1 - cpt_c / cpt_m), 1),
            'saving_per_yr': round((cpt_m - cpt_c) * convoy_t), 'break_even_efficiency': round(convoy_cost / manual_cost, 3),
            'tons_per_operator_hr': f"{tph:.0f} -> {n * tph * eff:.0f}", 'leadhaul_fee_per_yr': round(fee),
            'price_per_t_equiv': round(price / (tph * eff), 3),
            'labour_short_extra_t': round(extra_t), 'labour_short_value': round(extra_t * a['margin_per_t'])}


def follower_economics(stage, a=A):
    price = {'pilot': 20.0, 'scale': a['price_per_follower_hr']}[stage]
    kit, inst, sup = (a['kit_pilot'], a['install_pilot'], a['support_pilot']) if stage == 'pilot' else (a['kit_scale'], a['install_scale'], a['support_scale'])
    rev = price * a['billable_hours_per_follower']
    capex = kit + inst + a['lead_kit'] / a['followers_per_convoy']
    cogs = capex / a['kit_life_yr'] + sup + a['cloud_insurance']
    return {'stage': stage, 'price_per_hr': price, 'revenue_per_yr': round(rev), 'cogs_per_yr': round(cogs),
            'gross_margin_pct': round(100 * (rev - cogs) / rev, 1), 'capex': round(capex),
            'capex_payback_months': round(12 * capex / (rev - sup - a['cloud_insurance']), 1)}


def market(a=A):
    per = a['price_per_follower_hr'] * a['billable_hours_per_follower']
    f = a['followers_per_convoy'] / (a['followers_per_convoy'] + 1)
    fleet = a['global_rigid_adt_sales_yr'] * a['fleet_life_yr']
    us = [a['us_quarry_trucks_low'] * a['icp_share'] * f, a['us_quarry_trucks_high'] * a['icp_share'] * f]
    return {'revenue_per_follower_yr': per, 'global_fleet_est': fleet, 'tam': round(fleet * f * per),
            'icp_sam': round(fleet * a['icp_share'] * f * per),
            'us_aggregates_icp_followers': [round(x) for x in us], 'us_aggregates_icp_arr': [round(x * per) for x in us]}


def plan(a=A):
    rows = [(2027, 0, 0, 0.0), (2028, 2, 6, 20.0), (2029, 6, 24, 22.0), (2030, 15, 70, 24.0), (2031, 35, 180, 24.0)]
    m = market(a)
    out = [{'year': y, 'sites': s, 'followers': f, 'price': p, 'exit_arr': round(f * p * a['billable_hours_per_follower'])} for y, s, f, p in rows]
    lo, hi = m['us_aggregates_icp_followers']
    return {'years': out, 'share_of_us_aggregates_icp_2031': [round(180 / hi, 2), round(180 / lo, 2)],
            'capital_to_2031_est': 40e6,
            'raises': [{'round': 'pre-seed', 'when': 'Q1 2027', 'amount': 2.5e6, 'cap': 18e6},
                       {'round': 'seed', 'when': 'start month 12 (Q1 2028), close by month 16', 'amount': 8e6},
                       {'round': 'Series A', 'when': '2029-30', 'amount': 25e6}]}


def preseed_budget():
    items = [('Team: founder, 3 robotics/controls engineers, field-ops/mining lead from month 3 (loaded, 18 mo)', 1_270_000),
             ('By-wire engineering partner and OEM/J1939 access (steering, brakes, hoist)', 190_000),
             ('Vehicles: 2 used 40 t ADTs or leases + 3 by-wire UTVs, transport, site costs', 300_000),
             ('Autonomy kits, 3 sets (sensors, compute, V2V, actuation)', 225_000),
             ('Safety: HARA, UL 4600-structured case, third-party pre-audit, insurance, test site', 250_000),
             ('Legal/IP (Stanford OTL check), customer discovery, travel', 100_000),
             ('Contingency', 165_000)]
    return {'items': items, 'total': sum(v for _, v in items)}


def main():
    keys = ['icp_remote', 'icp_median', 'icp_fifo', 'icp_road', 'icp_single', 'icp_path', 'avoid']
    roi = {k: customer_roi(k) for k in keys}
    fe = [follower_economics('pilot'), follower_economics('scale')]
    res = {'assumptions': A, 'sites': SITES, 'customer_roi': roi, 'follower_economics': fe, 'market': market(),
           'plan': plan(), 'preseed_budget': preseed_budget()}
    here = Path(__file__).parent
    (here / 'model_output.json').write_text(json.dumps(res, indent=1))
    L = ['# Leadhaul model output', '', '## Customer ROI per convoy (1 operator + 3 followers, 4,160 h/yr, cost incl. truck ownership)', '',
         '| case | operator $/h | convoy eff. | $/t manual -> convoy | saving | $/yr | break-even eff. | t per operator-h | labour-short: extra t/yr, value at $5/t |',
         '|---|---|---|---|---|---|---|---|---|']
    for r in roi.values():
        L.append(f"| {r['label']} | {r['operator_cost_per_hr']} | {r['convoy_efficiency']} | {r['manual_cost_per_t']} -> {r['convoy_cost_per_t']} | {r['cost_per_t_saving_pct']}% | {r['saving_per_yr']:,} | {r['break_even_efficiency']} | {r['tons_per_operator_hr']} | {r['labour_short_extra_t']:,} t, ${r['labour_short_value']:,} |")
    b = roi['icp_remote']
    L += ['', f"Leadhaul fee per convoy: ${b['leadhaul_fee_per_yr']:,}/yr (2 shifts) = ${b['price_per_t_equiv']}/t on follower tons.", '',
          '## Leadhaul economics per follower (3,000 billable h/yr)', '', '| stage | $/h | revenue/yr | COGS/yr | GM % | capex | capex payback (mo) |', '|---|---|---|---|---|---|---|']
    for f in fe:
        L.append(f"| {f['stage']} | {f['price_per_hr']} | {f['revenue_per_yr']:,} | {f['cogs_per_yr']:,} | {f['gross_margin_pct']} | {f['capex']:,} | {f['capex_payback_months']} |")
    m = res['market']
    L += ['', '## Market (estimates)', '', f"- Global off-highway rigid + ADT fleet: {m['global_fleet_est']:,}; TAM ${m['tam'] / 1e9:.1f}B/yr; ICP-qualified SAM ${m['icp_sam'] / 1e9:.1f}B/yr",
          f"- US aggregates ICP: {m['us_aggregates_icp_followers'][0]:,}-{m['us_aggregates_icp_followers'][1]:,} followers = ${m['us_aggregates_icp_arr'][0] / 1e6:.0f}-{m['us_aggregates_icp_arr'][1] / 1e6:.0f}M ARR",
          '', '## Plan', '', '| year | sites | followers | $/h | exit ARR |', '|---|---|---|---|---|']
    for y in res['plan']['years']:
        L.append(f"| {y['year']} | {y['sites']} | {y['followers']} | {y['price']} | ${y['exit_arr'] / 1e6:.2f}M |")
    s = res['plan']['share_of_us_aggregates_icp_2031']
    L += ['', f"2031 followers = {int(s[0] * 100)}-{int(s[1] * 100)}% of the US aggregates ICP fleet, so the plan needs mining and heavy civil from 2029.",
          f"Equity needed to 2031 (est.): ${res['plan']['capital_to_2031_est'] / 1e6:.0f}M.", '', '## Pre-seed budget (18 months)', '']
    bb = res['preseed_budget']
    L += [f'- {k}: ${v:,}' for k, v in bb['items']] + [f"- **Total: ${bb['total']:,}**"]
    (here / 'model_output.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
