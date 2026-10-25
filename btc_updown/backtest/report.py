"""Render calibration (calib.fit) and backtest (backtest.engine) JSON into a markdown report section.

    python -m btc_updown.backtest.report --title "E0 zero-shot small, val" --calib data/calib/e0.json \
        --backtest data/backtests/val_e0/summary.json --out data/reports/e0_val.md
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional


def _f(x, nd=4) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (x != x):
            return "—"
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    except Exception:
        return str(x)


def render_calib(c: Dict) -> List[str]:
    out = [f"**Calibration fit** — features `{', '.join(c['features'])}`, market term: {c['include_market']}; "
           f"{c['n_rows']:,} rows, {c['n_windows']:,} windows, {c['n_days']} days, base rate {_f(c['base_rate'], 3)}", ""]
    b = c["brier"]
    out += ["| Brier | market | null | blend (in-sample) | blend (leave-one-day-out) | constant |", "|---|---|---|---|---|---|",
            f"| | {_f(b.get('market'))} | {_f(b['null'])} | {_f(b['blend_insample'])} | {_f(b['blend_leave_one_day_out'])} | {_f(b['constant'])} |", ""]
    out += ["| coefficient | estimate | 95% CI (day-block bootstrap) |", "|---|---|---|"]
    ci = c["bootstrap"]["coef_ci95"]
    for k, v in c["coef"].items():
        lo, hi = ci.get(k, [None, None])
        star = " **(excludes 0)**" if lo is not None and (lo > 0 or hi < 0) and k not in ("intercept",) else ""
        out.append(f"| {k} | {_f(v, 3)} | [{_f(lo, 3)}, {_f(hi, 3)}]{star} |")
    out.append("")
    g = c["bootstrap"]
    out.append(f"Brier gain of the blend vs market 95% CI: {g.get('brier_gain_vs_market_ci95')}; vs null: {g.get('brier_gain_vs_null_ci95')}.")
    if c.get("by_family"):
        out += ["", "| family | n | market | null | blend |", "|---|---|---|---|---|"]
        for fam, r in c["by_family"].items():
            out.append(f"| {fam} | {r['n']:,} | {_f(r.get('market'))} | {_f(r['null'])} | {_f(r['blend'])} |")
    out.append("")
    return out


def render_backtest(s: Dict) -> List[str]:
    out = [f"**Backtest** — clip ${s['clip_usd']:.0f}, margin {s['margin']}, calibration `{os.path.basename(s['calib'])}`", ""]
    bt = s.get("brier", {})
    if bt:
        out += [f"Brier on the table: blend {_f(bt.get('blend'))}, market {_f(bt.get('market'))}, null {_f(bt.get('null'))} (n = {bt.get('n'):,})", ""]
        if bt.get("by_h"):
            out += ["| minutes left | n | blend | market | null |", "|---|---|---|---|---|"]
            for h, r in sorted(bt["by_h"].items(), key=lambda kv: int(kv[0])):
                out.append(f"| {h} | {r['n']:,} | {_f(r['blend'])} | {_f(r.get('market'))} | {_f(r['null'])} |")
            out.append("")
    out += ["| execution | trades | days | mean $/day | sd | positive days | ¢/share | hit rate | notional $ | t vs hurdle | p (one-sided) | bootstrap lb95 $/day | 0.5×mean > hurdle |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in s["execution"].items():
        if r.get("n_trades", 0) == 0:
            out.append(f"| {name} | 0 | | | | | | | | | | | |")
            continue
        out.append(f"| {name} | {r['n_trades']:,} | {r['n_days']} | {_f(r['mean_per_day'], 1)} | {_f(r.get('sd_per_day'), 1)} | {r['positive_days']} | "
                   f"{_f(r['cents_per_share'], 2)} | {_f(r['hit_rate'], 3)} | {_f(r['notional'], 0)} | {_f(r.get('t_vs_hurdle'), 2)} | {_f(r.get('p_one_sided'), 3)} | "
                   f"{_f(r['bootstrap_lb95_mean_per_day'], 1)} | {r['half_mean_rule_passes']} |")
    out.append("")
    for name, r in s["execution"].items():
        if r.get("by_minutes_left"):
            out += [f"*{name}: by minutes to expiry*", "", "| h | trades | P&L $ | ¢/share |", "|---|---|---|---|"]
            for h, v in sorted(r["by_minutes_left"].items(), key=lambda kv: int(kv[0])):
                out.append(f"| {h} | {v['n']} | {_f(v['pnl'], 1)} | {_f(100 * v['cps'], 2)} |")
            out.append("")
        if r.get("by_price_bucket"):
            out += [f"*{name}: by entry price*", "", "| bucket | trades | P&L $ | ¢/share |", "|---|---|---|---|"]
            for b, v in r["by_price_bucket"].items():
                out.append(f"| {b} | {v['n']} | {_f(v['pnl'], 1)} | {_f(100 * v['cps'], 2)} |")
            out.append("")
    return out


def render(title: str, calib: Optional[Dict], backtest: Optional[Dict], label: str = "dev, exploratory") -> str:
    lines = [f"## {title}", "", f"_Label: {label}._", ""]
    if calib:
        lines += render_calib(calib)
    if backtest:
        lines += render_backtest(backtest)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--calib", default=None); ap.add_argument("--backtest", default=None)
    ap.add_argument("--label", default="dev, exploratory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.load(open(a.calib)) if a.calib else None
    b = json.load(open(a.backtest)) if a.backtest else None
    md = render(a.title, c, b, a.label)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
