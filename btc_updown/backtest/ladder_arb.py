"""Static arbitrage scan on the Kalshi ladders (free, Mac only).

A yes contract at a lower strike dominates one at a higher strike, so yes_bid(K_j) must not exceed yes_ask(K_i) for
K_i < K_j. Where it does, buying yes at K_i and buying no at K_j (= selling yes at K_j) locks in bid_j - ask_i per
contract before fees. Tiers, per (event, minute), from the 1-minute candles:
    close    both strikes have a candle in the minute; violation on the candles' closing bid/ask (the two closes are not
             simultaneous inside the minute, so this is an upper bound)
    minute   bid_low(K_j) > ask_high(K_i) over the whole minute: the crossed state persisted through the minute
    ffill    the forward-filled book (what a naive reader of the candles sees; stale legs included)
Net of taker fees 0.07·p(1−p) on both legs. Validation against the taker prints (ATM ± 5 strikes only): in a flagged
minute, was yes at K_i actually bought at ≤ ask_i and yes at K_j sold at ≥ bid_j?

    python -m btc_updown.backtest.ladder_arb --archive data/kalshi_ladder --out data/backtests/ladder_arb
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

FEE = 0.07


def fee(p: np.ndarray) -> np.ndarray:
    return FEE * p * (1 - p)


def _violations(g: pd.DataFrame, bid_col: str, ask_col: str) -> pd.DataFrame:
    """g: one (event, t) with columns strike, bid_col, ask_col (and ticker, volume, age if present), sorted by strike."""
    a = g[ask_col].to_numpy(); b = g[bid_col].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return g.iloc[0:0]
    g = g[ok]; a = a[ok]; b = b[ok]
    # best (lowest) ask among strictly lower strikes
    run_min = np.minimum.accumulate(a)
    prev_min = np.concatenate([[np.inf], run_min[:-1]])
    arg = np.concatenate([[-1], np.array([int(np.argmin(a[:j])) for j in range(1, len(a))])]) if len(a) > 1 else np.array([-1])
    gross = b - prev_min
    hit = gross > 1e-9
    if not hit.any():
        return g.iloc[0:0]
    out = g[hit].copy()
    out["ask_low_strike"] = prev_min[hit]; out["low_strike"] = g["strike"].to_numpy()[arg[hit]]
    out["low_ticker"] = g["ticker"].to_numpy()[arg[hit]] if "ticker" in g else None
    out["gross"] = gross[hit]
    out["net"] = gross[hit] - fee(prev_min[hit]) - fee(b[hit])
    return out


def scan_day(c: pd.DataFrame, hours: Dict[str, float], max_hours: float = 2.0) -> Dict[str, pd.DataFrame]:
    c = c[c["event_ticker"].map(hours).fillna(99) <= max_hours]
    c = c[(c["m"] >= 1) & (c["m"] <= 59)]
    res: Dict[str, List[pd.DataFrame]] = {"close": [], "minute": [], "ffill": []}
    for ev, g in c.groupby("event_ticker", sort=False):
        for t, gt in g.sort_values("strike").groupby("t", sort=False):
            if len(gt) >= 2:
                v = _violations(gt, "bid", "ask")
                if len(v): res["close"].append(v.assign(tier="close"))
                v = _violations(gt, "bid_low", "ask_high")
                if len(v): res["minute"].append(v.assign(tier="minute"))
        # forward-filled book over the ladder's minutes
        piv_b = g.pivot_table(index="m", columns="strike", values="bid").reindex(range(1, 60)).ffill()
        piv_a = g.pivot_table(index="m", columns="strike", values="ask").reindex(range(1, 60)).ffill()
        tick = g.drop_duplicates("strike").set_index("strike")["ticker"]
        o = int(g["t"].iloc[0]) - 60 * int(g["m"].iloc[0])
        for m in range(1, 60):
            gt = pd.DataFrame({"strike": piv_b.columns, "bid": piv_b.loc[m].to_numpy(), "ask": piv_a.loc[m].to_numpy()})
            gt["ticker"] = gt["strike"].map(tick).to_numpy(); gt["event_ticker"] = ev; gt["t"] = o + 60 * m; gt["m"] = m
            v = _violations(gt.sort_values("strike"), "bid", "ask")
            if len(v): res["ffill"].append(v.assign(tier="ffill"))
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in res.items()}


def check_fills(v: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """For each flagged (minute, high strike, low strike): prints in (t-60, t] that would have let both legs trade."""
    if v.empty or fills.empty:
        return v.assign(buy_low_ok=False, sell_high_ok=False, both_in_fills=False)
    f = fills[~fills["block"]]
    f = f.assign(m=(np.ceil(f["ts"] / 60.0) * 60).astype(np.int64))
    have = set(f["ticker"].unique())
    buy = f[f["taker_side"] == "yes"].groupby(["ticker", "m"])["yes_price"].min()      # takers bought yes at this low
    sell = f[f["taker_side"] == "no"].groupby(["ticker", "m"])["yes_price"].max()      # takers sold yes at this high
    v = v.copy()
    v["both_in_fills"] = v["ticker"].isin(have) & v["low_ticker"].isin(have)
    lo = pd.MultiIndex.from_arrays([v["low_ticker"], v["t"]]); hi = pd.MultiIndex.from_arrays([v["ticker"], v["t"]])
    v["low_buy_px"] = buy.reindex(lo).to_numpy(); v["high_sell_px"] = sell.reindex(hi).to_numpy()
    v["buy_low_ok"] = v["low_buy_px"] <= v["ask_low_strike"] + 1e-9
    v["sell_high_ok"] = v["high_sell_px"] >= v["bid"] - 1e-9
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/kalshi_ladder"); ap.add_argument("--out", default="data/backtests/ladder_arb")
    ap.add_argument("--start", default="2026-08-20"); ap.add_argument("--end", default="2026-10-25")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allv = {"close": [], "minute": [], "ffill": []}; n_min = 0; n_ev = 0
    for f in sorted(glob.glob(os.path.join(a.archive, "candles", "*.parquet"))):
        day = os.path.basename(f)[:10]
        if not (a.start <= day < a.end):
            continue
        c = pd.read_parquet(f)
        mk = pd.read_parquet(os.path.join(a.archive, "markets", f"{day}.parquet"))
        hours = dict(zip(mk["event_ticker"], mk["hours"]))
        hourly = c[c["event_ticker"].map(hours).fillna(99) <= 2.0]
        n_ev += hourly["event_ticker"].nunique(); n_min += hourly[["event_ticker", "t"]].drop_duplicates().shape[0]
        r = scan_day(c, hours)
        fp = os.path.join(a.archive, "fills", f"{day}.parquet")
        fills = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()
        if len(fills) and "ticker" not in fills:
            fills = pd.DataFrame()
        for k, v in r.items():
            if len(v):
                v = check_fills(v, fills); v["label_day"] = day; allv[k].append(v)
        print(day, {k: int(len(v)) for k, v in r.items()}, flush=True)
    res = {"hourly_events": n_ev, "candle_minutes_with_quotes": n_min, "tiers": {}}
    for k, vs in allv.items():
        v = pd.concat(vs, ignore_index=True) if vs else pd.DataFrame()
        v.to_parquet(os.path.join(a.out, f"violations_{k}.parquet"), index=False)
        if v.empty:
            res["tiers"][k] = {"n": 0}; continue
        pos = v[v["net"] > 0]
        bf = v[v["both_in_fills"]]; bfp = pos[pos["both_in_fills"]]
        res["tiers"][k] = {
            "n_rows": int(len(v)), "n_event_minutes": int(v[["event_ticker", "t"]].drop_duplicates().shape[0]), "n_events": int(v["event_ticker"].nunique()),
            "gross_cents_quantiles": [float(x * 100) for x in v["gross"].quantile([0.5, 0.9, 0.99, 1.0])],
            "n_net_positive_rows": int(len(pos)), "n_net_positive_event_minutes": int(pos[["event_ticker", "t"]].drop_duplicates().shape[0]),
            "net_positive_cents_quantiles": [float(x * 100) for x in pos["net"].quantile([0.5, 0.9, 1.0])] if len(pos) else None,
            "net_positive_by_day": {d: int(n) for d, n in pos.groupby("label_day").size().items()} if len(pos) else {},
            "strike_gap_quantiles": [float(x) for x in ((v["strike"] - v["low_strike"]) / 100.0).quantile([0.5, 0.9, 1.0])],
            "fills_check": {"rows_with_both_strikes_in_fills": int(len(bf)), "net_positive_with_both_in_fills": int(len(bfp)),
                            "both_legs_printed_at_or_through": int((bfp["buy_low_ok"] & bfp["sell_high_ok"]).sum()) if len(bfp) else 0,
                            "low_leg_printed": int(bfp["buy_low_ok"].sum()) if len(bfp) else 0, "high_leg_printed": int(bfp["sell_high_ok"].sum()) if len(bfp) else 0},
        }
        if k == "close" and len(pos):
            res["tiers"][k]["net_positive_by_h"] = {str(b): int(n) for b, n in pos.groupby(pd.cut(pos["m"], [0, 9, 20, 35, 59])).size().items()}
            res["tiers"][k]["net_positive_by_bid_level"] = {str(b): int(n) for b, n in pos.groupby(pd.cut(pos["bid"], [0, 0.1, 0.3, 0.7, 0.9, 1.0])).size().items()}
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
