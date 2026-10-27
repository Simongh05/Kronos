"""Maker-execution study on the Kalshi KXBTCD ladders: rest limit orders priced off the frozen blend and fill them
against the archived taker prints (`kalshi_ladder.py fills`), hold to settlement.

Model.  At every decision minute t (book = the candle closing at t), for every strike with a blend probability p:
  yes bid  at  min(floor(p - margin), ask - 1 tick)  if that is >= bid + 1 tick ("improve": a new best level, first in
           queue) or == bid ("join": back of an unknown queue);
  yes ask  at  max(ceil(p + margin), bid + 1 tick)   if that is <= ask - 1 tick (improve) or == ask (join).
An order rests until the desired price changes (replaced; a replaced "join" order loses its queue) or the ladder
closes.  Fills come from the taker prints in the minute after each decision: a print with taker_side "no" (a taker
selling yes into the bid) at yes_price <= our bid fills our bid (improve: we were the best price, they would have hit
us; join: only prints strictly below our price, i.e. the level was traded through), symmetrically for the ask with
taker_side "yes".  Fill size = min(remaining, print count); one position per market, hold to settlement; maker fee
0 or round-up(0.0175 C P (1 - P)) (Kalshi's maker schedule, applied only where the series charges it).

Controls: --prob-col p_null (the free pricer posts the same way) and --prob-col mkt_prob_up (post off the mid: pure
market making, measures the adverse selection a maker eats on this book regardless of model).

    python -m btc_updown.backtest.maker --table data/tables/ladder_e1b_dev.parquet --fills data/kalshi_ladder/fills \
        --calib data/calib/ladder_e1b_market_val.json --mode improve --margin-ticks 1 --out data/backtests/maker_e1b_dev
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .engine import blended_prob, day_stats

TICK = 0.01


def load_fills(archive: str, start: str, end: str) -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(os.path.join(archive, "fills", "*.parquet"))) if start <= os.path.basename(f)[:10] < end]
    if not fs:
        return pd.DataFrame(columns=["ticker", "ts", "yes_price", "count", "taker_side"])
    cols = ["ticker", "ts", "yes_price", "count", "taker_side", "block"]
    parts = [pd.read_parquet(f) for f in fs]
    parts = [p[cols] for p in parts if len(p)]  # a day with no prints is written as an empty, column-less file
    if not parts:
        return pd.DataFrame(columns=["ticker", "ts", "yes_price", "count", "taker_side"])
    d = pd.concat(parts)
    d = d[~d["block"]].drop(columns="block")
    return d.sort_values(["ticker", "ts"]).reset_index(drop=True)


def _r(x: float) -> float:
    return round(x, 2)


def simulate(table: pd.DataFrame, fills: pd.DataFrame, prob: np.ndarray, mode: str = "improve", margin_ticks: int = 1,
             clip_usd: float = 100.0, maker_fee: float = 0.0, max_k: Optional[int] = 5, min_minutes_left: int = 1,
             max_minutes_left: int = 59) -> pd.DataFrame:
    t = table.copy()
    t["p"] = prob
    t = t[t["two_sided"] & t["p"].notna() & t["h"].between(min_minutes_left, max_minutes_left)]
    if max_k is not None:
        t = t[t["k"].abs() <= max_k]
    margin = margin_ticks * TICK
    by_tk = {k: g.sort_values("ts") for k, g in fills.groupby("ticker")}
    out: List[Dict] = []
    for tk, g in t.groupby("market_id", sort=False):
        f = by_tk.get(tk)
        if f is None or len(f) == 0:
            continue
        f_ts = f["ts"].to_numpy(); f_px = f["yes_price"].to_numpy(); f_n = f["count"].to_numpy(); f_side = f["taker_side"].to_numpy()
        g = g.sort_values("t")
        outcome = int(g["outcome"].iloc[0]); day = g["label_day"].iloc[0]; k = int(g["k"].iloc[0])
        rest = {"yes": None, "no": None}        # resting order per side: dict(price, post_t, post_h, filled, size)
        pos = None
        for row in g.itertuples(index=False):
            if pos is not None:
                break
            bid, ask, p, tt, h = float(row.bid), float(row.ask), float(row.p), int(row.t), int(row.h)
            want = {}
            lo = math.floor((p - margin) / TICK + 1e-9) * TICK           # highest bid consistent with the blend
            hi = math.ceil((p + margin) / TICK - 1e-9) * TICK            # lowest ask consistent with the blend
            if mode == "improve":
                pb = _r(min(lo, ask - TICK)); pa = _r(max(hi, bid + TICK))
                want["yes"] = pb if pb >= _r(bid + TICK) and pb < ask - 1e-9 else None
                want["no"] = pa if pa <= _r(ask - TICK) and pa > bid + 1e-9 else None
            else:
                want["yes"] = _r(bid) if lo >= bid - 1e-9 else None
                want["no"] = _r(ask) if hi <= ask + 1e-9 else None
            for side in ("yes", "no"):
                w = want[side]
                r = rest[side]
                if w is None:
                    rest[side] = None
                    continue
                if r is None or abs(r["price"] - w) > 1e-9:
                    size = int(clip_usd / (w if side == "yes" else 1.0 - w))
                    rest[side] = {"price": w, "post_t": tt, "post_h": h, "filled": 0.0, "size": max(size, 1)}
            # match prints in (t, t + 60] against the resting orders
            i0 = int(np.searchsorted(f_ts, tt, side="right")); i1 = int(np.searchsorted(f_ts, tt + 60, side="right"))
            for i in range(i0, i1):
                for side in ("yes", "no"):
                    r = rest[side]
                    if r is None or r["filled"] >= r["size"]:
                        continue
                    if side == "yes":
                        hit = f_side[i] == "no" and (f_px[i] <= r["price"] + 1e-9 if mode == "improve" else f_px[i] < r["price"] - 1e-9)
                    else:
                        hit = f_side[i] == "yes" and (f_px[i] >= r["price"] - 1e-9 if mode == "improve" else f_px[i] > r["price"] + 1e-9)
                    if not hit:
                        continue
                    q = min(r["size"] - r["filled"], float(f_n[i]))
                    r["filled"] += q
                    if r.get("fill_t") is None:
                        r["fill_t"] = float(f_ts[i]); r["fill_h"] = h
                if any(rest[s] is not None and rest[s]["filled"] > 0 for s in ("yes", "no")):
                    break
            for side in ("yes", "no"):
                r = rest[side]
                if r is not None and r["filled"] > 0:
                    price = r["price"] if side == "yes" else 1.0 - r["price"]          # price paid per contract on our side
                    win = outcome == (1 if side == "yes" else 0)
                    n = r["filled"]
                    fee = math.ceil(maker_fee * n * price * (1 - price) * 100) / 100 if maker_fee > 0 else 0.0
                    pnl = n * ((1.0 if win else 0.0) - price) - fee
                    pos = {"market_id": tk, "label_day": day, "side": side, "k": k, "post_t": r["post_t"], "h": r["post_h"],
                           "fill_t": r["fill_t"], "fill_h": r["fill_h"], "price": price, "yes_price": r["price"], "shares": n,
                           "size": r["size"], "p_blend": p, "bid": bid, "ask": ask, "outcome": outcome, "win": bool(win),
                           "fee": fee, "pnl": pnl, "notional": n * price}
                    out.append(pos)
                    break
    cols = ["market_id", "label_day", "side", "k", "post_t", "h", "fill_t", "fill_h", "price", "yes_price", "shares", "size",
            "p_blend", "bid", "ask", "outcome", "win", "fee", "pnl", "notional"]
    return pd.DataFrame(out, columns=cols)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True); ap.add_argument("--fills", default="data/kalshi_ladder")
    ap.add_argument("--calib", default=None, help="frozen calibration JSON (blend); omit with --prob-col")
    ap.add_argument("--prob-col", default=None, help="use this table column as the probability instead of the blend")
    ap.add_argument("--mode", choices=["improve", "join"], default="improve")
    ap.add_argument("--margin-ticks", type=int, default=1); ap.add_argument("--clip", type=float, default=100.0)
    ap.add_argument("--maker-fee", type=float, default=0.0, help="0 or 0.0175")
    ap.add_argument("--max-k", type=int, default=5); ap.add_argument("--min-minutes-left", type=int, default=1)
    ap.add_argument("--max-minutes-left", type=int, default=59)
    ap.add_argument("--hurdle", type=float, default=150.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    table = pd.read_parquet(a.table)
    days = sorted(str(d) for d in table["label_day"].unique())
    fills = load_fills(a.fills, days[0], str(pd.Timestamp(days[-1]) + pd.Timedelta(days=1))[:10])
    if a.prob_col:
        prob = table[a.prob_col].to_numpy(dtype=float)
    else:
        calib = json.load(open(a.calib)); prob = blended_prob(table, calib)
    tr = simulate(table, fills, prob, a.mode, a.margin_ticks, a.clip, a.maker_fee, a.max_k, a.min_minutes_left, a.max_minutes_left)
    os.makedirs(a.out, exist_ok=True)
    tr.to_parquet(os.path.join(a.out, "fills.parquet"), index=False)
    fill_days = sorted(set(os.path.basename(f)[:10] for f in glob.glob(os.path.join(a.fills, "fills", "*.parquet"))))
    covered = [d for d in days if d in fill_days and len(fills[fills["ticker"].isin(table.loc[table["label_day"].astype(str) == d, "market_id"])]) > 0]
    res = {"table": a.table, "mode": a.mode, "margin_ticks": a.margin_ticks, "maker_fee": a.maker_fee, "prob": a.prob_col or a.calib,
           "days_with_fills": len(covered), "n_markets_with_prints": int(fills["ticker"].nunique()),
           "stats": day_stats(tr, a.hurdle, all_days=covered) if len(tr) else {"n_trades": 0}}
    if len(tr):
        res["by_side"] = tr.groupby("side").agg(n=("pnl", "size"), pnl=("pnl", "sum"), cps=("pnl", lambda x: 100 * x.sum() / tr.loc[x.index, "shares"].sum()), hit=("win", "mean")).to_dict("index")
        res["by_h"] = tr.groupby(pd.cut(tr["h"], [0, 9, 20, 35, 59])).agg(n=("pnl", "size"), pnl=("pnl", "sum"), hit=("win", "mean")).rename(index=str).to_dict("index")
        res["by_k"] = tr.groupby(tr["k"].abs()).agg(n=("pnl", "size"), pnl=("pnl", "sum"), hit=("win", "mean")).to_dict("index")
        res["fill_delay_min_median"] = float(((tr["fill_t"] - tr["post_t"]) / 60).median())
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_h", "by_k")}, indent=1, default=float))


if __name__ == "__main__":
    main()
