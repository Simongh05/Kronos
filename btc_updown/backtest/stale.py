"""Stale-quote robustness check for an alt ladder (free, Mac only).

The lag test prices the trade at the candle's closing quote of minute t. That quote may have been posted 50 s before
the Binance close it is compared with, so part of any "edge" is the time the book had not yet moved. Execution modes:
    close       the closing bid/ask of minute t (the lag test's assumption, optimistic)
    next_open   the opening bid/ask of minute t+1's candle when one exists (the book right after t), else the close
    next_tick   next_open plus one tick against us
    fresh       next_open, entries only where the quote at t was updated in minute t (age 0)
Breakdowns of the next_open P&L by minutes left, entry price, |k|, quote age, spread, day.

    python -m btc_updown.backtest.stale --table data/tables/ladder_sold.parquet --archive data/kalshi_ladder_KXSOLD \
        --coef data/backtests/ladder_lag_sold/summary.json --out data/backtests/stale_sold
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from ..calib.fit import logit, sigmoid
from .engine import day_stats, fee


def next_open(archive: str, start: str, end: str) -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(os.path.join(archive, "candles", "*.parquet"))) if start <= os.path.basename(f)[:10] < end]
    c = pd.concat([pd.read_parquet(f, columns=["ticker", "m", "bid_open", "ask_open", "bid_low", "ask_high"]) for f in fs], ignore_index=True)
    c = c.drop_duplicates(["ticker", "m"])
    c["m"] = c["m"] - 1                      # candle of minute m+1 attached to the decision minute m
    return c.rename(columns={"ticker": "market_id", "bid_open": "bid_next", "ask_open": "ask_next", "bid_low": "bid_low_next", "ask_high": "ask_high_next"})


def trades(te: pd.DataFrame, pcol: str, mode: str, hmin=5, hmax=55, margin=0.01, venue="kalshi") -> pd.DataFrame:
    rows = []
    for mid, g in te[(te["h"] >= hmin) & (te["h"] <= hmax)].sort_values(["market_id", "t"]).groupby("market_id", sort=False):
        for r in g.itertuples(index=False):
            p = getattr(r, pcol)
            ask, bid = r.ask, r.bid
            if mode != "close":
                ask = r.ask_next if np.isfinite(r.ask_next) else r.ask
                bid = r.bid_next if np.isfinite(r.bid_next) else r.bid
            if mode == "next_tick":
                ask, bid = min(ask + 0.01, 1.0), max(bid - 0.01, 0.0)
            if mode == "fresh" and r.age != 0:
                continue
            best = None
            for side, price, ps in (("up", ask, p), ("down", 1 - bid, 1 - p)):
                if not np.isfinite(price) or price <= 0 or price >= 1:
                    continue
                f = float(fee(venue, np.array([price]))[0]); edge = ps - price - f
                if edge > margin and (best is None or edge > best["edge"]):
                    best = {"side": side, "price": price, "fee": f, "edge": edge}
            if best is None:
                continue
            shares = 100.0 / best["price"]; win = (r.outcome == 1) if best["side"] == "up" else (r.outcome == 0)
            pnl_share = (1.0 if win else 0.0) - best["price"] - best["fee"]
            rows.append({"label_day": str(r.label_day), "market_id": mid, "h": r.h, "k": r.k, "age": r.age, "spread": r.ask - r.bid, "side": best["side"],
                         "price": best["price"], "edge": best["edge"], "p": p, "mid": r.mkt_prob_up, "win": bool(win), "pnl": pnl_share * shares,
                         "shares": shares, "notional": shares * best["price"]})
            break
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame, days) -> Dict:
    if not len(df):
        return {"n_trades": 0}
    st = day_stats(df, hurdle_per_day=150.0, n_boot=2000, all_days=days)
    return {k: st.get(k) for k in ("n_trades", "mean_per_day", "cents_per_share", "hit_rate", "bootstrap_lb95_mean_per_day", "positive_days", "n_days")}


def breakdown(df: pd.DataFrame) -> Dict:
    out = {}
    cuts = {"h": pd.cut(df["h"], [0, 9, 20, 35, 59]), "price": pd.cut(df["price"], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
            "abs_k": pd.cut(df["k"].abs(), [-1, 0.5, 1.5, 3.5, 12]), "age": pd.cut(df["age"], [-1, 0, 2, 10, 60]), "spread": pd.cut(df["spread"], [0, 0.02, 0.05, 0.1, 1.0]),
            "side": df["side"]}
    for name, c in cuts.items():
        g = df.groupby(c, observed=True)
        out[name] = {str(k): {"n": int(len(v)), "pnl": float(v["pnl"].sum()), "cps": float(v["pnl"].sum() / v["shares"].sum() * 100), "hit": float(v["win"].mean())} for k, v in g}
    pd_ = df.groupby("label_day")["pnl"].sum().sort_values()
    out["worst_days"] = {k: float(v) for k, v in pd_.head(3).items()}; out["best_days"] = {k: float(v) for k, v in pd_.tail(3).items()}
    out["pnl_share_of_top3_days"] = float(pd_.tail(3).sum() / pd_.sum()) if pd_.sum() > 0 else None
    return out


def run(table: str, archive: str, coef_path: str, split_day: str, out: str) -> Dict:
    os.makedirs(out, exist_ok=True)
    t = pd.read_parquet(table); t = t[t["outcome"].isin([0, 1]) & t["two_sided"] & t["p_null"].notna()].copy(); t["label_day"] = t["label_day"].astype(str)
    c = json.load(open(coef_path))["B_val"]["null"]["coef"]
    t["p_blend"] = sigmoid(c["intercept"] + c["logit_market"] * logit(t["mkt_prob_up"].to_numpy()) + c["logit_null"] * logit(t["p_null"].to_numpy()))
    t["m"] = t["j"]
    no = next_open(archive, t["label_day"].min(), "2026-12-31")
    t = t.merge(no, on=["market_id", "m"], how="left")
    res = {"coef": c, "rows": int(len(t)), "next_open_available": float(t["ask_next"].notna().mean())}
    for split, te in (("dev_frozen", t[t["label_day"] >= split_day]), ("val_insample", t[t["label_day"] < split_day])):
        days = sorted(te["label_day"].unique()); res[split] = {}
        for mode in ("close", "next_open", "next_tick", "fresh"):
            df = trades(te, "p_blend", mode); res[split][mode] = stats(df, days)
            if mode == "next_open":
                res[split]["next_open_breakdown"] = breakdown(df) if len(df) else {}
                df.to_parquet(os.path.join(out, f"trades_{split}_next_open.parquet"), index=False)
        # how much the quote moves between the close of t and the open of t+1, at the entry rows
        df = trades(te, "p_blend", "close")
        if len(df):
            e = te.set_index(["market_id", "h"]).loc[list(zip(df["market_id"], df["h"]))]
            res[split]["quote_move_close_to_next_open_cents"] = {"ask_mean": float(((e["ask_next"] - e["ask"]) * 100).mean()), "bid_mean": float(((e["bid_next"] - e["bid"]) * 100).mean()),
                                                                 "ask_q": [float(x * 100) for x in (e["ask_next"] - e["ask"]).quantile([0.1, 0.5, 0.9])]}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--archive", required=True); ap.add_argument("--coef", required=True)
    ap.add_argument("--split-day", default="2026-09-21"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.table, a.archive, a.coef, a.split_day, a.out), indent=1, default=float))


if __name__ == "__main__":
    main()
