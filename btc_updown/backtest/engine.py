"""Backtest: one entry per market window, hold to settlement, fees, day-level statistics.

Inputs: a decision table (backtest.assemble) and a frozen calibration JSON (calib.fit) whose
coefficients turn the row's features into P(Up).  Rule, per window, scanning decision minutes in
time order: skip if minutes_left < min_minutes_left or no executable price; edge_up = P - ask_up -
fee(ask_up) - margin; edge_down = (1-P) - ask_down - fee(ask_down) - margin; take the larger if > 0;
size = min(clip_usd / price, exec_size); stop after the first entry.  P&L per share = 1[win] - price - fee.

Two execution assumptions are reported: 'print' (the next print's price) and 'print+tick' (one tick worse).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..calib.fit import design, predict, brier

TICK = {"polymarket": 0.01, "kalshi": 0.01}


def fee_polymarket(p: np.ndarray) -> np.ndarray:
    return 0.07 * p * (1 - p)


def fee_kalshi(p: np.ndarray) -> np.ndarray:
    return np.ceil(0.07 * p * (1 - p) * 1e4) / 1e4   # rounded up to $0.0001 per contract (direct member)


def fee(venue: str, p: np.ndarray) -> np.ndarray:
    return fee_kalshi(p) if venue == "kalshi" else fee_polymarket(p)


def blended_prob(table: pd.DataFrame, calib: Dict, market_override: Optional[np.ndarray] = None) -> np.ndarray:
    """P(Up) from the frozen blend. `market_override` replaces mkt_prob_up (e.g. with the side's own ask) so that
    the trading rule never uses a market print later than the one it executes against."""
    t = table
    if market_override is not None and calib.get("include_market"):
        t = table.copy()
        t["mkt_prob_up"] = market_override
    X = design(t, calib["features"], calib["include_market"])
    names = ["intercept"] + (["logit_market"] if calib["include_market"] else []) + list(calib["features"])
    beta = np.array([calib["coef"][n] for n in names])
    return predict(beta, X)


def run_backtest(table: pd.DataFrame, calib: Dict, clip_usd: float = 100.0, margin: float = 0.0, min_minutes_left: int = 1,
                 haircut_ticks: int = 0, prob_col: Optional[str] = None) -> pd.DataFrame:
    t = table.copy()
    need = ["mkt_prob_up"] if calib.get("include_market") else []
    need += [f for f in calib["features"] if f not in ("logit_null", "logit_samp", "minutes_left")]
    t = t.dropna(subset=need + ["outcome"])
    if prob_col:
        t["p_up_side"] = t[prob_col].to_numpy()
        t["p_down_side"] = t[prob_col].to_numpy()
    else:
        # market input for the Up decision = the Up ask; for the Down decision = 1 - Down ask (same print we execute on)
        t["p_up_side"] = blended_prob(t, calib, market_override=t["exec_price_up"].to_numpy())
        t["p_down_side"] = blended_prob(t, calib, market_override=1.0 - t["exec_price_down"].to_numpy())
    t["p"] = blended_prob(t, calib) if not prob_col else t[prob_col].to_numpy()
    t = t.sort_values(["market_id", "t"])
    trades: List[Dict] = []
    for mid, g in t.groupby("market_id", sort=False):
        venue = g["venue"].iloc[0]
        tick = TICK.get(venue, 0.01) * haircut_ticks
        for r in g.itertuples(index=False):
            if r.h < min_minutes_left:
                continue
            best = None
            for side, price, size, p_side in (("up", r.exec_price_up, r.exec_size_up, r.p_up_side),
                                              ("down", r.exec_price_down, r.exec_size_down, 1 - r.p_down_side)):
                if not (isinstance(price, float) and np.isfinite(price)) or price <= 0 or price >= 1 or not np.isfinite(p_side):
                    continue
                px = min(price + tick, 0.999)
                f = float(fee(venue, np.array([px]))[0])
                edge = p_side - px - f - margin
                if edge > 0 and (best is None or edge > best["edge"]):
                    best = {"side": side, "price": px, "fee": f, "edge": edge, "size": size if np.isfinite(size) else np.inf}
            if best is None:
                continue
            shares = min(clip_usd / best["price"], best["size"])
            win = (r.outcome == 1) if best["side"] == "up" else (r.outcome == 0)
            pnl_share = (1.0 if win else 0.0) - best["price"] - best["fee"]
            trades.append({"venue": venue, "family": r.family, "market_id": mid, "label_day": str(r.label_day), "t": r.t, "j": r.j,
                           "h": r.h, "side": best["side"], "price": best["price"], "fee": best["fee"], "p_model": r.p,
                           "p_side": r.p if best["side"] == "up" else 1 - r.p, "edge": best["edge"], "shares": shares,
                           "notional": shares * best["price"], "win": bool(win), "pnl": pnl_share * shares, "pnl_share": pnl_share})
            break
    return pd.DataFrame(trades)


def day_stats(trades: pd.DataFrame, hurdle_per_day: float = 150.0, n_boot: int = 10000, seed: int = 0, all_days: Optional[List[str]] = None) -> Dict:
    if len(trades) == 0:
        return {"n_trades": 0}
    per_day = trades.groupby("label_day")["pnl"].sum()
    if all_days is not None:
        per_day = per_day.reindex(sorted(set(all_days) | set(per_day.index)), fill_value=0.0)
    x = per_day.to_numpy()
    n = len(x)
    t_stat, p_one = (np.nan, np.nan)
    if n > 1 and x.std(ddof=1) > 0:
        t_res = stats.ttest_1samp(x, hurdle_per_day)
        t_stat = float(t_res.statistic); p_one = float(1 - stats.t.cdf(t_stat, n - 1))
    rng = np.random.RandomState(seed)
    # stationary block bootstrap of days, mean block length 3
    boots = []
    for _ in range(n_boot):
        out, i = [], rng.randint(n)
        while len(out) < n:
            out.append(x[i])
            i = rng.randint(n) if rng.rand() < 1 / 3 else (i + 1) % n
        boots.append(np.mean(out))
    lb = float(np.quantile(boots, 0.05))
    return {
        "n_trades": int(len(trades)), "n_days": int(n), "mean_per_day": float(x.mean()), "sd_per_day": float(x.std(ddof=1)) if n > 1 else None,
        "positive_days": int((x > 0).sum()), "total_pnl": float(x.sum()), "cents_per_share": float(100 * trades["pnl"].sum() / trades["shares"].sum()),
        "hit_rate": float(trades["win"].mean()), "notional": float(trades["notional"].sum()),
        "t_vs_hurdle": t_stat, "p_one_sided": p_one, "bootstrap_lb95_mean_per_day": lb, "hurdle": hurdle_per_day,
        "half_mean_rule_passes": bool(0.5 * x.mean() > hurdle_per_day),
    }


def brier_table(table: pd.DataFrame, calib: Dict) -> Dict:
    t = table.dropna(subset=["outcome"]).copy()
    t = t[t["outcome"].isin([0, 1])]
    need = (["mkt_prob_up"] if calib.get("include_market") else []) + [f for f in calib["features"] if f not in ("logit_null", "logit_samp", "minutes_left")]
    t = t.dropna(subset=need)
    if len(t) == 0:
        return {}
    y = t["outcome"].to_numpy().astype(float)
    p = blended_prob(t, calib)
    out = {"n": int(len(t)), "blend": brier(p, y), "null": brier(t["p_null"].to_numpy(), y)}
    if t["mkt_prob_up"].notna().any():
        m = t["mkt_prob_up"].notna().to_numpy()
        out["market"] = brier(t["mkt_prob_up"].to_numpy()[m].astype(float), y[m]); out["blend_on_market_rows"] = brier(p[m], y[m]); out["n_market_rows"] = int(m.sum())
    pos = pd.Series(np.arange(len(t)), index=t.index)
    by_h = {}
    for h, g in t.groupby("h"):
        gi = pos.loc[g.index].to_numpy()
        row = {"n": int(len(gi)), "blend": brier(p[gi], y[gi]), "null": brier(t["p_null"].to_numpy()[gi], y[gi])}
        mm = t["mkt_prob_up"].notna().to_numpy()[gi]
        if mm.any():
            row["market"] = brier(t["mkt_prob_up"].to_numpy()[gi][mm].astype(float), y[gi][mm])
        by_h[int(h)] = row
    out["by_h"] = by_h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--clip", type=float, default=100.0); ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--min-minutes-left", type=int, default=1); ap.add_argument("--hurdle", type=float, default=150.0)
    ap.add_argument("--venue", default=None); ap.add_argument("--family", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    table = pd.read_parquet(a.table)
    if a.venue:
        table = table[table["venue"] == a.venue]
    if a.family:
        table = table[table["family"] == a.family]
    calib = json.load(open(a.calib))
    os.makedirs(a.out, exist_ok=True)
    all_days = sorted(table["label_day"].astype(str).unique())
    res = {"calib": a.calib, "clip_usd": a.clip, "margin": a.margin, "brier": brier_table(table, calib), "execution": {}}
    for name, hc in (("print", 0), ("print+tick", 1)):
        tr = run_backtest(table, calib, a.clip, a.margin, a.min_minutes_left, haircut_ticks=hc)
        tr.to_parquet(os.path.join(a.out, f"trades_{name.replace('+', '_')}.parquet"), index=False)
        res["execution"][name] = day_stats(tr, a.hurdle, all_days=all_days)
        if len(tr):
            res["execution"][name]["by_minutes_left"] = tr.groupby("h").agg(n=("pnl", "size"), pnl=("pnl", "sum"), cps=("pnl_share", "mean")).to_dict("index")
            tr["bucket"] = pd.cut(tr["price"], [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])
            res["execution"][name]["by_price_bucket"] = {str(k): v for k, v in tr.groupby("bucket", observed=True).agg(n=("pnl", "size"), pnl=("pnl", "sum"), cps=("pnl_share", "mean")).to_dict("index").items()}
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "execution"}, indent=1, default=float))
    for name, s in res["execution"].items():
        print(name, json.dumps({k: s[k] for k in s if k not in ("by_minutes_left", "by_price_bucket")}, default=float))


if __name__ == "__main__":
    main()
