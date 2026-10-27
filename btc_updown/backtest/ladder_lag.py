"""Cross-asset lag check on an alt-coin Kalshi ladder (KXETHD, KXSOLD): do the alt ladder's quotes lag BTC's spot moves
(or the alt's own spot) by a minute, and is that tradeable?

Inputs: the alt ladder table from `ladder.py` (alt klines series for close_t / sigma / p_null) and the BTC kline store.
Features known at t (bars closed by t): z-scored log returns of BTC spot over the last 1, 5, 15 minutes (zb1, zb5,
zb15; return / (σ_ewma √w)) and of the alt's own spot (za1, za5, za15).
A. Lead-lag of the quote: Δmid(t → t+1) at ATM strikes (|k| ≤ 2, two-sided and fresh at both minutes) regressed on
   the returns known at t and on Δmid(t−1 → t); day-block bootstrap. Contemporaneous regression on the same returns
   for scale (how much the quote moves with the current minute).
B. Outcome model: val fit of outcome ~ logit(mid) + logit(p_null) + z's, day-block bootstrap of the coefficients;
   frozen on dev: per-day Brier gain over the mid (paired day bootstrap), for (mid + null) and (mid + null + BTC).
C. Backtest on dev with the frozen fits: first qualifying minute per market (h 5–55, |blend − price| > fee + 1 tick),
   taker at the quote, hold to settlement, $100 clips, day stats vs the $150 hurdle.

    python -m btc_updown.backtest.ladder_lag --table data/tables/ladder_eth.parquet --alt-series spot_ETHUSDT \
        --out data/backtests/ladder_lag_eth
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from ..calib.fit import design, fit, predict, brier, logit
from ..calib.market_head import paired_day_bootstrap
from ..data.nulls import ewma_sigma
from ..train.dataset import SeriesStore
from .engine import day_stats, fee

WIN = (1, 5, 15)


def spot_z(store: SeriesStore, times: np.ndarray, prefix: str, halflife: float = 60.0) -> pd.DataFrame:
    ts = np.asarray(store.ts); close = np.asarray(store.x[:, 3], dtype=np.float64)
    lc = np.log(close); sig = ewma_sigma(close, halflife)
    pos = pd.Series(np.arange(len(ts)), index=ts)
    i = pos.reindex(times - 60).to_numpy()                   # bar with open t-60 closes at t
    ok = np.isfinite(i) & (i >= 16); i = np.where(ok, i, 16).astype(int)
    out = {"t": times}
    for w in WIN:
        r = lc[i] - lc[i - w]
        out[f"{prefix}{w}"] = np.where(ok, r / (sig[i] * np.sqrt(w) + 1e-12), np.nan)
    return pd.DataFrame(out).drop_duplicates("t")


def _ols_boot(days, X, y, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed); ud = np.unique(days); groups = {d: np.nonzero(days == d)[0] for d in ud}
    X1 = np.concatenate([np.ones((len(y), 1)), X], 1)
    def solve(idx): return np.linalg.lstsq(X1[idx], y[idx], rcond=None)[0]
    b = solve(np.arange(len(y))); bs = []
    for _ in range(n_boot):
        pick = rng.choice(ud, size=len(ud), replace=True); bs.append(solve(np.concatenate([groups[d] for d in pick])))
    bs = np.array(bs); resid = y - X1 @ b
    return b, np.quantile(bs, [0.025, 0.975], axis=0), 1 - resid.var() / y.var()


def lead_lag(t: pd.DataFrame, zcols) -> Dict:
    a = t[(t["k"].abs() <= 2) & t["two_sided"] & (t["age"] == 0)].copy()
    a["mid"] = a["mkt_prob_up"]
    a = a.sort_values(["market_id", "t"])
    key = pd.MultiIndex.from_arrays([a["market_id"], a["t"]])
    s = pd.Series(a["mid"].to_numpy(), index=key)
    a["mid_next"] = s.reindex(pd.MultiIndex.from_arrays([a["market_id"], a["t"] + 60])).to_numpy()
    a["mid_prev"] = s.reindex(pd.MultiIndex.from_arrays([a["market_id"], a["t"] - 60])).to_numpy()
    a["d_next"] = (a["mid_next"] - a["mid"]) * 100; a["d_prev"] = (a["mid"] - a["mid_prev"]) * 100     # in cents
    a = a.dropna(subset=["d_next", "d_prev"] + list(zcols))
    days = a["label_day"].astype(str).to_numpy(); y = a["d_next"].to_numpy()
    out = {"n_rows": int(len(a)), "n_days": int(len(set(days)))}
    X = a[list(zcols) + ["d_prev"]].to_numpy(); b, ci, r2 = _ols_boot(days, X, y)
    out["next_minute"] = {"coef": {n: float(v) for n, v in zip(["intercept"] + list(zcols) + ["d_prev"], b)},
                          "ci95": {n: [float(ci[0, i]), float(ci[1, i])] for i, n in enumerate(["intercept"] + list(zcols) + ["d_prev"])}, "r2": float(r2)}
    X = a[list(zcols)].to_numpy(); yc = a["d_prev"].to_numpy(); b, ci, r2 = _ols_boot(days, X, yc)
    out["same_minute"] = {"coef": {n: float(v) for n, v in zip(["intercept"] + list(zcols), b)}, "r2": float(r2)}
    out["sd_d_next_cents"] = float(np.std(y)); out["corr"] = {c: float(np.corrcoef(a[c], y)[0, 1]) for c in zcols}
    return out


def backtest(te: pd.DataFrame, pcol: str, venue: str = "kalshi", hmin: int = 5, hmax: int = 55, margin: float = 0.01) -> Dict:
    trades = []
    for mid, g in te[(te["h"] >= hmin) & (te["h"] <= hmax)].sort_values(["market_id", "t"]).groupby("market_id", sort=False):
        for r in g.itertuples(index=False):
            p = getattr(r, pcol); best = None
            for side, price, ps in (("up", r.exec_price_up, p), ("down", r.exec_price_down, 1 - p)):
                if not (isinstance(price, float) and np.isfinite(price)) or price <= 0 or price >= 1:
                    continue
                f = float(fee(venue, np.array([price]))[0]); edge = ps - price - f
                if edge > margin and (best is None or edge > best["edge"]):
                    best = {"side": side, "price": price, "fee": f, "edge": edge}
            if best is None:
                continue
            shares = 100.0 / best["price"]; win = (r.outcome == 1) if best["side"] == "up" else (r.outcome == 0)
            pnl_share = (1.0 if win else 0.0) - best["price"] - best["fee"]
            trades.append({"label_day": str(r.label_day), "market_id": mid, "h": r.h, "side": best["side"], "price": best["price"], "win": bool(win),
                           "pnl": pnl_share * shares, "shares": shares, "notional": shares * best["price"]})
            break
    df = pd.DataFrame(trades)
    st = day_stats(df, hurdle_per_day=150.0, n_boot=2000, all_days=sorted(te["label_day"].astype(str).unique())) if len(df) else {"n_trades": 0}
    return {k: st.get(k) for k in ("n_trades", "mean_per_day", "cents_per_share", "hit_rate", "bootstrap_lb95_mean_per_day", "positive_days", "n_days")}


def run(table: str, klines: str, btc_series: str, alt_series: str, split_day: str, out: str, n_boot: int) -> Dict:
    os.makedirs(out, exist_ok=True)
    t = pd.read_parquet(table)
    t = t[t["outcome"].isin([0, 1])].copy(); t["label_day"] = t["label_day"].astype(str)
    zb = spot_z(SeriesStore.open(klines, btc_series), t["t"].drop_duplicates().to_numpy(), "zb")
    za = spot_z(SeriesStore.open(klines, alt_series), t["t"].drop_duplicates().to_numpy(), "za")
    t = t.merge(zb, on="t", how="left").merge(za, on="t", how="left")
    zcols = [f"zb{w}" for w in WIN] + [f"za{w}" for w in WIN]
    t = t[t["two_sided"] & t["mkt_prob_up"].notna() & t["p_null"].notna()].dropna(subset=zcols)
    val = t[t["label_day"] < split_day].copy(); dev = t[t["label_day"] >= split_day].copy()
    res = {"rows": {"val": int(len(val)), "dev": int(len(dev))}, "days": {"val": int(val["label_day"].nunique()), "dev": int(dev["label_day"].nunique())},
           "markets": {"val": int(val["market_id"].nunique()), "dev": int(dev["market_id"].nunique())}}
    res["A_lead_lag_val"] = lead_lag(val, zcols); res["A_lead_lag_dev"] = lead_lag(dev, zcols)
    sets = {"null": ["logit_null"], "null_btc": ["logit_null", "zb1", "zb5", "zb15"], "null_alt": ["logit_null", "za1", "za5", "za15"], "null_btc_alt": ["logit_null"] + zcols}
    res["B_val"] = {}; res["B_dev_frozen"] = {"brier_market": brier(dev["mkt_prob_up"].to_numpy(), dev["outcome"].to_numpy().astype(float)),
                                            "brier_null": brier(dev["p_null"].to_numpy(), dev["outcome"].to_numpy().astype(float))}
    y = dev["outcome"].to_numpy().astype(float); d = dev["label_day"].to_numpy()
    for name, feats in sets.items():
        r = fit(val, feats, include_market=True, n_boot=n_boot)
        res["B_val"][name] = {"coef": r["coef"], "coef_ci95": r["bootstrap"]["coef_ci95"], "brier": r["brier"]}
        beta = np.array([r["coef"]["intercept"], r["coef"]["logit_market"]] + [r["coef"][f] for f in feats])
        dev[f"p_{name}"] = predict(beta, design(dev, feats, True))
        res["B_dev_frozen"][name] = {"brier": brier(dev[f"p_{name}"].to_numpy(), y),
                                     "minus_market": {"mean": brier(dev[f"p_{name}"].to_numpy(), y) - res["B_dev_frozen"]["brier_market"],
                                                      "ci95": paired_day_bootstrap(d, dev[f"p_{name}"].to_numpy(), dev["mkt_prob_up"].to_numpy(), y)}}
        res["B_dev_frozen"][name]["backtest"] = backtest(dev, f"p_{name}")
    res["B_dev_frozen"]["null_btc_alt_minus_null"] = paired_day_bootstrap(d, dev["p_null_btc_alt"].to_numpy(), dev["p_null"].to_numpy(), y)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--klines", default="data/klines"); ap.add_argument("--btc-series", default="spot_BTCUSDC")
    ap.add_argument("--alt-series", required=True); ap.add_argument("--split-day", default="2026-09-21"); ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.table, a.klines, a.btc_series, a.alt_series, a.split_day, a.out, a.n_boot), indent=1, default=float))


if __name__ == "__main__":
    main()
