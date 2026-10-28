"""Cross-venue test (SCOPE-ROUND2 item 1): the Kalshi KXBTCD ladder vs the Polymarket hourly BTC market on the same hour.

For every Polymarket btc-1h window and decision minute t (h = minutes left), the ladder's yes-mid is interpolated in logit
space at the Polymarket strike expressed in BRTI terms (K* = strike_usdt x usdc/usdt at the open + basis), giving a
Kalshi-implied P(up); the Polymarket mid at t comes from the CLOB price history. Then:
  A. lead-lag both ways (next-minute change of one venue's price on the gap to the other, EWMA null as control);
  B. calibration on val: logit(pm mid) + logit(kalshi implied) + logit(null); frozen on dev: Brier vs the pm mid;
  C. taker backtest on Polymarket at the executable print (engine rule) and its prints check (taker orders at or better
     than our price in the next 1 / 5 min -> fill-size P&L);
  D. the mirror on Kalshi: ladder rows |k| <= 2 with the Polymarket price mapped to the ladder strike through the null's
     shape; blend, backtest at the ladder's exec prices, prints check against the Kalshi fills archive.

    python -m btc_updown.backtest.cross_venue --out data/backtests/cross_venue
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..calib.fit import brier, fit, logit, predict, design
from ..calib.market_head import paired_day_bootstrap
from ..data.nulls import null_prob
from ..train.dataset import SeriesStore
from .engine import day_stats, run_backtest
from .ladder_lag import _ols_boot
from .maker import load_fills

SPLIT = "2026-09-21"


def _close_at(store: SeriesStore, times: np.ndarray) -> np.ndarray:
    """Close of the 1-min bar that ends at `times` (bar open = t - 60)."""
    pos = np.searchsorted(store.ts, times - 60)
    pos = np.clip(pos, 0, len(store.ts) - 1)
    ok = store.ts[pos] == times - 60
    out = store.x[pos, 3].astype(float)
    out[~ok] = np.nan
    return out


def _asof(times_src: np.ndarray, vals_src: np.ndarray, times_q: np.ndarray, max_age: float = 1800.0) -> np.ndarray:
    i = np.searchsorted(times_src, times_q, side="right") - 1
    out = np.full(len(times_q), np.nan)
    ok = i >= 0
    out[ok] = vals_src[i[ok]]
    age = times_q[ok] - times_src[i[ok]]
    out[np.nonzero(ok)[0][age > max_age]] = np.nan
    return out


def pm_mid_asof(mid_dir: str, days: List[str], query: pd.DataFrame, col: str, offset: int = 0) -> np.ndarray:
    fs = [os.path.join(mid_dir, f"{d}.parquet") for d in days]
    parts = [pd.read_parquet(f) for f in fs if os.path.exists(f)]
    m = pd.concat(parts).sort_values(["market_id", "t"])
    out = np.full(len(query), np.nan)
    by = {k: (g["t"].to_numpy(), g["p"].to_numpy(dtype=float)) for k, g in m.groupby("market_id")}
    for mid_id, idx in query.groupby("market_id").indices.items():
        src = by.get(mid_id)
        if src is None:
            continue
        out[idx] = _asof(src[0], src[1], query["t"].to_numpy()[idx] + offset)
    query[col] = out
    return out


def ladder_implied(ladder: pd.DataFrame, kstar: pd.DataFrame) -> pd.DataFrame:
    """kstar: rows (event_ticker, t, K) -> interpolated yes mid/bid/ask (logit-linear between the bracketing two-sided
    strikes), the bracketing gap in $, and the ladder's close_t / sigma / basis at t."""
    L = ladder[ladder["two_sided"] & (ladder["bid"] > 0) & (ladder["ask"] < 1)].sort_values(["event_ticker", "t", "strike"])
    L = L[["event_ticker", "t", "strike", "bid", "ask"]]
    groups = L.groupby(["event_ticker", "t"]).indices
    S = L["strike"].to_numpy(); B = L["bid"].to_numpy(); A = L["ask"].to_numpy(); M = (B + A) / 2
    lB, lA, lM = logit(np.clip(B, 0.005, 0.995)), logit(np.clip(A, 0.005, 0.995)), logit(np.clip(M, 0.005, 0.995))
    n = len(kstar); out = {c: np.full(n, np.nan) for c in ["k_mid", "k_bid", "k_ask", "gap_usd", "n_two_sided"]}
    ev = kstar["event_ticker"].to_numpy(); tt = kstar["t"].to_numpy(); K = kstar["K"].to_numpy(dtype=float)
    for i in range(n):
        idx = groups.get((ev[i], tt[i]))
        if idx is None or len(idx) < 2:
            continue
        s = S[idx]
        if not (s[0] <= K[i] <= s[-1]):
            continue
        j = int(np.searchsorted(s, K[i]))
        j = min(max(j, 1), len(s) - 1)
        w = (K[i] - s[j - 1]) / (s[j] - s[j - 1])
        out["k_mid"][i] = 1 / (1 + np.exp(-((1 - w) * lM[idx[j - 1]] + w * lM[idx[j]])))
        out["k_bid"][i] = 1 / (1 + np.exp(-((1 - w) * lB[idx[j - 1]] + w * lB[idx[j]])))
        out["k_ask"][i] = 1 / (1 + np.exp(-((1 - w) * lA[idx[j - 1]] + w * lA[idx[j]])))
        out["gap_usd"][i] = s[j] - s[j - 1]; out["n_two_sided"][i] = len(idx)
    return pd.DataFrame(out, index=kstar.index)


def build(windows: str, prices: str, mid_dir: str, ladder_table: str, klines: str, start: str, end: str) -> pd.DataFrame:
    w = pd.read_parquet(windows)
    w = w[(w["label_day"].astype(str) >= start) & (w["label_day"].astype(str) < end) & w["strike"].notna() & w["outcome"].isin([0, 1])]
    p = pd.read_parquet(prices)
    p = p[p["market_id"].isin(w["market_id"])]
    t = p.merge(w[["market_id", "label_day", "strike", "outcome", "volume_usd"]], on="market_id")
    t["h"] = t["minutes_left"].astype(int); t["j"] = 60 - t["h"]
    t = t[(t["h"] >= 1) & (t["h"] <= 59)].copy()
    t["venue"] = "polymarket"; t["family"] = "btc-1h"
    # the pm strike in USDC terms, then in BRTI terms with the ladder's basis
    usdt = SeriesStore.open(klines, "spot_BTCUSDT"); usdc = SeriesStore.open(klines, "spot_BTCUSDC")
    ws = t["window_start"].to_numpy()
    ratio = _close_at(usdc, ws) / _close_at(usdt, ws)
    t["k_usdc"] = t["strike"] * ratio
    lad = pd.read_parquet(ladder_table, columns=["event_ticker", "t", "strike", "bid", "ask", "two_sided", "close_t", "sigma_ewma", "basis"])
    per_t = lad.drop_duplicates(["event_ticker", "t"])[["event_ticker", "t", "close_t", "sigma_ewma", "basis"]]
    # map window_end -> event_ticker: the ladder's event closes at the same top of the hour (t + 60 h for any row)
    full = pd.read_parquet(ladder_table, columns=["event_ticker", "t", "h"]).drop_duplicates(["event_ticker"])
    full["window_end"] = full["t"] + 60 * full["h"]
    ev_map = dict(zip(full["window_end"], full["event_ticker"]))
    t["event_ticker"] = t["window_end"].map(ev_map)
    t = t[t["event_ticker"].notna()].copy()
    t = t.merge(per_t, on=["event_ticker", "t"], how="left")
    t["K"] = t["k_usdc"] + t["basis"].fillna(0.0)
    imp = ladder_implied(lad, t[["event_ticker", "t", "K"]])
    for c in imp.columns:
        t[c] = imp[c].to_numpy()
    t["p_null"] = null_prob(t["close_t"].to_numpy(), t["k_usdc"].to_numpy(), t["sigma_ewma"].to_numpy(), t["h"].to_numpy(dtype=float))
    days = sorted(t["label_day"].astype(str).unique())
    pm_mid_asof(mid_dir, days, t, "pm_mid", 0)
    pm_mid_asof(mid_dir, days, t, "pm_mid_next", 60)
    pm_mid_asof(mid_dir, days, t, "pm_mid_prev", -60)
    t = t.sort_values(["market_id", "t"]).reset_index(drop=True)
    t["k_mid_next"] = t.groupby("market_id")["k_mid"].shift(-1)
    t["k_mid_prev"] = t.groupby("market_id")["k_mid"].shift(1)      # the candle that closed a minute earlier: strictly older than the pm point
    t["mkt_prob_up"] = t["pm_mid"]
    t["logit_kalshi"] = logit(np.clip(t["k_mid"].to_numpy(dtype=float), 0.005, 0.995))
    t["label_day"] = t["label_day"].astype(str)
    return t


def lead_lag(t: pd.DataFrame, n_boot: int = 1000) -> Dict:
    d = t.dropna(subset=["pm_mid", "pm_mid_next", "pm_mid_prev", "k_mid", "k_mid_next", "p_null"])
    d = d[(d["pm_mid"] > 0.03) & (d["pm_mid"] < 0.97)]
    days = d["label_day"].to_numpy()
    res = {"n_rows": int(len(d)), "n_days": int(len(np.unique(days)))}
    y = 100 * (d["pm_mid_next"] - d["pm_mid"]).to_numpy()
    X = np.column_stack([100 * (d["k_mid"] - d["pm_mid"]), 100 * (d["p_null"] - d["pm_mid"]), 100 * (d["pm_mid"] - d["pm_mid_prev"])])
    b, ci, r2 = _ols_boot(days, X, y, n_boot)
    names = ["intercept", "kalshi_minus_pm", "null_minus_pm", "d_prev"]
    res["pm_next_on_gap"] = {"coef": dict(zip(names, map(float, b))), "ci95": {k: [float(ci[0][i]), float(ci[1][i])] for i, k in enumerate(names)}, "r2": float(r2),
                             "sd_y_cents": float(y.std()), "sd_gap_cents": float(X[:, 0].std())}
    b, ci, r2 = _ols_boot(days, X[:, :1], y, n_boot)
    res["pm_next_on_gap_only"] = {"coef": dict(zip(names[:2], map(float, b))), "ci95": {k: [float(ci[0][i]), float(ci[1][i])] for i, k in enumerate(names[:2])}, "r2": float(r2)}
    # conservative alignment: the pm point at <= t (0-60 s old) against the Kalshi candle closed at t - 60 (60-120 s older than t)
    dc = d.dropna(subset=["k_mid_prev"])
    yc = 100 * (dc["pm_mid_next"] - dc["pm_mid"]).to_numpy()
    Xc = np.column_stack([100 * (dc["k_mid_prev"] - dc["pm_mid"]), 100 * (dc["p_null"] - dc["pm_mid"]), 100 * (dc["pm_mid"] - dc["pm_mid_prev"])])
    b, ci, r2 = _ols_boot(dc["label_day"].to_numpy(), Xc, yc, n_boot)
    res["pm_next_on_gap_kalshi_one_minute_older"] = {"coef": dict(zip(names, map(float, b))), "ci95": {k: [float(ci[0][i]), float(ci[1][i])] for i, k in enumerate(names)}, "r2": float(r2), "n": int(len(dc))}
    y2 = 100 * (d["k_mid_next"] - d["k_mid"]).to_numpy()
    X2 = np.column_stack([100 * (d["pm_mid"] - d["k_mid"]), 100 * (d["p_null"] - d["k_mid"])])
    b, ci, r2 = _ols_boot(days, X2, y2, n_boot)
    names2 = ["intercept", "pm_minus_kalshi", "null_minus_kalshi"]
    res["kalshi_next_on_gap"] = {"coef": dict(zip(names2, map(float, b))), "ci95": {k: [float(ci[0][i]), float(ci[1][i])] for i, k in enumerate(names2)}, "r2": float(r2), "sd_y_cents": float(y2.std())}
    res["gap_stats_cents"] = {"mean": float(X[:, 0].mean()), "sd": float(X[:, 0].std()), "abs_median": float(np.median(np.abs(X[:, 0]))),
                              "corr_pm_kalshi": float(np.corrcoef(d["pm_mid"], d["k_mid"])[0, 1])}
    return res


def prints_check_pm(trades: pd.DataFrame, trades_dir: str, days: List[str], windows=(60, 300)) -> Dict:
    fs = [os.path.join(trades_dir, f"{d}.parquet") for d in days]
    T = pd.concat([pd.read_parquet(f) for f in fs if os.path.exists(f)]).sort_values("timestamp")
    by = {k: g for k, g in T.groupby("market_id")}
    out = {}
    for W in windows:
        qty = np.zeros(len(trades)); anyp = np.zeros(len(trades), bool)
        for i, r in enumerate(trades.itertuples(index=False)):
            g = by.get(r.market_id)
            if g is None:
                continue
            w = g[(g["timestamp"] > r.t) & (g["timestamp"] <= r.t + W)]
            anyp[i] = len(w) > 0
            if r.side == "up":
                ok = w[((w["side"] == "BUY") & (w["outcome"] == "Up") & (w["price"] <= r.price + 1e-9)) | ((w["side"] == "SELL") & (w["outcome"] == "Down") & (w["price"] >= 1 - r.price - 1e-9))]
            else:
                ok = w[((w["side"] == "BUY") & (w["outcome"] == "Down") & (w["price"] <= r.price + 1e-9)) | ((w["side"] == "SELL") & (w["outcome"] == "Up") & (w["price"] >= 1 - r.price - 1e-9))]
            qty[i] = float(ok["size"].sum())
        sh = np.minimum(trades["shares"].to_numpy(), qty)
        d = trades.assign(shares=sh, pnl=sh * trades["pnl_share"].to_numpy(), notional=sh * trades["price"].to_numpy())
        d = d[d["shares"] > 0]
        out[f"w{W // 60}"] = {"frac_any_print": float(anyp.mean()), "frac_print_at_or_better": float((qty > 0).mean()),
                              "frac_fully_fillable": float((qty >= trades["shares"].to_numpy()).mean()),
                              "fillable_shares": float(sh.sum()), "sim_shares": float(trades["shares"].sum()),
                              "day_stats": day_stats(d, 150, 2000, all_days=days) if len(d) else {"n_trades": 0}}
    return out


def prints_check_kalshi(trades: pd.DataFrame, fills_archive: str, days: List[str], windows=(60, 300)) -> Dict:
    F = load_fills(fills_archive, days[0], str(pd.Timestamp(days[-1]) + pd.Timedelta(days=1))[:10])
    by = {k: g for k, g in F.groupby("ticker")}
    out = {}
    for W in windows:
        qty = np.zeros(len(trades)); anyp = np.zeros(len(trades), bool)
        for i, r in enumerate(trades.itertuples(index=False)):
            g = by.get(r.market_id)
            if g is None:
                continue
            w = g[(g["ts"] > r.t) & (g["ts"] <= r.t + W)]
            anyp[i] = len(w) > 0
            yes_px = r.price if r.side == "up" else 1 - r.price
            ok = w[(w["taker_side"] == "yes") & (w["yes_price"] <= yes_px + 1e-9)] if r.side == "up" else w[(w["taker_side"] == "no") & (w["yes_price"] >= yes_px - 1e-9)]
            qty[i] = float(ok["count"].sum())
        sh = np.minimum(trades["shares"].to_numpy(), qty)
        d = trades.assign(shares=sh, pnl=sh * trades["pnl_share"].to_numpy(), notional=sh * trades["price"].to_numpy())
        d = d[d["shares"] > 0]
        out[f"w{W // 60}"] = {"frac_any_print": float(anyp.mean()), "frac_print_at_or_better": float((qty > 0).mean()),
                              "frac_fully_fillable": float((qty >= trades["shares"].to_numpy()).mean()),
                              "fillable_shares": float(sh.sum()), "sim_shares": float(trades["shares"].sum()),
                              "day_stats": day_stats(d, 150, 2000, all_days=days) if len(d) else {"n_trades": 0}}
    return out


def _calib(fit_res: Dict) -> Dict:
    return {"features": fit_res["features"], "include_market": fit_res["include_market"], "coef": fit_res["coef"]}


def _frozen(te: pd.DataFrame, calib: Dict, mkt_col: str = "mkt_prob_up") -> Dict:
    need = [c for c in calib["features"] if c not in ("logit_null", "minutes_left")] + ["p_null", mkt_col, "outcome"]
    d = te.dropna(subset=need)
    y = d["outcome"].to_numpy(dtype=float); days = d["label_day"].to_numpy()
    p = predict(np.array([calib["coef"][n] for n in ["intercept", "logit_market"] + list(calib["features"])]), design(d, calib["features"], True))
    m = d[mkt_col].to_numpy(dtype=float)
    res = {"n": int(len(d)), "brier_blend": brier(p, y), "brier_market": brier(m, y), "brier_null": brier(d["p_null"].to_numpy(dtype=float), y),
           "blend_minus_market_ci95": paired_day_bootstrap(days, p, m, y), "by_h": {}}
    for lo, hi in [(1, 9), (10, 20), (21, 35), (36, 59)]:
        s = (d["h"] >= lo) & (d["h"] <= hi)
        if s.sum():
            res["by_h"][f"{lo}-{hi}"] = {"n": int(s.sum()), "blend": brier(p[s], y[s]), "market": brier(m[s], y[s]), "null": brier(d["p_null"].to_numpy(dtype=float)[s], y[s])}
    return res


def _bt(te: pd.DataFrame, calib: Dict, days: List[str], margin: float, ticks: int) -> Dict:
    tr = run_backtest(te, calib, clip_usd=100.0, margin=margin, min_minutes_left=1, haircut_ticks=ticks)
    return tr, (day_stats(tr, 150, 2000, all_days=days) if len(tr) else {"n_trades": 0})


def kalshi_mirror(ladder_table: str, t_pm: pd.DataFrame) -> pd.DataFrame:
    """Ladder rows |k| <= 2, two-sided, fresh, with the Polymarket mid mapped to the row's strike through the null's shape."""
    lad = pd.read_parquet(ladder_table)
    lad = lad[lad["two_sided"] & (lad["k"].abs() <= 2) & (lad["age"] == 0) & lad["outcome"].isin([0, 1]) & lad["mkt_prob_up"].notna()].copy()
    src = t_pm.dropna(subset=["pm_mid"])[["event_ticker", "t", "pm_mid", "K"]].drop_duplicates(["event_ticker", "t"])
    lad = lad.merge(src, on=["event_ticker", "t"], how="inner")
    h_eff = np.maximum(lad["h"].to_numpy(dtype=float) - 2 / 3, 1 / 3)
    z_star = norm.ppf(np.clip(lad["pm_mid"].to_numpy(dtype=float), 0.005, 0.995))
    z = z_star + np.log(lad["K"].to_numpy() / lad["strike"].to_numpy()) / (lad["sigma_ewma"].to_numpy() * np.sqrt(h_eff) + 1e-12)
    lad["pm_implied"] = norm.cdf(z)
    lad["logit_pm"] = logit(np.clip(lad["pm_implied"].to_numpy(), 0.005, 0.995))
    lad["label_day"] = lad["label_day"].astype(str)
    return lad


def run(out: str, windows: str, prices: str, mid_dir: str, trades_dir: str, ladder_table: str, klines: str, fills: str,
        start: str = "2026-08-20", end: str = "2026-10-25", n_boot: int = 500) -> Dict:
    os.makedirs(out, exist_ok=True)
    t = build(windows, prices, mid_dir, ladder_table, klines, start, end)
    t.to_parquet(os.path.join(out, "table_pm.parquet"), index=False)
    va, de = t[t["label_day"] < SPLIT], t[t["label_day"] >= SPLIT]
    days_va, days_de = sorted(va["label_day"].unique()), sorted(de["label_day"].unique())
    res = {"rows": {"val": int(len(va)), "dev": int(len(de))}, "days": {"val": len(days_va), "dev": len(days_de)},
           "coverage": {"kalshi_implied_available": float(t["k_mid"].notna().mean()), "pm_mid_available": float(t["pm_mid"].notna().mean()),
                        "gap_usd_median": float(t["gap_usd"].median()), "basis_median": float(t["basis"].median())}}
    res["A_lead_lag_val"] = lead_lag(va, n_boot); res["A_lead_lag_dev"] = lead_lag(de, n_boot)
    # B: calibration on val, frozen on dev (pm side)
    sets = {"kalshi_null": ["logit_kalshi", "logit_null"], "kalshi": ["logit_kalshi"], "null": ["logit_null"]}
    res["B_val"] = {}; res["B_dev_frozen"] = {}; res["C_backtest_dev"] = {}
    vfit = va.dropna(subset=["pm_mid", "k_mid", "p_null"])
    dfit = de.dropna(subset=["pm_mid", "k_mid", "p_null"])
    for name, feats in sets.items():
        f = fit(vfit, feats, True, n_boot)
        res["B_val"][name] = {"coef": f["coef"], "coef_ci95": f["bootstrap"].get("coef_ci95"), "brier": f["brier"], "n_rows": f["n_rows"]}
        calib = _calib(f)
        res["B_dev_frozen"][name] = _frozen(dfit, calib)
        json.dump(calib, open(os.path.join(out, f"calib_{name}.json"), "w"), indent=1)
        res["C_backtest_dev"][name] = {}
        for margin, ticks in [(0.0, 0), (0.01, 0), (0.01, 1)]:
            tr, st = _bt(dfit, calib, days_de, margin, ticks)
            key = f"margin{int(margin * 100)}_tick{ticks}"
            res["C_backtest_dev"][name][key] = st
            if name == "kalshi_null" and margin == 0.01 and ticks == 0 and len(tr):
                tr.to_parquet(os.path.join(out, "trades_dev_kalshi_null.parquet"), index=False)
                res["C_prints_check_dev_kalshi_null"] = prints_check_pm(tr, trades_dir, days_de)
                res["C_backtest_dev"][name][key + "_by_h"] = tr.groupby(pd.cut(tr["h"], [0, 9, 20, 35, 59])).agg(n=("pnl", "size"), pnl=("pnl", "sum"), hit=("win", "mean")).rename(index=str).to_dict("index")
    # D: Kalshi mirror
    lad = kalshi_mirror(ladder_table, t)
    lva, lde = lad[lad["label_day"] < SPLIT], lad[lad["label_day"] >= SPLIT]
    ldays = sorted(lde["label_day"].unique())
    res["D_kalshi_val"] = {}; res["D_kalshi_dev_frozen"] = {}; res["D_kalshi_backtest_dev"] = {}
    for name, feats in {"pm_null": ["logit_pm", "logit_null"], "null": ["logit_null"]}.items():
        f = fit(lva, feats, True, n_boot)
        res["D_kalshi_val"][name] = {"coef": f["coef"], "coef_ci95": f["bootstrap"].get("coef_ci95"), "brier": f["brier"], "n_rows": f["n_rows"]}
        calib = _calib(f)
        res["D_kalshi_dev_frozen"][name] = _frozen(lde, calib)
        res["D_kalshi_backtest_dev"][name] = {}
        for margin, ticks in [(0.01, 0), (0.01, 1)]:
            tr, st = _bt(lde, calib, ldays, margin, ticks)
            res["D_kalshi_backtest_dev"][name][f"margin{int(margin * 100)}_tick{ticks}"] = st
            if name == "pm_null" and ticks == 0 and len(tr):
                tr.to_parquet(os.path.join(out, "trades_dev_kalshi_side.parquet"), index=False)
                res["D_prints_check_dev"] = prints_check_kalshi(tr, fills, ldays)
    res["D_rows"] = {"val": int(len(lva)), "dev": int(len(lde))}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/backtests/cross_venue")
    ap.add_argument("--windows", default="data/pm_long/windows_btc-1h.parquet")
    ap.add_argument("--prices", default="data/pm_long/prices_btc-1h.parquet")
    ap.add_argument("--mid-dir", default="data/pm_long/mid/btc-1h")
    ap.add_argument("--trades-dir", default="data/pm_long/trades/btc-1h")
    ap.add_argument("--ladder-table", default="data/tables/ladder_e1b.parquet")
    ap.add_argument("--klines", default="data/klines")
    ap.add_argument("--fills", default="data/kalshi_ladder")
    ap.add_argument("--start", default="2026-08-20"); ap.add_argument("--end", default="2026-10-25")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    res = run(a.out, a.windows, a.prices, a.mid_dir, a.trades_dir, a.ladder_table, a.klines, a.fills, a.start, a.end, a.n_boot)
    print(json.dumps({k: res[k] for k in ["rows", "days", "coverage", "A_lead_lag_val", "A_lead_lag_dev"]}, indent=1, default=float))
    for k in ["B_val", "B_dev_frozen", "C_backtest_dev", "C_prints_check_dev_kalshi_null", "D_kalshi_val", "D_kalshi_dev_frozen", "D_kalshi_backtest_dev", "D_prints_check_dev"]:
        if k in res:
            print(k, json.dumps(res[k], default=float)[:3000])


if __name__ == "__main__":
    main()
