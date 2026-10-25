"""Proxy validation (Mac only, reads the research clone read-only through its guard).

Question: how well does the 1-minute bar proxy (typical price of the Binance bar that closes at T)
reproduce the exact 60-second settlement average and the recorded final_price, and how often does
the bar proxy flip the outcome sign?  The flip rate is the ceiling for any bar-based model.

Sources (polymarket-5m): Binance 1-s klines BTCUSDT (data/h1k/binance_BTCUSDT_1s + data/settlement/
exchange_raw/binance_BTCUSDT_1s) and BTCUSDC (data/settlement/exchange_raw/binance_BTCUSDC_1s),
loaded with scripts/settlement/hist_common.load_binance_1s; labels from our data/labels/windows.parquet.

Run with the research venv:
    /Users/streetteam/Documents/perpcity-arb-bot/polymarket-5m/.venv/bin/python -m btc_updown.data.proxy_check \
        --windows data/labels/windows.parquet --out data/controls/proxy_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

RESEARCH_ROOT = "/Users/streetteam/Documents/perpcity-arb-bot/polymarket-5m"


def _guard_ok(dataset: str, t0: int, t1: int) -> bool:
    sys.path.insert(0, RESEARCH_ROOT)
    try:
        from scripts.common import guard as G  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"guard import failed: {e}", file=sys.stderr)
        return False
    try:
        G.check_data(None, dataset, t0, t1)
        return True
    except Exception as e:
        print(f"guard refused {dataset} {t0}-{t1}: {e}", file=sys.stderr)
        return False


def load_1s(symbol: str) -> pd.DataFrame:
    """Returns a frame indexed by second (UTC s) with columns open, high, low, close, vol via hist_common."""
    sys.path.insert(0, os.path.join(RESEARCH_ROOT, "scripts", "settlement"))
    sys.path.insert(0, RESEARCH_ROOT)
    import hist_common  # type: ignore
    df = hist_common.load_binance_1s(symbol)
    return df


def exact_window_mean(close_1s: pd.Series, T: int, k: int = 3) -> float:
    """Mean of the 60 per-second closes over [T-62, T-3] (k=3) or [T-61, T-2] (k=2); NaN if any second missing."""
    idx = np.arange(T - 59 - k, T - k + 1)
    vals = close_1s.reindex(idx)
    if vals.isna().any():
        # linear interpolation of interior gaps as the Chainlink TWAP does; give up if edges are missing
        if np.isnan(vals.iloc[0]) or np.isnan(vals.iloc[-1]):
            return float("nan")
        vals = vals.interpolate()
    return float(vals.mean())


def bar_typical(ohlc_min: pd.DataFrame, T: int) -> float:
    """(O+H+L+C)/4 of the 1-minute bar [T-60, T) (frame indexed by minute open, built from 1-s closes)."""
    if (T - 60) not in ohlc_min.index:
        return float("nan")
    r = ohlc_min.loc[T - 60]
    if r["n"] < 30:
        return float("nan")
    return float((r["open"] + r["high"] + r["low"] + r["close"]) / 4)


def run(windows_path: str, symbols=("BTCUSDT", "BTCUSDC"), max_windows: int = 20000) -> Dict:
    w = pd.read_parquet(windows_path)
    w = w[(w["venue"] == "polymarket") & w["regime"].isin(["btc-5m-twap-60", "btc-15m-twap-60"]) & w["outcome"].isin([0, 1])]
    w = w.dropna(subset=["strike", "final"])
    out: Dict = {"n_windows_considered": int(len(w)), "symbols": {}}
    for sym in symbols:
        if not _guard_ok("binance_klines", int(w["window_start"].min()), int(w["window_end"].max())):
            out["symbols"][sym] = {"error": "guard refused"}
            continue
        df = load_1s(sym)
        # hist_common.load_binance_1s: per-second frame indexed by kline open second with columns c (close), v, n
        close_1s = df["c"].astype(float).dropna()
        close_1s = close_1s[~close_1s.index.duplicated()].sort_index()
        # 1-minute OHLC from the 1-s closes (first/max/min/last within [T-60, T))
        minute = (close_1s.index.to_numpy() // 60) * 60
        g = close_1s.groupby(minute)
        ohlc = pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last(), "n": g.size()})
        ohlc.index = ohlc.index.astype(np.int64)
        close = close_1s
        lo, hi = int(close_1s.index.min()), int(close_1s.index.max())
        ww = w[(w["window_start"] - 120 >= lo) & (w["window_end"] + 5 <= hi)]
        if len(ww) > max_windows:
            ww = ww.sample(max_windows, random_state=0)
        rows = []
        for r in ww.itertuples(index=False):
            T0, T1 = int(r.window_start), int(r.window_end)
            ex0, ex1 = exact_window_mean(close, T0), exact_window_mean(close, T1)
            tp0, tp1 = bar_typical(ohlc, T0), bar_typical(ohlc, T1)
            c1 = float(close.get(T1 - 1, np.nan))
            rows.append((r.family, T0, T1, r.strike, r.final, int(r.outcome), ex0, ex1, tp0, tp1, c1))
        d = pd.DataFrame(rows, columns=["family", "T0", "T1", "strike", "final", "outcome", "exact0", "exact1", "tp0", "tp1", "close1"]).dropna()
        res = {"n": int(len(d))}
        for fam, g in d.groupby("family"):
            basis = g["final"] - g["exact1"]
            e_exact = g["exact1"] - g["final"]            # exchange exact 60-s mean vs Chainlink final (basis + venue noise)
            e_tp_vs_exact = g["tp1"] - g["exact1"]        # bar proxy vs exact mean, same exchange
            e_close = g["close1"] - g["exact1"]
            # outcome under each proxy, using the series' own strike proxy (basis cancels)
            o_exact = (g["exact1"] >= g["exact0"]).astype(int)
            o_tp = (g["tp1"] >= g["tp0"]).astype(int)
            o_close = (g["close1"] >= g["tp0"]).astype(int)
            margin = (g["final"] - g["strike"]).abs()
            q = lambda s: {k: float(np.quantile(s, v)) for k, v in (("p50", .5), ("p90", .9), ("p99", .99))}
            res[fam] = {
                "n": int(len(g)),
                "basis_final_minus_exchange_exact_usd": {"median": float(basis.median()), "sd": float(basis.std())},
                "abs_err_exact_vs_final_usd": q(e_exact.abs()),
                "abs_err_tp_vs_exact_usd": q(e_tp_vs_exact.abs()),
                "abs_err_close_vs_exact_usd": q(e_close.abs()),
                "median_abs_margin_usd": float(margin.median()),
                "flip_rate_exact_series": float((o_exact != g["outcome"]).mean()),
                "flip_rate_tp_proxy": float((o_tp != g["outcome"]).mean()),
                "flip_rate_close_proxy": float((o_close != g["outcome"]).mean()),
                "flip_rate_tp_when_margin_lt_20usd": float((o_tp != g["outcome"])[margin < 20].mean()) if (margin < 20).any() else None,
            }
        out["symbols"][sym] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/labels/windows.parquet")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "BTCUSDC"])
    ap.add_argument("--max-windows", type=int, default=20000)
    ap.add_argument("--out", default="data/controls/proxy_check.json")
    a = ap.parse_args()
    res = run(a.windows, a.symbols, a.max_windows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
