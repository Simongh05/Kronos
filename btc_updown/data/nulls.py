"""Free baselines the model must beat: an EWMA-vol, TWAP-aware digital pricer.

P(Up | t) for a market with strike K (in the series' own price level, i.e. the typical price of the
bar that closed at the window start) and h bars to expiry:

    P = Phi( ln(c_t / K) / (sigma_1m * sqrt(h_eff)) ),   h_eff = h - 2/3

sigma_1m is an EWMA of squared 1-minute log returns (halflife in minutes); h - 2/3 is the variance
of the mean of the final minute relative to the current close (Brownian motion averaged over the last
unit interval has variance h - 1 + 1/3).  Fat tails are handled by an optional Student-t with `df`.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from ..train.dataset import BAR_SECONDS, SeriesStore, typical_price


def ewma_sigma(close: np.ndarray, halflife_min: float) -> np.ndarray:
    """Per-bar EWMA volatility of 1-minute log returns (uses returns up to and including bar i)."""
    lr = np.diff(np.log(close), prepend=np.log(close[0]))
    lam = 0.5 ** (1.0 / halflife_min)
    var = np.empty_like(lr)
    v = float(np.var(lr[1:200])) if len(lr) > 200 else float(np.var(lr[1:])) + 1e-12
    for i in range(len(lr)):
        v = lam * v + (1 - lam) * lr[i] ** 2
        var[i] = v
    return np.sqrt(var)


def null_prob(close_t: np.ndarray, strike: np.ndarray, sigma: np.ndarray, h: np.ndarray, df: float | None = None) -> np.ndarray:
    h_eff = np.maximum(h - 2.0 / 3.0, 1.0 / 3.0)
    z = np.log(close_t / strike) / (sigma * np.sqrt(h_eff) + 1e-12)
    if df is None:
        return norm.cdf(z)
    scale = np.sqrt(df / (df - 2)) if df > 2 else 1.0  # keep the variance equal to the Gaussian's
    return student_t.cdf(z * scale, df)


def score_on_windows(store: SeriesStore, windows: pd.DataFrame, halflife_min: float = 60.0, df: float | None = None,
                     min_minutes_left: int = 1) -> pd.DataFrame:
    """One row per (window, decision minute): null probability vs outcome."""
    ts = np.asarray(store.ts)
    close = np.asarray(store.x[:, 3], dtype=np.float64)
    sigma = ewma_sigma(close, halflife_min)
    tp = typical_price(np.asarray(store.x, dtype=np.float64))
    rows = []
    for ws, we, oc, fam, venue in zip(windows["window_start"].to_numpy(), windows["window_end"].to_numpy(),
                                      windows["outcome"].to_numpy(), windows["family"].to_numpy(), windows["venue"].to_numpy()):
        i0 = store.index_of_bar_closing_at(int(ws))
        if i0 < 0 or oc not in (0, 1):
            continue
        K = tp[i0]
        dur = int((we - ws) // BAR_SECONDS)
        for k in range(1, dur - min_minutes_left + 1):
            t = int(ws + BAR_SECONDS * k)
            it = store.index_of_bar_closing_at(t)
            if it < 0:
                continue
            h = dur - k
            rows.append((venue, fam, int(ws), int(we), t, k, h, close[it], K, sigma[it], int(oc)))
    df_rows = pd.DataFrame(rows, columns=["venue", "family", "window_start", "window_end", "t", "j", "h", "close_t", "strike_series",
                                          "sigma1m", "outcome"])
    if len(df_rows):
        df_rows["p_null"] = null_prob(df_rows["close_t"].to_numpy(), df_rows["strike_series"].to_numpy(),
                                      df_rows["sigma1m"].to_numpy(), df_rows["h"].to_numpy().astype(float), df)
    return df_rows


def summarize(df: pd.DataFrame) -> Dict:
    out = {}
    for (venue, fam), g in df.groupby(["venue", "family"]):
        b = float(np.mean((g["p_null"] - g["outcome"]) ** 2))
        out[f"{venue}/{fam}"] = {"n": int(len(g)), "brier_null": b, "brier_const": float(np.mean((g["outcome"].mean() - g["outcome"]) ** 2)),
                                 "by_h": {int(h): float(np.mean((gg["p_null"] - gg["outcome"]) ** 2)) for h, gg in g.groupby("h")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/klines"); ap.add_argument("--series", default="spot_BTCUSDC")
    ap.add_argument("--windows", default="data/labels/windows.parquet")
    ap.add_argument("--start", default="2026-08-07"); ap.add_argument("--end", default="2026-09-07")
    ap.add_argument("--halflife", type=float, default=60.0); ap.add_argument("--df", type=float, default=None)
    ap.add_argument("--out", default="data/controls/null_pricer_val.parquet")
    a = ap.parse_args()
    store = SeriesStore.open(a.root, a.series)
    w = pd.read_parquet(a.windows)
    t0 = int(np.datetime64(a.start, "s").astype(np.int64)); t1 = int(np.datetime64(a.end, "s").astype(np.int64))
    w = w[(w["window_start"] >= t0) & (w["window_end"] <= t1)]
    w = w[w["regime"].isin(["btc-5m-twap-60", "btc-15m-twap-60", "brti60"])]
    df = score_on_windows(store, w, a.halflife, a.df)
    df.to_parquet(a.out, index=False)
    print(json.dumps(summarize(df), indent=1))


if __name__ == "__main__":
    main()
