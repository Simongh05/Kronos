"""Build the decision table: one row per (market window, decision minute) joining labels, market
prices, cube outputs (sampling summary / typical-price samples / DirectHead logits) and the null pricer.

Columns (all optional sources are NaN when absent):
    venue, family, market_id, window_start, window_end, label_day, t, j, h, outcome, strike, final,
    strike_series            typical price of the series' bar that closed at window_start (K in Binance terms)
    close_t, sigma1          from the cube summary (or the store when no cube)
    exec_price_up/down, exec_size_up/down, mkt_prob_up, mkt_prob_src       from prices_*.parquet
    tp_mean, tp_sd, tp_q50   sampling summary at (t, h)
    z_model                  (tp_mean - strike_series) / tp_sd
    log_rel_sd               log(tp_sd / close_t)
    p_samp                   fraction of sorted tp samples >= strike_series (from tp_samples/*.npy)
    logit_direct             DirectHead logit at (t, j, h)
    p_null                   EWMA-vol digital pricer
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..data.nulls import ewma_sigma, null_prob
from ..train.dataset import BAR_SECONDS, SeriesStore, typical_price

REGIMES_OK = {"btc-5m-twap-60", "btc-15m-twap-60", "brti60", "binance-1h-candle"}
# families whose strike is the OPEN of the bar at window start (Binance candle markets), not the previous bar's typical price
OPEN_STRIKE_FAMILIES = {"btc-1h"}


def _epoch(s: str) -> int:
    return int(np.datetime64(s, "s").astype(np.int64))


def decision_rows(windows: pd.DataFrame, store: SeriesStore, min_minutes_left: int = 1) -> pd.DataFrame:
    ts = np.asarray(store.ts)
    tp_all = typical_price(np.asarray(store.x, dtype=np.float64))
    rows = []
    for r in windows.itertuples(index=False):
        i0 = store.index_of_bar_closing_at(int(r.window_start))
        if i0 < 0:
            continue
        dur = int((r.window_end - r.window_start) // BAR_SECONDS)
        if r.family in OPEN_STRIKE_FAMILIES:
            i_open = store.index_of_bar_closing_at(int(r.window_start) + BAR_SECONDS)
            if i_open < 0:
                continue
            strike_series = float(store.x[i_open, 0])
        else:
            strike_series = float(tp_all[i0])
        for k in range(1, dur - min_minutes_left + 1):
            t = int(r.window_start + BAR_SECONDS * k)
            rows.append((r.venue, r.family, r.market_id, int(r.window_start), int(r.window_end), r.label_day, t, k, dur - k,
                         int(r.outcome), float(r.strike) if pd.notna(r.strike) else np.nan,
                         float(r.final) if pd.notna(r.final) else np.nan, strike_series))
    cols = ["venue", "family", "market_id", "window_start", "window_end", "label_day", "t", "j", "h", "outcome", "strike",
            "final", "strike_series"]
    return pd.DataFrame(rows, columns=cols)


def attach_prices(table: pd.DataFrame, prices: Optional[pd.DataFrame]) -> pd.DataFrame:
    if prices is None or len(prices) == 0:
        for c in ["exec_price_up", "exec_size_up", "exec_price_down", "exec_size_down", "mkt_prob_up", "n_prints_30s"]:
            table[c] = np.nan
        table["mkt_prob_src"] = None
        return table
    cols = ["market_id", "t", "exec_price_up", "exec_size_up", "exec_price_down", "exec_size_down", "mkt_prob_up",
            "mkt_prob_src", "n_prints_30s"]
    p = prices[[c for c in cols if c in prices.columns]].drop_duplicates(["market_id", "t"])
    return table.merge(p, on=["market_id", "t"], how="left")


def attach_cube(table: pd.DataFrame, cube_dir: Optional[str]) -> pd.DataFrame:
    for c in ["tp_mean", "tp_sd", "tp_q50", "close_t", "sigma1", "p_samp", "logit_direct"]:
        table[c] = np.nan
    if not cube_dir:
        return table
    # sampling summary + samples
    summ_files = sorted(glob.glob(os.path.join(cube_dir, "summary", "*.parquet")))
    if summ_files:
        s = pd.concat([pd.read_parquet(f, columns=["t", "h", "tp_mean", "tp_sd", "tp_q50", "close_t", "sigma1"]) for f in summ_files])
        table = table.drop(columns=["tp_mean", "tp_sd", "tp_q50", "close_t", "sigma1"]).merge(s, on=["t", "h"], how="left")
        # exact tail probability from the sorted samples
        p_samp = np.full(len(table), np.nan)
        by_day: Dict[str, int] = {}
        for f in sorted(glob.glob(os.path.join(cube_dir, "tp_samples", "*.t.npy"))):
            day = os.path.basename(f)[:10]
            t_arr = np.load(f)
            if len(t_arr) == 0:
                continue
            samp = np.load(os.path.join(cube_dir, "tp_samples", f"{day}.npy"), mmap_mode="r")
            sel = np.nonzero((table["t"].to_numpy() >= t_arr[0]) & (table["t"].to_numpy() <= t_arr[-1]))[0]
            if len(sel) == 0:
                continue
            idx = np.searchsorted(t_arr, table["t"].to_numpy()[sel])
            ok = (idx < len(t_arr)) & (t_arr[np.clip(idx, 0, len(t_arr) - 1)] == table["t"].to_numpy()[sel])
            sel, idx = sel[ok], idx[ok]
            hh = table["h"].to_numpy()[sel] - 1
            ok2 = (hh >= 0) & (hh < samp.shape[1])
            sel, idx, hh = sel[ok2], idx[ok2], hh[ok2]
            K = table["strike_series"].to_numpy()[sel]
            for a, b, c, k in zip(sel, idx, hh, K):
                row = samp[b, c]
                p_samp[a] = 1.0 - np.searchsorted(row, k, side="left") / row.shape[0]
            by_day[day] = int(len(sel))
        table["p_samp"] = p_samp
    direct_files = sorted(glob.glob(os.path.join(cube_dir, "direct", "*.parquet")))
    if direct_files:
        d = pd.concat([pd.read_parquet(f, columns=["t", "j", "h", "logit"]) for f in direct_files])
        d = d.rename(columns={"logit": "logit_direct"}).drop_duplicates(["t", "j", "h"])
        table = table.drop(columns=["logit_direct"]).merge(d, on=["t", "j", "h"], how="left")
    table["z_model"] = (table["tp_mean"] - table["strike_series"]) / table["tp_sd"]
    table["log_rel_sd"] = np.log(table["tp_sd"] / table["close_t"])
    return table


def attach_null(table: pd.DataFrame, store: SeriesStore, halflife_min: float = 60.0) -> pd.DataFrame:
    close = np.asarray(store.x[:, 3], dtype=np.float64)
    sigma = ewma_sigma(close, halflife_min)
    idx = np.array([store.index_of_bar_closing_at(int(t)) for t in table["t"].to_numpy()])
    ok = idx >= 0
    c = np.where(ok, close[np.clip(idx, 0, len(close) - 1)], np.nan)
    s = np.where(ok, sigma[np.clip(idx, 0, len(close) - 1)], np.nan)
    table["close_series_t"] = c
    table["sigma_ewma"] = s
    table["p_null"] = null_prob(c, table["strike_series"].to_numpy(), s, table["h"].to_numpy().astype(float))
    if table["close_t"].isna().all():
        table["close_t"] = c
    return table


def build(windows_path: str, prices_path: Optional[str], cube_dir: Optional[str], root: str, series: str,
          start: str, end: str, venues: Iterable[str] = ("polymarket", "kalshi"), regimes: Iterable[str] = REGIMES_OK) -> pd.DataFrame:
    store = SeriesStore.open(root, series)
    w = pd.read_parquet(windows_path)
    t0, t1 = _epoch(start), _epoch(end)
    w = w[(w["window_start"] >= t0) & (w["window_end"] <= t1) & w["venue"].isin(list(venues)) & w["regime"].isin(list(regimes))
          & w["outcome"].isin([0, 1])]
    table = decision_rows(w, store)
    prices = pd.read_parquet(prices_path) if prices_path and os.path.exists(prices_path) else None
    table = attach_prices(table, prices)
    table = attach_cube(table, cube_dir)
    table = attach_null(table, store)
    table["minutes_left"] = table["h"].astype(float)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/labels/windows.parquet")
    ap.add_argument("--prices", default="data/labels/prices_polymarket.parquet")
    ap.add_argument("--cube", default=None, help="cube run_dir (summary/, tp_samples/, direct/)")
    ap.add_argument("--root", default="data/klines"); ap.add_argument("--series", default="spot_BTCUSDC")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--venues", nargs="+", default=["polymarket", "kalshi"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = build(a.windows, a.prices, a.cube, a.root, a.series, a.start, a.end, a.venues)
    t.to_parquet(a.out, index=False)
    print(json.dumps({"rows": int(len(t)), "windows": int(t["market_id"].nunique()),
                      "with_market_price": int(t["mkt_prob_up"].notna().sum()),
                      "with_cube": int(t["tp_mean"].notna().sum()), "with_direct": int(t["logit_direct"].notna().sum())}))


if __name__ == "__main__":
    main()
