"""Decision table for Kalshi KXBTCD hourly strike ladders: one row per (ladder, decision minute, strike).

Inputs (Mac only): the ladder archive (`data/kalshi_ladder/{markets,candles}`), the Binance BTCUSDC kline store, and
optionally a backbone hidden-state dump (`cube.py` with dump_hidden) plus a DirectHead, which is applied at every strike.

Timing: decision minute m = 1..59 after the open, t = open + 60 m, h = 60 - m minutes to the close.  The book at t is
the forward-filled candle ending at t (the last state inside minute m).  Model inputs use bars with open <= t - 60 s.
Strikes are moved onto Binance levels with the basis of the previous ladder's settlement (BRTI average minus the
Binance typical price of the bar closing at that time), which is known before the open.

    python -m btc_updown.backtest.ladder --archive data/kalshi_ladder --klines data/klines --series spot_BTCUSDC \
        --hidden data/cloud/cubes/cube_e1b_hidden_kalshi/hidden --head data/cloud/runs/probe_small_1h/best/head.pt \
        --start 2026-08-20 --end 2026-10-25 --max-k 12 --out data/tables/ladder_e1b.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..data.nulls import ewma_sigma, null_prob
from ..train.dataset import BAR_SECONDS, SeriesStore, direct_head_features, typical_price
from ..train.loop import load_head

VENUE, FAMILY = "kalshi", "kxbtcd-1h"
MINUTES = np.arange(1, 60)
DEFAULT_BASIS = -14.0        # BRTI - Binance BTCUSDC, measured 2026-10-18..24 (SCOPE-KALSHI-LADDER.md)


def _read_days(d: str, start: str, end: str, columns=None) -> pd.DataFrame:
    files = [f for f in sorted(glob.glob(os.path.join(d, "*.parquet"))) if start <= os.path.basename(f)[:10] < end]
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_ladders(archive: str, start: str, end: str) -> pd.DataFrame:
    """Hourly ladders (one row per market) with the previous ladder's settlement attached."""
    m = _read_days(os.path.join(archive, "markets"), start, end)
    m = m[(m["hours"] <= 2.0) & m["expiration_value"].notna() & m["result"].isin(["yes", "no"])].copy()
    ev = m.groupby("event_ticker").agg(open_time=("open_time", "first"), close_time=("close_time", "first"),
                                       exp=("expiration_value", "first")).sort_values("close_time")
    ev["prev_close"] = ev["close_time"].shift(1)
    ev["prev_exp"] = ev["exp"].shift(1)
    consecutive = (ev["close_time"] - ev["prev_close"]) == 3600
    ev.loc[~consecutive, ["prev_close", "prev_exp"]] = np.nan
    m = m.merge(ev[["prev_close", "prev_exp"]], left_on="event_ticker", right_index=True, how="left")
    m["strike_eff"] = m["strike"] + 0.01           # "> 80299.99" == ">= 80300"
    return m


def book_rows(candles: pd.DataFrame, ladders: pd.DataFrame, max_k: Optional[int]) -> pd.DataFrame:
    """Forward-fill each (event, strike) book over minutes 1..59; keep strikes within max_k of the ATM strike."""
    out = []
    lad = ladders.set_index("ticker")
    for ev, g in candles.groupby("event_ticker", sort=True):
        if ev not in set(lad["event_ticker"]):
            continue
        row0 = lad[lad["event_ticker"] == ev].iloc[0]
        o = int(row0["open_time"]); ref = row0["prev_exp"]
        if not np.isfinite(ref):
            ref = lad.loc[g["ticker"].iloc[g["volume"].to_numpy().argmax()], "strike"] if "volume" in g else np.nan
        for tk, gg in g.groupby("ticker", sort=False):
            if tk not in lad.index:
                continue
            k = (lad.loc[tk, "strike"] - ref) / 100.0 if np.isfinite(ref) else np.nan
            if max_k is not None and (not np.isfinite(k) or abs(k) > max_k):
                continue
            gg = gg[gg["m"] <= 59].sort_values("m")
            if gg.empty:
                continue
            b = gg.set_index("m")[["bid", "ask", "px", "volume"]].reindex(MINUTES).ffill()
            b = b[b["bid"].notna()]
            b["ticker"] = tk; b["event_ticker"] = ev; b["k"] = k
            b["t"] = o + 60 * b.index.to_numpy()
            out.append(b.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def kline_features(store: SeriesStore, times: np.ndarray, lookback: int = 256, halflife_min: float = 60.0) -> pd.DataFrame:
    ts = np.asarray(store.ts); X = np.asarray(store.x, dtype=np.float64)
    close = X[:, 3]
    sig_ewma = ewma_sigma(close, halflife_min)
    lr = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.nan)
    rows = []
    for t in np.unique(times):
        i = store.index_of_bar_closing_at(int(t))
        if i < lookback:
            rows.append({"t": int(t), "close_t": np.nan, "sigma1": np.nan, "sigma_ewma": np.nan, "tp_t": np.nan}); continue
        win = lr[i - lookback + 2:i + 1]                          # the lookback window's 255 log returns
        rows.append({"t": int(t), "close_t": close[i], "sigma1": max(float(np.nanstd(win)), 1e-6), "sigma_ewma": sig_ewma[i],
                     "tp_t": float(typical_price(X[i]))})
    return pd.DataFrame(rows)


def tp_at(store: SeriesStore, times: np.ndarray) -> Dict[int, float]:
    X = np.asarray(store.x, dtype=np.float64); out = {}
    for t in np.unique(times[np.isfinite(times)]):
        i = store.index_of_bar_closing_at(int(t))
        out[int(t)] = float(typical_price(X[i])) if i >= 0 else np.nan
    return out


def apply_head(table: pd.DataFrame, hidden_dir: str, head_path: str) -> np.ndarray:
    from ..calib.market_head import load_hidden
    ts, H = load_hidden(hidden_dir)
    pos = pd.Series(np.arange(len(ts)), index=ts)
    idx = pos.reindex(table["t"].to_numpy()).to_numpy()
    head = load_head(head_path, torch.device("cpu")).eval()
    logits = np.full(len(table), np.nan, dtype=np.float32)
    ok = np.isfinite(idx) & table["close_t"].notna().to_numpy() & table["strike_series"].notna().to_numpy()
    for (j, h), g in table[ok].groupby(["j", "h"]):
        gi = g.index.to_numpy()
        ctx = torch.from_numpy(H[idx[gi].astype(int)])
        pairs = np.array([[int(j), int(h)]])
        feats = direct_head_features(torch.as_tensor(g["close_t"].to_numpy(), dtype=torch.float32),
                                     torch.as_tensor(g["sigma1"].to_numpy(), dtype=torch.float32),
                                     torch.as_tensor(g["strike_series"].to_numpy(), dtype=torch.float32)[:, None], pairs, head.max_h)
        with torch.no_grad():
            logits[gi] = head(ctx, torch.as_tensor(pairs[:, 1]), feats).numpy()[:, 0]
    return logits


def build(archive: str, klines: str, series: str, start: str, end: str, max_k: Optional[int], hidden: Optional[str],
          head: Optional[str], halflife_min: float = 60.0) -> pd.DataFrame:
    lad = load_ladders(archive, start, end)
    cand = _read_days(os.path.join(archive, "candles"), start, end, columns=["ticker", "event_ticker", "m", "bid", "ask", "px", "volume"])
    b = book_rows(cand, lad, max_k)
    if b.empty:
        return b
    store = SeriesStore.open(klines, series)
    tbl = b.merge(lad[["ticker", "event_ticker", "strike", "strike_eff", "open_time", "close_time", "expiration_value", "result",
                       "prev_close", "prev_exp", "label_day"]].drop(columns="event_ticker"), on="ticker", how="left")
    tbl["j"] = tbl["m"].astype(int); tbl["h"] = (60 - tbl["m"]).astype(int)
    tbl["outcome"] = (tbl["result"] == "yes").astype(int)
    tbl["two_sided"] = (tbl["bid"] > 0) & (tbl["ask"] < 1)
    tbl["mkt_prob_up"] = np.where(tbl["two_sided"], (tbl["bid"] + tbl["ask"]) / 2.0, np.nan)
    tbl["exec_price_up"] = np.where(tbl["ask"] < 1, tbl["ask"], np.nan)
    tbl["exec_price_down"] = np.where(tbl["bid"] > 0, 1.0 - tbl["bid"], np.nan)
    tbl["exec_size_up"] = 1e4; tbl["exec_size_down"] = 1e4
    tbl["exec_time_up"] = tbl["t"].astype(float); tbl["exec_time_down"] = tbl["t"].astype(float)
    kf = kline_features(store, tbl["t"].to_numpy(), halflife_min=halflife_min)
    tbl = tbl.merge(kf, on="t", how="left")
    tpp = tp_at(store, tbl["prev_close"].to_numpy(dtype=float))
    tbl["basis"] = tbl["prev_exp"] - tbl["prev_close"].map(lambda x: tpp.get(int(x), np.nan) if np.isfinite(x) else np.nan)
    tbl["basis_src"] = np.where(tbl["basis"].notna(), "prev_settle", "default")
    tbl["basis"] = tbl["basis"].fillna(DEFAULT_BASIS)
    tbl["strike_series"] = tbl["strike_eff"] - tbl["basis"]
    tbl["k_final"] = ((tbl["strike"] - tbl["expiration_value"]) / 100.0).round().astype(int)
    tbl["p_null"] = null_prob(tbl["close_t"].to_numpy(), tbl["strike_series"].to_numpy(), tbl["sigma_ewma"].to_numpy(),
                              tbl["h"].to_numpy().astype(float))
    tbl["logit_direct"] = apply_head(tbl, hidden, head) if hidden and head else np.nan
    tbl["venue"] = VENUE; tbl["family"] = FAMILY; tbl["market_id"] = tbl["ticker"]; tbl["minutes_left"] = tbl["h"]
    tbl["window_start"] = tbl["open_time"]; tbl["window_end"] = tbl["close_time"]
    tbl["label_day"] = pd.to_datetime(tbl["close_time"], unit="s").dt.date
    cols = ["venue", "family", "market_id", "event_ticker", "label_day", "window_start", "window_end", "t", "j", "h", "minutes_left",
            "strike", "strike_series", "basis", "basis_src", "k", "k_final", "outcome", "two_sided", "bid", "ask", "px", "mkt_prob_up",
            "exec_price_up", "exec_price_down", "exec_size_up", "exec_size_down", "exec_time_up", "exec_time_down",
            "close_t", "sigma1", "sigma_ewma", "p_null", "logit_direct", "expiration_value"]
    return tbl[cols].sort_values(["event_ticker", "t", "strike"]).reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="data/kalshi_ladder"); ap.add_argument("--klines", default="data/klines")
    ap.add_argument("--series", default="spot_BTCUSDC")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--max-k", type=int, default=12, help="keep strikes within this many $100 steps of the ATM (None = all)")
    ap.add_argument("--hidden", default=None); ap.add_argument("--head", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = build(a.archive, a.klines, a.series, a.start, a.end, a.max_k, a.hidden, a.head)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_parquet(a.out, index=False)
    print({"rows": int(len(t)), "ladders": int(t["event_ticker"].nunique()) if len(t) else 0, "days": int(t["label_day"].nunique()) if len(t) else 0,
           "two_sided": float(t["two_sided"].mean()) if len(t) else 0, "with_head": int(t["logit_direct"].notna().sum()) if len(t) else 0})


if __name__ == "__main__":
    main()
