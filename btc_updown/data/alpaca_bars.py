"""US equity 1-minute bars from Alpaca Market Data (free "Basic" plan, IEX feed) into the kline SeriesStore layout
(SCOPE-ROUND2 item 3: SPY and QQQ as the settlement proxies for the Kalshi S&P 500 / Nasdaq-100 hourly ladders).

    python -m btc_updown.data.alpaca_bars --symbols SPY QQQ --start 2016-01-01 --end 2026-10-28

Keys: ALPACA_KEY / ALPACA_SECRET in /Users/streetteam/Kronos/.env (never printed, never committed). Endpoint
GET https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Min&feed=iex&adjustment=raw&start=&end=&limit=10000
paged by page_token; 200 requests/minute on the free plan. Bars are UTC with `t` = bar open; written as
data/klines/eq_<SYMBOL>/{bars.parquet, x.f32.npy, stamp.i16.npy, ts.i64.npy, meta.json} with amount = volume x vwap.
Market hours only: sessions are separated by gaps, which the training dataset's contiguity mask rejects across.
Resumable: bars already written are kept and only the range after the last bar is fetched unless --force.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from .binance_klines import BAR_COLUMNS, compute_gaps, write_outputs

URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


def _env():
    p = Path(__file__).resolve().parents[2] / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    key, sec = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
    if not key or not sec:
        raise SystemExit("ALPACA_KEY / ALPACA_SECRET not set (put them in .env)")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def fetch(symbol: str, start: str, end: str, headers: dict, min_interval: float = 0.32) -> pd.DataFrame:
    rows: List[dict] = []; token: Optional[str] = None; n = 0
    while True:
        params = {"timeframe": "1Min", "feed": "iex", "adjustment": "raw", "start": f"{start}T00:00:00Z", "end": f"{end}T00:00:00Z", "limit": 10000}
        if token:
            params["page_token"] = token
        for attempt in range(6):
            r = requests.get(URL.format(symbol=symbol), params=params, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status(); break
        else:
            raise RuntimeError("rate limited six times")
        d = r.json(); n += 1
        rows.extend(d.get("bars") or [])
        token = d.get("next_page_token")
        if n % 20 == 0:
            print(f"{symbol}: {len(rows)} bars, {n} requests, last {rows[-1]['t'] if rows else None}", flush=True)
        if not token:
            break
        time.sleep(min_interval)
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)
    b = pd.DataFrame(rows)
    out = pd.DataFrame({
        "ts": (pd.to_datetime(b["t"], utc=True).astype("int64") // 10**9).astype(np.int64),
        "open": b["o"].astype(float), "high": b["h"].astype(float), "low": b["l"].astype(float), "close": b["c"].astype(float),
        "volume": b["v"].astype(float), "amount": (b["v"].astype(float) * b["vw"].astype(float)),
    })
    return out[BAR_COLUMNS].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def build(symbol: str, start: str, end: str, out: Path, force: bool, headers: dict) -> dict:
    sdir = out / f"eq_{symbol}"
    old = pd.DataFrame(columns=BAR_COLUMNS)
    if (sdir / "bars.parquet").exists() and not force:
        old = pd.read_parquet(sdir / "bars.parquet")
        if len(old):
            start = max(start, dt.datetime.fromtimestamp(int(old["ts"].max()), dt.timezone.utc).strftime("%Y-%m-%d"))
    new = fetch(symbol, start, end, headers)
    bars = pd.concat([old, new], ignore_index=True).drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1)]
    ts = bars["ts"].to_numpy(dtype=np.int64)
    meta = {"series": f"eq_{symbol}", "market": "us_equity", "symbol": symbol, "interval": "1m", "source": "alpaca iex v2 bars",
            "code_version": "btc_updown.data.alpaca_bars/1.0.0", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "first_ts": dt.datetime.fromtimestamp(int(ts[0]), dt.timezone.utc).isoformat() if len(ts) else None,
            "last_ts": dt.datetime.fromtimestamp(int(ts[-1]), dt.timezone.utc).isoformat() if len(ts) else None,
            "n": int(len(ts)), "gaps": compute_gaps(ts, top=5), "note": "market hours only; a session boundary is a gap"}
    write_outputs(sdir, bars, meta)
    return {k: meta[k] for k in ("series", "first_ts", "last_ts", "n")} | {"n_gaps": meta["gaps"]["n_gaps"], "new_bars": int(len(new))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--start", default="2016-01-01"); ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--out", type=Path, default=Path("data/klines")); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    headers = _env()
    for s in a.symbols:
        print(json.dumps(build(s, a.start, a.end, a.out, a.force, headers)), flush=True)


if __name__ == "__main__":
    main()
