"""Archive of Kalshi KXBTCD strike ladders from the public API (no key, Mac only, never committed).

Per UTC label day (the day the ladder closes) under --out:
    markets/<day>.parquet   every market of every ladder closing that day
    candles/<day>.parquet   1-minute bid/ask/price candles for ALL strikes of every ladder (batch endpoint, <= 100 tickers)
    fills/<day>.parquet     taker fills for the ATM +/- --fill-strikes strikes of every hourly ladder
Candles exist only for minutes with a change; the next candle after a gap opens at the previous close, so a forward-fill
of the book is exact.  Market-level data is retained by Kalshi for ~66 days only (the window slides daily): rerun the same
command to extend the archive; existing day files are skipped unless --force.

    python -m btc_updown.data.kalshi_ladder markets --start 2026-08-20 --end 2026-10-26
    python -m btc_updown.data.kalshi_ladder candles --start 2026-08-20 --end 2026-10-26 --workers 4
    python -m btc_updown.data.kalshi_ladder fills   --start 2026-08-20 --end 2026-10-26 --workers 4
    python -m btc_updown.data.kalshi_ladder report
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTCD"
BATCH = 100
MAX_CANDLES = 10_000     # per batch request: tickers x periods
HOURLY_MAX_H = 2.0
MAX_LADDER_HOURS = 30.0    # candles/fills only for ladders up to the daily one (25 h)          # open->close hours; the daily ladder is 25 h


def ts(x: str) -> int:
    return int(dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp())


def day_bounds(day: str):
    t0 = int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp())
    return t0, t0 + 86400


def days_between(start: str, end: str) -> List[str]:
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days)]


class Http:
    def __init__(self, min_interval: float = 0.2):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "btc-updown-research/0.1"
        self.lock = threading.Lock()
        self.min_interval = min_interval
        self.last = 0.0
        self.n = 0

    def get(self, path: str, **params) -> dict:
        for attempt in range(8):
            with self.lock:
                wait = self.min_interval - (time.time() - self.last)
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
                self.n += 1
            try:
                r = self.s.get(BASE + path, params=params, timeout=60)
            except requests.RequestException:
                time.sleep(2.0 * (attempt + 1)); continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2.0 * (attempt + 1)); continue
            if r.status_code != 200:
                raise RuntimeError(f"{path} {r.status_code}: {r.text[:200]}")
            return r.json()
        raise RuntimeError(f"{path}: retries exhausted")


def _f(x, default=np.nan) -> float:
    try:
        return float(x) if x not in (None, "") else default
    except (TypeError, ValueError):
        return default


def fetch_markets(http: Http, day: str) -> pd.DataFrame:
    t0, t1 = day_bounds(day)
    rows, cursor = [], None
    while True:
        r = http.get("/markets", series_ticker=SERIES, status="settled", min_close_ts=t0, max_close_ts=t1, limit=1000,
                     **({"cursor": cursor} if cursor else {}))
        for m in r.get("markets", []):
            o, c = ts(m["open_time"]), ts(m["close_time"])
            rows.append({"ticker": m["ticker"], "event_ticker": m["event_ticker"], "strike": _f(m.get("floor_strike")),
                         "strike_type": m.get("strike_type"), "open_time": o, "close_time": c, "hours": (c - o) / 3600.0,
                         "expiration_value": _f(m.get("expiration_value")), "result": m.get("result"), "status": m.get("status"),
                         "volume": _f(m.get("volume_fp"), 0.0), "open_interest": _f(m.get("open_interest_fp"), 0.0),
                         "settlement_ts": m.get("settlement_ts"), "label_day": day})
        cursor = r.get("cursor")
        if not cursor:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["close_time", "event_ticker", "strike"]).reset_index(drop=True)
    return df


def parse_candle(c: dict) -> dict:
    yb, ya, pr = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
    return {"t": int(c["end_period_ts"]),
            "bid_open": _f(yb.get("open_dollars")), "bid_high": _f(yb.get("high_dollars")), "bid_low": _f(yb.get("low_dollars")), "bid": _f(yb.get("close_dollars")),
            "ask_open": _f(ya.get("open_dollars")), "ask_high": _f(ya.get("high_dollars")), "ask_low": _f(ya.get("low_dollars")), "ask": _f(ya.get("close_dollars")),
            "px_open": _f(pr.get("open_dollars")), "px_high": _f(pr.get("high_dollars")), "px_low": _f(pr.get("low_dollars")), "px": _f(pr.get("close_dollars")),
            "px_mean": _f(pr.get("mean_dollars")), "volume": _f(c.get("volume_fp"), 0.0), "open_interest": _f(c.get("open_interest_fp"), 0.0)}


def _batches(http: Http, tickers: List[str], batch: int, start_ts: int, end_ts: int):
    """Batch candlestick requests; on the server's 'max candlesticks' refusal, split the batch and retry."""
    i = 0
    while i < len(tickers):
        chunk = tickers[i:i + batch]
        try:
            yield http.get("/markets/candlesticks", market_tickers=",".join(chunk), start_ts=start_ts, end_ts=end_ts, period_interval=1)
            i += len(chunk)
        except RuntimeError as e:
            if "max candlesticks" in str(e) and batch > 1:
                batch = max(1, batch // 2)
            else:
                raise


def fetch_candles(http: Http, markets: pd.DataFrame, day: str) -> pd.DataFrame:
    frames = []
    for ev, g in markets.groupby("event_ticker", sort=True):
        o, c = int(g["open_time"].iloc[0]), int(g["close_time"].iloc[0])
        if (c - o) / 3600.0 > MAX_LADDER_HOURS:                  # a few oddly long events (not the hourly/daily ladders)
            continue
        strike = dict(zip(g["ticker"], g["strike"]))
        tickers = list(g["ticker"])
        periods = (c + 120 - (o - 120)) // 60 + 20
        batch = max(1, min(BATCH, MAX_CANDLES // periods))      # the endpoint caps tickers x periods at 10,000
        for chunk in _batches(http, tickers, batch, o - 120, c + 120):
            for mk in chunk.get("markets", []):
                tk = mk.get("market_ticker") or mk.get("ticker")
                cs = mk.get("candlesticks") or []
                if not cs:
                    continue
                df = pd.DataFrame([parse_candle(x) for x in cs])
                df.insert(0, "ticker", tk); df.insert(1, "event_ticker", ev); df.insert(2, "strike", strike.get(tk, np.nan))
                df["m"] = ((df["t"] - o) // 60).astype(np.int32)
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["label_day"] = day
    return out.sort_values(["event_ticker", "strike", "t"]).reset_index(drop=True)


def atm_strikes(markets: pd.DataFrame, k: int) -> pd.DataFrame:
    """ATM +/- k strikes per hourly ladder; ATM = strike nearest the previous ladder's settlement (known at the open),
    else the ladder's own highest-volume strike."""
    hourly = markets[markets["hours"] <= HOURLY_MAX_H]
    events = hourly.groupby("event_ticker").agg(close_time=("close_time", "first"), exp=("expiration_value", "first")).sort_values("close_time")
    prev_exp = events["exp"].shift(1)
    out = []
    for ev, g in hourly.groupby("event_ticker"):
        ref = prev_exp.get(ev, np.nan)
        if not np.isfinite(ref):
            ref = g.loc[g["volume"].idxmax(), "strike"]
        g = g.assign(dist=(g["strike"] - ref).abs()).sort_values("dist")
        out.append(g.head(2 * k + 1))
    return pd.concat(out) if out else hourly.iloc[0:0]


def fetch_fills(http: Http, markets: pd.DataFrame, day: str, k: int) -> pd.DataFrame:
    sel = atm_strikes(markets, k)
    rows = []
    for m in sel.itertuples(index=False):
        cursor = None
        while True:
            r = http.get("/markets/trades", ticker=m.ticker, limit=1000, **({"cursor": cursor} if cursor else {}))
            for tr in r.get("trades", []):
                rows.append({"ticker": m.ticker, "event_ticker": m.event_ticker, "strike": m.strike,
                             "ts": dt.datetime.fromisoformat(tr["created_time"].replace("Z", "+00:00")).timestamp(),
                             "yes_price": _f(tr.get("yes_price_dollars")), "no_price": _f(tr.get("no_price_dollars")),
                             "count": _f(tr.get("count_fp"), 0.0), "taker_side": tr.get("taker_side"),
                             "taker_book_side": tr.get("taker_book_side"), "block": bool(tr.get("is_block_trade", False)),
                             "trade_id": tr.get("trade_id")})
            cursor = r.get("cursor")
            if not cursor:
                break
    df = pd.DataFrame(rows)
    if len(df):
        df["label_day"] = day
        df = df.sort_values(["event_ticker", "strike", "ts"]).reset_index(drop=True)
    return df


def _path(out: str, kind: str, day: str) -> str:
    os.makedirs(os.path.join(out, kind), exist_ok=True)
    return os.path.join(out, kind, f"{day}.parquet")


def run_day(http: Http, kind: str, day: str, out: str, force: bool, fill_strikes: int) -> str:
    p = _path(out, kind, day)
    if os.path.exists(p) and not force:
        return f"{day} {kind}: exists"
    if kind == "markets":
        df = fetch_markets(http, day)
    else:
        mp = _path(out, "markets", day)
        if not os.path.exists(mp):
            pd.DataFrame(fetch_markets(http, day)).to_parquet(mp, index=False)
        markets = pd.read_parquet(mp)
        if markets.empty:
            return f"{day} {kind}: no markets"
        df = fetch_candles(http, markets, day) if kind == "candles" else fetch_fills(http, markets, day, fill_strikes)
    tmp = p + ".tmp"
    df.to_parquet(tmp, index=False); os.replace(tmp, p)
    return f"{day} {kind}: {len(df)} rows"


def report(out: str) -> str:
    lines = []
    for kind in ("markets", "candles", "fills"):
        d = os.path.join(out, kind)
        files = sorted(f for f in os.listdir(d)) if os.path.isdir(d) else []
        n = 0
        for f in files:
            try:
                n += pd.read_parquet(os.path.join(d, f), columns=["ticker"]).shape[0]
            except Exception:
                pass
        lines.append(f"{kind}: {len(files)} days ({files[0][:10] if files else '-'} .. {files[-1][:10] if files else '-'}), {n:,} rows")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["markets", "candles", "fills", "report"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--out", default="data/kalshi_ladder")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-interval", type=float, default=0.2)
    ap.add_argument("--fill-strikes", type=int, default=5, help="ATM +/- this many strikes get their fills")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        print(report(a.out)); return
    if not (a.start and a.end):
        ap.error("--start and --end are required")
    http = Http(a.min_interval)
    days = days_between(a.start, a.end)
    t0 = time.time()
    if a.cmd == "markets":
        for d in days:
            print(run_day(http, "markets", d, a.out, a.force, a.fill_strikes), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for msg in ex.map(lambda d: run_day(http, a.cmd, d, a.out, a.force, a.fill_strikes), days):
                print(msg, flush=True)
    print(f"done: {len(days)} days, {http.n} requests, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
