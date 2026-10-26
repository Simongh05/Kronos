"""Polymarket long-window BTC up/down families from PUBLIC endpoints (no recorder, no research repo, no key).

    python -m btc_updown.data.pm_long windows --family btc-1h --start 2025-05-28 --end 2026-10-27
    python -m btc_updown.data.pm_long prices  --family btc-1h --start 2026-08-01 --end 2026-10-27 [--workers 4]
    python -m btc_updown.data.pm_long report  --family btc-1h

Families (gamma series ids; windows from the slug):
    btc-1h   series 10114, slug bitcoin-up-or-down-<month>-<d>[-<yyyy>]-<h><am|pm>-et[-candle]; window = the Binance 1 h candle
             starting at that New York hour; settles Up iff candle close >= candle open (Binance BTCUSDT) -> strike/final from
             the local spot_BTCUSDT klines, outcome from gamma's outcomePrices, cross-checked.
    btc-4h   series 10331, slug btc-updown-4h-<unix start>; Chainlink TWAP-60 at end vs start (strike/final left NaN here).
    btc-1d   series 41, slug bitcoin-up-or-down-on-<month>-<d>-<yyyy>; noon ET -> noon ET; Binance 1-min close at both ends.

Outputs under --out (default data/pm_long):
    windows_<family>.parquet        one row per market, same columns as data/labels/windows.parquet (+ volume_usd, outcome_klines)
    trades/<family>/<day>.parquet   every TAKER order of the markets whose window starts on that UTC day (data-api /trades:
                                    one row per taker order, side = taker side, timestamp = block time, price = order avg price)
    mid/<family>/<day>.parquet      CLOB prices-history of the Up token at fidelity 1 (t, p)
    prices_<family>.parquet         decision-minute rows in the data/labels/prices_polymarket.parquet layout:
                                    executable price of each side = first taker BUY of that side with timestamp in
                                    [t+3 s, t+30 s] (price = that order's price, size = its shares); mkt_prob_up = mid of
                                    the first Up-buy and 1 - first Down-buy; n_prints_30s = taker BUY orders of either side.
Resumable: a day whose trades file exists is skipped. Only public data; nothing here reads the research repo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
NY = ZoneInfo("America/New_York")
EXEC_LO, EXEC_HI = 3, 30
MAX_OFFSET = 10_000  # data-api: "max historical trades offset of 10000 exceeded"

FAMILIES = {
    "btc-1h": {"series_id": 10114, "seconds": 3600, "regime": "binance-1h-candle"},
    "btc-4h": {"series_id": 10331, "seconds": 14400, "regime": "btc-4h-twap-60"},
    "btc-1d": {"series_id": 41, "seconds": 86400, "regime": "binance-1m-noon-et"},
}
MONTHS = {m.lower(): i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

_HOURLY = re.compile(r"-([a-z]+)-(\d{1,2})(?:-(\d{4}))?-(\d{1,2})-?(am|pm)-et")
_DAILY = re.compile(r"-on-([a-z]+)-(\d{1,2})-(\d{4})")


def _month(name: str) -> Optional[int]:
    return MONTHS.get(name[:3].lower())


def window_from_slug(family: str, slug: str, year_hint: Optional[int] = None, end_hint: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """(window_start, window_end) epoch seconds from the slug; None if it does not parse. `year_hint` / `end_hint`
    (gamma endDate year and epoch) disambiguate hourly slugs that carry no year."""
    secs = FAMILIES[family]["seconds"]
    if family == "btc-4h":
        try:
            ws = int(slug.rsplit("-", 1)[1])
        except (ValueError, IndexError):
            return None
        return ws, ws + secs
    if family == "btc-1h":
        m = _HOURLY.search(slug)
        if not m:
            return None
        mon, day, yr, hr, ap = m.groups()
        mo = _month(mon)
        if mo is None:
            return None
        h = int(hr) % 12 + (12 if ap == "pm" else 0)
        if yr:
            years = [int(yr)]
        elif year_hint is not None:
            years = [year_hint, year_hint - 1]   # slugs without a year: gamma's endDate year, or the year before around New Year
        else:
            return None
        cands = []
        for year in years:
            try:
                cands.append(int(dt.datetime(year, mo, int(day), h, tzinfo=NY).timestamp()))
            except ValueError:
                pass
        if not cands:
            return None
        if end_hint is not None and len(cands) > 1:
            ws = min(cands, key=lambda c: abs(c - end_hint))
        else:
            ws = cands[0]
        return ws, ws + secs
    if family == "btc-1d":
        m = _DAILY.search(slug)
        if not m:
            return None
        mon, day, yr = m.groups()
        mo = _month(mon)
        if mo is None:
            return None
        we = int(dt.datetime(int(yr), mo, int(day), 12, tzinfo=NY).timestamp())
        return we - secs, we
    return None


# ----------------------------------------------------------------------------------------------- HTTP
class Http:
    def __init__(self, min_interval: float = 0.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "btc-updown-research/0.1"
        self.lock = threading.Lock()
        self.min_interval = min_interval
        self.last = 0.0
        self.calls = 0

    def get(self, url: str, params: dict, tries: int = 6):
        for k in range(tries):
            with self.lock:
                wait = self.min_interval - (time.time() - self.last)
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
                self.calls += 1
            try:
                r = self.s.get(url, params=params, timeout=60)
            except requests.RequestException as e:
                if k == tries - 1:
                    raise
                time.sleep(2.0 * (k + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60.0, 2.0 ** k))
                continue
            raise RuntimeError(f"HTTP {r.status_code} {url} {params}: {r.text[:120]}")
        raise RuntimeError(f"gave up on {url} {params}")


# ------------------------------------------------------------------------------------------ windows
def _month_edges(start: str, end: str):
    d0 = dt.date.fromisoformat(start).replace(day=1)
    d1 = dt.date.fromisoformat(end)
    while d0 <= d1:
        nxt = (d0.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield d0, nxt
        d0 = nxt


def fetch_markets(http: Http, family: str, start: str, end: str) -> List[dict]:
    """All closed gamma markets of the family's series whose (gamma) endDate falls in [start, end]; paginated by month
    (gamma refuses offsets beyond ~2000)."""
    sid = FAMILIES[family]["series_id"]
    out, seen = [], set()
    for d0, d1 in _month_edges(start, end):
        off = 0
        while True:
            ev = http.get(f"{GAMMA}/events", {"series_id": sid, "closed": "true", "limit": 100, "offset": off,
                                                "end_date_min": f"{d0}T00:00:00Z", "end_date_max": f"{d1}T00:00:00Z"})
            if not ev:
                break
            for e in ev:
                for m in e.get("markets", []):
                    if m.get("conditionId") and m["conditionId"] not in seen:
                        seen.add(m["conditionId"])
                        out.append(m)
            if len(ev) < 100:
                break
            off += 100
    return out


def _outcome_from_prices(m: dict) -> int:
    try:
        op = json.loads(m.get("outcomePrices") or "[]")
        if len(op) != 2:
            return -1
        a, b = float(op[0]), float(op[1])
    except (ValueError, TypeError):
        return -1
    if a > 0.9 and b < 0.1:
        return 1
    if b > 0.9 and a < 0.1:
        return 0
    return -1


def klines_strike_final(family: str, ws: int, we: int, ts: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Settlement values in the settlement series' own terms from local 1-min klines (NaN when a bar is missing)."""
    def bar(t: int) -> int:
        i = int(np.searchsorted(ts, t))
        return i if i < len(ts) and int(ts[i]) == t else -1
    if family == "btc-1h":
        i0, i1 = bar(ws), bar(we - 60)
        if i0 < 0 or i1 < 0:
            return np.nan, np.nan
        return float(x[i0, 0]), float(x[i1, 3])          # candle open, candle close
    if family == "btc-1d":
        i0, i1 = bar(ws), bar(we)
        if i0 < 0 or i1 < 0:
            return np.nan, np.nan
        return float(x[i0, 3]), float(x[i1, 3])          # close of the noon 1-min candle, both ends
    return np.nan, np.nan


def build_windows(http: Http, family: str, start: str, end: str, klines_root: str, settle_series: str = "spot_BTCUSDT") -> pd.DataFrame:
    ms = fetch_markets(http, family, start, end)
    ts = np.load(os.path.join(klines_root, settle_series, "ts.i64.npy"), mmap_mode="r")
    x = np.load(os.path.join(klines_root, settle_series, "x.f32.npy"), mmap_mode="r")
    rows = []
    for m in ms:
        slug = m.get("slug", "")
        year_hint = int((m.get("endDate") or "0000")[:4]) or None
        try:
            end_hint = int(dt.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp())
        except (KeyError, ValueError, AttributeError):
            end_hint = None
        win = window_from_slug(family, slug, year_hint, end_hint)
        if win is None:
            continue
        ws, we = win
        toks = json.loads(m.get("clobTokenIds") or "[]")
        outs = json.loads(m.get("outcomes") or '["Up","Down"]')
        if len(toks) != 2 or outs[0] != "Up":
            continue
        strike, final = klines_strike_final(family, ws, we, ts, x)
        outcome = _outcome_from_prices(m)
        rows.append({
            "venue": "polymarket", "family": family, "market_id": slug, "condition_id": m["conditionId"],
            "token_up": toks[0], "token_down": toks[1], "window_start": ws, "window_end": we,
            "strike": strike, "final": final, "outcome": outcome,
            "outcome_klines": (-1 if np.isnan(strike) or np.isnan(final) else int(final >= strike)),
            "regime": FAMILIES[family]["regime"], "tape_regime": m.get("umaResolutionStatus") or "",
            "source": "gamma", "label_day": dt.datetime.fromtimestamp(ws, dt.timezone.utc).date(),
            "volume_usd": float(m.get("volumeNum") or m.get("volume") or 0.0),
        })
    w = pd.DataFrame(rows).drop_duplicates("market_id")
    # relisted hours (a slug with and without the year, a "-666" twin, the DST hour that does not exist): keep the
    # market with the most volume per window
    w = w.sort_values(["window_start", "volume_usd"], ascending=[True, False]).drop_duplicates("window_start").reset_index(drop=True)
    w["outcome"] = w["outcome"].astype("int8")
    return w


# ------------------------------------------------------------------------------------------- prices
def fetch_trades(http: Http, condition_id: str) -> Tuple[List[dict], bool]:
    """All taker orders of a market (data-api /trades, taker rows only). Beyond the 10k offset cap the fetch is
    repeated per taker side (BUY / SELL) which doubles the capacity; `truncated` says a partition still hit the cap."""
    def page(params) -> Tuple[List[dict], bool]:
        rows, off = [], 0
        while True:
            b = http.get(f"{DATA_API}/trades", {**params, "limit": 1000, "offset": off})
            rows += b
            if len(b) < 1000:
                return rows, False
            off += 1000
            if off >= MAX_OFFSET:
                return rows, True
    rows, trunc = page({"market": condition_id})
    if not trunc:
        return rows, False
    rows, trunc = [], False
    for side in ("BUY", "SELL"):
        r, t = page({"market": condition_id, "side": side})
        rows += r
        trunc = trunc or t
    return rows, trunc


def fetch_mid(http: Http, token: str, ws: int, we: int) -> List[dict]:
    h = http.get(f"{CLOB}/prices-history", {"market": token, "startTs": ws - 600, "endTs": we + 120, "fidelity": 1})
    return h.get("history", []) if isinstance(h, dict) else []


TRADE_COLS = ["market_id", "condition_id", "timestamp", "side", "outcome", "price", "size", "tx", "wallet"]


def trades_frame(market_id: str, condition_id: str, rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TRADE_COLS)
    df = pd.DataFrame({
        "market_id": market_id, "condition_id": condition_id,
        "timestamp": [int(r["timestamp"]) for r in rows], "side": [r.get("side") for r in rows],
        "outcome": [r.get("outcome") for r in rows], "price": [float(r["price"]) for r in rows],
        "size": [float(r["size"]) for r in rows], "tx": [r.get("transactionHash") for r in rows],
        "wallet": [r.get("proxyWallet") for r in rows],
    })
    return df.sort_values(["timestamp", "tx"]).reset_index(drop=True)


def decision_prices(window: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    """The executable-price rule on one market's taker orders: rows for t = ws + 60k, k = 1 .. n-1."""
    ws, we = int(window.window_start), int(window.window_end)
    n = (we - ws) // 60
    ts_dec = ws + 60 * np.arange(1, n)
    buys = trades[trades["side"] == "BUY"]
    out = {"market_id": window.market_id, "family": window.family, "window_start": ws, "window_end": we, "t": ts_dec,
           "minutes_left": (we - ts_dec) / 60.0}
    first = {}
    for side in ("Up", "Down"):
        b = buys[buys["outcome"] == side].sort_values("timestamp")
        bt = b["timestamp"].to_numpy()
        lo = np.searchsorted(bt, ts_dec + EXEC_LO, side="left")
        ok = (lo < len(bt)) & (bt[np.clip(lo, 0, max(0, len(bt) - 1))] <= ts_dec + EXEC_HI) if len(bt) else np.zeros(len(ts_dec), bool)
        px = np.full(len(ts_dec), np.nan); sz = np.full(len(ts_dec), np.nan); et = np.full(len(ts_dec), np.nan)
        if ok.any():
            idx = lo[ok]
            px[ok] = b["price"].to_numpy()[idx]; sz[ok] = b["size"].to_numpy()[idx]; et[ok] = bt[idx]
        key = "up" if side == "Up" else "down"
        out[f"exec_price_{key}"], out[f"exec_size_{key}"], out[f"exec_time_{key}"] = px, sz, et
        first[side] = px
    pu, pdn = first["Up"], first["Down"]
    both = ~np.isnan(pu) & ~np.isnan(pdn)
    mkt = np.where(both, (pu + (1 - pdn)) / 2, np.where(~np.isnan(pu), pu, 1 - pdn))
    src = ["both" if b else ("up_only" if not np.isnan(u) else ("down_only" if not np.isnan(d) else None))
           for b, u, d in zip(both, pu, pdn)]
    out["mkt_prob_up"], out["mkt_prob_src"] = mkt, pd.array(src, dtype="string")
    bt_all = np.sort(buys["timestamp"].to_numpy())
    out["n_prints_30s"] = (np.searchsorted(bt_all, ts_dec + EXEC_HI, side="right") - np.searchsorted(bt_all, ts_dec + EXEC_LO, side="left")).astype(np.int64)
    return pd.DataFrame(out)


def build_prices(http: Http, family: str, windows: pd.DataFrame, start: str, end: str, out: str, workers: int, with_mid: bool) -> dict:
    tdir, mdir = os.path.join(out, "trades", family), os.path.join(out, "mid", family)
    os.makedirs(tdir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    w = windows[(windows["label_day"].astype(str) >= start) & (windows["label_day"].astype(str) < end)]
    days = sorted(w["label_day"].astype(str).unique())
    rep = {"days": len(days), "days_done_before": 0, "days_built": 0, "markets": 0, "orders": 0, "truncated_markets": [], "calls": 0}
    t_start = time.time()
    for di, day in enumerate(days):
        tf = os.path.join(tdir, f"{day}.parquet")
        if os.path.exists(tf):
            rep["days_done_before"] += 1
            continue
        wd = w[w["label_day"].astype(str) == day]
        frames, mids = [], []

        def one(row):
            rows, trunc = fetch_trades(http, row.condition_id)
            mid = fetch_mid(http, row.token_up, int(row.window_start), int(row.window_end)) if with_mid else []
            return row, trades_frame(row.market_id, row.condition_id, rows), trunc, mid

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(one, r) for r in wd.itertuples(index=False)]):
                row, df, trunc, mid = fut.result()
                frames.append(df)
                if trunc:
                    rep["truncated_markets"].append(row.market_id)
                if mid:
                    mids.append(pd.DataFrame({"market_id": row.market_id, "t": [int(h["t"]) for h in mid], "p": [float(h["p"]) for h in mid]}))
        tr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRADE_COLS)
        tr.to_parquet(tf + ".tmp", index=False); os.replace(tf + ".tmp", tf)
        if mids:
            pd.concat(mids, ignore_index=True).to_parquet(os.path.join(mdir, f"{day}.parquet"), index=False)
        rep["days_built"] += 1; rep["markets"] += len(wd); rep["orders"] += len(tr)
        el = time.time() - t_start
        print(f"[pm_long] {family} {day}: markets={len(wd)} orders={len(tr)} calls={http.calls} elapsed={el/60:.1f} min "
              f"({(di + 1)}/{len(days)} days)", flush=True)
    rep["calls"] = http.calls
    return rep


def assemble_prices(family: str, windows: pd.DataFrame, out: str) -> pd.DataFrame:
    tdir = os.path.join(out, "trades", family)
    frames = []
    wmap = windows.set_index("market_id")
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".parquet"):
            continue
        tr = pd.read_parquet(os.path.join(tdir, f))
        day = f[:10]
        for mid in windows[windows["label_day"].astype(str) == day]["market_id"]:
            frames.append(decision_prices(_row(wmap, mid), tr[tr["market_id"] == mid]))
    p = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return p


def _row(wmap: pd.DataFrame, market_id: str) -> pd.Series:
    s = wmap.loc[market_id].copy()
    s["market_id"] = market_id
    return s


# ------------------------------------------------------------------------------------------- report
def report(family: str, out: str) -> str:
    wp = os.path.join(out, f"windows_{family}.parquet"); pp = os.path.join(out, f"prices_{family}.parquet")
    lines = [f"# pm_long coverage: {family} ({dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')})", ""]
    if os.path.exists(wp):
        w = pd.read_parquet(wp)
        ok = w[(w["outcome"] >= 0) & (w["outcome_klines"] >= 0)]
        lines += [f"- windows: {len(w):,} markets, {w['label_day'].astype(str).min()} .. {w['label_day'].astype(str).max()}, "
                  f"outcome counts {w['outcome'].value_counts().to_dict()}, volume ${w['volume_usd'].sum():,.0f}",
                  f"- label check (gamma outcome == klines close >= open): {int((ok['outcome'] == ok['outcome_klines']).sum()):,} / {len(ok):,} agree; "
                  f"{int((w['outcome_klines'] < 0).sum())} without klines"]
    if os.path.exists(pp):
        p = pd.read_parquet(pp)
        lines += [f"- prices: {len(p):,} decision rows, {p['market_id'].nunique():,} markets, days {pd.to_datetime(p['t'], unit='s').dt.date.min()} .. {pd.to_datetime(p['t'], unit='s').dt.date.max()}",
                  f"- exec coverage up {p['exec_price_up'].notna().mean():.3f} / down {p['exec_price_down'].notna().mean():.3f}; mkt_prob_src {p['mkt_prob_src'].value_counts(dropna=False).to_dict()}",
                  f"- median n_prints_30s {p['n_prints_30s'].median():.0f}; rows with an exec time before t+3 s: {int(((p['exec_time_up'] < p['t'] + 3) | (p['exec_time_down'] < p['t'] + 3)).sum())}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["windows", "prices", "report"])
    ap.add_argument("--family", default="btc-1h", choices=list(FAMILIES))
    ap.add_argument("--start", default="2025-05-28"); ap.add_argument("--end", default="2026-10-27")
    ap.add_argument("--out", default="data/pm_long"); ap.add_argument("--klines", default="data/klines")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--no-mid", action="store_true")
    ap.add_argument("--min-interval", type=float, default=0.05, help="seconds between HTTP calls (global)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    http = Http(a.min_interval)
    wp = os.path.join(a.out, f"windows_{a.family}.parquet")
    if a.cmd == "windows":
        w = build_windows(http, a.family, a.start, a.end, a.klines)
        if os.path.exists(wp):  # merge with what exists (other date ranges)
            old = pd.read_parquet(wp)
            w = pd.concat([old[~old["market_id"].isin(w["market_id"])], w], ignore_index=True)
            w = w.sort_values(["window_start", "volume_usd"], ascending=[True, False]).drop_duplicates("window_start").reset_index(drop=True)
        w.to_parquet(wp, index=False)
        print(json.dumps({"markets": len(w), "first": str(w["label_day"].min()), "last": str(w["label_day"].max()),
                          "calls": http.calls, "label_agree": int(((w["outcome"] == w["outcome_klines"]) & (w["outcome"] >= 0)).sum()),
                          "labelled": int((w["outcome"] >= 0).sum())}))
    elif a.cmd == "prices":
        w = pd.read_parquet(wp)
        rep = build_prices(http, a.family, w, a.start, a.end, a.out, a.workers, not a.no_mid)
        p = assemble_prices(a.family, w, a.out)
        p.to_parquet(os.path.join(a.out, f"prices_{a.family}.parquet"), index=False)
        rep["decision_rows"] = int(len(p))
        print(json.dumps(rep))
    open(os.path.join(a.out, f"coverage_{a.family}.md"), "w").write(report(a.family, a.out))
    if a.cmd == "report":
        print(report(a.family, a.out))


if __name__ == "__main__":
    main()
