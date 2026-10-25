#!/usr/bin/env python3
"""Labels and executable market prices for the BTC 5m/15m up-down markets (Mac only, research venv).

Run from the Kronos root with the research repo's interpreter:

    /Users/streetteam/Documents/perpcity-arb-bot/polymarket-5m/.venv/bin/python -m btc_updown.data.labels \
        --out data/labels [--venues polymarket kalshi] [--start YYYY-MM-DD --end YYYY-MM-DD] \
        [--only windows prices bbo kalshi] [--resume]

Outputs under --out:
  windows.parquet           one row per market window (both venues, every label source)
  prices_polymarket.parquet executable print prices per (window, decision minute) from the on-chain fill tape
  bbo_polymarket.parquet    recorder best bid/ask at t+3 s and first print after t+3 s (recorder days 10-10, 10-11)
  haircut.json              first_print - best_ask distribution per family and side (the execution haircut)
  prices_kalshi.parquet     Kalshi ticker (1 Hz) yes ask / 1 - yes bid at t+3 s (recorder days 10-11 .. 10-17)
  coverage.md / coverage.json   days covered, row counts, gaps, guard refusals, mismatch counts, runtimes
  guard_log.jsonl           every scripts.common.guard.check_data call (ours and the readers') and every GuardError

Every read of research-repo data goes through `scripts.common.guard.check_data(None, dataset, start, end)`; a
GuardError means the range is skipped and logged, never bypassed. Nothing is ever written inside the research repo.

Anti-look-ahead rule (do not relax): the executable price for buying side S at decision time t is the FIRST
taker-BUY taker order of outcome S with block_time in [t + 3 s, t + 30 s], priced at the lowest fill price inside
that taker order and sized at the order's total shares; null when there is none.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import glob
import json
import math
import os
import re
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_ROOT = Path(os.environ.get("PM5_ROOT", "/Users/streetteam/Documents/perpcity-arb-bot/polymarket-5m"))
FAMILIES = {"btc-5m": 300, "btc-15m": 900}          # family -> window length in seconds
KALSHI_SERIES = "KXBTC15M"
KALSHI_FAMILY = "kxbtc15m"
DAY = 86400
EXEC_LO, EXEC_HI = 3, 30                            # [t + 3 s, t + 30 s]
BBO_DAYS = ["2026-10-10", "2026-10-11"]             # pm_recorder days with labels
KALSHI_REC_DAYS = [f"2026-10-{d:02d}" for d in range(11, 18)]
PM_REC_QUARANTINE = "2026-10-17"
KALSHI_REC_QUARANTINE = "2026-10-18"

WINDOW_COLS = ["venue", "family", "market_id", "condition_id", "token_up", "token_down", "window_start", "window_end",
               "strike", "final", "outcome", "regime", "tape_regime", "source", "label_day"]
PRICE_COLS = ["market_id", "family", "window_start", "window_end", "t", "minutes_left",
              "exec_price_up", "exec_size_up", "exec_time_up", "exec_price_down", "exec_size_down", "exec_time_down",
              "mkt_prob_up", "mkt_prob_src", "n_prints_30s"]
BBO_COLS = ["market_id", "family", "window_start", "window_end", "t", "minutes_left",
            "ask_up", "bid_up", "bbo_time_up", "ask_down", "bid_down", "bbo_time_down",
            "print_price_up", "print_size_up", "print_time_up", "print_price_down", "print_size_down", "print_time_down"]


# ------------------------------------------------------------------------------------------------ small helpers
def day_s(d: str) -> int:
    """'YYYY-MM-DD' -> unix seconds at 00:00 UTC."""
    return int(dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def s_day(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def date_range(d0: str, d1: str) -> list[str]:
    a, b = dt.date.fromisoformat(d0), dt.date.fromisoformat(d1)
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def compress_days(days) -> str:
    """['2026-08-14', '2026-08-15', '2026-08-17'] -> '2026-08-14..2026-08-15, 2026-08-17'."""
    days = sorted(set(days))
    if not days:
        return "(none)"
    out, run0, prev = [], days[0], days[0]
    for d in days[1:]:
        if dt.date.fromisoformat(d) - dt.date.fromisoformat(prev) == dt.timedelta(days=1):
            prev = d
            continue
        out.append(run0 if run0 == prev else f"{run0}..{prev}")
        run0 = prev = d
    out.append(run0 if run0 == prev else f"{run0}..{prev}")
    return ", ".join(out)


def to_float(x):
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def iso_to_s(s: str) -> int | None:
    if not s:
        return None
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


# ------------------------------------------------------------------------------------------------ pure logic
def decision_times(window_start: int, window_end: int, step: int = 60) -> list[int]:
    """1-minute bar closes inside the window with at least `step` seconds to expiry: k = 1 .. duration/step - 1."""
    n = (int(window_end) - int(window_start)) // step
    return [int(window_start) + step * k for k in range(1, n)]


def build_decisions(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in windows.itertuples(index=False):
        for t in decision_times(w.window_start, w.window_end):
            rows.append((w.market_id, w.family, int(w.window_start), int(w.window_end), t, (int(w.window_end) - t) / 60.0))
    return pd.DataFrame(rows, columns=["market_id", "family", "window_start", "window_end", "t", "minutes_left"])


def compose_mkt_prob(up_price, down_price):
    """Market P(Up) from the first Up-buy print and the first Down-buy print after t+3 s: the mid of p_up and
    1 - p_down when both exist, else the one that exists."""
    if up_price is not None and down_price is not None:
        return (float(up_price) + (1.0 - float(down_price))) / 2.0, "both"
    if up_price is not None:
        return float(up_price), "up_only"
    if down_price is not None:
        return 1.0 - float(down_price), "down_only"
    return None, None


def outcome_from(final, strike) -> int:
    """1 = Up/yes (final >= strike), 0 = Down/no, -1 unknown."""
    if final is None or strike is None:
        return -1
    return 1 if float(final) >= float(strike) else 0


def outcome_mismatches(windows: pd.DataFrame) -> pd.DataFrame:
    """Per source: rows where outcome, final and strike all exist, and how many violate outcome == (final >= strike)."""
    w = windows[(windows["outcome"] >= 0) & windows["final"].notna() & windows["strike"].notna()]
    if w.empty:
        return pd.DataFrame(columns=["source", "checked", "mismatch"])
    bad = (w["final"] >= w["strike"]).astype(int) != w["outcome"].astype(int)
    out = w.assign(bad=bad.astype(int)).groupby("source").agg(checked=("bad", "size"), mismatch=("bad", "sum")).reset_index()
    return out


def exec_price_sql(dec_src: str, orders_src: str, fills_src: str, lo: int = EXEC_LO, hi: int = EXEC_HI) -> str:
    """DuckDB SQL implementing the executable-price rule. `dec_src`: decisions (market_id, family, window_start,
    window_end, t, minutes_left); `orders_src`: taker orders (slug, tx_hash, log_index, tx_index, block_time, side,
    outcome, shares, avg_price); `fills_src`: fills (tx_hash, taker_log_index, log_index, price, shares)."""
    return f"""
WITH d AS (SELECT * FROM {dec_src}),
o AS (SELECT slug, tx_hash, log_index, tx_index, block_time, outcome, shares, avg_price
      FROM {orders_src} WHERE side = 'BUY' AND outcome IN ('Up', 'Down')),
cand AS (
  SELECT d.market_id, d.t, o.outcome, o.tx_hash, o.log_index, o.block_time, o.shares, o.avg_price,
         row_number() OVER (PARTITION BY d.market_id, d.t, o.outcome
                            ORDER BY o.block_time, o.tx_index, o.log_index, o.tx_hash) AS rn
  FROM d JOIN o ON o.slug = d.market_id AND o.block_time >= d.t + {int(lo)} AND o.block_time <= d.t + {int(hi)}
),
np AS (SELECT market_id, t, count(*) AS n_prints_30s FROM cand GROUP BY 1, 2),
first AS (SELECT * FROM cand WHERE rn = 1),
fp AS (
  SELECT f.tx_hash, f.taker_log_index, min(f.price) AS min_price, arg_min(f.price, f.log_index) AS first_price
  FROM {fills_src} f
  JOIN (SELECT DISTINCT tx_hash, log_index FROM first) k ON k.tx_hash = f.tx_hash AND k.log_index = f.taker_log_index
  GROUP BY 1, 2
),
side AS (
  SELECT first.market_id, first.t, first.outcome,
         coalesce(fp.min_price, first.avg_price) AS exec_price, first.shares AS exec_size,
         first.block_time AS exec_time, coalesce(fp.first_price, first.avg_price) AS first_price,
         CASE WHEN fp.tx_hash IS NULL THEN 1 ELSE 0 END AS fill_missing
  FROM first LEFT JOIN fp ON fp.tx_hash = first.tx_hash AND fp.taker_log_index = first.log_index
)
SELECT d.market_id, d.family, d.window_start, d.window_end, d.t, d.minutes_left,
       u.exec_price AS exec_price_up, u.exec_size AS exec_size_up, u.exec_time AS exec_time_up,
       w.exec_price AS exec_price_down, w.exec_size AS exec_size_down, w.exec_time AS exec_time_down,
       CASE WHEN u.first_price IS NOT NULL AND w.first_price IS NOT NULL THEN (u.first_price + (1 - w.first_price)) / 2
            WHEN u.first_price IS NOT NULL THEN u.first_price
            WHEN w.first_price IS NOT NULL THEN 1 - w.first_price END AS mkt_prob_up,
       CASE WHEN u.first_price IS NOT NULL AND w.first_price IS NOT NULL THEN 'both'
            WHEN u.first_price IS NOT NULL THEN 'up_only'
            WHEN w.first_price IS NOT NULL THEN 'down_only' END AS mkt_prob_src,
       coalesce(np.n_prints_30s, 0)::BIGINT AS n_prints_30s,
       (coalesce(u.fill_missing, 0) + coalesce(w.fill_missing, 0))::BIGINT AS fill_missing
FROM d
LEFT JOIN side u ON u.market_id = d.market_id AND u.t = d.t AND u.outcome = 'Up'
LEFT JOIN side w ON w.market_id = d.market_id AND w.t = d.t AND w.outcome = 'Down'
LEFT JOIN np ON np.market_id = d.market_id AND np.t = d.t
ORDER BY d.market_id, d.t
"""


def pick_exec_prices(con, decisions: pd.DataFrame, orders: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """The executable-price rule on in-memory frames (used by the tests; production feeds the tape views)."""
    con.register("_dec", decisions)
    con.register("_ord", orders)
    con.register("_fill", fills)
    try:
        return con.execute(exec_price_sql("_dec", "_ord", "_fill")).df()
    finally:
        for n in ("_dec", "_ord", "_fill"):
            con.unregister(n)


def state_at(ts: np.ndarray, t: float) -> int:
    """Index of the last record with timestamp <= t, or -1 (carried state as of t)."""
    return int(np.searchsorted(ts, t, side="right")) - 1


def first_in(ts: np.ndarray, lo: float, hi: float) -> int:
    """Index of the first record with lo <= timestamp <= hi, or -1."""
    i = int(np.searchsorted(ts, lo, side="left"))
    return i if i < len(ts) and ts[i] <= hi else -1


def parse_kalshi_meta_lines(raw_lines) -> tuple[dict, list[dict]]:
    """Kalshi market metadata day file: first line is '#' + JSON header, then one JSON market per line."""
    it = iter(raw_lines)
    first = next(it, None)
    if first is None:
        raise ValueError("empty kalshi meta file")
    if isinstance(first, bytes):
        first = first.decode("utf-8")
    first = first.strip()
    if not first.startswith("#"):
        raise ValueError("kalshi meta file does not start with a '#' JSON header")
    head = json.loads(first[1:])
    markets = []
    for ln in it:
        if isinstance(ln, bytes):
            ln = ln.decode("utf-8")
        ln = ln.strip()
        if ln:
            markets.append(json.loads(ln))
    n = head.get("n")
    if n is not None and n != len(markets):
        raise ValueError(f"kalshi meta file: {len(markets)} markets, header says {n}")
    return head, markets


def parse_kalshi_meta_zst(path) -> tuple[dict, list[dict]]:
    try:
        from compression import zstd            # Python 3.14 stdlib
        opener = zstd.open
    except ImportError:                         # pragma: no cover
        import zstandard
        opener = zstandard.open
    with opener(path, "rb") as f:
        return parse_kalshi_meta_lines(f)


def kalshi_window_row(m: dict, source: str) -> dict | None:
    """One windows row from a Kalshi market dict (meta day file or recorder `market` object)."""
    ticker = m.get("ticker")
    ws, we = iso_to_s(m.get("open_time")), iso_to_s(m.get("close_time"))
    if not ticker or ws is None or we is None:
        return None
    strike, final = to_float(m.get("floor_strike")), to_float(m.get("expiration_value"))
    res = (m.get("result") or "").lower()
    outcome = {"yes": 1, "no": 0}.get(res, -1)
    return dict(venue="kalshi", family=KALSHI_FAMILY, market_id=ticker, condition_id=m.get("event_ticker"),
                token_up=None, token_down=None, window_start=ws, window_end=we, strike=strike, final=final,
                outcome=outcome, regime="brti60", tape_regime=None, source=source, label_day=s_day(ws))


def best_of_book(levels, pick) -> float | None:
    ps = [to_float(l.get("price")) for l in levels or [] if to_float(l.get("size")) not in (None, 0.0)]
    ps = [p for p in ps if p is not None]
    return pick(ps) if ps else None


# ------------------------------------------------------------------------------------------------ research repo access
class Research:
    """The research repo's guard + readers, imported from its root; every check_data call is logged."""

    def __init__(self, root: Path, log_path: Path):
        self.root = Path(root)
        if not (self.root / "scripts" / "common" / "guard.py").is_file():
            raise SystemExit(f"research repo not found at {self.root} (set PM5_ROOT)")
        for p in (self.root, self.root / "recorder", self.root / "kalshi-recorder", self.root / "scripts" / "tape"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from scripts.common import guard as G          # noqa: N812
        self.G = G
        self.GuardError = G.GuardError
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "a")
        self.calls, self.refusals = 0, []
        self._context = ""
        self._orig_check = G.check_data
        G.check_data = self._logged_check                 # readers (pmr_read, kr_read) call G.check_file -> check_data

    # the wrapper never changes the decision; it records it
    def _logged_check(self, handle, dataset, start_ts, end_ts=None, market_id=None):
        G = self.G
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "dataset": dataset,
               "start": G.iso(G.to_ns(start_ts)), "end": None if end_ts is None else G.iso(G.to_ns(end_ts)),
               "handle": None if handle is None else getattr(handle, "id", "?"), "context": self._context}
        try:
            ok = self._orig_check(handle, dataset, start_ts, end_ts, market_id)
            rec["ok"] = bool(ok)
        except G.GuardError as e:
            rec["ok"] = False
            rec["error"] = str(e)
            self.refusals.append(rec)
            self._write(rec)
            raise
        self._write(rec)
        return ok

    def _write(self, rec):
        self.calls += 1
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()

    def allowed(self, dataset: str, start_s: int, end_s: int, context: str) -> bool:
        """check_data(None, dataset, [start, end)) -> True, or False after logging the GuardError."""
        self._context = context
        try:
            return bool(self.G.check_data(None, dataset, int(start_s), int(end_s)))
        except self.G.GuardError:
            return False

    def refusal_summary(self) -> dict:
        out = defaultdict(list)
        for r in self.refusals:
            out[r["dataset"]].append((r["start"][:10], (r["end"] or r["start"])[:10], r.get("error", "")))
        summ = {}
        for ds, items in out.items():
            days = sorted({a for a, _, _ in items})
            reasons = sorted({re.sub(r"\[.*?\)", "[..]", e.split(": ", 1)[-1])[:140] for _, _, e in items})
            summ[ds] = {"n_refusals": len(items), "days": compress_days(days), "reasons": reasons}
        return summ


# ------------------------------------------------------------------------------------------------ parts / resume
class Parts:
    def __init__(self, out: Path, table: str, resume: bool):
        self.dir = out / "_parts" / table
        self.dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.keys = []

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.parquet"

    def have(self, key: str) -> bool:
        return self.resume and self.path(key).is_file()

    def write(self, key: str, df: pd.DataFrame, cols: list[str]):
        df = df.reindex(columns=cols)
        tmp = self.path(key).with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, self.path(key))
        self.keys.append(key)

    def read(self, key: str) -> pd.DataFrame:
        self.keys.append(key)
        return pd.read_parquet(self.path(key))

    def concat(self, cols: list[str], dtypes: dict | None = None) -> pd.DataFrame:
        frames = [pd.read_parquet(self.path(k)) for k in self.keys if self.path(k).is_file()]
        frames = [f for f in frames if len(f)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        df = df.reindex(columns=cols)
        for c, t in (dtypes or {}).items():
            if c in df:
                df[c] = df[c].astype(t)
        return df


WINDOW_DTYPES = {"window_start": "int64", "window_end": "int64", "strike": "float64", "final": "float64",
                 "outcome": "int8", "venue": "string", "family": "string", "market_id": "string",
                 "condition_id": "string", "token_up": "string", "token_down": "string", "regime": "string",
                 "tape_regime": "string", "source": "string"}
PRICE_DTYPES = {"window_start": "int64", "window_end": "int64", "t": "int64", "minutes_left": "float64",
                "exec_price_up": "float64", "exec_size_up": "float64", "exec_time_up": "float64",
                "exec_price_down": "float64", "exec_size_down": "float64", "exec_time_down": "float64",
                "mkt_prob_up": "float64", "mkt_prob_src": "string", "n_prints_30s": "int64"}


# ------------------------------------------------------------------------------------------------ windows
def build_windows(R: Research, out: Path, args, report: dict) -> pd.DataFrame:
    t_start = time.time()
    parts = Parts(out, "windows", resume=False)
    rep = report["windows"] = {"sources": {}}
    frames = []

    if "polymarket" in args.venues:
        import duckdb
        from tape import connect
        con = connect(R.root / "data" / "tape")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        for fam in FAMILIES:
            files = sorted(glob.glob(str(R.root / "data" / "tape" / "meta" / f"family={fam}" / "*.parquet")))
            file_days = [Path(f).stem for f in files]
            days = [d for d in file_days if args.start <= d <= args.end]
            got_days, n_rows = [], 0
            for d in days:
                d0 = day_s(d)
                if not R.allowed("pm_tape", d0, d0 + DAY, f"windows meta {fam} {d}"):
                    continue
                df = con.execute(f"""
                    SELECT 'polymarket' AS venue, family, slug AS market_id, condition_id, token_up, token_down,
                           window_start, window_end, price_to_beat AS strike, final_price AS final,
                           CASE winner WHEN 'Up' THEN 1 WHEN 'Down' THEN 0 ELSE -1 END AS outcome,
                           coalesce(config_id, regime) AS regime, regime AS tape_regime, 'tape_meta' AS source
                    FROM meta WHERE family = ? AND window_start >= ? AND window_start < ?
                    ORDER BY window_start""", [fam, d0, d0 + DAY]).df()
                if len(df):
                    df["label_day"] = d
                    frames.append(df)
                    got_days.append(d)
                    n_rows += len(df)
            total = con.execute("SELECT count(*) FROM meta WHERE family = ?", [fam]).fetchone()[0]
            rep["sources"][f"tape_meta/{fam}"] = {
                "dataset": "pm_tape", "meta_files": len(files), "file_days": compress_days(file_days),
                "days_with_rows": compress_days(got_days), "n_days": len(got_days), "rows": int(n_rows),
                "rows_in_view_total": int(total),
                "gaps_in_file_range": compress_days(sorted(set(date_range(file_days[0], file_days[-1])) - set(got_days))) if file_days else "(none)"}
        con.close()

    if "kalshi" in args.venues:
        # (a) the Kalshi market-metadata day files (dataset kalshi_trade_tape)
        files = sorted(glob.glob(str(R.root / "data" / "kalshi_meta" / f"kalshi-{KALSHI_SERIES}-markets-*.txt.zst")))
        file_days = [re.search(r"(\d{4}-\d{2}-\d{2})\.txt\.zst$", f).group(1) for f in files]
        rows, got_days, refused_days, bad_files = [], [], [], []
        try:
            from scripts.sim_kalshi import build_cs_tapes as B      # the repo's own draft/registry holds
        except Exception:                                            # pragma: no cover
            B = None
        for f, d in zip(files, file_days):
            if not (args.start <= d <= args.end):
                continue
            d0 = day_s(d)
            if not R.allowed("kalshi_trade_tape", d0, d0 + DAY, f"windows kalshi_meta {d}"):
                refused_days.append(d)
                continue
            if B is not None:                                        # the repo's own extra check (drafts too)
                held = B.registry_holds("kalshi_trade_tape", d0 * 10**9, (d0 + DAY) * 10**9)
                if held:
                    refused_days.append(d)
                    R.refusals.append({"dataset": "kalshi_trade_tape", "start": d, "end": d,
                                       "error": f"registry_holds (build_cs_tapes.check_day convention): held by {held}"})
                    continue
            try:
                head, markets = parse_kalshi_meta_zst(f)
            except Exception as e:
                bad_files.append(f"{Path(f).name}: {e}")
                continue
            if head.get("day") != d:
                bad_files.append(f"{Path(f).name}: header day {head.get('day')!r}")
            n0 = len(rows)
            for m in markets:
                r = kalshi_window_row(m, "kalshi_meta")
                if r is not None:
                    rows.append(r)
            if len(rows) > n0:
                got_days.append(d)
        rep["sources"]["kalshi_meta"] = {
            "dataset": "kalshi_trade_tape", "files": len(files), "file_days": compress_days(file_days),
            "days_read": compress_days(got_days), "n_days": len(got_days), "rows": len(rows),
            "days_refused": compress_days(refused_days), "n_refused": len(refused_days), "bad_files": bad_files}
        if rows:
            frames.append(pd.DataFrame(rows))

        # (b) the Kalshi recorder's meta stream (dataset kalshi_recorder), days 2026-10-11 .. 10-17
        rows = build_kalshi_recorder_windows(R, args, rep)
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        windows = pd.DataFrame(columns=WINDOW_COLS)
    else:
        windows = pd.concat(frames, ignore_index=True).reindex(columns=WINDOW_COLS)
    # de-duplicate: a Kalshi ticker present in both the meta day file and the recorder keeps the meta row
    prio = {"tape_meta": 0, "kalshi_meta": 0, "kalshi_recorder_meta": 1}
    windows["_p"] = windows["source"].map(prio).fillna(9)
    before = len(windows)
    windows = windows.sort_values(["venue", "market_id", "_p"]).drop_duplicates(["venue", "market_id"], keep="first")
    windows = windows.drop(columns="_p").sort_values(["venue", "family", "window_start"]).reset_index(drop=True)
    rep["duplicates_dropped"] = int(before - len(windows))
    for c, t in WINDOW_DTYPES.items():
        windows[c] = windows[c].astype(t)
    windows["label_day"] = pd.to_datetime(windows["label_day"]).dt.date

    mm = outcome_mismatches(windows)
    rep["outcome_mismatch"] = {r.source: {"checked": int(r.checked), "mismatch": int(r.mismatch)} for r in mm.itertuples()}
    rep["rows"] = int(len(windows))
    rep["rows_by_family_source"] = {f"{k[0]}/{k[1]}": int(v) for k, v in windows.groupby(["family", "source"]).size().items()}
    rep["outcome_counts"] = {f"{k[0]}/{int(k[1])}": int(v) for k, v in windows.groupby(["family", "outcome"]).size().items()}
    rep["days_by_family"] = {fam: compress_days(sorted({str(d) for d in g["label_day"]})) for fam, g in windows.groupby("family")}
    rep["regimes"] = {f"{k[0]}/{k[1]}": int(v) for k, v in windows.groupby(["family", "regime"]).size().items()}
    rep["runtime_s"] = round(time.time() - t_start, 1)
    windows.to_parquet(out / "windows.parquet", index=False)
    return windows


def build_kalshi_recorder_windows(R: Research, args, rep: dict) -> list[dict]:
    import kr_read
    kdir = str(R.root / "data" / "rec_mirror" / "kalshi")
    days = [d for d in KALSHI_REC_DAYS if args.start <= d <= args.end]
    latest, strikes, results, finals, got_days, refused = {}, {}, {}, {}, [], []
    lifecycle_results = {}
    for d in days:
        d0 = day_s(d)
        d1 = min(d0 + DAY, day_s(KALSHI_REC_QUARANTINE))
        if d1 <= d0 or not R.allowed("kalshi_recorder", d0, d1, f"windows kalshi recorder meta {d}"):
            refused.append(d)
            continue
        n = 0
        try:
            for j in kr_read.iter_lines(kdir, "meta", d0 * 10**9, d1 * 10**9 - 1):
                if j.get("series") != KALSHI_SERIES:
                    continue
                m = j.get("market") or {}
                tk = j.get("ticker") or m.get("ticker")
                if not tk:
                    continue
                n += 1
                m = dict(m)
                m.setdefault("ticker", tk)
                latest[tk] = m
                if to_float(m.get("floor_strike")) is not None:
                    strikes[tk] = m["floor_strike"]
                if (m.get("result") or "").lower() in ("yes", "no"):
                    results[tk] = m["result"]
                if to_float(m.get("expiration_value")) is not None:
                    finals[tk] = m["expiration_value"]
            for j in kr_read.iter_lines(kdir, "kalshi_lifecycle", d0 * 10**9, d1 * 10**9 - 1):
                msg = (j.get("m") or {}).get("msg") or {}
                tk = msg.get("market_ticker", "")
                if tk.startswith(KALSHI_SERIES + "-") and msg.get("event_type") == "determined" and \
                        (msg.get("result") or "").lower() in ("yes", "no"):
                    lifecycle_results[tk] = msg["result"]
        except R.GuardError as e:                       # a reader-internal refusal: skip the day
            refused.append(d)
            rep.setdefault("reader_guard_errors", []).append(f"kalshi_recorder meta {d}: {e}")
            continue
        if n:
            got_days.append(d)
    rows, n_lifecycle, n_quarantined = [], 0, 0
    for tk, m in latest.items():
        m = dict(m)
        if tk in strikes:
            m["floor_strike"] = strikes[tk]
        if tk in results:
            m["result"] = results[tk]
        elif tk in lifecycle_results:
            m["result"] = lifecycle_results[tk]
            n_lifecycle += 1
        if tk in finals:
            m["expiration_value"] = finals[tk]
        r = kalshi_window_row(m, "kalshi_recorder_meta")
        if r is None:
            continue
        if r["window_start"] >= day_s(KALSHI_REC_QUARANTINE):          # discovered early; its data is quarantined
            n_quarantined += 1
            continue
        rows.append(r)
    rep["sources"]["kalshi_recorder_meta"] = {
        "windows_starting_in_quarantine_dropped": n_quarantined,
        "dataset": "kalshi_recorder", "days_requested": compress_days(days), "days_read": compress_days(got_days),
        "n_days": len(got_days), "days_refused": compress_days(refused), "rows": len(rows),
        "results_from_lifecycle_only": n_lifecycle,
        "rows_without_result": sum(1 for r in rows if r["outcome"] < 0),
        "rows_without_strike": sum(1 for r in rows if r["strike"] is None)}
    return rows


def load_windows(out: Path) -> pd.DataFrame:
    p = out / "windows.parquet"
    if not p.is_file():
        raise SystemExit(f"{p} missing: run without --only or include 'windows'")
    return pd.read_parquet(p)


# ------------------------------------------------------------------------------------------------ Polymarket prints
def build_prices_polymarket(R: Research, out: Path, args, windows: pd.DataFrame, report: dict):
    import duckdb
    from tape import connect
    t_start = time.time()
    rep = report["prices_polymarket"] = {"families": {}}
    con = connect(R.root / "data" / "tape")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    parts = Parts(out, "prices_polymarket", args.resume)
    pw = windows[windows["venue"] == "polymarket"]
    per_day = {}
    for fam in FAMILIES:
        dirs = sorted(glob.glob(str(R.root / "data" / "tape" / "fills" / f"family={fam}" / "date=*")))
        part_days = [Path(d).name.split("=", 1)[1] for d in dirs]
        days = [d for d in part_days if args.start <= d <= args.end]
        done, skipped, times, fill_missing = [], [], {}, 0
        for d in days:
            key = f"{fam}_{d}"
            if parts.have(key):
                parts.keys.append(key)
                done.append(d)
                continue
            d0 = day_s(d)
            if not R.allowed("pm_tape", d0, d0 + DAY, f"prices {fam} {d}"):
                skipped.append(d)
                continue
            t0 = time.time()
            wd = pw[(pw["family"] == fam) & (pw["window_start"] >= d0) & (pw["window_start"] < d0 + DAY)]
            dec = build_decisions(wd)
            if dec.empty:
                parts.write(key, pd.DataFrame(columns=PRICE_COLS), PRICE_COLS)
                done.append(d)
                continue
            con.register("_dec", dec)
            orders_src = f"(SELECT * FROM taker_orders WHERE family = '{fam}' AND date = DATE '{d}')"
            fills_src = f"(SELECT * FROM fills WHERE family = '{fam}' AND date = DATE '{d}')"
            df = con.execute(exec_price_sql("_dec", orders_src, fills_src)).df()
            con.unregister("_dec")
            fill_missing += int(df["fill_missing"].sum())
            parts.write(key, df.drop(columns=["fill_missing"]), PRICE_COLS)
            times[d] = round(time.time() - t0, 1)
            per_day[key] = len(df)
            done.append(d)
            print(f"  prices {fam} {d}: {len(df)} rows, {times[d]} s", flush=True)
        rng = date_range(part_days[0], part_days[-1]) if part_days else []
        rep["families"][fam] = {
            "fills_partitions": len(part_days), "partition_days": compress_days(part_days),
            "days_built": compress_days(done), "n_days": len(done), "days_refused": compress_days(skipped),
            "missing_days_in_range": compress_days(sorted(set(rng) - set(part_days))),
            "orders_without_fill_join": fill_missing, "median_day_runtime_s": float(np.median(list(times.values()))) if times else None}
    con.close()
    df = parts.concat(PRICE_COLS, PRICE_DTYPES)
    df.to_parquet(out / "prices_polymarket.parquet", index=False)
    rep["rows"] = int(len(df))
    if len(df):
        g = df.groupby("family")
        rep["rows_by_family"] = {k: int(v) for k, v in g.size().items()}
        rep["exec_up_coverage"] = {k: round(float(v), 4) for k, v in g["exec_price_up"].apply(lambda s: s.notna().mean()).items()}
        rep["exec_down_coverage"] = {k: round(float(v), 4) for k, v in g["exec_price_down"].apply(lambda s: s.notna().mean()).items()}
        rep["mkt_prob_src"] = {k: int(v) for k, v in df["mkt_prob_src"].fillna("none").value_counts().items()}
        rep["days_by_family"] = {k: compress_days(sorted({s_day(x) for x in v["window_start"].unique()})) for k, v in g}
        rep["rows_per_day_min_max"] = {k: [int(v["window_start"].map(s_day).value_counts().min()), int(v["window_start"].map(s_day).value_counts().max())] for k, v in g}
        bad = int(((df["exec_time_up"] < df["t"] + EXEC_LO) | (df["exec_time_down"] < df["t"] + EXEC_LO)).sum())
        rep["rows_with_exec_before_t_plus_3"] = bad
        assert bad == 0, "look-ahead violation in prices_polymarket"
    rep["runtime_s"] = round(time.time() - t_start, 1)


# ------------------------------------------------------------------------------------------------ Polymarket recorder BBO
_ASSET_RE = re.compile(rb'"asset_id": ?"(\d+)"')


class Series:
    __slots__ = ("ts", "a", "b")

    def __init__(self):
        self.ts, self.a, self.b = array("d"), array("d"), array("d")

    def push(self, t, a, b):
        self.ts.append(t)
        self.a.append(float("nan") if a is None else a)
        self.b.append(float("nan") if b is None else b)

    def arrays(self):
        ts = np.frombuffer(self.ts, dtype="d")
        order = np.argsort(ts, kind="stable")
        return ts[order], np.frombuffer(self.a, dtype="d")[order], np.frombuffer(self.b, dtype="d")[order]


def build_bbo_polymarket(R: Research, out: Path, args, windows: pd.DataFrame, report: dict):
    import pmr_read
    t_start = time.time()
    rep = report["bbo_polymarket"] = {"days": {}}
    pdir = R.root / "data" / "rec_mirror" / "pm"
    parts = Parts(out, "bbo_polymarket", args.resume)
    pw = windows[windows["venue"] == "polymarket"]
    days = [d for d in BBO_DAYS if args.start <= d <= args.end]
    for d in days:
        key = d
        if parts.have(key):
            parts.keys.append(key)
            rep["days"][d] = {"resumed": True}
            continue
        d0 = day_s(d)
        d1 = min(d0 + DAY, day_s(PM_REC_QUARANTINE))
        if d1 <= d0 or not R.allowed("pm_recorder", d0, d1, f"bbo day {d}"):
            rep["days"][d] = {"refused": True}
            continue
        t0 = time.time()
        # windows whose decision minutes fall inside the day
        wd = pw[(pw["window_end"] > d0) & (pw["window_start"] < d1)]
        wd = wd[wd["token_up"].notna() & wd["token_down"].notna()]
        dec = build_decisions(wd)
        dec = dec[(dec["t"] >= d0) & (dec["t"] < d1)]
        tok = {}
        tokens = pmr_read.TokenMap()
        for w in wd.itertuples(index=False):
            tok[str(w.token_up)] = (w.market_id, w.family, "up")
            tok[str(w.token_down)] = (w.market_id, w.family, "down")
            tokens.add_pair(str(w.token_up), str(w.token_down), w.condition_id)
        tok_bytes = {k.encode(): k for k in tok}
        # recorder meta of the day: a cross-check of the token -> market map (Up first in clobTokenIds)
        meta_mismatch, meta_seen = 0, 0
        for f in sorted(glob.glob(str(pdir / "meta" / d.replace("-", "") / "*.jsonl.gz"))):
            s, e = R.G.hour_file_span(f)
            if not R.allowed("pm_recorder", s // 10**9, e // 10**9, f"bbo meta {Path(f).parent.name}/{Path(f).name}"):
                continue
            try:
                for rec in pmr_read.iter_lines(f):
                    if rec.get("family") not in FAMILIES:
                        continue
                    try:
                        mk = rec["event"]["markets"][0]
                        a, b = json.loads(mk["clobTokenIds"])
                    except Exception:
                        continue
                    meta_seen += 1
                    if (a in tok and tok[a] != (rec.get("slug"), rec.get("family"), "up")) or \
                            (b in tok and tok[b] != (rec.get("slug"), rec.get("family"), "down")):
                        meta_mismatch += 1
            except R.GuardError as e:
                rep.setdefault("reader_guard_errors", []).append(f"pm meta {f}: {e}")
        # pm_clob hour files from one hour before the day to one hour after it (carried state, t+30 s overrun)
        files = []
        for f in sorted(glob.glob(str(pdir / "pm_clob" / "*" / "*.jsonl.gz"))):
            s, e = R.G.hour_file_span(f)
            if e <= (d0 - 3600) * 10**9 or s >= (d1 + 3600) * 10**9:
                continue
            files.append((f, s, e))
        bbo, trades = defaultdict(Series), defaultdict(list)
        n_lines = n_parsed = 0
        kinds = defaultdict(int)
        files_read, files_refused = [], []
        for f, s, e in files:
            if not R.allowed("pm_recorder", s // 10**9, min(e, day_s(PM_REC_QUARANTINE) * 10**9) // 10**9,
                             f"bbo pm_clob {Path(f).parent.name}/{Path(f).name}"):
                files_refused.append(f"{Path(f).parent.name}/{Path(f).name}")
                continue
            try:
                for ln in pmr_read.iter_raw(f):
                    n_lines += 1
                    hit = False
                    for m in _ASSET_RE.finditer(ln):
                        if m.group(1) in tok_bytes:
                            hit = True
                            break
                    if not hit:
                        continue
                    try:
                        rec = json.loads(ln)
                    except ValueError:
                        continue
                    n_parsed += 1
                    m = rec.get("m") or {}
                    et = m.get("event_type")
                    r_s = rec.get("r", 0) / 1e9
                    kinds[et] += 1
                    if et == "best_bid_ask":
                        a = m.get("asset_id")
                        if a in tok:
                            bbo[a].push(r_s, to_float(m.get("best_ask")), to_float(m.get("best_bid")))
                    elif et == "price_change":
                        try:
                            entries = pmr_read.expand_price_change(m, tokens)
                        except ValueError:
                            entries = m.get("price_changes", [])
                        for pc in entries:
                            a = pc.get("asset_id")
                            if a in tok and (pc.get("best_ask") is not None or pc.get("best_bid") is not None):
                                bbo[a].push(r_s, to_float(pc.get("best_ask")), to_float(pc.get("best_bid")))
                    elif et == "book":
                        a = m.get("asset_id")
                        if a in tok:
                            bbo[a].push(r_s, best_of_book(m.get("asks"), min), best_of_book(m.get("bids"), max))
                    elif et == "last_trade_price":
                        a = m.get("asset_id")
                        if a in tok:
                            trades[a].append((r_s, to_float(m.get("price")), to_float(m.get("size")), m.get("side")))
                files_read.append(f"{Path(f).parent.name}/{Path(f).name}")
            except R.GuardError as e:
                files_refused.append(f"{Path(f).parent.name}/{Path(f).name}: {e}")
        # the recorder's live span on this day (first/last receive time of any scanned line we parsed)
        all_ts = [s.ts for s in bbo.values() if len(s.ts)]
        live_lo = min(min(x) for x in all_ts) if all_ts else None
        live_hi = max(max(x) for x in all_ts) if all_ts else None
        n_dec_all = len(dec)
        if live_lo is not None:                      # decision minutes outside the recorder's coverage are meaningless
            dec = dec[(dec["t"] + EXEC_LO >= live_lo) & (dec["t"] + EXEC_HI <= live_hi)]
        else:
            dec = dec.iloc[0:0]
        # resolve each decision
        arrs = {a: s.arrays() for a, s in bbo.items()}
        tr = {}
        for a, lst in trades.items():
            lst.sort(key=lambda x: x[0])
            tr[a] = (np.array([x[0] for x in lst]), lst)
        side_tok = {}
        for a, (mid, fam, side) in tok.items():
            side_tok[(mid, side)] = a
        rows = []
        for q in dec.itertuples(index=False):
            row = {"market_id": q.market_id, "family": q.family, "window_start": q.window_start,
                   "window_end": q.window_end, "t": q.t, "minutes_left": q.minutes_left}
            for side in ("up", "down"):
                a = side_tok.get((q.market_id, side))
                ask = bid = bt = pp = ps = pt = None
                if a in arrs:
                    ts, asks, bids = arrs[a]
                    i = state_at(ts, q.t + EXEC_LO)
                    if i >= 0:
                        ask = None if np.isnan(asks[i]) else float(asks[i])
                        bid = None if np.isnan(bids[i]) else float(bids[i])
                        bt = float(ts[i])
                if a in tr:
                    ts, lst = tr[a]
                    i = first_in(ts, q.t + EXEC_LO, q.t + EXEC_HI)
                    while i >= 0 and i < len(lst) and lst[i][0] <= q.t + EXEC_HI:
                        if lst[i][3] == "BUY":
                            pt, pp, ps = lst[i][0], lst[i][1], lst[i][2]
                            break
                        i += 1
                row.update({f"ask_{side}": ask, f"bid_{side}": bid, f"bbo_time_{side}": bt,
                            f"print_price_{side}": pp, f"print_size_{side}": ps, f"print_time_{side}": pt})
            rows.append(row)
        df = pd.DataFrame(rows, columns=BBO_COLS)
        parts.write(key, df, BBO_COLS)
        rep["days"][d] = {"rows": len(df), "decisions_in_day": n_dec_all,
                          "recorder_live_span": None if live_lo is None else [iso(live_lo), iso(live_hi)],
                          "windows": int(len(wd)), "pm_clob_files_read": len(files_read),
                          "pm_clob_files_refused": files_refused, "lines_scanned": n_lines, "lines_parsed": n_parsed,
                          "events": dict(kinds), "meta_markets_seen": meta_seen, "meta_token_map_mismatch": meta_mismatch,
                          "ask_up_coverage": round(float(df["ask_up"].notna().mean()), 4) if len(df) else None,
                          "runtime_s": round(time.time() - t0, 1)}
        print(f"  bbo {d}: {len(df)} rows, {n_lines} lines scanned, {round(time.time() - t0)} s", flush=True)
    df = parts.concat(BBO_COLS)
    for c in ("window_start", "window_end", "t"):
        if c in df:
            df[c] = df[c].astype("int64")
    df.to_parquet(out / "bbo_polymarket.parquet", index=False)
    rep["rows"] = int(len(df))
    if len(df):
        bad = int(((df["print_time_up"] < df["t"] + EXEC_LO) | (df["print_time_down"] < df["t"] + EXEC_LO) |
                   (df["bbo_time_up"] > df["t"] + EXEC_LO) | (df["bbo_time_down"] > df["t"] + EXEC_LO)).sum())
        rep["rows_violating_timing"] = bad
        assert bad == 0, "timing violation in bbo_polymarket"
    rep["runtime_s"] = round(time.time() - t_start, 1)
    haircut = compute_haircut(df)
    (out / "haircut.json").write_text(json.dumps(haircut, indent=2))
    report["haircut"] = haircut


def compute_haircut(bbo: pd.DataFrame) -> dict:
    """first_print_price - best_ask_at_t+3 per family and side, plus the no-print fraction."""
    out = {"definition": "first taker-BUY last_trade_price of the side's token with receive time in [t+3 s, t+30 s] "
                         "minus the carried best ask of that token at t+3 s (recorder receive times); "
                         "no_print_frac = decision minutes without such a print within 30 s",
           "days": sorted({s_day(x) for x in bbo["t"].unique()}) if len(bbo) else [], "by_family_side": {}}
    if not len(bbo):
        return out
    for fam, g in bbo.groupby("family"):
        for side in ("up", "down"):
            pp, ask = g[f"print_price_{side}"], g[f"ask_{side}"]
            ok = pp.notna() & ask.notna()
            diff = (pp[ok] - ask[ok]).astype(float)
            q = diff.quantile([0.5, 0.75, 0.9]).tolist() if len(diff) else [None, None, None]
            out["by_family_side"][f"{fam}/{side}"] = {
                "n_decisions": int(len(g)), "n_with_print": int(pp.notna().sum()), "n_with_bbo": int(ask.notna().sum()),
                "n_compared": int(ok.sum()), "no_print_frac": round(float(pp.isna().mean()), 4),
                "median": None if not len(diff) else round(float(q[0]), 4), "p75": None if not len(diff) else round(float(q[1]), 4),
                "p90": None if not len(diff) else round(float(q[2]), 4),
                "mean": None if not len(diff) else round(float(diff.mean()), 4),
                "frac_print_above_ask": None if not len(diff) else round(float((diff > 1e-9).mean()), 4),
                "frac_print_below_ask": None if not len(diff) else round(float((diff < -1e-9).mean()), 4)}
        any_print = g["print_price_up"].notna() | g["print_price_down"].notna()
        out["by_family_side"][f"{fam}/any"] = {"n_decisions": int(len(g)), "no_print_frac": round(float((~any_print).mean()), 4)}
    return out


# ------------------------------------------------------------------------------------------------ Kalshi ticker prices
def build_prices_kalshi(R: Research, out: Path, args, windows: pd.DataFrame, report: dict):
    import kr_read
    t_start = time.time()
    rep = report["prices_kalshi"] = {"days": {}}
    kdir = str(R.root / "data" / "rec_mirror" / "kalshi")
    parts = Parts(out, "prices_kalshi", args.resume)
    kw = windows[windows["venue"] == "kalshi"]
    days = [d for d in KALSHI_REC_DAYS if args.start <= d <= args.end]
    for d in days:
        key = d
        if parts.have(key):
            parts.keys.append(key)
            rep["days"][d] = {"resumed": True}
            continue
        d0 = day_s(d)
        d1 = min(d0 + DAY, day_s(KALSHI_REC_QUARANTINE))
        if d1 <= d0 or not R.allowed("kalshi_recorder", d0, d1, f"prices kalshi {d}"):
            rep["days"][d] = {"refused": True}
            continue
        t0 = time.time()
        wd = kw[(kw["window_start"] >= d0) & (kw["window_start"] < d0 + DAY)]
        dec = build_decisions(wd)
        since = max(d0 - 3600, day_s("2026-10-10")) * 10**9
        until = min(d1 + 3600, day_s(KALSHI_REC_QUARANTINE)) * 10**9 - 1
        if not R.allowed("kalshi_recorder", since // 10**9, until // 10**9 + 1, f"prices kalshi ticker span {d}"):
            rep["days"][d] = {"refused": True}
            continue
        ser = defaultdict(lambda: {"ts": array("d"), "bid": array("d"), "ask": array("d"), "bs": array("d"),
                                   "as": array("d"), "vol": array("d")})
        n = 0
        try:
            for j in kr_read.iter_lines(kdir, "kalshi_ticker", since, until):
                m = j.get("m") or {}
                if m.get("type") != "ticker":
                    continue
                msg = m.get("msg") or {}
                tk = msg.get("market_ticker", "")
                if not tk.startswith(KALSHI_SERIES + "-"):
                    continue
                s = ser[tk]
                s["ts"].append(j.get("r", 0) / 1e9)
                for k, f in (("bid", "yes_bid_dollars"), ("ask", "yes_ask_dollars"), ("bs", "yes_bid_size_fp"),
                             ("as", "yes_ask_size_fp"), ("vol", "volume_fp")):
                    v = to_float(msg.get(f))
                    s[k].append(float("nan") if v is None else v)
                n += 1
        except R.GuardError as e:
            rep["days"][d] = {"refused": True, "error": str(e)}
            continue
        except FileNotFoundError as e:
            rep["days"][d] = {"missing": str(e)}
            continue
        arrs = {}
        for tk, s in ser.items():
            ts = np.frombuffer(s["ts"], dtype="d")
            o = np.argsort(ts, kind="stable")
            arrs[tk] = {k: np.frombuffer(v, dtype="d")[o] for k, v in s.items()}
        rows = []
        for q in dec.itertuples(index=False):
            row = {"market_id": q.market_id, "family": q.family, "window_start": q.window_start,
                   "window_end": q.window_end, "t": q.t, "minutes_left": q.minutes_left,
                   "exec_price_up": None, "exec_size_up": None, "exec_time_up": None,
                   "exec_price_down": None, "exec_size_down": None, "exec_time_down": None,
                   "mkt_prob_up": None, "mkt_prob_src": None, "n_prints_30s": 0}
            a = arrs.get(q.market_id)
            if a is not None:
                i = state_at(a["ts"], q.t + EXEC_LO)
                if i >= 0:
                    ask, bid = a["ask"][i], a["bid"][i]
                    if not np.isnan(ask):
                        row.update(exec_price_up=float(ask), exec_size_up=None if np.isnan(a["as"][i]) else float(a["as"][i]),
                                   exec_time_up=float(a["ts"][i]))
                    if not np.isnan(bid):
                        row.update(exec_price_down=float(1.0 - bid), exec_size_down=None if np.isnan(a["bs"][i]) else float(a["bs"][i]),
                                   exec_time_down=float(a["ts"][i]))
                    if not np.isnan(ask) and not np.isnan(bid):
                        row.update(mkt_prob_up=float((ask + bid) / 2.0), mkt_prob_src="yes_mid")
                    elif not np.isnan(ask):
                        row.update(mkt_prob_up=float(ask), mkt_prob_src="yes_ask_only")
                    elif not np.isnan(bid):
                        row.update(mkt_prob_up=float(bid), mkt_prob_src="yes_bid_only")
                lo = int(np.searchsorted(a["ts"], q.t + EXEC_LO, side="left"))
                hi = int(np.searchsorted(a["ts"], q.t + EXEC_HI, side="right"))
                if hi > lo:
                    vol = a["vol"][max(lo - 1, 0):hi]
                    row["n_prints_30s"] = int(np.sum(np.diff(vol) > 0)) if len(vol) > 1 else 0
            rows.append(row)
        df = pd.DataFrame(rows, columns=PRICE_COLS)
        parts.write(key, df, PRICE_COLS)
        rep["days"][d] = {"rows": len(df), "windows": int(len(wd)), "ticker_records": n, "markets_with_ticker": len(arrs),
                          "exec_up_coverage": round(float(df["exec_price_up"].notna().mean()), 4) if len(df) else None,
                          "runtime_s": round(time.time() - t0, 1)}
        print(f"  kalshi {d}: {len(df)} rows, {n} ticker records, {round(time.time() - t0)} s", flush=True)
    df = parts.concat(PRICE_COLS, PRICE_DTYPES)
    df.to_parquet(out / "prices_kalshi.parquet", index=False)
    rep["rows"] = int(len(df))
    if len(df):
        bad = int(((df["exec_time_up"] > df["t"] + EXEC_LO) | (df["exec_time_down"] > df["t"] + EXEC_LO)).sum())
        rep["rows_with_state_after_t_plus_3"] = bad
        assert bad == 0
    rep["runtime_s"] = round(time.time() - t_start, 1)


# ------------------------------------------------------------------------------------------------ coverage report
def write_coverage(out: Path, report: dict, R: Research, args):
    report["guard"] = {"check_data_calls_logged_this_run": R.calls, "n_refusals_this_run": len(R.refusals),
                       "refusals_by_dataset_this_run": R.refusal_summary(), "log": str(out / "guard_log.jsonl")}
    # the whole log (all runs into this --out), so coverage.md is complete after partial / resumed runs
    tot, ref, by_ds = 0, 0, defaultdict(lambda: {"n": 0, "days": set(), "reasons": set()})
    if R.log_path.is_file():
        for ln in open(R.log_path):
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            tot += 1
            if not rec.get("ok"):
                ref += 1
                b = by_ds[rec["dataset"]]
                b["n"] += 1
                b["days"].add(rec["start"][:10])
                b["reasons"].add(re.sub(r"\[.*?\)", "[..]", rec.get("error", "").split(": ", 1)[-1])[:140])
    report["guard"]["log_total"] = {"check_data_calls": tot, "refusals": ref, "refusals_by_dataset": {
        ds: {"n_refusals": b["n"], "days": compress_days(sorted(b["days"])), "reasons": sorted(b["reasons"])} for ds, b in by_ds.items()}}
    report["args"] = {k: (v if isinstance(v, (str, int, float, bool, list)) or v is None else str(v)) for k, v in vars(args).items()}
    (out / "coverage.json").write_text(json.dumps(report, indent=2, default=str))
    L = []
    L.append(f"# Label and market-price coverage\n\nBuilt {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} "
             f"by `btc_updown/data/labels.py` from `{R.root}` (read-only, guard-checked). Args: `{report['args']}`.\n")
    g = report["guard"]
    gt = g["log_total"]
    L.append(f"## Guard\n\n- check_data calls logged (all runs into this folder, ours and the readers' own via the wrapper; "
             f"see `guard_log.jsonl`): {gt['check_data_calls']}; this run: {g['check_data_calls_logged_this_run']}\n"
             f"- refusals (all runs): {gt['refusals']}; this run: {g['n_refusals_this_run']}")
    for ds, s in gt["refusals_by_dataset"].items():
        L.append(f"  - `{ds}`: {s['n_refusals']} refusals, days {s['days']}; reasons: {'; '.join(s['reasons'])}")
    L.append(f"- runs: {report.get('runs')}")
    if "windows" in report:
        w = report["windows"]
        L.append(f"\n## windows.parquet ({w.get('rows')} rows, {w.get('runtime_s')} s)\n")
        for src, s in w["sources"].items():
            L.append(f"- **{src}** (dataset `{s['dataset']}`): " + ", ".join(f"{k}={v}" for k, v in s.items() if k != "dataset"))
        L.append(f"- rows by family/source: {w.get('rows_by_family_source')}")
        L.append(f"- days by family: {w.get('days_by_family')}")
        L.append(f"- outcome counts (family/outcome): {w.get('outcome_counts')}")
        L.append(f"- regimes: {w.get('regimes')}")
        L.append(f"- duplicates dropped (same venue+market_id, meta file preferred over recorder): {w.get('duplicates_dropped')}")
        L.append(f"- **outcome == (final >= strike) mismatches per source**: {w.get('outcome_mismatch')}")
        if w.get("reader_guard_errors"):
            L.append(f"- reader-internal guard errors: {w['reader_guard_errors']}")
    if "prices_polymarket" in report:
        p = report["prices_polymarket"]
        L.append(f"\n## prices_polymarket.parquet ({p.get('rows')} rows, {p.get('runtime_s')} s)\n")
        L.append("Rule: first taker-BUY taker order of the side with block_time in [t+3 s, t+30 s]; price = lowest fill "
                 "price inside that order (fills joined on tx_hash + taker_log_index), size = the order's shares; "
                 "mkt_prob_up = mid of the first Up-buy fill price and 1 - first Down-buy fill price; "
                 "n_prints_30s = taker-BUY orders of either side in that interval.")
        for fam, s in p["families"].items():
            L.append(f"- **{fam}**: " + ", ".join(f"{k}={v}" for k, v in s.items()))
        for k in ("rows_by_family", "days_by_family", "rows_per_day_min_max", "exec_up_coverage", "exec_down_coverage",
                  "mkt_prob_src", "rows_with_exec_before_t_plus_3"):
            if k in p:
                L.append(f"- {k}: {p[k]}")
    if "bbo_polymarket" in report:
        b = report["bbo_polymarket"]
        L.append(f"\n## bbo_polymarket.parquet ({b.get('rows')} rows, {b.get('runtime_s')} s)\n")
        for d, s in b["days"].items():
            L.append(f"- **{d}**: " + ", ".join(f"{k}={v}" for k, v in s.items()))
        if "rows_violating_timing" in b:
            L.append(f"- rows violating timing: {b['rows_violating_timing']}")
        if b.get("reader_guard_errors"):
            L.append(f"- reader-internal guard errors: {b['reader_guard_errors']}")
    if "haircut" in report:
        h = report["haircut"]
        L.append(f"\n## haircut.json\n\n{h['definition']}\n\n| family/side | n_decisions | n_compared | median | p75 | p90 | mean | no_print_frac |\n|---|---|---|---|---|---|---|---|")
        for k, s in h["by_family_side"].items():
            L.append(f"| {k} | {s.get('n_decisions')} | {s.get('n_compared', '')} | {s.get('median', '')} | {s.get('p75', '')} | "
                     f"{s.get('p90', '')} | {s.get('mean', '')} | {s.get('no_print_frac')} |")
    if "prices_kalshi" in report:
        k = report["prices_kalshi"]
        L.append(f"\n## prices_kalshi.parquet ({k.get('rows')} rows, {k.get('runtime_s')} s)\n")
        L.append("Rule: carried kalshi_ticker state (receive time) at t+3 s; exec_price_up = yes ask, exec_price_down = "
                 "1 - yes bid, sizes = the quoted sizes, mkt_prob_up = yes mid; n_prints_30s = ticker updates in "
                 "[t+3, t+30] whose volume_fp increased.")
        for d, s in k["days"].items():
            L.append(f"- **{d}**: " + ", ".join(f"{kk}={v}" for kk, v in s.items()))
    (out / "coverage.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/labels")
    ap.add_argument("--research-root", default=str(RESEARCH_ROOT))
    ap.add_argument("--venues", nargs="+", default=["polymarket", "kalshi"], choices=["polymarket", "kalshi"])
    ap.add_argument("--start", default="2025-01-01", help="first label day (UTC) to include")
    ap.add_argument("--end", default="2026-10-17", help="last label day (UTC) to include")
    ap.add_argument("--only", nargs="+", default=["windows", "prices", "bbo", "kalshi"],
                    choices=["windows", "prices", "bbo", "kalshi"])
    ap.add_argument("--resume", action="store_true", help="skip days whose per-day part file exists under _parts/")
    ap.add_argument("--memory-limit", default="3GB")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Research(Path(args.research_root), out / "guard_log.jsonl")
    report = {}
    if (out / "coverage.json").is_file():               # keep the sections of tables not rebuilt in this run
        try:
            report = json.loads((out / "coverage.json").read_text())
        except ValueError:
            report = {}
    report["started"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    report.setdefault("runs", []).append({"started": report["started"], "only": args.only, "start": args.start,
                                          "end": args.end, "resume": args.resume})
    t0 = time.time()
    if "windows" in args.only:
        print("building windows ...", flush=True)
        windows = build_windows(R, out, args, report)
    else:
        windows = load_windows(out)
    print(f"windows: {len(windows)} rows", flush=True)
    if "prices" in args.only and "polymarket" in args.venues:
        print("building prices_polymarket ...", flush=True)
        build_prices_polymarket(R, out, args, windows, report)
    if "bbo" in args.only and "polymarket" in args.venues:
        print("building bbo_polymarket ...", flush=True)
        build_bbo_polymarket(R, out, args, windows, report)
    if "kalshi" in args.only and "kalshi" in args.venues:
        print("building prices_kalshi ...", flush=True)
        build_prices_kalshi(R, out, args, windows, report)
    report["total_runtime_s"] = round(time.time() - t0, 1)
    write_coverage(out, report, R, args)
    print(f"done in {report['total_runtime_s']} s; guard calls {R.calls}, refusals {len(R.refusals)}", flush=True)
    print((out / "coverage.md").read_text())


if __name__ == "__main__":
    main()
