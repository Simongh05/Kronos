"""Download Binance 1-minute klines from data.binance.vision and build per-series arrays.

Plain Python with numpy / pandas / pyarrow / requests only, so the same file runs
on the Mac and unattended on a RunPod pod.  Idempotent and resumable: every zip
is cached under ``<out>/_zips/<series>/`` next to its ``.CHECKSUM`` and is only
re-downloaded when the local copy fails the sha256 check.

Outputs per series under ``<out>/<series>/``:

* ``bars.parquet``   ts (int64, UTC seconds of bar open), open, high, low, close,
                     volume, amount (float64); sorted, unique ts, no NaN, prices > 0
* ``x.f32.npy``      float32 [N, 6] = open, high, low, close, volume, amount
* ``stamp.i16.npy``  int16  [N, 5] = minute, hour, weekday (Mon=0), day, month (UTC)
* ``ts.i64.npy``     int64  [N]    = bar open time in seconds
* ``meta.json``      coverage, gaps, source files with sha256, code version

Usage::

    python -m btc_updown.data.binance_klines --series spot_BTCUSDC spot_BTCUSDT um_BTCUSDT \
        --start 2019-01 --end 2026-10 --out data/klines --workers 4
    python -m btc_updown.data.binance_klines --series spot_BTCUSDC --check

Kline CSV quirks handled here: ``open_time`` is in milliseconds up to 2024-12 and
in microseconds from 2025-01 for spot (futures stayed in milliseconds); the unit
is detected per row by magnitude, never by date.  Futures files carry a header
row, spot files do not.  ``quote_asset_volume`` is stored as ``amount``.
"""
from __future__ import annotations

import argparse
import calendar
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import random
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests

CODE_VERSION = "btc_updown.data.binance_klines/1.0.0"
BASE_URL = "https://data.binance.vision"
DEFAULT_OUT = Path("/Users/streetteam/Kronos/data/klines")

RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume", "amount"]
X_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
STAMP_COLUMNS = ["minute", "hour", "weekday", "day", "month"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# open_time magnitude thresholds (values are ~1.5e9 s, ~1.5e12 ms, ~1.5e15 us
# for any date between 2001 and 2286).
_US_THRESHOLD = 10**14
_MS_THRESHOLD = 10**11

log = logging.getLogger("btc_updown.binance_klines")


# --------------------------------------------------------------------------- series


@dataclasses.dataclass(frozen=True)
class Series:
    """One kline series on data.binance.vision."""

    name: str
    market: str  # "spot" or "futures/um"
    symbol: str

    @property
    def monthly_prefix(self) -> str:
        return f"data/{self.market}/monthly/klines/{self.symbol}/1m"

    @property
    def daily_prefix(self) -> str:
        return f"data/{self.market}/daily/klines/{self.symbol}/1m"

    def monthly_name(self, year: int, month: int) -> str:
        return f"{self.symbol}-1m-{year:04d}-{month:02d}.zip"

    def daily_name(self, day: dt.date) -> str:
        return f"{self.symbol}-1m-{day.isoformat()}.zip"

    def monthly_url(self, year: int, month: int) -> str:
        return f"{BASE_URL}/{self.monthly_prefix}/{self.monthly_name(year, month)}"

    def daily_url(self, day: dt.date) -> str:
        return f"{BASE_URL}/{self.daily_prefix}/{self.daily_name(day)}"


SERIES: dict[str, Series] = {
    "spot_BTCUSDC": Series("spot_BTCUSDC", "spot", "BTCUSDC"),
    "spot_BTCUSDT": Series("spot_BTCUSDT", "spot", "BTCUSDT"),
    "um_BTCUSDT": Series("um_BTCUSDT", "futures/um", "BTCUSDT"),
    "spot_ETHUSDT": Series("spot_ETHUSDT", "spot", "ETHUSDT"),
    "spot_SOLUSDT": Series("spot_SOLUSDT", "spot", "SOLUSDT"),
}


# --------------------------------------------------------------------------- helpers


def parse_month(text: str) -> tuple[int, int]:
    year, month = text.split("-")
    y, m = int(year), int(month)
    if not 1 <= m <= 12:
        raise ValueError(f"bad month {text!r}")
    return y, m


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from start to end."""
    out = []
    y, m = start
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def months_back(ym: tuple[int, int], n: int) -> tuple[int, int]:
    y, m = ym
    idx = y * 12 + (m - 1) - n
    return idx // 12, idx % 12 + 1


def iso_utc(ts_seconds: int | float) -> str:
    return dt.datetime.fromtimestamp(int(ts_seconds), tz=dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def parse_checksum(text: str) -> str:
    """Return the sha256 hex digest from a Binance ``.CHECKSUM`` file (``<sha256>  <name>``)."""
    for line in text.splitlines():
        parts = line.split()
        if parts:
            digest = parts[0].lower()
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                return digest
    raise ValueError("no sha256 digest found in CHECKSUM text")


def to_seconds(open_time: np.ndarray) -> np.ndarray:
    """Convert raw Binance ``open_time`` values to int64 seconds, detecting the unit per row.

    Values >= 1e14 are microseconds, >= 1e11 milliseconds, anything smaller seconds.
    """
    raw = np.asarray(open_time, dtype=np.int64)
    out = raw.copy()
    is_us = raw >= _US_THRESHOLD
    is_ms = (raw >= _MS_THRESHOLD) & ~is_us
    out[is_us] = raw[is_us] // 1_000_000
    out[is_ms] = raw[is_ms] // 1_000
    return out


def detect_units(open_time: np.ndarray) -> dict[str, int]:
    raw = np.asarray(open_time, dtype=np.int64)
    n_us = int((raw >= _US_THRESHOLD).sum())
    n_ms = int(((raw >= _MS_THRESHOLD) & (raw < _US_THRESHOLD)).sum())
    return {"us": n_us, "ms": n_ms, "s": int(raw.size - n_us - n_ms)}


# --------------------------------------------------------------------------- download


class Fetcher:
    """HTTP downloader with per-thread sessions, retries and exponential backoff."""

    def __init__(self, timeout: float = 60.0, retries: int = 6, backoff: float = 1.0, max_backoff: float = 60.0):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = CODE_VERSION
            self._local.session = s
        return s

    def _sleep(self, attempt: int) -> None:
        delay = min(self.max_backoff, self.backoff * (2 ** attempt)) * (0.5 + random.random())
        time.sleep(delay)

    def get(self, url: str, dest: Path) -> str:
        """Download ``url`` to ``dest`` atomically. Returns ``"ok"`` or ``"missing"`` (HTTP 404)."""
        if not url.startswith(BASE_URL + "/"):
            raise ValueError(f"refusing to download from outside {BASE_URL}: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self.session.get(url, stream=True, timeout=self.timeout) as r:
                    if r.status_code == 404:
                        return "missing"
                    if r.status_code != 200:
                        raise requests.HTTPError(f"HTTP {r.status_code} for {url}")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            if chunk:
                                fh.write(chunk)
                os.replace(tmp, dest)
                return "ok"
            except (requests.RequestException, OSError) as exc:  # network, timeout, disk
                last_err = exc
                log.warning("attempt %d/%d failed for %s: %s", attempt + 1, self.retries, url, exc)
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                if attempt + 1 < self.retries:
                    self._sleep(attempt)
        raise RuntimeError(f"giving up on {url}: {last_err}")


@dataclasses.dataclass
class Archive:
    """One zip on the archive: where it lives remotely and locally, and its status."""

    series: str
    kind: str  # "monthly" | "daily"
    key: str  # "YYYY-MM" or "YYYY-MM-DD"
    url: str
    path: Path
    status: str = "pending"  # "ok" | "missing" | "pending"
    sha256: str | None = None
    cached: bool = False

    @property
    def checksum_path(self) -> Path:
        return self.path.with_name(self.path.name + ".CHECKSUM")

    def to_meta(self) -> dict:
        return {"name": self.path.name, "kind": self.kind, "key": self.key, "url": self.url, "sha256": self.sha256}


def _verify_local(archive: Archive) -> bool:
    """True if the cached zip exists and matches its cached CHECKSUM."""
    if not (archive.path.exists() and archive.checksum_path.exists()):
        return False
    try:
        expected = parse_checksum(archive.checksum_path.read_text())
    except (ValueError, OSError):
        return False
    actual = sha256_file(archive.path)
    if actual != expected:
        return False
    archive.sha256 = actual
    return True


def fetch_archive(archive: Archive, fetcher: Fetcher, verify_attempts: int = 3) -> Archive:
    """Ensure ``archive`` is present and verified locally, downloading only if needed."""
    if _verify_local(archive):
        archive.status, archive.cached = "ok", True
        log.debug("cached   %s", archive.path.name)
        return archive

    status = fetcher.get(archive.url + ".CHECKSUM", archive.checksum_path)
    if status == "missing":
        archive.status = "missing"
        for p in (archive.path, archive.checksum_path):
            if p.exists():
                p.unlink()
        log.info("missing  %s", archive.path.name)
        return archive
    expected = parse_checksum(archive.checksum_path.read_text())

    for attempt in range(verify_attempts):
        status = fetcher.get(archive.url, archive.path)
        if status == "missing":
            archive.status = "missing"
            archive.checksum_path.unlink(missing_ok=True)
            log.info("missing  %s (checksum present, zip absent)", archive.path.name)
            return archive
        actual = sha256_file(archive.path)
        if actual == expected:
            archive.sha256, archive.status = actual, "ok"
            log.info("fetched  %s (%d bytes)", archive.path.name, archive.path.stat().st_size)
            return archive
        log.warning("sha256 mismatch for %s (attempt %d/%d), re-downloading", archive.path.name, attempt + 1, verify_attempts)
        archive.path.unlink(missing_ok=True)
    raise RuntimeError(f"sha256 mismatch persists for {archive.url}")


def _run_parallel(archives: Sequence[Archive], fetcher: Fetcher, workers: int) -> list[Archive]:
    if not archives:
        return []
    results: list[Archive] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch_archive, a, fetcher): a for a in archives}
        for fut in as_completed(futures):
            results.append(fut.result())  # re-raises download failures
    return results


def plan_and_download(
    series_list: Sequence[Series],
    months: Sequence[tuple[int, int]],
    out: Path,
    fetcher: Fetcher,
    workers: int = 4,
    today: dt.date | None = None,
    daily_fallback_months: int = 2,
) -> dict[str, list[Archive]]:
    """Download monthly zips for every series/month, then daily zips for recent months
    whose monthly archive does not exist yet.  Returns all archives per series.

    A monthly 404 older than ``daily_fallback_months`` is recorded as absent and
    not probed day by day (e.g. futures months before the market existed).
    """
    today = today or dt.datetime.now(dt.timezone.utc).date()
    current_month = (today.year, today.month)
    fallback_from = months_back(current_month, daily_fallback_months)

    monthly: list[Archive] = []
    for s in series_list:
        cache = out / "_zips" / s.name
        for (y, m) in months:
            if (y, m) > current_month:
                continue  # nothing can exist in the future
            monthly.append(Archive(s.name, "monthly", f"{y:04d}-{m:02d}", s.monthly_url(y, m), cache / s.monthly_name(y, m)))
    log.info("checking %d monthly archives with %d workers", len(monthly), workers)
    done = _run_parallel(monthly, fetcher, workers)

    daily: list[Archive] = []
    by_series = {s.name: s for s in series_list}
    for a in done:
        if a.status != "missing":
            continue
        ym = parse_month(a.key)
        if ym < fallback_from:
            log.info("monthly %s/%s absent and older than fallback window: recorded as missing", a.series, a.key)
            continue
        s = by_series[a.series]
        y, m = ym
        last_day = calendar.monthrange(y, m)[1]
        for d in range(1, last_day + 1):
            day = dt.date(y, m, d)
            if day >= today:
                break  # today's file is published after UTC midnight
            daily.append(Archive(s.name, "daily", day.isoformat(), s.daily_url(day), (out / "_zips" / s.name) / s.daily_name(day)))
    if daily:
        log.info("fetching %d daily archives for months without a monthly file", len(daily))
    done += _run_parallel(daily, fetcher, workers)

    result: dict[str, list[Archive]] = {s.name: [] for s in series_list}
    for a in done:
        result[a.series].append(a)
    for name in result:
        result[name].sort(key=lambda a: (0 if a.kind == "monthly" else 1, a.key))
    return result


# --------------------------------------------------------------------------- parsing


def read_kline_zip(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse one Binance kline zip into a DataFrame with ``BAR_COLUMNS`` plus parse stats."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected exactly one csv member, got {members}")
        with zf.open(members[0]) as fh:
            df = pd.read_csv(fh, header=None, dtype=str, keep_default_na=False, skip_blank_lines=True)
    if df.shape[1] < 12:
        raise ValueError(f"{path.name}: expected 12 columns, got {df.shape[1]}")
    df = df.iloc[:, :12]
    df.columns = RAW_COLUMNS

    raw_rows = len(df)
    open_time = pd.to_numeric(df["open_time"], errors="coerce")
    header_rows = int(open_time.isna().sum())  # header rows / junk lines
    keep = open_time.notna()
    df = df.loc[keep]
    open_time = open_time.loc[keep].astype(np.int64).to_numpy()

    units = detect_units(open_time)
    ts = to_seconds(open_time)
    non_aligned = int((ts % 60 != 0).sum())

    out = pd.DataFrame({"ts": ts})
    for src, dst in (("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"),
                     ("volume", "volume"), ("quote_asset_volume", "amount")):
        out[dst] = pd.to_numeric(df[src].to_numpy(), errors="coerce").astype(np.float64)
    stats = {"file": path.name, "rows": raw_rows, "header_rows": header_rows, "units": units, "non_aligned": non_aligned}
    return out, stats


def assemble(frames: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Concatenate, drop bad rows, sort by ts and deduplicate (first occurrence wins)."""
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=np.int64 if c == "ts" else np.float64) for c in BAR_COLUMNS}), {
            "rows_in": 0, "dropped_nan": 0, "dropped_nonpositive_price": 0, "dropped_duplicates": 0}
    df = pd.concat(frames, ignore_index=True)
    rows_in = len(df)
    nan_mask = df[BAR_COLUMNS[1:]].isna().any(axis=1) | df["ts"].isna()
    df = df.loc[~nan_mask]
    bad_price = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    df = df.loc[~bad_price]
    df = df.sort_values("ts", kind="stable")
    before = len(df)
    df = df.drop_duplicates(subset="ts", keep="first").reset_index(drop=True)
    df["ts"] = df["ts"].astype(np.int64)
    for c in BAR_COLUMNS[1:]:
        df[c] = df[c].astype(np.float64)
    stats = {
        "rows_in": rows_in,
        "dropped_nan": int(nan_mask.sum()),
        "dropped_nonpositive_price": int(bad_price.sum()),
        "dropped_duplicates": int(before - len(df)),
    }
    return df[BAR_COLUMNS], stats


def compute_stamp(ts: np.ndarray) -> np.ndarray:
    """int16 [N, 5] = minute, hour, weekday (Monday=0), day, month in UTC, as pandas computes them."""
    idx = pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="s", utc=True)
    cols = [idx.minute, idx.hour, idx.day_of_week, idx.day, idx.month]  # day_of_week == weekday (Mon=0)
    return np.stack([np.asarray(c, dtype=np.int16) for c in cols], axis=1)


def compute_gaps(ts: np.ndarray, top: int = 20) -> dict:
    """Gap statistics for a sorted, unique array of bar-open seconds.

    A gap is any pair of consecutive bars whose ts differ from 60 s.  Missing minutes
    counts the bars that would fill the hole; steps shorter than 60 s or not a
    multiple of 60 s are reported separately as ``irregular``.
    """
    ts = np.asarray(ts, dtype=np.int64)
    if ts.size < 2:
        return {"n_gaps": 0, "missing_minutes": 0, "irregular": 0, "largest": []}
    d = np.diff(ts)
    gap_idx = np.flatnonzero(d != 60)
    long = d[gap_idx] > 60
    missing = int(((d[gap_idx][long] // 60) - 1).sum())
    irregular = int(((d[gap_idx] < 60) | (d[gap_idx] % 60 != 0)).sum())
    order = gap_idx[np.argsort(-d[gap_idx], kind="stable")][:top]
    largest = [
        {
            "start": iso_utc(ts[i]),
            "end": iso_utc(ts[i + 1]),
            "missing_minutes": int(max(d[i] // 60 - 1, 0)),
            "seconds": int(d[i]),
        }
        for i in order
    ]
    return {"n_gaps": int(gap_idx.size), "missing_minutes": missing, "irregular": irregular, "largest": largest}


# --------------------------------------------------------------------------- outputs


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def write_outputs(series_dir: Path, bars: pd.DataFrame, meta: dict) -> dict:
    """Write bars.parquet, x.f32.npy, stamp.i16.npy, ts.i64.npy and meta.json atomically."""
    series_dir.mkdir(parents=True, exist_ok=True)
    ts = bars["ts"].to_numpy(dtype=np.int64)
    x = np.ascontiguousarray(bars[X_COLUMNS].to_numpy(dtype=np.float64).astype(np.float32))
    stamp = compute_stamp(ts)

    tmp_parquet = series_dir / "bars.parquet.tmp"
    bars.to_parquet(tmp_parquet, engine="pyarrow", index=False)
    os.replace(tmp_parquet, series_dir / "bars.parquet")
    _atomic_save_npy(series_dir / "x.f32.npy", x)
    _atomic_save_npy(series_dir / "stamp.i16.npy", stamp)
    _atomic_save_npy(series_dir / "ts.i64.npy", ts)

    sizes = {name: (series_dir / name).stat().st_size for name in ("bars.parquet", "x.f32.npy", "stamp.i16.npy", "ts.i64.npy")}
    meta = dict(meta, output_bytes=sizes)
    tmp_meta = series_dir / "meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, series_dir / "meta.json")
    return sizes


def build_series(series: Series, archives: Sequence[Archive], out: Path) -> dict:
    """Parse all verified archives of a series and write its outputs. Returns the meta dict."""
    ok = [a for a in archives if a.status == "ok"]
    missing = [a for a in archives if a.status == "missing"]
    frames, parse_stats = [], []
    for a in ok:
        df, st = read_kline_zip(a.path)
        frames.append(df)
        parse_stats.append(st)
    bars, asm = assemble(frames)
    ts = bars["ts"].to_numpy(dtype=np.int64)
    gaps = compute_gaps(ts)
    n = int(len(bars))
    expected = int((ts[-1] - ts[0]) // 60 + 1) if n else 0
    meta = {
        "series": series.name,
        "market": series.market,
        "symbol": series.symbol,
        "interval": "1m",
        "code_version": CODE_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "first_ts": iso_utc(ts[0]) if n else None,
        "last_ts": iso_utc(ts[-1]) if n else None,
        "n": n,
        "expected_minutes": expected,
        "coverage": (n / expected) if expected else None,
        "n_gaps": gaps["n_gaps"],
        "missing_minutes": gaps["missing_minutes"],
        "irregular_steps": gaps["irregular"],
        "largest_gaps": gaps["largest"],
        "assembly": asm,
        "units_detected": {
            "us": sum(s["units"]["us"] for s in parse_stats),
            "ms": sum(s["units"]["ms"] for s in parse_stats),
            "s": sum(s["units"]["s"] for s in parse_stats),
        },
        "header_rows_skipped": sum(s["header_rows"] for s in parse_stats),
        "non_aligned_rows": sum(s["non_aligned"] for s in parse_stats),
        "source_files": [a.to_meta() | {"rows": st["rows"] - st["header_rows"]} for a, st in zip(ok, parse_stats)],
        "missing_archives": [{"kind": a.kind, "key": a.key, "url": a.url} for a in missing],
        "columns": {"bars": BAR_COLUMNS, "x": X_COLUMNS, "stamp": STAMP_COLUMNS},
    }
    meta["output_bytes"] = write_outputs(out / series.name, bars, meta)
    log.info("%s: %d bars, %d gaps, %d missing minutes, %d files", series.name, n, gaps["n_gaps"], gaps["missing_minutes"], len(ok))
    return meta


# --------------------------------------------------------------------------- check


def check_outputs(series_dir: Path) -> dict:
    """Verify existing outputs are mutually consistent; returns a coverage row or raises."""
    meta = json.loads((series_dir / "meta.json").read_text())
    ts = np.load(series_dir / "ts.i64.npy", mmap_mode="r")
    x = np.load(series_dir / "x.f32.npy", mmap_mode="r")
    stamp = np.load(series_dir / "stamp.i16.npy", mmap_mode="r")
    bars = pd.read_parquet(series_dir / "bars.parquet")
    n = int(ts.shape[0])
    problems = []
    if ts.dtype != np.int64 or x.dtype != np.float32 or stamp.dtype != np.int16:
        problems.append(f"dtypes {ts.dtype} {x.dtype} {stamp.dtype}")
    if x.shape != (n, 6) or stamp.shape != (n, 5) or len(bars) != n or meta.get("n") != n:
        problems.append(f"shape mismatch ts={n} x={x.shape} stamp={stamp.shape} parquet={len(bars)} meta={meta.get('n')}")
    if list(bars.columns) != BAR_COLUMNS:
        problems.append(f"parquet columns {list(bars.columns)}")
    if n:
        ts_arr = np.asarray(ts)
        if not np.all(np.diff(ts_arr) > 0):
            problems.append("ts not strictly increasing")
        if not np.array_equal(ts_arr, bars["ts"].to_numpy(dtype=np.int64)):
            problems.append("ts.npy != parquet ts")
        xa = np.asarray(x)
        if not np.all(np.isfinite(xa)):
            problems.append("non-finite values in x")
        if not np.all(xa[:, :4] > 0):
            problems.append("non-positive prices in x")
        if not np.array_equal(np.asarray(stamp), compute_stamp(ts_arr)):
            problems.append("stamp does not match recomputation from ts")
        gaps = compute_gaps(ts_arr)
        if gaps["n_gaps"] != meta.get("n_gaps") or gaps["missing_minutes"] != meta.get("missing_minutes"):
            problems.append("gap statistics in meta.json are stale")
    if problems:
        raise RuntimeError(f"{series_dir.name}: " + "; ".join(problems))
    return meta


def coverage_row(meta: dict) -> dict:
    return {
        "series": meta["series"],
        "first": (meta.get("first_ts") or "-")[:16],
        "last": (meta.get("last_ts") or "-")[:16],
        "N": meta["n"],
        "gaps": meta["n_gaps"],
        "missing_min": meta["missing_minutes"],
        "coverage": f"{100 * meta['coverage']:.3f}%" if meta.get("coverage") else "-",
    }


def format_table(rows: Sequence[dict]) -> str:
    if not rows:
        return "(no series)"
    cols = list(rows[0].keys())

    def cell(value) -> str:
        return f"{value:,}" if isinstance(value, int) else str(value)

    widths = {c: max(len(c), *(len(cell(r[c])) for r in rows)) for c in cols}

    def fmt(r: dict, c: str) -> str:
        text = cell(r[c])
        return text.rjust(widths[c]) if isinstance(r[c], int) else text.ljust(widths[c])

    lines = ["  ".join(c.ljust(widths[c]) for c in cols), "  ".join("-" * widths[c] for c in cols)]
    lines += ["  ".join(fmt(r, c) for c in cols) for r in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES), help="series to build")
    p.add_argument("--start", default="2019-01", help="first month YYYY-MM (inclusive)")
    p.add_argument("--end", default=None, help="last month YYYY-MM (inclusive; default: current UTC month)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads")
    p.add_argument("--check", action="store_true", help="only verify existing outputs and print coverage")
    p.add_argument("--daily-fallback-months", type=int, default=2,
                   help="probe daily zips for months without a monthly zip only this many months back from today")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    p.add_argument("--retries", type=int, default=6, help="download attempts per file")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    out: Path = args.out
    series_list = [SERIES[name] for name in args.series]

    if args.check:
        rows, failed = [], False
        for s in series_list:
            try:
                rows.append(coverage_row(check_outputs(out / s.name)))
            except Exception as exc:  # report every series, then fail
                failed = True
                log.error("check failed: %s", exc)
        print(format_table(rows))
        return 1 if failed else 0

    today = dt.datetime.now(dt.timezone.utc).date()
    start = parse_month(args.start)
    end = parse_month(args.end) if args.end else (today.year, today.month)
    end = min(end, (today.year, today.month))
    if start > end:
        raise SystemExit(f"--start {args.start} is after --end {args.end}")
    months = month_range(start, end)

    fetcher = Fetcher(timeout=args.timeout, retries=args.retries)
    archives = plan_and_download(series_list, months, out, fetcher, workers=args.workers, today=today,
                                 daily_fallback_months=args.daily_fallback_months)
    rows = []
    for s in series_list:
        meta = build_series(s, archives[s.name], out)
        rows.append(coverage_row(meta))
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
