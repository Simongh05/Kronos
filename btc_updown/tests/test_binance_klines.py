"""Offline tests for btc_updown.data.binance_klines (no network unless BTC_UPDOWN_NETWORK=1)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_updown.data import binance_klines as bk

network = pytest.mark.network
skip_without_network = pytest.mark.skipif(
    not os.environ.get("BTC_UPDOWN_NETWORK"), reason="set BTC_UPDOWN_NETWORK=1 to run network tests"
)

T0 = 1735689600  # 2025-01-01 00:00:00 UTC


def _row(ts_raw: int, price: float, vol: float = 1.5, quote: float = 150000.0) -> str:
    close_raw = ts_raw + (59_999_999 if ts_raw >= 10**14 else 59_999)
    return (f"{ts_raw},{price:.8f},{price + 10:.8f},{price - 10:.8f},{price + 5:.8f},{vol:.8f},"
            f"{close_raw},{quote:.8f},42,{vol / 2:.8f},{quote / 2:.8f},0")


def make_fake_zip(directory: Path, name: str = "BTCUSDC-1m-2025-01.zip", header: bool = True) -> tuple[Path, list[int]]:
    """Fake kline zip: header row, 3 ms rows, 4 us rows, a duplicated row, a 3-minute gap.

    Returns the zip path and the list of unique bar-open seconds it encodes.
    """
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                     "taker_buy_volume,taker_buy_quote_volume,ignore")
    # minutes 0,1,2 in milliseconds
    for k in range(3):
        lines.append(_row((T0 + 60 * k) * 1000, 93000.0 + k))
    # minute 2 duplicated (same ts, different quoting unit) -> dedup
    lines.append(_row((T0 + 120) * 1_000_000, 93002.0))
    # minutes 3 and 4 in microseconds, then a 3-minute gap (5,6,7 missing), then 8 and 9
    for k in (3, 4, 8, 9):
        lines.append(_row((T0 + 60 * k) * 1_000_000, 93000.0 + k))
    csv = "\n".join(lines) + "\n"
    path = directory / name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name.replace(".zip", ".csv"), csv)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / (name + ".CHECKSUM")).write_text(f"{digest}  {name}")  # Binance spot style: no newline
    expected = [T0 + 60 * k for k in (0, 1, 2, 3, 4, 8, 9)]
    return path, expected


# --------------------------------------------------------------------------- unit detection


def test_to_seconds_detects_unit_per_row():
    raw = np.array([T0, T0 * 1000, T0 * 1_000_000, 1546300800000, 1792800000000000], dtype=np.int64)
    out = bk.to_seconds(raw)
    assert out.tolist() == [T0, T0, T0, 1546300800, 1792800000]
    assert bk.detect_units(raw) == {"us": 2, "ms": 2, "s": 1}


def test_parse_checksum_with_and_without_newline():
    digest = "a" * 64
    assert bk.parse_checksum(f"{digest}  BTCUSDC-1m-2025-01.zip") == digest
    assert bk.parse_checksum(f"{digest}  BTCUSDC-1m-2025-01.zip\n") == digest
    with pytest.raises(ValueError):
        bk.parse_checksum("not a checksum\n")


# --------------------------------------------------------------------------- parsing


def test_read_kline_zip_header_units_and_amount(tmp_path):
    path, expected = make_fake_zip(tmp_path)
    df, stats = bk.read_kline_zip(path)
    assert list(df.columns) == bk.BAR_COLUMNS
    assert stats["header_rows"] == 1
    assert stats["units"] == {"ms": 3, "us": 5, "s": 0}
    assert stats["non_aligned"] == 0
    assert len(df) == 8  # header dropped, duplicate still present before assembly
    assert df["ts"].iloc[0] == T0 and df["ts"].iloc[-1] == T0 + 540
    # quote_asset_volume is carried as amount; prices parsed as float64
    assert df["amount"].iloc[0] == pytest.approx(150000.0)
    assert df["open"].iloc[1] == pytest.approx(93001.0)
    assert df.dtypes["ts"] == np.int64 and all(df.dtypes[c] == np.float64 for c in bk.BAR_COLUMNS[1:])


def test_read_kline_zip_without_header(tmp_path):
    path, expected = make_fake_zip(tmp_path, header=False)
    df, stats = bk.read_kline_zip(path)
    assert stats["header_rows"] == 0 and len(df) == 8


def test_assemble_dedups_sorts_and_drops_bad_rows(tmp_path):
    path, expected = make_fake_zip(tmp_path)
    df, _ = bk.read_kline_zip(path)
    # shuffle, add a NaN row and a zero-price row that must be dropped
    shuffled = df.sample(frac=1, random_state=0)
    bad = pd.DataFrame({"ts": [T0 + 600, T0 + 660], "open": [np.nan, 0.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
                        "close": [1.0, 1.0], "volume": [1.0, 1.0], "amount": [1.0, 1.0]})
    bars, stats = bk.assemble([shuffled, bad])
    assert bars["ts"].tolist() == expected
    assert bars["ts"].is_monotonic_increasing and bars["ts"].is_unique
    assert stats == {"rows_in": 10, "dropped_nan": 1, "dropped_nonpositive_price": 1, "dropped_duplicates": 1}
    assert not bars.isna().any().any()
    # first occurrence wins on the duplicated minute
    assert bars.loc[bars["ts"] == T0 + 120, "open"].item() == pytest.approx(93002.0)


# --------------------------------------------------------------------------- gaps and stamps


def test_compute_gaps_counts_one_gap_of_three_minutes(tmp_path):
    path, expected = make_fake_zip(tmp_path)
    df, _ = bk.read_kline_zip(path)
    bars, _ = bk.assemble([df])
    gaps = bk.compute_gaps(bars["ts"].to_numpy())
    assert gaps["n_gaps"] == 1
    assert gaps["missing_minutes"] == 3
    assert gaps["irregular"] == 0
    assert gaps["largest"] == [{
        "start": "2025-01-01T00:04:00+00:00", "end": "2025-01-01T00:08:00+00:00", "missing_minutes": 3, "seconds": 240,
    }]
    assert bk.compute_gaps(np.array([T0], dtype=np.int64))["n_gaps"] == 0


def test_compute_gaps_irregular_steps():
    ts = np.array([T0, T0 + 60, T0 + 90, T0 + 150, T0 + 330], dtype=np.int64)
    gaps = bk.compute_gaps(ts)
    assert gaps["n_gaps"] == 2  # diffs 60, 30, 60, 180: the 30 s and 180 s steps
    assert gaps["irregular"] == 1
    assert gaps["missing_minutes"] == 2


def test_stamp_matches_pandas_dt_accessors():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.integers(1_546_300_800, 1_800_000_000, size=5000)) // 60 * 60
    ts = np.concatenate([ts, [T0, 1735689540, 1546300800, 1711929540]]).astype(np.int64)  # year ends, leap day eve
    stamp = bk.compute_stamp(ts)
    s = pd.Series(pd.to_datetime(ts, unit="s", utc=True))
    assert stamp.dtype == np.int16 and stamp.shape == (ts.size, 5)
    np.testing.assert_array_equal(stamp[:, 0], s.dt.minute.to_numpy())
    np.testing.assert_array_equal(stamp[:, 1], s.dt.hour.to_numpy())
    np.testing.assert_array_equal(stamp[:, 2], s.dt.day_of_week.to_numpy())  # == .dt.weekday, Mon=0
    np.testing.assert_array_equal(stamp[:, 3], s.dt.day.to_numpy())
    np.testing.assert_array_equal(stamp[:, 4], s.dt.month.to_numpy())
    assert stamp[-4].tolist() == [0, 0, 2, 1, 1]  # 2025-01-01 is a Wednesday


# --------------------------------------------------------------------------- end-to-end offline


def test_build_series_writes_mmapable_outputs_and_meta(tmp_path):
    out = tmp_path / "klines"
    series = bk.SERIES["spot_BTCUSDC"]
    cache = out / "_zips" / series.name
    cache.mkdir(parents=True)
    path, expected = make_fake_zip(cache)
    archive = bk.Archive(series.name, "monthly", "2025-01", series.monthly_url(2025, 1), path)
    assert bk._verify_local(archive)  # cached zip + CHECKSUM accepted without network
    archive.status = "ok"
    missing = bk.Archive(series.name, "monthly", "2019-01", series.monthly_url(2019, 1), cache / "x.zip", status="missing")

    meta = bk.build_series(series, [archive, missing], out)
    d = out / series.name
    ts = np.load(d / "ts.i64.npy", mmap_mode="r")
    x = np.load(d / "x.f32.npy", mmap_mode="r")
    stamp = np.load(d / "stamp.i16.npy", mmap_mode="r")
    bars = pd.read_parquet(d / "bars.parquet")
    assert ts.dtype == np.int64 and x.dtype == np.float32 and stamp.dtype == np.int16
    assert ts.tolist() == expected and x.shape == (7, 6) and stamp.shape == (7, 5)
    assert list(bars.columns) == bk.BAR_COLUMNS and bars["ts"].tolist() == expected
    np.testing.assert_allclose(np.asarray(x)[:, 0], bars["open"].to_numpy(), rtol=1e-6)

    on_disk = json.loads((d / "meta.json").read_text())
    assert on_disk["n"] == 7 and on_disk["n_gaps"] == 1 and on_disk["missing_minutes"] == 3
    assert on_disk["first_ts"] == "2025-01-01T00:00:00+00:00" and on_disk["last_ts"] == "2025-01-01T00:09:00+00:00"
    assert on_disk["code_version"] == bk.CODE_VERSION
    assert on_disk["source_files"][0]["sha256"] == archive.sha256 and len(on_disk["source_files"][0]["sha256"]) == 64
    assert on_disk["missing_archives"] == [{"kind": "monthly", "key": "2019-01", "url": series.monthly_url(2019, 1)}]
    assert on_disk["assembly"]["dropped_duplicates"] == 1 and on_disk["header_rows_skipped"] == 1
    assert on_disk["output_bytes"]["x.f32.npy"] == (d / "x.f32.npy").stat().st_size
    assert meta["output_bytes"] == on_disk["output_bytes"]

    # --check mode accepts what was just written and is idempotent on re-run
    assert bk.check_outputs(d)["n"] == 7
    assert bk.main(["--series", series.name, "--out", str(out), "--check"]) == 0
    meta2 = bk.build_series(series, [archive], out)
    assert meta2["n"] == 7


def test_check_outputs_detects_corruption(tmp_path):
    out = tmp_path / "klines"
    series = bk.SERIES["spot_BTCUSDT"]
    cache = out / "_zips" / series.name
    cache.mkdir(parents=True)
    path, _ = make_fake_zip(cache, name="BTCUSDT-1m-2025-01.zip")
    archive = bk.Archive(series.name, "monthly", "2025-01", series.monthly_url(2025, 1), path, status="ok")
    bk.build_series(series, [archive], out)
    d = out / series.name
    stamp = np.load(d / "stamp.i16.npy")
    stamp[0, 1] += 1
    np.save(d / "stamp.i16.npy", stamp)
    with pytest.raises(RuntimeError, match="stamp"):
        bk.check_outputs(d)


# --------------------------------------------------------------------------- planning (no network)


class _FakeFetcher:
    """Pretends monthly files exist except for 2019-01 and the current month; dailies exist before today."""

    def __init__(self, today: dt.date):
        self.today = today
        self.requested: list[str] = []

    def get(self, url: str, dest: Path) -> str:
        self.requested.append(url)
        name = url.rsplit("/", 1)[1].replace(".CHECKSUM", "")
        key = name.split("-1m-")[1].replace(".zip", "")
        if "/monthly/" in url:
            if key == "2019-01" or key == f"{self.today.year:04d}-{self.today.month:02d}":
                return "missing"
        else:
            if dt.date.fromisoformat(key) >= self.today:
                return "missing"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".CHECKSUM"):
            dest.write_text("0" * 64 + "  " + name)
        else:
            dest.write_bytes(b"zip")
        return "ok"


def test_plan_downloads_monthly_then_daily_fallback_only_for_recent_months(tmp_path, monkeypatch):
    today = dt.date(2026, 10, 25)
    fetcher = _FakeFetcher(today)
    # sha256 of b"zip" is not 64 zeros: short-circuit verification so the planner logic is what we test
    monkeypatch.setattr(bk, "sha256_file", lambda p: "0" * 64)
    series = bk.SERIES["um_BTCUSDT"]
    months = bk.month_range((2019, 1), (2026, 12))  # end past today must be clamped by the planner
    result = bk.plan_and_download([series], months, tmp_path, fetcher, workers=2, today=today, daily_fallback_months=2)
    archives = result[series.name]
    monthly = [a for a in archives if a.kind == "monthly"]
    daily = [a for a in archives if a.kind == "daily"]
    assert [a.key for a in monthly][0] == "2019-01" and [a.key for a in monthly][-1] == "2026-10"
    assert len(monthly) == 94  # 2019-01 .. 2026-10, nothing in the future
    assert {a.status for a in monthly if a.key not in ("2019-01", "2026-10")} == {"ok"}
    # old 404 recorded as missing without probing days; current month probed day by day up to yesterday
    assert [a.key for a in daily] == [dt.date(2026, 10, d).isoformat() for d in range(1, 25)]
    assert all(a.status == "ok" for a in daily)
    assert not any("2019-01-" in u for u in fetcher.requested)
    assert not any("2026-10-25" in u for u in fetcher.requested)
    assert all(u.startswith(bk.BASE_URL) for u in fetcher.requested)
    # idempotent: a second run touches nothing on the network
    fetcher.requested.clear()
    bk.plan_and_download([series], months, tmp_path, fetcher, workers=2, today=today, daily_fallback_months=2)
    assert fetcher.requested  # the two absent monthly files are re-probed (they may appear later)
    assert all("2019-01.zip" in u or "2026-10.zip" in u for u in fetcher.requested)


def test_month_helpers():
    assert bk.month_range((2024, 11), (2025, 2)) == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
    assert bk.months_back((2026, 1), 2) == (2025, 11)
    assert bk.months_back((2026, 10), 0) == (2026, 10)
    assert bk.parse_month("2019-01") == (2019, 1)
    with pytest.raises(ValueError):
        bk.parse_month("2019-13")


def test_fetcher_refuses_other_hosts(tmp_path):
    with pytest.raises(ValueError):
        bk.Fetcher().get("https://example.com/x.zip", tmp_path / "x.zip")


# --------------------------------------------------------------------------- network (opt-in)


@network
@skip_without_network
def test_download_one_real_daily_file(tmp_path):
    series = bk.SERIES["spot_BTCUSDC"]
    day = dt.date(2026, 10, 24)
    archive = bk.Archive(series.name, "daily", day.isoformat(), series.daily_url(day), tmp_path / series.daily_name(day))
    bk.fetch_archive(archive, bk.Fetcher())
    assert archive.status == "ok" and archive.sha256
    df, stats = bk.read_kline_zip(archive.path)
    assert len(df) == 1440 and stats["units"]["us"] == 1440
