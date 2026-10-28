"""Resample 1-minute SeriesStore klines into coarser bars in the same layout (SCOPE long-horizon test: hourly bars).

    python -m btc_updown.data.resample --root data/klines --series spot_BTCUSDC spot_BTCUSDT um_BTCUSDT --seconds 3600

Writes <root>/h1_<series>/{bars.parquet, x.f32.npy, stamp.i16.npy, ts.i64.npy, meta.json} with ts = bucket open (UTC seconds,
a multiple of --seconds). A bucket is kept only if its first and last minute bars are present and at least --min-bars minutes
exist (open/close exact, volume and amount summed); missing buckets are gaps that the dataset's contiguity mask rejects.
Readers must run with BTC_UPDOWN_BAR_SECONDS=<seconds> (meta.json carries interval "1h"; SeriesStore.open checks it).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .binance_klines import BAR_COLUMNS, write_outputs

CODE_VERSION = "btc_updown.data.resample/1.0.0"


def resample(bars: pd.DataFrame, seconds: int, min_bars: int) -> pd.DataFrame:
    b = bars.sort_values("ts")
    bucket = (b["ts"].to_numpy(dtype=np.int64) // seconds) * seconds
    g = b.assign(bucket=bucket).groupby("bucket", sort=True)
    agg = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), amount=("amount", "sum"), n=("ts", "size"), first_ts=("ts", "first"), last_ts=("ts", "last"))
    idx = agg.index.to_numpy(dtype=np.int64)
    ok = (agg["n"].to_numpy() >= min_bars) & (agg["first_ts"].to_numpy() == idx) & (agg["last_ts"].to_numpy() == idx + seconds - 60)
    out = agg[ok].reset_index().rename(columns={"bucket": "ts"})[BAR_COLUMNS]
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
    return out


def gaps(ts: np.ndarray, seconds: int) -> dict:
    d = np.diff(ts)
    bad = np.flatnonzero(d != seconds)
    return {"n_gaps": int(len(bad)), "missing_bars": int(((d[bad] // seconds) - 1).sum()) if len(bad) else 0,
            "largest": [{"after": dt.datetime.fromtimestamp(int(ts[i]), dt.timezone.utc).isoformat(), "bars": int(d[i] // seconds - 1)}
                        for i in bad[np.argsort(-d[bad])[:5]]]}


def build(root: Path, series: str, seconds: int, min_bars: int, prefix: str) -> dict:
    src = root / series
    bars = pd.read_parquet(src / "bars.parquet")
    out = resample(bars, seconds, min_bars)
    ts = out["ts"].to_numpy(dtype=np.int64)
    src_meta = json.load(open(src / "meta.json")) if (src / "meta.json").exists() else {}
    name = f"{prefix}{series}"
    meta = {"series": name, "interval": {3600: "1h"}.get(seconds, f"{seconds}s"), "bar_seconds": seconds, "source_series": series,
            "source_code_version": src_meta.get("code_version"), "source_n": int(len(bars)), "min_bars": min_bars,
            "code_version": CODE_VERSION, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "first_ts": dt.datetime.fromtimestamp(int(ts[0]), dt.timezone.utc).isoformat() if len(ts) else None,
            "last_ts": dt.datetime.fromtimestamp(int(ts[-1]), dt.timezone.utc).isoformat() if len(ts) else None,
            "n": int(len(ts)), "gaps": gaps(ts, seconds)}
    write_outputs(root / name, out, meta)
    return {k: meta[k] for k in ("series", "first_ts", "last_ts", "n")} | {"n_gaps": meta["gaps"]["n_gaps"], "missing_bars": meta["gaps"]["missing_bars"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/klines"))
    ap.add_argument("--series", nargs="+", default=["spot_BTCUSDC", "spot_BTCUSDT", "um_BTCUSDT"])
    ap.add_argument("--seconds", type=int, default=3600); ap.add_argument("--min-bars", type=int, default=55)
    ap.add_argument("--prefix", default="h1_")
    a = ap.parse_args(argv)
    for s in a.series:
        print(json.dumps(build(a.root, s, a.seconds, a.min_bars, a.prefix)), flush=True)


if __name__ == "__main__":
    main()
