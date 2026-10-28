"""Memmap-backed window dataset for fine-tuning Kronos on 1-minute bars.

Data contract (produced by ``btc_updown.data.binance_klines``), one directory per series::

    <root>/<series>/x.f32.npy      float32 [N, 6]  open, high, low, close, volume, amount
    <root>/<series>/stamp.i16.npy  int16   [N, 5]  minute, hour, weekday, day, month (UTC)
    <root>/<series>/ts.i64.npy     int64   [N]     bar OPEN time, UTC seconds, sorted, unique

A window of ``window`` bars starting at row ``i`` is valid only if the bars are contiguous
1-minute steps, i.e. ``ts[i + window - 1] - ts[i] == 60 * (window - 1)``, and (optionally) if the
whole window lies inside a time range.  Normalization uses the mean/std of the *lookback* part
only (never the future bars), as ``finetune/dataset.py`` does, so training matches inference.

Sampling is a deterministic function of ``(idx, epoch)`` so that a resumed run reproduces the
exact sample sequence of an uninterrupted one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURES = ("open", "high", "low", "close", "volume", "amount")
TIME_FEATURES = ("minute", "hour", "weekday", "day", "month")
BAR_SECONDS = int(os.environ.get("BTC_UPDOWN_BAR_SECONDS", "60"))   # 60 for 1-minute bars; 3600 for the h1_* hourly series
EPS = 1e-5


def typical_price(x: np.ndarray) -> np.ndarray:
    """(O+H+L+C)/4 per bar; proxy for the 60-second settlement average of that minute."""
    return x[..., :4].mean(axis=-1)


@dataclass
class SeriesStore:
    """One kline series opened as read-only memmaps."""

    name: str
    x: np.ndarray       # [N, 6] float32 (memmap)
    stamp: np.ndarray   # [N, 5] int16 (memmap)
    ts: np.ndarray      # [N] int64 (memmap)

    @classmethod
    def open(cls, root: str, name: str) -> "SeriesStore":
        d = os.path.join(root, name)
        x = np.load(os.path.join(d, "x.f32.npy"), mmap_mode="r")
        stamp = np.load(os.path.join(d, "stamp.i16.npy"), mmap_mode="r")
        ts = np.load(os.path.join(d, "ts.i64.npy"), mmap_mode="r")
        if not (x.shape[0] == stamp.shape[0] == ts.shape[0]):
            raise ValueError(f"{name}: inconsistent lengths x={x.shape} stamp={stamp.shape} ts={ts.shape}")
        if x.shape[1] != 6 or stamp.shape[1] != 5:
            raise ValueError(f"{name}: expected x[N,6] and stamp[N,5], got {x.shape} {stamp.shape}")
        meta_path = os.path.join(d, "meta.json")
        if os.path.exists(meta_path):
            interval = json.load(open(meta_path)).get("interval", "1m")
            secs = {"1m": 60, "1h": 3600}.get(interval)
            if secs is not None and secs != BAR_SECONDS:
                raise ValueError(f"{name}: series interval {interval} but BAR_SECONDS={BAR_SECONDS}; set BTC_UPDOWN_BAR_SECONDS={secs}")
        return cls(name=name, x=x, stamp=stamp, ts=ts)

    @classmethod
    def from_arrays(cls, name: str, x: np.ndarray, stamp: np.ndarray, ts: np.ndarray) -> "SeriesStore":
        return cls(name=name, x=np.asarray(x, dtype=np.float32), stamp=np.asarray(stamp, dtype=np.int16),
                   ts=np.asarray(ts, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def valid_starts(self, window: int, t_min: Optional[int] = None, t_max: Optional[int] = None) -> np.ndarray:
        """Row indices ``i`` such that bars ``i .. i+window-1`` are contiguous and inside [t_min, t_max].

        ``t_min`` / ``t_max`` are UTC seconds and bound the OPEN time of the first bar and the CLOSE
        time (open + 60) of the last bar respectively, so a window never straddles a split boundary.
        """
        n = len(self)
        if n < window:
            return np.zeros(0, dtype=np.int64)
        ts = np.asarray(self.ts)
        span = ts[window - 1:] - ts[: n - window + 1]
        ok = span == BAR_SECONDS * (window - 1)
        if t_min is not None:
            ok &= ts[: n - window + 1] >= t_min
        if t_max is not None:
            ok &= (ts[window - 1:] + BAR_SECONDS) <= t_max
        return np.nonzero(ok)[0].astype(np.int64)

    def index_of_bar_closing_at(self, t: int) -> int:
        """Row of the bar whose CLOSE time is ``t`` (open = t - 60); -1 if absent."""
        i = int(np.searchsorted(self.ts, t - BAR_SECONDS))
        if i < len(self) and int(self.ts[i]) == t - BAR_SECONDS:
            return i
        return -1


def normalize_window(x: np.ndarray, lookback: int, clip: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score the whole window with statistics of the first ``lookback`` bars only."""
    past = x[:lookback]
    mean = past.mean(axis=0)
    std = past.std(axis=0)
    z = (x - mean) / (std + EPS)
    z = np.clip(z, -clip, clip).astype(np.float32)
    return z, mean.astype(np.float32), std.astype(np.float32)


class WindowDataset(Dataset):
    """Windows of ``lookback + predict`` bars drawn from several series.

    Each item is a dict:
        x       float32 [W, 6]   normalized (lookback stats), clipped
        stamp   float32 [W, 5]   time features
        tp      float32 [W]      raw typical price of every bar (for DirectHead labels / strikes)
        close_t float32          raw close of the last lookback bar (decision time t)
        sigma1  float32          std of 1-minute log returns of close over the lookback (vol scale)
        series  int64            series index
        start   int64            row index of the first bar in its series
    """

    def __init__(
        self,
        stores: Sequence[SeriesStore],
        lookback: int,
        predict: int,
        n_samples: int,
        clip: float = 5.0,
        seed: int = 100,
        t_min: Optional[int] = None,
        t_max: Optional[int] = None,
        deterministic_grid: bool = False,
    ):
        if lookback < 2 or predict < 1:
            raise ValueError("lookback must be >= 2 and predict >= 1")
        self.stores = list(stores)
        self.lookback = int(lookback)
        self.predict = int(predict)
        self.window = self.lookback + self.predict
        self.clip = float(clip)
        self.seed = int(seed)
        self.epoch = 0
        self.deterministic_grid = deterministic_grid

        self.valid: List[np.ndarray] = [s.valid_starts(self.window, t_min, t_max) for s in self.stores]
        counts = np.array([len(v) for v in self.valid], dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(counts)])
        self.total_valid = int(self.cum[-1])
        if self.total_valid == 0:
            raise ValueError("no valid windows in the requested range")
        self.n_samples = int(n_samples) if n_samples > 0 else self.total_valid

    # -- sampling -----------------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_index(self, idx: int) -> int:
        if self.deterministic_grid:
            # evenly spread fixed subset (validation): same windows every epoch
            return int((idx * self.total_valid) // self.n_samples) % self.total_valid
        # large-prime hash of (idx, epoch, seed): deterministic, roughly uniform, resumable
        g = (idx * 9973 + (self.epoch + 1) * 104729 + self.seed * 7919) % self.total_valid
        return int(g)

    def locate(self, g: int) -> Tuple[int, int]:
        s = int(np.searchsorted(self.cum, g, side="right") - 1)
        start = int(self.valid[s][g - self.cum[s]])
        return s, start

    def __len__(self) -> int:
        return self.n_samples

    def fetch(self, s: int, start: int) -> Dict[str, torch.Tensor]:
        store = self.stores[s]
        end = start + self.window
        x = np.asarray(store.x[start:end], dtype=np.float32)
        stamp = np.asarray(store.stamp[start:end], dtype=np.float32)
        z, _, _ = normalize_window(x, self.lookback, self.clip)
        tp = typical_price(x).astype(np.float32)
        close = x[: self.lookback, 3]
        lr = np.diff(np.log(np.clip(close, 1e-9, None)))
        sigma1 = float(lr.std()) if lr.size > 1 else 0.0
        return {
            "x": torch.from_numpy(z),
            "stamp": torch.from_numpy(stamp),
            "tp": torch.from_numpy(tp),
            "close_t": torch.tensor(float(close[-1]), dtype=torch.float32),
            "sigma1": torch.tensor(max(sigma1, 1e-6), dtype=torch.float32),
            "series": torch.tensor(s, dtype=torch.int64),
            "start": torch.tensor(start, dtype=torch.int64),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s, start = self.locate(self._global_index(idx))
        return self.fetch(s, start)


# -- strike / horizon grid for the DirectHead ----------------------------------------------------
def strike_horizon_pairs(max_minutes: int = 15) -> np.ndarray:
    """All (j, h) with j >= 0 minutes since window start, h >= 1 minutes to expiry, j + h <= max_minutes.

    j is how many bars ago the window started (its strike is the typical price of the bar that closed
    at the window start); h is how many bars ahead settlement is.  Real markets: 5m => j + h == 5,
    15m => j + h == 15; the grid covers both plus intermediate pairs for augmentation.
    """
    pairs = [(j, h) for j in range(0, max_minutes) for h in range(1, max_minutes + 1) if j + h <= max_minutes]
    return np.array(pairs, dtype=np.int64)


def direct_head_targets(tp: torch.Tensor, lookback: int, pairs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Labels and strikes for every (j, h) pair.

    tp: [B, W] raw typical prices; the decision bar is index lookback-1.
    Returns (labels [B, P] float in {0,1}, strikes [B, P] float).
    """
    t_idx = lookback - 1
    j = torch.as_tensor(pairs[:, 0], device=tp.device)
    h = torch.as_tensor(pairs[:, 1], device=tp.device)
    strike = tp[:, t_idx - j]            # [B, P]
    future = tp[:, t_idx + h]            # [B, P]
    labels = (future >= strike).to(torch.float32)
    return labels, strike


def direct_head_features(close_t: torch.Tensor, sigma1: torch.Tensor, strike: torch.Tensor, pairs: np.ndarray,
                         max_h: int = 15) -> torch.Tensor:
    """Numeric features per (j, h): [(K-c)/(c*sigma1*sqrt(h)), (K-c)/c in bps/100, log sigma1, h/max_h, j/max_h]."""
    j = torch.as_tensor(pairs[:, 0], device=close_t.device, dtype=torch.float32)
    h = torch.as_tensor(pairs[:, 1], device=close_t.device, dtype=torch.float32)
    c = close_t[:, None]
    s = sigma1[:, None]
    rel = (strike - c) / c                                   # relative distance to strike
    f0 = rel / (s * torch.sqrt(h)[None, :])                  # in units of expected h-minute vol
    f1 = rel * 100.0                                         # in percent
    f2 = torch.log(s).expand_as(rel)
    f3 = (h / float(max_h))[None, :].expand_as(rel)
    f4 = (j / float(max_h))[None, :].expand_as(rel)
    feats = torch.stack([f0, f1, f2, f3, f4], dim=-1)
    return torch.nan_to_num(feats, nan=0.0, posinf=50.0, neginf=-50.0).clamp_(-50.0, 50.0)


def load_stores(root: str, names: Sequence[str]) -> List[SeriesStore]:
    return [SeriesStore.open(root, n) for n in names]


def describe_stores(stores: Sequence[SeriesStore]) -> str:
    rows = []
    for s in stores:
        if len(s) == 0:
            rows.append(f"{s.name}: empty")
            continue
        t0 = np.datetime64(int(s.ts[0]), "s")
        t1 = np.datetime64(int(s.ts[-1]), "s")
        rows.append(f"{s.name}: {len(s):,} bars, {t0} .. {t1}")
    return "\n".join(rows)


def iso_to_epoch(s: Optional[str]) -> Optional[int]:
    """'YYYY-MM-DD' or ISO datetime (UTC) -> epoch seconds; None passes through."""
    if s is None or s == "":
        return None
    return int(np.datetime64(s, "s").astype("datetime64[s]").astype(np.int64))


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
