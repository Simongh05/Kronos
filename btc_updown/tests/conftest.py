"""Shared fixtures: tiny random Kronos models and a synthetic 1-minute series (no downloads)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model import Kronos, KronosTokenizer  # noqa: E402
from btc_updown.train.dataset import SeriesStore  # noqa: E402

S1_BITS = 4
S2_BITS = 4


def pytest_configure(config):
    config.addinivalue_line("markers", "network: tests that need internet access (skipped by default)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m", default=None):
        return
    skip = pytest.mark.skip(reason="network test; run with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


def make_tokenizer(seed: int = 0) -> KronosTokenizer:
    torch.manual_seed(seed)
    return KronosTokenizer(d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=2, n_dec_layers=2,
                           ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
                           s1_bits=S1_BITS, s2_bits=S2_BITS, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)


def make_model(seed: int = 0, dropout: float = 0.0) -> Kronos:
    torch.manual_seed(seed)
    return Kronos(s1_bits=S1_BITS, s2_bits=S2_BITS, n_layers=1, d_model=16, n_heads=2, ff_dim=32,
                  ffn_dropout_p=dropout, attn_dropout_p=0.0, resid_dropout_p=dropout, token_dropout_p=0.0, learn_te=True)


def synthetic_series(n: int = 4000, t0: int = 1_700_000_040, seed: int = 0, gap_at: int | None = 2000, gap_len: int = 7):
    """Random-walk OHLCVA at 1-minute steps with one gap of `gap_len` missing bars after row `gap_at`."""
    rng = np.random.RandomState(seed)
    ts = t0 + 60 * np.arange(n, dtype=np.int64)
    if gap_at is not None:
        ts[gap_at:] += 60 * gap_len
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1e-4, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1e-4, n)))
    vol = rng.lognormal(0, 0.5, n)
    x = np.stack([open_, hi, lo, close, vol, vol * close], axis=1).astype(np.float32)
    dt = pd.to_datetime(ts, unit="s", utc=True)
    stamp = np.stack([dt.minute, dt.hour, dt.weekday, dt.day, dt.month], axis=1).astype(np.int16)
    return x, stamp, ts


def write_store(root: str, name: str, x, stamp, ts) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.f32.npy"), x)
    np.save(os.path.join(d, "stamp.i16.npy"), stamp)
    np.save(os.path.join(d, "ts.i64.npy"), ts)
    return d


@pytest.fixture
def tiny_models(tmp_path):
    tok = make_tokenizer()
    model = make_model()
    tok_dir = tmp_path / "tok"
    model_dir = tmp_path / "model"
    tok.save_pretrained(str(tok_dir))
    model.save_pretrained(str(model_dir))
    return {"tokenizer": tok, "model": model, "tok_dir": str(tok_dir), "model_dir": str(model_dir)}


@pytest.fixture
def store_root(tmp_path):
    x, stamp, ts = synthetic_series()
    root = tmp_path / "klines"
    write_store(str(root), "synthetic", x, stamp, ts)
    return str(root)


@pytest.fixture
def store(store_root):
    return SeriesStore.open(store_root, "synthetic")
