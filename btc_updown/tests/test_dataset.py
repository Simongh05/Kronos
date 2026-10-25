import numpy as np
import torch

from btc_updown.train.dataset import (
    WindowDataset,
    direct_head_features,
    direct_head_targets,
    normalize_window,
    strike_horizon_pairs,
    typical_price,
)
from .conftest import synthetic_series


def test_valid_starts_reject_gaps_and_respect_range(store):
    W = 50
    starts = store.valid_starts(W)
    ts = np.asarray(store.ts)
    # every returned window is contiguous
    assert np.all(ts[starts + W - 1] - ts[starts] == 60 * (W - 1))
    # no window straddles the gap at row 2000
    assert not np.any((starts < 2000) & (starts + W - 1 >= 2000))
    # count: rows 0..2000-W plus rows 2000..n-W
    n = len(store)
    assert len(starts) == (2000 - W + 1) + (n - 2000 - W + 1)
    # time range bounds: first open >= t_min, last close <= t_max
    t_min, t_max = int(ts[100]), int(ts[700]) + 60
    s2 = store.valid_starts(W, t_min, t_max)
    assert s2.min() == 100 and ts[s2.max() + W - 1] + 60 <= t_max


def test_index_of_bar_closing_at(store):
    ts = np.asarray(store.ts)
    assert store.index_of_bar_closing_at(int(ts[10]) + 60) == 10
    assert store.index_of_bar_closing_at(int(ts[10]) + 61) == -1
    # inside the gap: no bar
    assert store.index_of_bar_closing_at(int(ts[1999]) + 120) == -1


def test_normalization_uses_lookback_only():
    x, _, _ = synthetic_series(n=300, gap_at=None)
    L = 200
    w = x[:L + 15].copy()
    z1, m1, s1 = normalize_window(w, L, 5.0)
    w2 = w.copy()
    w2[L:] *= 10.0  # corrupt the future bars
    z2, m2, s2 = normalize_window(w2, L, 5.0)
    assert np.allclose(m1, m2) and np.allclose(s1, s2)
    assert np.allclose(z1[:L], z2[:L])
    assert np.abs(z2).max() <= 5.0


def test_window_dataset_items_and_determinism(store):
    ds = WindowDataset([store], lookback=64, predict=15, n_samples=100, seed=1)
    a = ds[7]
    assert a["x"].shape == (79, 6) and a["stamp"].shape == (79, 5) and a["tp"].shape == (79,)
    assert float(a["x"][:64].mean()) == pytest.approx(0.0, abs=1e-3)
    # same (idx, epoch) -> same window; different epoch -> (almost surely) different window
    b = ds[7]
    assert int(a["start"]) == int(b["start"])
    ds.set_epoch(1)
    c = ds[7]
    assert int(c["start"]) != int(a["start"])
    # grid mode is independent of epoch
    dv = WindowDataset([store], lookback=64, predict=15, n_samples=10, deterministic_grid=True)
    s0 = [int(dv[i]["start"]) for i in range(10)]
    dv.set_epoch(5)
    assert s0 == [int(dv[i]["start"]) for i in range(10)]
    assert len(set(s0)) == 10


def test_direct_head_targets_and_pairs():
    pairs = strike_horizon_pairs(15)
    assert pairs.shape[0] == 120 and np.all(pairs[:, 0] + pairs[:, 1] <= 15) and pairs[:, 1].min() == 1
    L = 20
    tp = torch.arange(1.0, 1.0 + L + 15)[None, :]  # strictly increasing typical price
    labels, strike = direct_head_targets(tp, L, pairs)
    assert labels.shape == (1, 120) and torch.all(labels == 1.0)  # future always above any past strike
    assert torch.allclose(strike[0], tp[0, L - 1 - torch.as_tensor(pairs[:, 0])])
    labels2, _ = direct_head_targets(-tp, L, pairs)
    assert torch.all(labels2 == 0.0)
    feats = direct_head_features(torch.tensor([100.0]), torch.tensor([1e-3]), strike, pairs)
    assert feats.shape == (1, 120, 5) and torch.isfinite(feats).all()


def test_typical_price():
    x = np.array([[1.0, 3.0, 0.0, 2.0, 9, 9]], dtype=np.float32)
    assert typical_price(x)[0] == 1.5


import pytest  # noqa: E402  (used by approx above)
