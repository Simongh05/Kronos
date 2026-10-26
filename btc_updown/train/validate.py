"""Validation: leakage-free CE, DirectHead metrics on the (j, h) grid, and the real-market task metric."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    BAR_SECONDS,
    SeriesStore,
    direct_head_features,
    direct_head_targets,
    normalize_window,
    strike_horizon_pairs,
    typical_price,
)
from .heads import DirectHead, brier
from .model_io import autocast_ctx, backbone_hidden, dual_logits, encode_tokens


@torch.no_grad()
def evaluate_windows(model, tokenizer, head: Optional[DirectHead], loader: DataLoader, lookback: int,
                     device: torch.device, bf16: bool) -> Dict[str, float]:
    """Teacher-forced, causal CE in eval mode plus DirectHead BCE/Brier/accuracy on the (j,h) grid."""
    was_training = model.training
    model.eval()
    if head is not None:
        head.eval()
    pairs = strike_horizon_pairs(head.max_h if head is not None else 15)
    h_idx = torch.as_tensor(pairs[:, 1], device=device)
    sum_jh = pairs[:, 0] + pairs[:, 1]
    sel5 = torch.as_tensor(sum_jh == 5, device=device)
    sel15 = torch.as_tensor(sum_jh == 15, device=device)

    n = 0
    tot = {"ce": 0.0, "ce_s1": 0.0, "ce_s2": 0.0}
    hsum = {"bce": 0.0, "brier": 0.0, "acc": 0.0, "brier_5": 0.0, "brier_15": 0.0, "base_rate": 0.0}
    for batch in loader:
        x = batch["x"].to(device)
        stamp = batch["stamp"].to(device)
        s1, s2 = encode_tokens(tokenizer, x)
        s1_in, s2_in, st_in = s1[:, :-1], s2[:, :-1], stamp[:, :-1]
        s1_t, s2_t = s1[:, 1:], s2[:, 1:]
        with autocast_ctx(device, bf16):
            hidden = backbone_hidden(model, s1_in, s2_in, st_in)
            l1, l2 = dual_logits(model, hidden, s1_t, teacher_forcing=True, causal=True)
            ce, ce1, ce2 = model.head.compute_loss(l1.float(), l2.float(), s1_t, s2_t)
        b = x.shape[0]
        tot["ce"] += float(ce) * b
        tot["ce_s1"] += float(ce1) * b
        tot["ce_s2"] += float(ce2) * b
        if head is not None:
            tp = batch["tp"].to(device)
            labels, strike = direct_head_targets(tp, lookback, pairs)
            feats = direct_head_features(batch["close_t"].to(device), batch["sigma1"].to(device), strike, pairs, head.max_h)
            ctx = hidden[:, lookback - 1].float()
            logits = head(ctx, h_idx, feats)
            p = torch.sigmoid(logits)
            hsum["bce"] += float(DirectHead.loss(logits, labels)) * b
            hsum["brier"] += float(brier(p, labels)) * b
            hsum["acc"] += float(((p > 0.5).float() == labels).float().mean()) * b
            hsum["brier_5"] += float(brier(p[:, sel5], labels[:, sel5])) * b
            hsum["brier_15"] += float(brier(p[:, sel15], labels[:, sel15])) * b
            hsum["base_rate"] += float(labels.mean()) * b
        n += b
    out = {f"val/{k}": v / max(n, 1) for k, v in tot.items()}
    if head is not None:
        out.update({f"val/head_{k}": v / max(n, 1) for k, v in hsum.items()})
    out["val/n_windows"] = float(n)
    if was_training:
        model.train()
        if head is not None:
            head.train()
    return out


def market_decision_points(windows, t_min: int, t_max: int, n_points: int, seed: int) -> np.ndarray:
    """Fixed subsample of (window_start, window_end, t, outcome) rows from a labels frame.

    `windows` is a pandas DataFrame with columns window_start, window_end (UTC s), outcome in {0,1}.
    Decision times are bar closes t = start + 60k with k >= 1 and at least 60 s to expiry.
    """
    w = windows[(windows["window_start"] >= t_min) & (windows["window_end"] <= t_max) & (windows["outcome"].isin([0, 1]))]
    rows = []
    for ws, we, oc in zip(w["window_start"].to_numpy(), w["window_end"].to_numpy(), w["outcome"].to_numpy()):
        dur = int((we - ws) // BAR_SECONDS)
        for k in range(1, dur):
            rows.append((int(ws), int(we), int(ws + BAR_SECONDS * k), int(oc)))
    if not rows:
        return np.zeros((0, 4), dtype=np.int64)
    arr = np.array(rows, dtype=np.int64)
    rng = np.random.RandomState(seed)
    if len(arr) > n_points:
        arr = arr[np.sort(rng.choice(len(arr), size=n_points, replace=False))]
    return arr


@torch.no_grad()
def evaluate_markets(model, tokenizer, head: DirectHead, store: SeriesStore, points: np.ndarray, lookback: int,
                     clip: float, device: torch.device, bf16: bool, batch_size: int = 64) -> Dict[str, float]:
    """Brier / log-loss of the DirectHead on real market windows.

    For each point (window_start, window_end, t, outcome): context = lookback bars closing at <= t,
    K = typical price of the bar that closed at window_start (the strike in the series' own price level,
    i.e. the exact-strike basis is assumed constant over the window), h = minutes from t to expiry.
    """
    if points.shape[0] == 0:
        return {}
    was_training = model.training
    model.eval()
    head.eval()
    probs, labels = [], []
    skipped = 0
    for i0 in range(0, len(points), batch_size):
        chunk = points[i0:i0 + batch_size]
        xs, stamps, tps, closes, sigmas, hs, ks, ocs = [], [], [], [], [], [], [], []
        for ws, we, t, oc in chunk:
            it = store.index_of_bar_closing_at(int(t))
            i0b = store.index_of_bar_closing_at(int(ws))
            if it < 0 or i0b < 0 or it - lookback + 1 < 0:
                skipped += 1
                continue
            start = it - lookback + 1
            if int(store.ts[it]) - int(store.ts[start]) != BAR_SECONDS * (lookback - 1):
                skipped += 1
                continue
            x = np.asarray(store.x[start:it + 1], dtype=np.float32)
            z, _, _ = normalize_window(x, lookback, clip)
            xs.append(z)
            stamps.append(np.asarray(store.stamp[start:it + 1], dtype=np.float32))
            close = x[:, 3]
            lr = np.diff(np.log(np.clip(close, 1e-9, None)))
            closes.append(float(close[-1]))
            sigmas.append(max(float(lr.std()), 1e-6))
            ks.append(float(typical_price(np.asarray(store.x[i0b:i0b + 1], dtype=np.float32))[0]))
            hs.append(int((we - t) // BAR_SECONDS))
            ocs.append(int(oc))
        if not xs:
            continue
        x = torch.from_numpy(np.stack(xs)).to(device)
        stamp = torch.from_numpy(np.stack(stamps)).to(device)
        s1, s2 = encode_tokens(tokenizer, x)
        with autocast_ctx(device, bf16):
            hidden = backbone_hidden(model, s1, s2, stamp)
        ctx = hidden[:, -1].float()
        h = torch.as_tensor(hs, device=device).clamp(1, head.max_h)
        close_t = torch.as_tensor(closes, device=device, dtype=torch.float32)
        sigma1 = torch.as_tensor(sigmas, device=device, dtype=torch.float32)
        strike = torch.as_tensor(ks, device=device, dtype=torch.float32)
        # per-row (j, h) features with P = 1; j (minutes since window start) enters only as j/15 and
        # is set to 0 here because the strike is taken from the real window start, not from the grid
        rel = (strike - close_t) / close_t
        feats = torch.stack([
            rel / (sigma1 * torch.sqrt(h.float())),
            rel * 100.0,
            torch.log(sigma1),
            h.float() / float(head.max_h),
            torch.zeros_like(rel),
        ], dim=-1)[:, None, :]
        feats = torch.nan_to_num(feats, nan=0.0, posinf=50.0, neginf=-50.0).clamp_(-50.0, 50.0)
        logits = head(ctx, h[:, None], feats)[:, 0]
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(np.array(ocs, dtype=np.float32))
    if was_training:
        model.train()
        head.train()
    if not probs:
        return {"val/mkt_n": 0.0, "val/mkt_skipped": float(skipped)}
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    return {
        "val/mkt_brier": float(np.mean((p - y) ** 2)),
        "val/mkt_logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "val/mkt_acc": float(np.mean((p > 0.5) == (y > 0.5))),
        "val/mkt_base_rate": float(y.mean()),
        "val/mkt_n": float(len(p)),
        "val/mkt_skipped": float(skipped),
    }
