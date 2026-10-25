"""Tokenizer round-trip control: can the pretrained BSQ codebook resolve 1-minute BTC moves?

Encodes real 1-minute windows (lookback-only z-score, as at inference), decodes them, and reports
the reconstruction error of the close and typical price in dollars and in basis points, against the
median |final - strike| of the markets (~$58, KALSHI-15M.md).  Large error => the sampling readout
is hopeless without tokenizer fine-tuning / a shorter lookback.

    python -m btc_updown.infer.roundtrip --root data/klines --series spot_BTCUSDC --start 2026-08-07 --end 2026-09-07 \
        --lookbacks 512 256 --n 400
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..train.dataset import SeriesStore, iso_to_epoch, normalize_window, typical_price
from ..train.model_io import load_tokenizer, pick_device


def run(root: str, series: str, start: str, end: str, lookbacks, n: int, tokenizer: str, revision, device_name: str, seed: int = 0):
    store = SeriesStore.open(root, series)
    device = pick_device(device_name)
    tok = load_tokenizer(tokenizer, revision).to(device).float().eval()
    rng = np.random.RandomState(seed)
    out = {}
    for L in lookbacks:
        starts = store.valid_starts(L, iso_to_epoch(start), iso_to_epoch(end))
        sel = np.sort(rng.choice(starts, size=min(n, len(starts)), replace=False))
        err_close, err_tp, err_close_bps, err_tp_bps, moves = [], [], [], [], []
        used_s1, used_full = set(), set()
        for b0 in range(0, len(sel), 32):
            xs, means, stds = [], [], []
            for s in sel[b0:b0 + 32]:
                x = np.asarray(store.x[s:s + L], dtype=np.float32)
                z, m, sd = normalize_window(x, L, 5.0)
                xs.append(z); means.append(m); stds.append(sd)
            xt = torch.from_numpy(np.stack(xs)).to(device)
            with torch.no_grad():
                ids = tok.encode(xt, half=True)
                rec = tok.decode(ids, half=True).cpu().numpy()
            used_s1.update(ids[0].reshape(-1).tolist()); used_full.update((ids[0] * 1024 + ids[1]).reshape(-1).tolist())
            mean = np.stack(means)[:, None, :]; sd = np.stack(stds)[:, None, :]
            x_raw = np.stack(xs) * (sd + 1e-5) + mean
            rec_raw = rec * (sd + 1e-5) + mean
            # last 16 bars of each window (where a forecast would be read)
            c, cr = x_raw[:, -16:, 3], rec_raw[:, -16:, 3]
            tp, tpr = typical_price(x_raw[:, -16:]), typical_price(rec_raw[:, -16:])
            err_close.append(np.abs(c - cr).ravel()); err_tp.append(np.abs(tp - tpr).ravel())
            err_close_bps.append((np.abs(c - cr) / c * 1e4).ravel()); err_tp_bps.append((np.abs(tp - tpr) / tp * 1e4).ravel())
            moves.append(np.abs(np.diff(x_raw[:, -16:, 3], axis=1)).ravel())
        ec, et = np.concatenate(err_close), np.concatenate(err_tp)
        ecb, etb = np.concatenate(err_close_bps), np.concatenate(err_tp_bps)
        mv = np.concatenate(moves)
        q = lambda a: {k: float(np.quantile(a, v)) for k, v in (("p50", .5), ("p75", .75), ("p90", .9), ("p99", .99))}
        out[f"lookback_{L}"] = {
            "n_windows": int(len(sel)),
            "close_err_usd": q(ec), "tp_err_usd": q(et), "close_err_bps": q(ecb), "tp_err_bps": q(etb),
            "abs_1min_close_move_usd": q(mv),
            "share_tp_err_gt_58usd": float((et > 58).mean()),
            "share_tp_err_gt_1min_move": float((et > np.median(mv)).mean()),
            "distinct_s1_tokens": len(used_s1), "distinct_full_tokens": len(used_full),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/klines"); ap.add_argument("--series", default="spot_BTCUSDC")
    ap.add_argument("--start", default="2026-08-07"); ap.add_argument("--end", default="2026-09-07")
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base"); ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.root, a.series, a.start, a.end, a.lookbacks, a.n, a.tokenizer, a.revision, a.device)
    s = json.dumps(res, indent=2)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
