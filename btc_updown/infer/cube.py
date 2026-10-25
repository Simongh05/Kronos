"""Forecast cube: for every decision minute in a date range, sample N paths (and/or DirectHead logits).

Outputs under <run_dir>/ (one set of files per UTC day, resumable; a day is complete when
<run_dir>/days/<YYYY-MM-DD>.done exists):

    summary/<day>.parquet      per (t, h): tp_mean, tp_sd, close_mean, close_sd, tp_q01..tp_q99 (13 quantiles), n
    tp_samples/<day>.npy       float32 [n_t, pred_len, n_samples]  sorted typical-price samples per horizon
    tp_samples/<day>.t.npy     int64 [n_t] decision times (bar close, UTC seconds)
    paths/<day>.parquet        (optional, store_paths: true) raw OHLCVA per (t, sample, h)
    direct/<day>.parquet       per (t, j, h): logit of P(TP(t+h) >= TP(t-60j)) from the DirectHead

Decision time t is a bar CLOSE; the context is the `lookback` bars closing at <= t (open <= t-60).
Everything is in the series' own price level (Binance); strikes are mapped into that level by the
backtest via the typical price of the bar that closed at the market's window start.

    python -m btc_updown.infer.cube --config btc_updown/configs/cube_e0_small_val.yaml [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.kronos import auto_regressive_inference  # noqa: E402

from ..train.config import Config, load_config
from ..train.dataset import (BAR_SECONDS, SeriesStore, direct_head_features, normalize_window, strike_horizon_pairs,
                             typical_price, write_json)
from ..train.loop import load_head
from ..train.model_io import autocast_ctx, backbone_hidden, encode_tokens, load_model, load_tokenizer, pick_device

QS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])


class _AutocastModel:
    """Proxy that runs the predictor's decode_s1/decode_s2 under bf16 autocast (tokenizer stays fp32)."""

    def __init__(self, model, device: torch.device, enabled: bool):
        self.m = model
        self.device = device
        self.enabled = enabled

    def decode_s1(self, *a, **k):
        with autocast_ctx(self.device, self.enabled):
            logits, ctx = self.m.decode_s1(*a, **k)
        return logits.float(), ctx

    def decode_s2(self, *a, **k):
        with autocast_ctx(self.device, self.enabled):
            logits = self.m.decode_s2(*a, **k)
        return logits.float()


def stamps_for_times(open_times: np.ndarray) -> np.ndarray:
    """[..., 5] float32 minute, hour, weekday, day, month for bar OPEN times (UTC seconds)."""
    dt = pd.to_datetime(open_times.reshape(-1), unit="s", utc=True)
    arr = np.stack([dt.minute, dt.hour, dt.weekday, dt.day, dt.month], axis=-1).astype(np.float32)
    return arr.reshape(*open_times.shape, 5)


def day_decision_times(store: SeriesStore, day_start: int, day_end: int, stride: int, lookback: int) -> np.ndarray:
    """Bar-close times t in [day_start, day_end) on the stride grid with a contiguous lookback context."""
    ts = np.asarray(store.ts)
    cand = np.arange(day_start, day_end, stride, dtype=np.int64)
    idx = np.searchsorted(ts, cand - BAR_SECONDS)
    ok = (idx < len(ts)) & (idx >= lookback - 1)
    idx_c = np.clip(idx, 0, len(ts) - 1)
    ok &= ts[idx_c] == cand - BAR_SECONDS
    start = np.clip(idx_c - lookback + 1, 0, len(ts) - 1)
    ok &= (ts[idx_c] - ts[start]) == BAR_SECONDS * (lookback - 1)
    return cand[ok]


def run(cfg: Config, resume: bool = True, max_days: Optional[int] = None) -> Dict:
    run_dir = cfg.run_dir
    for sub in ("summary", "tp_samples", "direct", "paths", "days"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    write_json(os.path.join(run_dir, "config.json"), cfg.to_dict())
    device = pick_device(cfg.device)
    bf16 = bool(cfg.bf16) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(cfg.seed))

    cc = cfg.cube
    lookback = int(cfg.data.lookback)
    clip = float(cfg.data.clip)
    pred_len = int(cc.pred_len)
    n_samples = int(cc.n_samples)
    stride = int(cc.stride_s)
    do_sampling = bool(cc.sampling)
    do_direct = bool(cc.direct) and bool(cfg.model.get("head"))
    store_paths = bool(cc.get("store_paths", False))
    if not (do_sampling or do_direct):
        raise ValueError("nothing to do: enable cube.sampling and/or cube.direct with model.head")

    store = SeriesStore.open(cfg.data.root, cfg.data.series)
    tokenizer = load_tokenizer(cfg.model.tokenizer, cfg.model.get("tokenizer_revision")).to(device).float().eval()
    model = load_model(cfg.model.predictor, cfg.model.get("predictor_revision")).to(device).eval()
    model_ac = _AutocastModel(model, device, bf16)
    head = load_head(cfg.model.head, device).eval() if do_direct else None
    pairs = strike_horizon_pairs(15)
    h_idx = torch.as_tensor(pairs[:, 1], device=device)

    t_start = int(np.datetime64(cc.start, "s").astype(np.int64))
    t_end = int(np.datetime64(cc.end, "s").astype(np.int64))
    days = np.arange(t_start, t_end, 86400, dtype=np.int64)
    if max_days is not None:
        days = days[:max_days]
    bp = int(cc.batch_points)
    manifest: List[Dict] = []
    t_run0 = time.time()
    for d0 in days:
        day = str(np.datetime64(int(d0), "s"))[:10]
        done_marker = os.path.join(run_dir, "days", f"{day}.done")
        if resume and os.path.exists(done_marker):
            manifest.append(json.load(open(done_marker)))
            continue
        tday = time.time()
        times = day_decision_times(store, int(d0), int(min(d0 + 86400, t_end)), stride, lookback)
        summ_rows, direct_rows, path_frames = [], [], []
        tp_all = np.zeros((len(times), pred_len, n_samples), dtype=np.float32) if do_sampling else None
        for b0 in range(0, len(times), bp):
            tb = times[b0:b0 + bp]
            xs, means, stds, stamps_x, closes, sigmas, tps = [], [], [], [], [], [], []
            for t in tb:
                it = store.index_of_bar_closing_at(int(t))
                start = it - lookback + 1
                x = np.asarray(store.x[start:it + 1], dtype=np.float32)
                z, mean, std = normalize_window(x, lookback, clip)
                xs.append(z); means.append(mean); stds.append(std)
                stamps_x.append(np.asarray(store.stamp[start:it + 1], dtype=np.float32))
                closes.append(float(x[-1, 3]))
                lr = np.diff(np.log(np.clip(x[:, 3], 1e-9, None)))
                sigmas.append(max(float(lr.std()), 1e-6))
                tps.append(typical_price(x))
            x_t = torch.from_numpy(np.stack(xs)).to(device)
            xs_t = torch.from_numpy(np.stack(stamps_x)).to(device)
            B = x_t.shape[0]
            if do_sampling:
                future_open = tb[:, None] + BAR_SECONDS * np.arange(0, pred_len)[None, :]   # open time of bar h = t + 60(h-1)
                ys_t = torch.from_numpy(stamps_for_times(future_open)).to(device)
                with torch.no_grad():
                    preds = auto_regressive_inference(tokenizer, model_ac, x_t, xs_t, ys_t, max_context=lookback,
                                                      pred_len=pred_len, clip=clip, T=float(cc.T), top_k=int(cc.top_k),
                                                      top_p=float(cc.top_p), sample_count=n_samples, verbose=False,
                                                      return_samples=True)
                preds = preds[:, :, -pred_len:, :]                                         # [B, N, H, 6]
                mean = np.stack(means)[:, None, None, :]; std = np.stack(stds)[:, None, None, :]
                preds = preds * (std + 1e-5) + mean
                tp = preds[..., :4].mean(-1)                                               # [B, N, H]
                tp_sorted = np.sort(np.transpose(tp, (0, 2, 1)), axis=-1)                  # [B, H, N]
                tp_all[b0:b0 + B] = tp_sorted
                close = preds[..., 3]
                q = np.quantile(tp_sorted, QS, axis=-1)                                     # [13, B, H]
                for i in range(B):
                    for h in range(pred_len):
                        row = {"t": int(tb[i]), "h": h + 1, "tp_mean": float(tp[i, :, h].mean()), "tp_sd": float(tp[i, :, h].std()),
                               "close_mean": float(close[i, :, h].mean()), "close_sd": float(close[i, :, h].std()),
                               "close_t": closes[i], "sigma1": sigmas[i], "n": n_samples}
                        for qi, qv in enumerate(QS):
                            row[f"tp_q{int(round(qv * 100)):02d}"] = float(q[qi, i, h])
                        summ_rows.append(row)
                if store_paths:
                    Bn, Nn, Hn, _ = preds.shape
                    df = pd.DataFrame({
                        "t": np.repeat(tb, Nn * Hn), "sample": np.tile(np.repeat(np.arange(Nn, dtype=np.int16), Hn), Bn),
                        "h": np.tile(np.arange(1, Hn + 1, dtype=np.int8), Bn * Nn),
                        **{c: preds[..., k].reshape(-1).astype(np.float32) for k, c in enumerate(["open", "high", "low", "close", "volume", "amount"])},
                    })
                    path_frames.append(df)
            if do_direct:
                with torch.no_grad():
                    s1, s2 = encode_tokens(tokenizer, x_t)
                    with autocast_ctx(device, bf16):
                        hidden = backbone_hidden(model, s1, s2, xs_t)
                    ctx = hidden[:, -1].float()
                    tp_hist = torch.from_numpy(np.stack(tps)).to(device)                    # [B, L]
                    j = torch.as_tensor(pairs[:, 0], device=device)
                    strike = tp_hist[:, lookback - 1 - j]                                  # [B, P]
                    close_t = torch.as_tensor(closes, device=device, dtype=torch.float32)
                    sigma1 = torch.as_tensor(sigmas, device=device, dtype=torch.float32)
                    feats = direct_head_features(close_t, sigma1, strike, pairs)
                    logits = head(ctx, h_idx, feats).float().cpu().numpy()                # [B, P]
                P = pairs.shape[0]
                direct_rows.append(pd.DataFrame({
                    "t": np.repeat(tb, P), "j": np.tile(pairs[:, 0].astype(np.int8), B), "h": np.tile(pairs[:, 1].astype(np.int8), B),
                    "strike": strike.float().cpu().numpy().reshape(-1).astype(np.float32),
                    "logit": logits.reshape(-1).astype(np.float32),
                }))
        if do_sampling:
            pd.DataFrame(summ_rows).to_parquet(os.path.join(run_dir, "summary", f"{day}.parquet"), index=False)
            np.save(os.path.join(run_dir, "tp_samples", f"{day}.npy"), tp_all)
            np.save(os.path.join(run_dir, "tp_samples", f"{day}.t.npy"), times)
            if store_paths and path_frames:
                pd.concat(path_frames).to_parquet(os.path.join(run_dir, "paths", f"{day}.parquet"), index=False)
        if do_direct and direct_rows:
            pd.concat(direct_rows).to_parquet(os.path.join(run_dir, "direct", f"{day}.parquet"), index=False)
        rec = {"day": day, "n_points": int(len(times)), "seconds": round(time.time() - tday, 1),
               "sampling": do_sampling, "direct": do_direct}
        write_json(done_marker, rec)
        manifest.append(rec)
        print(json.dumps(rec), flush=True)
    out = {"days": manifest, "total_seconds": round(time.time() - t_run0, 1)}
    write_json(os.path.join(run_dir, "manifest.json"), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = run(cfg, resume=args.resume, max_days=args.max_days)
    print(json.dumps({"days": len(out["days"]), "total_seconds": out["total_seconds"]}))


if __name__ == "__main__":
    main()
