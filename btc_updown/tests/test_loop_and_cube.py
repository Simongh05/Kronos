"""End-to-end on synthetic data: training with resume, validation, cube job."""
import json
import os

import numpy as np
import pandas as pd
import torch

from btc_updown.train.config import config_from_dict
from btc_updown.train import loop as L
from btc_updown.train import validate as V
from btc_updown.train.dataset import SeriesStore
from btc_updown.infer import cube as C


def _cfg(tmp_path, tiny, store_root, steps, exp, **over):
    d = {
        "exp_name": exp, "run_dir": str(tmp_path / "runs" / "{exp_name}"), "seed": 3, "device": "cpu", "bf16": False,
        "data": {"root": store_root, "train_series": ["synthetic"], "val_series": ["synthetic"], "lookback": 32,
                 "predict": 15, "clip": 5.0, "train_end": "2023-11-16", "val_start": "2023-11-16", "val_end": "2023-11-18",
                 "n_val_windows": 16, "num_workers": 0},
        "model": {"tokenizer": tiny["tok_dir"], "predictor": tiny["model_dir"], "direct_head": True, "head_hidden": 16,
                  "head_dropout": 0.0},
        "train": {"steps": steps, "batch_size": 4, "lr": 1e-3, "head_lr": 1e-3, "teacher_forcing": True,
                  "log_every": 1, "val_every": 100, "ckpt_every": 3, "keep_last": 2, "val_batch_size": 8},
    }
    for k, v in over.items():
        d[k] = v
    return config_from_dict(d)


def _losses(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))]
    return {r["step"]: r["loss"] for r in rows if "loss" in r}


def test_train_then_resume_reproduces_uninterrupted_run(tmp_path, tiny_models, store_root):
    torch.use_deterministic_algorithms(True)
    cfg_a = _cfg(tmp_path, tiny_models, store_root, steps=6, exp="a")
    L.run(cfg_a)
    la = _losses(cfg_a.run_dir)
    assert set(la) == {1, 2, 3, 4, 5, 6}

    cfg_b = _cfg(tmp_path, tiny_models, store_root, steps=6, exp="b")
    L.run(cfg_b, max_steps=3)
    assert set(_losses(cfg_b.run_dir)) == {1, 2, 3}
    assert os.path.exists(os.path.join(cfg_b.run_dir, "ckpt", "last.json"))
    L.run(cfg_b, resume=True)
    lb = _losses(cfg_b.run_dir)
    assert set(lb) == {1, 2, 3, 4, 5, 6}
    for s in (4, 5, 6):
        assert abs(la[s] - lb[s]) < 1e-5, f"step {s}: {la[s]} vs {lb[s]}"
    # checkpoints pruned to keep_last=2, best/ and done.json present
    ck = sorted(n for n in os.listdir(os.path.join(cfg_b.run_dir, "ckpt")) if n.startswith("step_"))
    assert len(ck) <= 2
    assert os.path.isdir(os.path.join(cfg_b.run_dir, "best", "model")) and os.path.exists(os.path.join(cfg_b.run_dir, "done.json"))
    torch.use_deterministic_algorithms(False)


def test_resume_refuses_changed_total_steps(tmp_path, tiny_models, store_root):
    cfg = _cfg(tmp_path, tiny_models, store_root, steps=6, exp="c")
    L.run(cfg, max_steps=3)
    cfg2 = _cfg(tmp_path, tiny_models, store_root, steps=9, exp="c")
    import pytest
    with pytest.raises(RuntimeError):
        L.run(cfg2, resume=True)


def test_frozen_backbone_probe_trains_only_head(tmp_path, tiny_models, store_root):
    cfg = _cfg(tmp_path, tiny_models, store_root, steps=3, exp="probe",
               model={"tokenizer": tiny_models["tok_dir"], "predictor": tiny_models["model_dir"], "direct_head": True,
                      "head_hidden": 16, "head_dropout": 0.0, "freeze_backbone": True})
    m = L.run(cfg)
    assert "val/head_brier" in m and m["val/score"] == m["val/head_brier"]
    # backbone weights unchanged
    from btc_updown.train.model_io import load_model
    a = load_model(tiny_models["model_dir"]).state_dict()
    b = load_model(os.path.join(cfg.run_dir, "best", "model")).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_market_points_and_evaluate_markets(tmp_path, tiny_models, store_root):
    store = SeriesStore.open(store_root, "synthetic")
    ts = np.asarray(store.ts)
    # fake 5m windows on the bar grid, outcome from the series' own typical prices
    starts = ts[100:400:5] + 60
    rows = []
    for ws in starts:
        we = ws + 300
        i0, i1 = store.index_of_bar_closing_at(int(ws)), store.index_of_bar_closing_at(int(we))
        if i0 < 0 or i1 < 0:
            continue
        tp0 = store.x[i0, :4].mean(); tp1 = store.x[i1, :4].mean()
        rows.append({"window_start": int(ws), "window_end": int(we), "outcome": int(tp1 >= tp0)})
    wdf = pd.DataFrame(rows)
    pts = V.market_decision_points(wdf, int(ts[0]), int(ts[-1]) + 60, n_points=50, seed=0)
    assert pts.shape == (50, 4) and np.all(pts[:, 2] > pts[:, 0]) and np.all(pts[:, 1] - pts[:, 2] >= 60)
    from btc_updown.train.heads import DirectHead
    from btc_updown.train.model_io import load_model, load_tokenizer
    model = load_model(tiny_models["model_dir"]).eval()
    tok = load_tokenizer(tiny_models["tok_dir"]).eval()
    head = DirectHead(model.d_model, hidden=16)
    m = V.evaluate_markets(model, tok, head, store, pts, lookback=32, clip=5.0, device=torch.device("cpu"), bf16=False)
    assert m["val/mkt_n"] == 50 and 0.0 <= m["val/mkt_brier"] <= 1.0


def test_cube_job_outputs_and_resume(tmp_path, tiny_models, store_root):
    # train a head for 2 steps so the cube can run the direct branch too
    cfg_t = _cfg(tmp_path, tiny_models, store_root, steps=2, exp="h")
    L.run(cfg_t)
    head_path = os.path.join(cfg_t.run_dir, "best", "head.pt")
    store = SeriesStore.open(store_root, "synthetic")
    t0 = int(np.asarray(store.ts)[0])
    day0 = (t0 // 86400 + 1) * 86400  # first full UTC day inside the series
    start = str(np.datetime64(day0, "s"))[:10]
    end = str(np.datetime64(day0 + 86400, "s"))[:10]
    cfg = config_from_dict({
        "exp_name": "cube", "run_dir": str(tmp_path / "cubes" / "{exp_name}"), "seed": 1, "device": "cpu", "bf16": False,
        "data": {"root": store_root, "series": "synthetic", "lookback": 32, "clip": 5.0},
        "model": {"tokenizer": tiny_models["tok_dir"], "predictor": tiny_models["model_dir"], "head": head_path},
        "cube": {"start": start, "end": end, "stride_s": 3600, "pred_len": 4, "n_samples": 5, "T": 1.0, "top_p": 1.0,
                 "top_k": 0, "batch_points": 7, "sampling": True, "direct": True, "store_paths": True},
    })
    out = C.run(cfg, resume=True)
    assert len(out["days"]) == 1 and out["days"][0]["n_points"] == 24
    s = pd.read_parquet(os.path.join(cfg.run_dir, "summary", f"{start}.parquet"))
    assert len(s) == 24 * 4 and {"tp_mean", "tp_sd", "tp_q50", "close_t"} <= set(s.columns)
    tp = np.load(os.path.join(cfg.run_dir, "tp_samples", f"{start}.npy"))
    assert tp.shape == (24, 4, 5) and np.all(np.diff(tp, axis=-1) >= 0)  # sorted
    d = pd.read_parquet(os.path.join(cfg.run_dir, "direct", f"{start}.parquet"))
    assert len(d) == 24 * 120 and np.isfinite(d["logit"]).all()
    p = pd.read_parquet(os.path.join(cfg.run_dir, "paths", f"{start}.parquet"))
    assert len(p) == 24 * 5 * 4
    # decision times are bar closes with no future bar used: context ends at t
    assert np.all(np.load(os.path.join(cfg.run_dir, "tp_samples", f"{start}.t.npy")) % 3600 == 0)
    # resume skips the finished day
    out2 = C.run(cfg, resume=True)
    assert out2["days"][0]["seconds"] == out["days"][0]["seconds"]
