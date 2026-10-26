"""YAML experiment configs with defaults; attribute access via dot-paths."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "exp_name": "exp",
    "run_dir": "data/runs/{exp_name}",
    "expected_hours": 1.0,
    "seed": 100,
    "device": "auto",
    "bf16": True,
    "data": {
        "root": "data/klines",
        "train_series": ["spot_BTCUSDC", "spot_BTCUSDT", "um_BTCUSDT"],
        "val_series": ["spot_BTCUSDC"],
        "lookback": 496,
        "predict": 15,
        "clip": 5.0,
        "train_start": None,
        "train_end": "2026-08-01",
        "val_start": "2026-08-07",
        "val_end": "2026-09-07",
        "n_val_windows": 2048,
        "num_workers": 4,
    },
    "model": {
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_revision": None,
        "predictor": "NeoQuasar/Kronos-small",
        "predictor_revision": None,
        "freeze_backbone": False,
        "direct_head": True,
        "head_hidden": 256,
        "head_dropout": 0.1,
        "head_max_h": 15,           # (j,h) grid: j + h <= head_max_h; 60 for hourly markets
        "head_init": None,          # path to a head.pt to start from
    },
    "train": {
        "steps": 30000,
        "batch_size": 64,
        "lr": 4.0e-5,
        "head_lr": 1.0e-3,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "grad_clip": 3.0,
        "pct_start": 0.03,
        "div_factor": 10.0,
        "teacher_forcing": False,
        "ce_weight": 1.0,
        "bce_weight": 1.0,
        "log_every": 50,
        "val_every": 2500,
        "ckpt_every": 2500,
        "keep_last": 2,
        "val_batch_size": 64,
    },
    "val_labels": None,             # optional: data/labels/windows.parquet for the real-market task metric
    "val_label_points": 500,
    "cube": {                       # only used by infer.cube
        "start": "2026-08-07",
        "end": "2026-09-07",
        "stride_s": 60,
        "pred_len": 15,
        "n_samples": 128,
        "T": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "batch_points": 16,
        "sampling": True,
        "direct": False,
        "store_paths": False,
    },
    "tokenizer_train": {            # only used by train_tokenizer
        "steps": 5000,
        "batch_size": 64,
        "lr": 2.0e-4,
        "weight_decay": 0.1,
        "grad_clip": 2.0,
        "pct_start": 0.03,
        "div_factor": 10.0,
        "log_every": 50,
        "val_every": 1000,
        "ckpt_every": 1000,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, d: Dict[str, Any]):
        self._d = d

    def __getattr__(self, k: str):
        d = self.__dict__.get("_d", {})
        if k in d:
            v = d[k]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(k)

    def get(self, k: str, default=None):
        return self._d.get(k, default)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self._d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    d = _merge(DEFAULTS, raw)
    if overrides:
        d = _merge(d, overrides)
    d["run_dir"] = str(d["run_dir"]).format(exp_name=d["exp_name"])
    d["_config_path"] = os.path.abspath(path)
    return Config(d)


def config_from_dict(d: Dict[str, Any]) -> Config:
    d = _merge(DEFAULTS, d)
    d["run_dir"] = str(d["run_dir"]).format(exp_name=d["exp_name"])
    return Config(d)
