"""Predictor fine-tuning loop: next-token CE (+ optional DirectHead BCE), bf16, resumable."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import Config
from .dataset import (
    WindowDataset,
    describe_stores,
    direct_head_features,
    direct_head_targets,
    iso_to_epoch,
    load_stores,
    strike_horizon_pairs,
    write_json,
)
from .heads import DirectHead
from .model_io import autocast_ctx, backbone_hidden, dual_logits, encode_tokens, load_model, load_tokenizer, pick_device
from . import validate as V


# ---------------------------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------------------------
def _ckpt_dir(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, "ckpt", f"step_{step:07d}")


def save_checkpoint(run_dir: str, step: int, model, head: Optional[DirectHead], optimizer, scheduler,
                    total_steps: int, cfg_digest: str, best_score: float, extra: Dict) -> str:
    final = _ckpt_dir(run_dir, step)
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(os.path.join(tmp, "model"), exist_ok=True)
    model.save_pretrained(os.path.join(tmp, "model"))
    if head is not None:
        torch.save(head_state(head), os.path.join(tmp, "head.pt"))
    state = {
        "step": step,
        "total_steps": total_steps,
        "config_digest": cfg_digest,
        "best_score": best_score,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "extra": extra,
    }
    torch.save(state, os.path.join(tmp, "state.pt"))
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)
    write_json(os.path.join(run_dir, "ckpt", "last.json"), {"step": step, "path": final})
    return final


def prune_checkpoints(run_dir: str, keep_last: int, keep_step: Optional[int]) -> None:
    d = os.path.join(run_dir, "ckpt")
    if not os.path.isdir(d):
        return
    steps = sorted(int(n.split("_")[1]) for n in os.listdir(d) if n.startswith("step_") and not n.endswith(".tmp"))
    for s in steps[:-keep_last] if keep_last > 0 else []:
        if keep_step is not None and s == keep_step:
            continue
        shutil.rmtree(_ckpt_dir(run_dir, s), ignore_errors=True)


def head_state(head: DirectHead) -> Dict:
    return {"state_dict": head.state_dict(), "d_model": head.d_model, "hidden": head.hidden,
            "max_h": head.max_h, "dropout": head.dropout_p}


def load_head(path: str, device: torch.device) -> DirectHead:
    obj = torch.load(path, map_location="cpu")
    head = DirectHead(obj["d_model"], hidden=obj["hidden"], max_h=obj["max_h"], dropout=float(obj.get("dropout", 0.1)))
    head.load_state_dict(obj["state_dict"])
    return head.to(device)


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _log(run_dir: str, rec: Dict) -> None:
    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def build_datasets(cfg: Config):
    d = cfg.data
    train_stores = load_stores(d.root, d.train_series)
    val_stores = load_stores(d.root, d.val_series)
    steps = int(cfg.train.steps)
    bs = int(cfg.train.batch_size)
    train_ds = WindowDataset(train_stores, d.lookback, d.predict, n_samples=steps * bs, clip=d.clip, seed=cfg.seed,
                             t_min=iso_to_epoch(d.get("train_start")), t_max=iso_to_epoch(d.train_end))
    val_ds = WindowDataset(val_stores, d.lookback, d.predict, n_samples=int(d.n_val_windows), clip=d.clip,
                           seed=cfg.seed + 1, t_min=iso_to_epoch(d.val_start), t_max=iso_to_epoch(d.val_end),
                           deterministic_grid=True)
    return train_stores, val_stores, train_ds, val_ds


def run(cfg: Config, resume: bool = False, max_steps: Optional[int] = None) -> Dict[str, float]:
    run_dir = cfg.run_dir
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
    write_json(os.path.join(run_dir, "config.json"), cfg.to_dict())
    _seed_all(int(cfg.seed))
    device = pick_device(cfg.device)
    bf16 = bool(cfg.bf16) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False  # keep tokenizer encode bit-exact with fp32 inference
        torch.backends.cudnn.allow_tf32 = False

    lookback = int(cfg.data.lookback)
    predict = int(cfg.data.predict)
    if lookback + predict > 512:
        raise ValueError("lookback + predict must be <= 512 (Kronos max context)")
    if cfg.model.direct_head and predict < 15:
        raise ValueError("data.predict must be >= 15 when model.direct_head is enabled (the (j,h) grid needs 15 future bars)")

    train_stores, val_stores, train_ds, val_ds = build_datasets(cfg)
    print(describe_stores(train_stores))
    print(f"train windows available: {train_ds.total_valid:,}; val windows: {val_ds.total_valid:,} (using {len(val_ds)})")

    tokenizer = load_tokenizer(cfg.model.tokenizer, cfg.model.get("tokenizer_revision")).to(device).float().eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    model = load_model(cfg.model.predictor, cfg.model.get("predictor_revision")).to(device)
    head: Optional[DirectHead] = None
    if cfg.model.direct_head:
        if cfg.model.get("head_init"):
            head = load_head(cfg.model.head_init, device)
        else:
            head = DirectHead(model.d_model, hidden=int(cfg.model.head_hidden), max_h=15,
                              dropout=float(cfg.model.head_dropout)).to(device)
    freeze = bool(cfg.model.freeze_backbone)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    else:
        model.train()

    groups = []
    if not freeze:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": float(cfg.train.lr)})
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": float(cfg.train.head_lr)})
    if not groups:
        raise ValueError("nothing to train: backbone frozen and no head")
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg.train.betas), weight_decay=float(cfg.train.weight_decay))
    total_steps = int(cfg.train.steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=total_steps,
        pct_start=float(cfg.train.pct_start), div_factor=float(cfg.train.div_factor))

    step = 0
    best_score = math.inf
    digest = cfg.digest()
    last_json = os.path.join(run_dir, "ckpt", "last.json")
    if resume and os.path.exists(last_json):
        info = json.load(open(last_json))
        ck = info["path"]
        state = torch.load(os.path.join(ck, "state.pt"), map_location="cpu", weights_only=False)
        if int(state["total_steps"]) != total_steps:
            raise RuntimeError(f"resume: total_steps changed ({state['total_steps']} -> {total_steps}); OneCycleLR cannot resume")
        if state["config_digest"] != digest:
            print(f"WARNING: config digest changed since checkpoint ({state['config_digest']} -> {digest})")
        model = load_model(os.path.join(ck, "model")).to(device)
        model.train() if not freeze else model.eval()
        if head is not None and os.path.exists(os.path.join(ck, "head.pt")):
            head = load_head(os.path.join(ck, "head.pt"), device)
        groups = []
        if not freeze:
            groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": float(cfg.train.lr)})
        if head is not None:
            groups.append({"params": list(head.parameters()), "lr": float(cfg.train.head_lr)})
        optimizer = torch.optim.AdamW(groups, betas=tuple(cfg.train.betas), weight_decay=float(cfg.train.weight_decay))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in groups], total_steps=total_steps,
            pct_start=float(cfg.train.pct_start), div_factor=float(cfg.train.div_factor))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        best_score = float(state["best_score"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        print(f"resumed from {ck} at step {step}")

    bs = int(cfg.train.batch_size)
    sub = Subset(train_ds, range(step * bs, len(train_ds)))
    loader = DataLoader(sub, batch_size=bs, shuffle=False, drop_last=True, num_workers=int(cfg.data.num_workers),
                        pin_memory=device.type == "cuda", persistent_workers=int(cfg.data.num_workers) > 0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.train.val_batch_size), shuffle=False, drop_last=False,
                            num_workers=min(2, int(cfg.data.num_workers)))

    mkt_points = None
    if cfg.get("val_labels") and head is not None:
        import pandas as pd
        wdf = pd.read_parquet(cfg.val_labels)
        mkt_points = V.market_decision_points(wdf, iso_to_epoch(cfg.data.val_start), iso_to_epoch(cfg.data.val_end),
                                             int(cfg.val_label_points), int(cfg.seed))
        print(f"real-market validation points: {len(mkt_points)}")

    pairs = strike_horizon_pairs(15)
    h_idx = torch.as_tensor(pairs[:, 1], device=device)
    ce_w = float(cfg.train.ce_weight) if not freeze else 0.0
    bce_w = float(cfg.train.bce_weight) if head is not None else 0.0
    tf = bool(cfg.train.teacher_forcing)
    log_every, val_every, ckpt_every = int(cfg.train.log_every), int(cfg.train.val_every), int(cfg.train.ckpt_every)
    stop_at = total_steps if max_steps is None else min(total_steps, step + int(max_steps))

    def do_validation() -> Dict[str, float]:
        m = V.evaluate_windows(model, tokenizer, head, val_loader, lookback, device, bf16)
        if mkt_points is not None and len(mkt_points) and head is not None:
            m.update(V.evaluate_markets(model, tokenizer, head, val_stores[0], mkt_points, lookback,
                                        float(cfg.data.clip), device, bf16))
        if "val/mkt_brier" in m:
            m["val/score"] = m["val/mkt_brier"]
        elif head is not None:
            m["val/score"] = m["val/head_brier"]
        else:
            m["val/score"] = m["val/ce"]
        return m

    t0 = time.time()
    seen = 0
    acc = {"loss": 0.0, "ce": 0.0, "ce_s1": 0.0, "ce_s2": 0.0, "bce": 0.0, "n": 0}
    metrics: Dict[str, float] = {}
    it = iter(loader)
    while step < stop_at:
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch["x"].to(device, non_blocking=True)
        stamp = batch["stamp"].to(device, non_blocking=True)
        s1, s2 = encode_tokens(tokenizer, x)
        s1_in, s2_in, st_in = s1[:, :-1], s2[:, :-1], stamp[:, :-1]
        s1_t, s2_t = s1[:, 1:], s2[:, 1:]

        with autocast_ctx(device, bf16):
            if freeze:
                with torch.no_grad():
                    hidden = backbone_hidden(model, s1_in, s2_in, st_in)
                ce = ce1 = ce2 = torch.zeros((), device=device)
            else:
                hidden = backbone_hidden(model, s1_in, s2_in, st_in)
                l1, l2 = dual_logits(model, hidden, s1_t, teacher_forcing=tf, causal=None)
                ce, ce1, ce2 = model.head.compute_loss(l1.float(), l2.float(), s1_t, s2_t)
        loss = ce_w * ce
        bce = torch.zeros((), device=device)
        if head is not None:
            tp = batch["tp"].to(device)
            labels, strike = direct_head_targets(tp, lookback, pairs)
            feats = direct_head_features(batch["close_t"].to(device), batch["sigma1"].to(device), strike, pairs)
            ctx = hidden[:, lookback - 1].float()
            logits = head(ctx, h_idx, feats)
            bce = DirectHead.loss(logits, labels)
            loss = loss + bce_w * bce

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, float(cfg.train.grad_clip))
        optimizer.step()
        scheduler.step()
        step += 1
        seen += x.shape[0]

        acc["loss"] += float(loss.detach()); acc["ce"] += float(ce.detach()); acc["ce_s1"] += float(ce1.detach())
        acc["ce_s2"] += float(ce2.detach()); acc["bce"] += float(bce.detach()); acc["n"] += 1
        if step % log_every == 0 or step == stop_at:
            n = max(acc["n"], 1)
            el = time.time() - t0
            rec = {"step": step, "loss": acc["loss"] / n, "ce": acc["ce"] / n, "ce_s1": acc["ce_s1"] / n,
                   "ce_s2": acc["ce_s2"] / n, "bce": acc["bce"] / n, "lr": scheduler.get_last_lr()[0],
                   "elapsed_s": el, "samples_per_s": seen / max(el, 1e-6), "time": time.time()}
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
            _log(run_dir, rec)
            acc = {k: 0.0 for k in acc}; acc["n"] = 0

        if step % val_every == 0 or step == stop_at:
            metrics = do_validation()
            metrics["step"] = step
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()}))
            _log(run_dir, metrics)
            if metrics["val/score"] < best_score:
                best_score = metrics["val/score"]
                best_dir = os.path.join(run_dir, "best")
                tmp = best_dir + ".tmp"
                if os.path.exists(tmp):
                    shutil.rmtree(tmp)
                os.makedirs(os.path.join(tmp, "model"))
                model.save_pretrained(os.path.join(tmp, "model"))
                if head is not None:
                    torch.save(head_state(head), os.path.join(tmp, "head.pt"))
                write_json(os.path.join(tmp, "metrics.json"), metrics)
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                os.replace(tmp, best_dir)

        if step % ckpt_every == 0 or step == stop_at:
            save_checkpoint(run_dir, step, model, head, optimizer, scheduler, total_steps, digest, best_score,
                            {"metrics": metrics})
            prune_checkpoints(run_dir, int(cfg.train.keep_last), keep_step=None)

    write_json(os.path.join(run_dir, "done.json"), {"step": step, "total_steps": total_steps,
                                                   "finished": step >= total_steps, "best_score": best_score,
                                                   "last_metrics": metrics})
    return metrics
