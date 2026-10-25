"""CLI: fine-tune the Kronos tokenizer (fp32 only; the BSQ quantizer is fragile in reduced precision).

Loss as upstream finetune_csv/finetune_tokenizer.py: (mse(z_pre, x) + mse(z, x) + bsq_loss) / 2.
Only run this if the tokenizer round-trip test shows the pretrained codebook cannot resolve 1-minute moves.

    python -m btc_updown.train.train_tokenizer --config btc_updown/configs/tok_1m.yaml [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import Config, load_config
from .dataset import describe_stores, write_json
from .loop import _seed_all, build_datasets
from .model_io import load_tokenizer, pick_device


@torch.no_grad()
def eval_tokenizer(tok, loader, device) -> Dict[str, float]:
    tok.eval()
    tot, n = 0.0, 0
    tot_tp = 0.0
    for b in loader:
        x = b["x"].to(device).float()
        (_, z), _, _, _ = tok(x)
        tot += float(F.mse_loss(z, x)) * x.shape[0]
        # reconstruction error of the typical price in normalized units
        tot_tp += float(F.mse_loss(z[..., :4].mean(-1), x[..., :4].mean(-1))) * x.shape[0]
        n += x.shape[0]
    tok.train()
    return {"val/recon_mse": tot / max(n, 1), "val/tp_mse": tot_tp / max(n, 1), "val/n": float(n)}


def run(cfg: Config, resume: bool = False, max_steps: Optional[int] = None) -> Dict[str, float]:
    run_dir = cfg.run_dir
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
    write_json(os.path.join(run_dir, "config.json"), cfg.to_dict())
    _seed_all(int(cfg.seed))
    device = pick_device(cfg.device)
    tc = cfg.tokenizer_train
    # reuse the predictor dataset builder with tokenizer step/batch settings
    cfg._d["train"]["steps"] = int(tc.steps)
    cfg._d["train"]["batch_size"] = int(tc.batch_size)
    train_stores, _, train_ds, val_ds = build_datasets(cfg)
    print(describe_stores(train_stores))

    tok = load_tokenizer(cfg.model.tokenizer, cfg.model.get("tokenizer_revision")).to(device).float().train()
    optimizer = torch.optim.AdamW(tok.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
    total_steps = int(tc.steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(tc.lr), total_steps=total_steps,
                                                    pct_start=float(tc.pct_start), div_factor=float(tc.div_factor))
    step, best = 0, math.inf
    last_json = os.path.join(run_dir, "ckpt", "last.json")
    if resume and os.path.exists(last_json):
        ck = json.load(open(last_json))["path"]
        st = torch.load(os.path.join(ck, "state.pt"), map_location="cpu", weights_only=False)
        if int(st["total_steps"]) != total_steps:
            raise RuntimeError("resume: total_steps changed")
        tok = load_tokenizer(os.path.join(ck, "model")).to(device).float().train()
        optimizer = torch.optim.AdamW(tok.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(tc.lr), total_steps=total_steps,
                                                        pct_start=float(tc.pct_start), div_factor=float(tc.div_factor))
        optimizer.load_state_dict(st["optimizer"]); scheduler.load_state_dict(st["scheduler"])
        step, best = int(st["step"]), float(st["best_score"])
        torch.set_rng_state(st["rng_torch"])
        print(f"resumed from {ck} at step {step}")

    bs = int(tc.batch_size)
    loader = DataLoader(Subset(train_ds, range(step * bs, len(train_ds))), batch_size=bs, shuffle=False, drop_last=True,
                        num_workers=int(cfg.data.num_workers))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    stop_at = total_steps if max_steps is None else min(total_steps, step + int(max_steps))
    t0, acc, n_acc, metrics = time.time(), 0.0, 0, {}
    for b in loader:
        if step >= stop_at:
            break
        x = b["x"].to(device).float()
        (z_pre, z), bsq_loss, _, _ = tok(x)
        recon = F.mse_loss(z_pre, x) + F.mse_loss(z, x)
        loss = (recon + bsq_loss) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), float(tc.grad_clip))
        optimizer.step(); scheduler.step(); step += 1
        acc += float(loss); n_acc += 1
        if step % int(tc.log_every) == 0 or step == stop_at:
            rec = {"step": step, "loss": acc / max(n_acc, 1), "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - t0}
            print(json.dumps(rec)); acc, n_acc = 0.0, 0
            with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        if step % int(tc.val_every) == 0 or step == stop_at:
            metrics = eval_tokenizer(tok, val_loader, device); metrics["step"] = step
            print(json.dumps(metrics))
            with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(metrics) + "\n")
            if metrics["val/recon_mse"] < best:
                best = metrics["val/recon_mse"]
                bd = os.path.join(run_dir, "best"); tmp = bd + ".tmp"
                os.makedirs(tmp, exist_ok=True); tok.save_pretrained(tmp)
                write_json(os.path.join(tmp, "metrics.json"), metrics)
                if os.path.exists(bd):
                    import shutil; shutil.rmtree(bd)
                os.replace(tmp, bd)
        if step % int(tc.ckpt_every) == 0 or step == stop_at:
            ck = os.path.join(run_dir, "ckpt", f"step_{step:07d}"); tmp = ck + ".tmp"
            os.makedirs(os.path.join(tmp, "model"), exist_ok=True)
            tok.save_pretrained(os.path.join(tmp, "model"))
            torch.save({"step": step, "total_steps": total_steps, "best_score": best, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng_torch": torch.get_rng_state()}, os.path.join(tmp, "state.pt"))
            if os.path.exists(ck):
                import shutil; shutil.rmtree(ck)
            os.replace(tmp, ck)
            write_json(last_json, {"step": step, "path": ck})
    write_json(os.path.join(run_dir, "done.json"), {"step": step, "finished": step >= total_steps, "best": best})
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(run(cfg, resume=args.resume, max_steps=args.max_steps), indent=2))


if __name__ == "__main__":
    main()
