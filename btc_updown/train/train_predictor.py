"""CLI: fine-tune the Kronos predictor (next-token CE) and/or the DirectHead.

    python -m btc_updown.train.train_predictor --config btc_updown/configs/ft_small_1m.yaml [--resume] [--max-steps N]
"""
from __future__ import annotations

import argparse
import json

from .config import load_config
from .loop import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <run_dir>/ckpt/last.json if present")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many additional steps (smoke tests)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, dotted keys, YAML-parsed value (e.g. train.steps=20)")
    args = ap.parse_args()

    overrides = {}
    import yaml
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = overrides
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    cfg = load_config(args.config, overrides)
    metrics = run(cfg, resume=args.resume, max_steps=args.max_steps)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
