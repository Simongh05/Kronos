"""Flow test (free, Mac only): does the ladder's own taker flow carry information beside its price?

From the public taker prints (`data/kalshi_ladder/fills/`), per ladder (event) and minute: signed taker count
(+count for a taker buying yes at any strike = bullish, −count for a taker buying no) and total count. Features at a
decision minute t (prints with ts ≤ t, the same information set as the candle's bid/ask at t):
    imb{W}      signed / total over the last W minutes of the ladder, W ∈ {1, 5, 15}, and over its whole life (imb_life)
    own_imb5    the same at the row's own strike over 5 minutes
    lvol5       log1p(total count over 5 minutes)
Readouts: (1) val fit of the frozen blend's features + flow, day-block bootstrap of the flow coefficients (fit.fit);
(2) the val fit frozen on dev: per-day Brier gain of (blend + flow) over the blend, paired day bootstrap; (3) dev refit
coefficients (exploratory). Also the same without the head (market + null + flow), to see the flow on its own.

    python -m btc_updown.calib.flow_test --val data/tables/ladder_e1b_val.parquet --dev data/tables/ladder_e1b_dev.parquet \
        --fills data/kalshi_ladder --out data/calib/flow_test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .fit import design, fit, fit_logistic, predict, brier
from .market_head import paired_day_bootstrap

WINDOWS = (1, 5, 15)


def minute_sums(archive: str, key: str) -> pd.DataFrame:
    """Per (key, minute bucket m) signed and total taker counts; bucket m holds prints with ts in (m-60, m]."""
    parts = []
    for f in sorted(glob.glob(os.path.join(archive, "fills", "*.parquet"))):
        d = pd.read_parquet(f)
        if not len(d):
            continue
        d = d[~d["block"]]
        m = (np.ceil(d["ts"].to_numpy() / 60.0) * 60).astype(np.int64)
        sgn = np.where(d["taker_side"].to_numpy() == "yes", 1.0, -1.0) * d["count"].to_numpy()
        g = pd.DataFrame({"key": d[key].to_numpy(), "m": m, "signed": sgn, "count": d["count"].to_numpy()}).groupby(["key", "m"], sort=False).sum()
        parts.append(g.reset_index())
    s = pd.concat(parts, ignore_index=True).groupby(["key", "m"], sort=True).sum().reset_index()
    return s


def window_feats(sums: pd.DataFrame, keys: np.ndarray, times: np.ndarray, windows, prefix: str) -> pd.DataFrame:
    """For each (key, t): signed/total over (t - 60W, t] via per-key cumulative sums."""
    out = {f"{prefix}imb{w}": np.zeros(len(keys)) for w in windows}
    out[f"{prefix}imb_life"] = np.zeros(len(keys)); out[f"{prefix}lvol5"] = np.zeros(len(keys))
    q = pd.DataFrame({"key": keys, "t": times, "i": np.arange(len(keys))})
    for key, gq in q.groupby("key", sort=False):
        gs = sums[sums["key"] == key]
        if gs.empty:
            continue
        m = gs["m"].to_numpy(); cs = np.concatenate([[0.0], np.cumsum(gs["signed"].to_numpy())]); cn = np.concatenate([[0.0], np.cumsum(gs["count"].to_numpy())])
        tq = gq["t"].to_numpy(); ii = gq["i"].to_numpy()
        hi = np.searchsorted(m, tq, side="right")
        for w in list(windows) + ["life"]:
            lo = np.zeros_like(hi) if w == "life" else np.searchsorted(m, tq - 60 * w, side="right")
            s = cs[hi] - cs[lo]; n = cn[hi] - cn[lo]
            out[f"{prefix}imb_life" if w == "life" else f"{prefix}imb{w}"][ii] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
            if w == 5:
                out[f"{prefix}lvol5"][ii] = np.log1p(n)
    return pd.DataFrame(out)


def add_flow(t: pd.DataFrame, ev_sums: pd.DataFrame, tk_sums: pd.DataFrame) -> pd.DataFrame:
    lm = t[["event_ticker", "t"]].drop_duplicates().reset_index(drop=True)
    f = window_feats(ev_sums, lm["event_ticker"].to_numpy(), lm["t"].to_numpy(), WINDOWS, "")
    lm = pd.concat([lm, f], axis=1)
    t = t.merge(lm, on=["event_ticker", "t"], how="left")
    own = t[["market_id", "t"]].drop_duplicates().reset_index(drop=True)
    fo = window_feats(tk_sums, own["market_id"].to_numpy(), own["t"].to_numpy(), (5,), "own_")[["own_imb5"]]
    own = pd.concat([own, fo], axis=1)
    return t.merge(own, on=["market_id", "t"], how="left")


def _prep(path: str) -> pd.DataFrame:
    t = pd.read_parquet(path)
    return t[t["two_sided"] & t["outcome"].isin([0, 1]) & t["mkt_prob_up"].notna() & t["p_null"].notna() & t["logit_direct"].notna()].copy()


def run(val_path: str, dev_path: str, archive: str, out: str, n_boot: int) -> Dict:
    os.makedirs(out, exist_ok=True)
    ev_sums = minute_sums(archive, "event_ticker"); tk_sums = minute_sums(archive, "ticker")
    val = add_flow(_prep(val_path), ev_sums, tk_sums); dev = add_flow(_prep(dev_path), ev_sums, tk_sums)
    flow = ["imb1", "imb5", "imb15", "imb_life", "own_imb5", "lvol5"]
    res = {"rows": {"val": int(len(val)), "dev": int(len(dev))},
           "flow_stats_val": {c: {"mean": float(val[c].mean()), "sd": float(val[c].std()), "nonzero": float((val[c] != 0).mean())} for c in flow},
           "corr_val": {c: {"with_market_logit": float(np.corrcoef(val[c], np.log(val["mkt_prob_up"] / (1 - val["mkt_prob_up"])))[0, 1]),
                            "with_outcome": float(np.corrcoef(val[c], val["outcome"])[0, 1])} for c in flow}}
    base = ["logit_direct", "logit_null"]
    for name, feats in (("blend_plus_flow", base + flow), ("market_plus_flow", ["logit_null"] + flow), ("blend", base), ("flow_imb5_only", base + ["imb5"])):
        r = fit(val, feats, include_market=True, n_boot=n_boot)
        res[f"val_{name}"] = {"coef": r["coef"], "coef_ci95": r["bootstrap"]["coef_ci95"], "brier": r["brier"]}
        # frozen on dev
        Xd = design(dev, feats, True); beta = np.array([r["coef"]["intercept"], r["coef"]["logit_market"]] + [r["coef"][f] for f in feats])
        dev[f"p_{name}"] = predict(beta, Xd)
    y = dev["outcome"].to_numpy().astype(float); d = dev["label_day"].astype(str).to_numpy()
    res["dev_frozen"] = {"brier": {k: brier(dev[f"p_{k}"].to_numpy(), y) for k in ("blend", "blend_plus_flow", "market_plus_flow", "flow_imb5_only")},
                         "market": brier(dev["mkt_prob_up"].to_numpy(), y),
                         "paired_diff": {"blend_plus_flow_minus_blend": {"mean": brier(dev["p_blend_plus_flow"].to_numpy(), y) - brier(dev["p_blend"].to_numpy(), y),
                                                                         "ci95": paired_day_bootstrap(d, dev["p_blend_plus_flow"].to_numpy(), dev["p_blend"].to_numpy(), y)},
                                         "market_plus_flow_minus_market": {"mean": brier(dev["p_market_plus_flow"].to_numpy(), y) - brier(dev["mkt_prob_up"].to_numpy(), y),
                                                                           "ci95": paired_day_bootstrap(d, dev["p_market_plus_flow"].to_numpy(), dev["mkt_prob_up"].to_numpy(), y)},
                                         "flow_imb5_only_minus_blend": {"mean": brier(dev["p_flow_imb5_only"].to_numpy(), y) - brier(dev["p_blend"].to_numpy(), y),
                                                                        "ci95": paired_day_bootstrap(d, dev["p_flow_imb5_only"].to_numpy(), dev["p_blend"].to_numpy(), y)}}}
    r = fit(dev, base + flow, include_market=True, n_boot=n_boot)
    res["dev_refit_exploratory"] = {"coef": r["coef"], "coef_ci95": r["bootstrap"]["coef_ci95"], "brier": r["brier"]}
    # lag structure: flow over the minutes (t-W, t-5] i.e. older flow, to see whether the price has absorbed it
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    val[["market_id", "t"] + flow].to_parquet(os.path.join(out, "val_flow.parquet"), index=False)
    dev[["market_id", "t"] + flow].to_parquet(os.path.join(out, "dev_flow.parquet"), index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--fills", default="data/kalshi_ladder")
    ap.add_argument("--n-boot", type=int, default=300); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.val, a.dev, a.fills, a.out, a.n_boot), indent=1, default=float))


if __name__ == "__main__":
    main()
