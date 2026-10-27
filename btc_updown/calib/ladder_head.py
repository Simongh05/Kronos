"""Market-conditioned head on the Kalshi KXBTCD ladders (free, Mac only).

Question: given the ladder's own prices (the market's implied distribution), the EWMA null, and the existing 60-minute
DirectHead, does the Kronos hidden state add anything on held-out days? Same design as `market_head.py` (option 2 on
the 5m markets), adapted to the ladder table: rows = two-sided (ladder, minute, strike) rows; train on the val days
(fit + early-stop), test on the dev days (dev is already open for the ladders → exploratory).

Features
    market   logit(mid) of the row's strike
    ladder   logit(mid) at the strikes k-5..k+5 of the same ladder-minute, relative to the row's own logit (0 if absent)
    numeric  k/5, (strike-close)/close bps, log sigma_ewma, h/60, logit(p_null), logit_direct (the frozen head), age
    hidden   backbone state at t (512-d, standardized, PCA)
Models  A market | B market+numeric+ladder (logistic) | D B+hidden linear | C B+hidden MLP | C0 B MLP (no hidden)
Readouts on the test days: Brier / log loss, paired day-block bootstrap of the differences, by h and |k| buckets.

    python -m btc_updown.calib.ladder_head --val data/tables/ladder_e1b_val.parquet --dev data/tables/ladder_e1b_dev.parquet \
        --hidden data/cloud/cubes/cube_e1b_hidden_kalshi/hidden --out data/calib/ladder_head
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .fit import logit, sigmoid
from .market_head import load_hidden, train_mlp, brier, logloss, paired_day_bootstrap


def ladder_vector(t: pd.DataFrame, K: int = 5) -> np.ndarray:
    own = logit(t["mkt_prob_up"].to_numpy())
    s = pd.Series(own, index=pd.MultiIndex.from_arrays([t["event_ticker"], t["t"], t["k"]]))
    cols = []
    for dk in range(-K, K + 1):
        if dk == 0:
            continue
        v = s.reindex(pd.MultiIndex.from_arrays([t["event_ticker"], t["t"], t["k"] + dk])).to_numpy()
        cols.append(np.nan_to_num(v - own, nan=0.0))
    return np.clip(np.stack(cols, 1), -10, 10).astype(np.float32)


def numeric_feats(t: pd.DataFrame) -> np.ndarray:
    rel = (t["strike_series"] - t["close_t"]) / t["close_t"] * 1e4
    return np.stack([t["k"].to_numpy() / 5.0, rel.to_numpy(), np.log(t["sigma_ewma"].to_numpy()), t["h"].to_numpy() / 60.0,
                     logit(t["p_null"].to_numpy()), t["logit_direct"].to_numpy().astype(float), t["age"].to_numpy() / 10.0], 1).astype(np.float32)


def _load(path: str) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t = t[t["two_sided"] & t["outcome"].isin([0, 1]) & t["mkt_prob_up"].notna() & t["p_null"].notna() & t["logit_direct"].notna()].copy()
    t["day"] = t["label_day"].astype(str)
    return t


def run(val_path: str, dev_path: str, hidden_dir: str, out: str, es_days: int, pca: int, seed: int) -> Dict:
    os.makedirs(out, exist_ok=True)
    t = pd.concat([_load(val_path).assign(split="val"), _load(dev_path).assign(split="dev")], ignore_index=True)
    ts, H = load_hidden(hidden_dir)
    pos = pd.Series(np.arange(len(ts)), index=ts)
    idx = pos.reindex(t["t"].to_numpy()).to_numpy()
    ok = ~np.isnan(idx); t = t[ok].reset_index(drop=True); Hrows = H[idx[ok].astype(int)]
    val_days = np.array(sorted(t.loc[t["split"] == "val", "day"].unique())); test_days = np.array(sorted(t.loc[t["split"] == "dev", "day"].unique()))
    es_set, fit_set = set(val_days[-es_days:]), set(val_days[:-es_days])
    m_fit, m_es, m_te = t["day"].isin(fit_set).to_numpy(), t["day"].isin(es_set).to_numpy(), (t["split"] == "dev").to_numpy()
    y = t["outcome"].to_numpy().astype(np.float32)
    xm = logit(t["mkt_prob_up"].to_numpy()).astype(np.float32)
    xn = np.concatenate([numeric_feats(t), ladder_vector(t)], 1)
    mu_n, sd_n = xn[m_fit].mean(0), xn[m_fit].std(0) + 1e-6; xn = ((xn - mu_n) / sd_n).astype(np.float32)
    mu_h, sd_h = Hrows[m_fit].mean(0), Hrows[m_fit].std(0) + 1e-6; xh = (Hrows - mu_h) / sd_h
    if pca and pca < xh.shape[1]:
        U, S, Vt = np.linalg.svd(xh[m_fit][::4], full_matrices=False); xh = xh @ Vt[:pca].T; xh = xh / (xh[m_fit].std(0) + 1e-6)
    xh = xh.astype(np.float32)
    res = {"rows": {"fit": int(m_fit.sum()), "early_stop": int(m_es.sum()), "test": int(m_te.sum())},
           "days": {"fit": sorted(fit_set), "early_stop": sorted(es_set), "test": list(test_days)}, "hidden_dim": int(xh.shape[1]),
           "n_numeric": int(xn.shape[1]), "base_rate_test": float(y[m_te].mean())}
    preds: Dict[str, np.ndarray] = {}
    tr = m_fit | m_es

    def logreg(X):
        lr = LogisticRegression(C=1.0, max_iter=3000); lr.fit(X[tr], y[tr]); return lr.predict_proba(X)[:, 1], lr
    preds["A_market"], lrA = logreg(xm[:, None])
    preds["B_market_numeric_ladder"], lrB = logreg(np.concatenate([xm[:, None], xn], 1))
    preds["D_B_plus_hidden_linear"], lrD = logreg(np.concatenate([xm[:, None], xn, xh], 1))
    res["B_coef"] = {"market": float(lrB.coef_[0][0]), "numeric_ladder": [float(v) for v in lrB.coef_[0][1:]]}
    Xr = np.concatenate([xn, xh], 1)
    mlp, info = train_mlp(xm[m_fit], Xr[m_fit], y[m_fit], xm[m_es], Xr[m_es], y[m_es], seed=seed)
    with torch.no_grad():
        preds["C_B_plus_hidden_mlp"] = sigmoid(mlp(torch.as_tensor(xm), torch.as_tensor(Xr)).numpy())
    res["C_train"] = info; res["C_market_skip_weight"] = float(mlp.market_w.item())
    mlp0, info0 = train_mlp(xm[m_fit], xn[m_fit], y[m_fit], xm[m_es], xn[m_es], y[m_es], seed=seed)
    with torch.no_grad():
        preds["C0_B_mlp_no_hidden"] = sigmoid(mlp0(torch.as_tensor(xm), torch.as_tensor(xn)).numpy())
    res["C0_train"] = info0
    preds["market_raw"] = t["mkt_prob_up"].to_numpy().astype(np.float32)
    # the frozen val blend (market + head + null) as the reference the ladder result was stated with
    cal = json.load(open("data/calib/ladder_e1b_market_val.json"))["coef"]
    preds["blend_frozen"] = sigmoid(cal["intercept"] + cal["logit_market"] * xm + cal["logit_direct"] * t["logit_direct"].to_numpy()
                                    + cal["logit_null"] * logit(t["p_null"].to_numpy())).astype(np.float32)
    dte = t["day"].to_numpy()[m_te]; yte = y[m_te]
    res["test"] = {k: {"brier": brier(v[m_te], yte), "logloss": logloss(v[m_te], yte)} for k, v in preds.items()}
    res["test"]["null"] = {"brier": brier(t["p_null"].to_numpy()[m_te], yte)}
    pairs = {"C_minus_B": ("C_B_plus_hidden_mlp", "B_market_numeric_ladder"), "C_minus_C0": ("C_B_plus_hidden_mlp", "C0_B_mlp_no_hidden"),
             "D_minus_B": ("D_B_plus_hidden_linear", "B_market_numeric_ladder"), "B_minus_A": ("B_market_numeric_ladder", "A_market"),
             "B_minus_blend_frozen": ("B_market_numeric_ladder", "blend_frozen"), "C_minus_blend_frozen": ("C_B_plus_hidden_mlp", "blend_frozen"),
             "blend_frozen_minus_market": ("blend_frozen", "market_raw"), "C0_minus_B": ("C0_B_mlp_no_hidden", "B_market_numeric_ladder")}
    res["paired_brier_diff"] = {k: {"mean": brier(preds[a][m_te], yte) - brier(preds[b][m_te], yte),
                                    "ci95": paired_day_bootstrap(dte, preds[a][m_te], preds[b][m_te], yte)} for k, (a, b) in pairs.items()}
    # where: by minutes-left and |k| buckets, C - B and D - B
    te = t[m_te]
    hb = pd.cut(te["h"], [0, 9, 20, 35, 59]).astype(str).to_numpy(); kb = pd.cut(te["k"].abs(), [-1, 0, 1, 3, 12]).astype(str).to_numpy()
    res["by_h"] = {}; res["by_abs_k"] = {}
    for name, lab in (("by_h", hb), ("by_abs_k", kb)):
        for b in sorted(set(lab)):
            m = lab == b
            res[name][b] = {"n": int(m.sum()), "C_minus_B": brier(preds["C_B_plus_hidden_mlp"][m_te][m], yte[m]) - brier(preds["B_market_numeric_ladder"][m_te][m], yte[m]),
                            "D_minus_B": brier(preds["D_B_plus_hidden_linear"][m_te][m], yte[m]) - brier(preds["B_market_numeric_ladder"][m_te][m], yte[m]),
                            "B_minus_A": brier(preds["B_market_numeric_ladder"][m_te][m], yte[m]) - brier(preds["A_market"][m_te][m], yte[m])}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    torch.save(mlp.state_dict(), os.path.join(out, "head_C.pt"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--hidden", required=True)
    ap.add_argument("--es-days", type=int, default=4); ap.add_argument("--pca", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.val, a.dev, a.hidden, a.out, a.es_days, a.pca, a.seed)
    print(json.dumps({k: r[k] for k in ("rows", "hidden_dim", "test", "paired_brier_diff", "by_h", "by_abs_k", "C_train", "C0_train")}, indent=1, default=float))


if __name__ == "__main__":
    main()
