"""Calibration / blend layer fitted on validation and frozen before dev is opened.

Model:  logit P(Up) = a + b * logit(p_market) + sum_k c_k * feature_k
Features are chosen per run (e.g. z_model, log_rel_sd, logit_direct, logit(p_null), minutes_left/15).
The edge test is whether the model feature's coefficient has a day-block bootstrap CI excluding 0 given
the market term, and whether the blend's Brier beats the market's with a CI excluding 0.

    python -m btc_updown.calib.fit --table data/tables/val.parquet --features logit_direct --out data/calib/direct.json
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def design(table: pd.DataFrame, features: Sequence[str], include_market: bool) -> np.ndarray:
    cols = []
    if include_market:
        cols.append(logit(table["mkt_prob_up"].to_numpy()))
    for f in features:
        if f == "logit_null":
            cols.append(logit(table["p_null"].to_numpy()))
        elif f == "logit_samp":
            cols.append(logit(table["p_samp"].to_numpy()))
        elif f == "minutes_left":
            cols.append(table["h"].to_numpy().astype(float) / 15.0)
        else:
            cols.append(table[f].to_numpy().astype(float))
    X = np.stack(cols, axis=1) if cols else np.zeros((len(table), 0))
    return np.nan_to_num(np.clip(X, -20, 20), nan=0.0)


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns [intercept, coefs...] (very weak L2 so the fit is essentially unpenalized)."""
    if X.shape[1] == 0:
        p = np.clip(y.mean(), EPS, 1 - EPS)
        return np.array([np.log(p / (1 - p))])
    lr = LogisticRegression(C=1e4, max_iter=2000)
    lr.fit(X, y)
    return np.concatenate([lr.intercept_, lr.coef_[0]])


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(beta[0] + X @ beta[1:])


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def day_block_bootstrap(table: pd.DataFrame, features: Sequence[str], include_market: bool, n_boot: int, seed: int,
                        block_days: int = 1) -> Dict:
    """Resample whole days (optionally blocks of consecutive days) with replacement; refit; collect coefs and Brier gains."""
    days = np.array(sorted(table["label_day"].astype(str).unique()))
    rng = np.random.RandomState(seed)
    groups = table.groupby(table["label_day"].astype(str)).indices
    X_all = design(table, features, include_market)
    y_all = table["outcome"].to_numpy().astype(float)
    pm_all = table["mkt_prob_up"].to_numpy().astype(float) if include_market else None
    coefs, gains_mkt, gains_null = [], [], []
    n_blocks = int(np.ceil(len(days) / block_days))
    for _ in range(n_boot):
        starts = rng.randint(0, len(days), size=n_blocks)
        pick = np.concatenate([np.arange(s, min(s + block_days, len(days))) for s in starts])
        idx = np.concatenate([groups[days[i]] for i in pick])
        X, y = X_all[idx], y_all[idx]
        beta = fit_logistic(X, y)
        coefs.append(beta)
        p = predict(beta, X)
        if include_market:
            gains_mkt.append(brier(pm_all[idx], y) - brier(p, y))
        gains_null.append(brier(table["p_null"].to_numpy()[idx], y) - brier(p, y))
    coefs = np.array(coefs)
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
    return {
        "coef_ci95": {name: q(coefs[:, i]) for i, name in enumerate(["intercept"] + (["logit_market"] if include_market else []) + list(features))},
        "coef_mean": {name: float(coefs[:, i].mean()) for i, name in enumerate(["intercept"] + (["logit_market"] if include_market else []) + list(features))},
        "brier_gain_vs_market_ci95": q(np.array(gains_mkt)) if include_market else None,
        "brier_gain_vs_null_ci95": q(np.array(gains_null)),
        "n_boot": n_boot, "n_days": int(len(days)),
    }


def leave_one_day_out(table: pd.DataFrame, features: Sequence[str], include_market: bool) -> float:
    """Out-of-sample Brier of the blend across days (each day scored by a fit on the others)."""
    X_all = design(table, features, include_market)
    y_all = table["outcome"].to_numpy().astype(float)
    days = table["label_day"].astype(str).to_numpy()
    if len(np.unique(days)) < 2:
        return float("nan")
    p_oos = np.full(len(table), np.nan)
    for d in np.unique(days):
        m = days == d
        beta = fit_logistic(X_all[~m], y_all[~m])
        p_oos[m] = predict(beta, X_all[m])
    return brier(p_oos, y_all)


def fit(table: pd.DataFrame, features: Sequence[str], include_market: bool = True, n_boot: int = 500, seed: int = 0,
        require_cols: Optional[Sequence[str]] = None) -> Dict:
    need = list(features)
    if include_market:
        need.append("mkt_prob_up")
    need = [c if c not in ("logit_null", "logit_samp", "minutes_left") else {"logit_null": "p_null", "logit_samp": "p_samp", "minutes_left": "h"}[c] for c in need]
    t = table.dropna(subset=need + ["outcome"]).copy()
    t = t[t["outcome"].isin([0, 1])]
    if len(t) == 0:
        return {"error": "no rows with all required columns", "features": list(features)}
    X = design(t, features, include_market)
    y = t["outcome"].to_numpy().astype(float)
    beta = fit_logistic(X, y)
    p = predict(beta, X)
    out = {
        "features": list(features), "include_market": include_market, "n_rows": int(len(t)), "n_windows": int(t["market_id"].nunique()),
        "n_days": int(t["label_day"].nunique()), "base_rate": float(y.mean()),
        "coef": {name: float(b) for name, b in zip(["intercept"] + (["logit_market"] if include_market else []) + list(features), beta)},
        "brier": {
            "blend_insample": brier(p, y),
            "blend_leave_one_day_out": leave_one_day_out(t, features, include_market),
            "market": brier(t["mkt_prob_up"].to_numpy().astype(float), y) if include_market else None,
            "null": brier(t["p_null"].to_numpy().astype(float), y),
            "constant": brier(np.full(len(y), y.mean()), y),
        },
        "by_family": {},
    }
    for fam, g in t.groupby("family"):
        idx = g.index
        gi = t.index.get_indexer(idx)
        out["by_family"][fam] = {"n": int(len(g)), "blend": brier(p[gi], y[gi]), "null": brier(g["p_null"].to_numpy(), y[gi]),
                                 "market": brier(g["mkt_prob_up"].to_numpy().astype(float), y[gi]) if include_market else None}
    out["bootstrap"] = day_block_bootstrap(t, features, include_market, n_boot, seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--features", nargs="*", default=["logit_null"])
    ap.add_argument("--no-market", action="store_true")
    ap.add_argument("--venue", default=None); ap.add_argument("--family", default=None)
    ap.add_argument("--n-boot", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = pd.read_parquet(a.table)
    if a.venue:
        t = t[t["venue"] == a.venue]
    if a.family:
        t = t[t["family"] == a.family]
    res = fit(t, a.features, include_market=not a.no_market, n_boot=a.n_boot, seed=a.seed)
    s = json.dumps(res, indent=1, default=float)
    print(s)
    if a.out:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
