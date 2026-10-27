"""Vol test (free, Mac only): does the Kronos hidden state forecast realized 60-minute variance better than the
standard vol models plus time of day?

Rows: every minute t with a hidden-state row (the ladder window, 2026-08-20 → 10-24). Target y = log RV_fwd, the sum
of squared 1-minute log returns of the 60 bars after t. Predictors (all on the same information set, bars closed by t):
    ewma    log of 60 × EWMA variance at halflives 15, 60 (the null's), 240, 1440 min
    har     log RV over the past 60, 360, 1440, 10080 min (HAR-RV at minute resolution)
    tod     hour-of-day and day-of-week one-hots (vol seasonality; Kronos sees time features, so this must be a control)
    hidden  512-d backbone state at t, standardized, PCA
Models (ridge on val days, alpha picked on the early-stop days; HistGradientBoosting for the nonlinear pair):
    N0 ewma60 | N1 ewma+har | N2 N1+tod | N3 N2+hidden | N4 tod+hidden (no vol features) | G2/G3 = HGB on N2/N3 features
Readouts on dev days: MSE of log RV, QLIKE, R²; paired day-block bootstrap of the MSE differences.

    python -m btc_updown.calib.vol_test --klines data/klines --series spot_BTCUSDC \
        --hidden data/cloud/cubes/cube_e1b_hidden_kalshi/hidden --out data/calib/vol_test
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from ..data.nulls import ewma_sigma
from ..train.dataset import SeriesStore
from .market_head import load_hidden

EPS = 1e-12


def _day_boot(days, a, b, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed); ud = np.unique(days); groups = {d: np.nonzero(days == d)[0] for d in ud}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ud, size=len(ud), replace=True); idx = np.concatenate([groups[d] for d in pick])
        diffs.append(float(np.mean(a[idx]) - np.mean(b[idx])))
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def run(klines: str, series: str, hidden_dir: str, out: str, split_day: str, es_days: int, pca: int, seed: int) -> Dict:
    os.makedirs(out, exist_ok=True)
    st = SeriesStore.open(klines, series)
    ts = np.asarray(st.ts); close = np.asarray(st.x[:, 3], dtype=np.float64)
    r = np.zeros(len(close)); r[1:] = np.diff(np.log(close)); r2 = r ** 2
    cs = np.concatenate([[0.0], np.cumsum(r2)])            # cs[i] = sum r2[:i]
    ht, H = load_hidden(hidden_dir)
    pos = pd.Series(np.arange(len(ts)), index=ts)
    # the decision at t uses bars with open <= t-60 (closed by t): bar index i_t = position of open time t-60
    it = pos.reindex(ht - 60).to_numpy()
    ok = ~np.isnan(it) & (it >= 10080 + 1) & (it + 61 < len(ts))
    it = it[ok].astype(int); ht = ht[ok]; H = H[ok]
    # contiguity: require the 60 forward bars and the 10080 back bars to be a regular 1-min grid
    ok2 = (ts[it + 60] - ts[it] == 3600) & (ts[it] - ts[it - 10080] == 10080 * 60)
    it, ht, H = it[ok2], ht[ok2], H[ok2]
    rv_fwd = cs[it + 61] - cs[it + 1]
    y = np.log(rv_fwd + EPS)
    feats = {}
    for hl in (15, 60, 240, 1440):
        feats[f"ewma{hl}"] = np.log(60 * ewma_sigma(close, hl)[it] ** 2 + EPS)
    for w in (60, 360, 1440, 10080):
        feats[f"har{w}"] = np.log((cs[it + 1] - cs[it + 1 - w]) * (60.0 / w) + EPS)
    hour = ((ht // 3600) % 24).astype(int); dow = ((ht // 86400 + 3) % 7).astype(int)   # 1970-01-01 was a Thursday
    tod = np.concatenate([np.eye(24)[hour], np.eye(7)[dow]], 1)
    day = np.array([pd.Timestamp(int(x), unit="s").strftime("%Y-%m-%d") for x in ht])
    days = np.array(sorted(set(day))); val_days = days[days < split_day]; test_days = days[days >= split_day]
    es_set, fit_set = set(val_days[-es_days:]), set(val_days[:-es_days])
    m_fit = np.isin(day, list(fit_set)); m_es = np.isin(day, list(es_set)); m_te = np.isin(day, test_days); tr = m_fit | m_es
    Xv = np.stack([feats[k] for k in feats], 1)
    mu, sd = Xv[m_fit].mean(0), Xv[m_fit].std(0) + 1e-9; Xv = (Xv - mu) / sd
    Hs = (H - H[m_fit].mean(0)) / (H[m_fit].std(0) + 1e-6)
    if pca and pca < Hs.shape[1]:
        U, S, Vt = np.linalg.svd(Hs[m_fit][::4], full_matrices=False); Hs = Hs @ Vt[:pca].T; Hs = Hs / (Hs[m_fit].std(0) + 1e-6)
    sets = {"N0_ewma60": Xv[:, [1]], "N1_ewma_har": Xv, "N2_N1_tod": np.concatenate([Xv, tod], 1),
            "N3_N2_hidden": np.concatenate([Xv, tod, Hs], 1), "N4_tod_hidden": np.concatenate([tod, Hs], 1), "N5_hidden_only": Hs}
    preds: Dict[str, np.ndarray] = {}
    res = {"rows": {"fit": int(m_fit.sum()), "es": int(m_es.sum()), "test": int(m_te.sum())}, "days": {"val": list(val_days), "test": list(test_days)},
           "hidden_dim": int(Hs.shape[1]), "alpha": {}}
    for name, X in sets.items():
        best = None
        for alpha in (0.1, 1, 10, 100, 1000):
            m = Ridge(alpha=alpha).fit(X[m_fit], y[m_fit]); e = float(np.mean((m.predict(X[m_es]) - y[m_es]) ** 2))
            if best is None or e < best[0]:
                best = (e, alpha)
        res["alpha"][name] = best[1]
        preds[name] = Ridge(alpha=best[1]).fit(X[tr], y[tr]).predict(X)
    for name, X in (("G2_hgb_N2", sets["N2_N1_tod"]), ("G3_hgb_N3", sets["N3_N2_hidden"])):
        g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, early_stopping=False, random_state=seed)
        preds[name] = g.fit(X[tr], y[tr]).predict(X)
    yte = y[m_te]; dte = day[m_te]; rv_te = rv_fwd[m_te]
    var_y = float(np.var(yte))
    res["test"] = {}
    for k, p in preds.items():
        pt = p[m_te]; sig2 = np.exp(pt)
        res["test"][k] = {"mse_logrv": float(np.mean((pt - yte) ** 2)), "r2_logrv": 1 - float(np.mean((pt - yte) ** 2)) / var_y,
                          "qlike": float(np.mean(rv_te / sig2 - np.log(rv_te / sig2) - 1))}
    def se(k): return (preds[k][m_te] - yte) ** 2
    pairs = {"N3_minus_N2": ("N3_N2_hidden", "N2_N1_tod"), "N2_minus_N1": ("N2_N1_tod", "N1_ewma_har"), "N1_minus_N0": ("N1_ewma_har", "N0_ewma60"),
             "N4_minus_N2": ("N4_tod_hidden", "N2_N1_tod"), "G3_minus_G2": ("G3_hgb_N3", "G2_hgb_N2"), "G2_minus_N2": ("G2_hgb_N2", "N2_N1_tod")}
    res["paired_mse_diff"] = {k: {"mean": float(np.mean(se(a)) - np.mean(se(b))), "ci95": _day_boot(dte, se(a), se(b))} for k, (a, b) in pairs.items()}
    def ql(k):
        sig2 = np.exp(preds[k][m_te]); return rv_te / sig2 - np.log(rv_te / sig2) - 1
    res["paired_qlike_diff"] = {k: {"mean": float(np.mean(ql(a)) - np.mean(ql(b))), "ci95": _day_boot(dte, ql(a), ql(b))} for k, (a, b) in pairs.items()}
    # bias diagnostic: QLIKE rewards a model whose variance level happens to sit closer to the test regime; report the
    # test-set bias of each model and the QLIKE pair after removing each model's own es-day bias (an honest level fix)
    res["bias_test_logrv"] = {k: float(np.mean(preds[k][m_te] - yte)) for k in preds}
    res["bias_es_logrv"] = {k: float(np.mean(preds[k][m_es] - y[m_es])) for k in preds}
    def ql_fix(k):
        sig2 = np.exp(preds[k][m_te] - res["bias_es_logrv"][k]); return rv_te / sig2 - np.log(rv_te / sig2) - 1
    res["paired_qlike_diff_es_bias_removed"] = {k: {"mean": float(np.mean(ql_fix(a)) - np.mean(ql_fix(b))), "ci95": _day_boot(dte, ql_fix(a), ql_fix(b))} for k, (a, b) in pairs.items()}
    res["test_qlike_es_bias_removed"] = {k: float(np.mean(ql_fix(k))) for k in preds}
    # by-month split of the N3 - N2 QLIKE difference (is it one regime?)
    mon = np.array([d[:7] for d in dte]); res["qlike_N3_minus_N2_by_month"] = {m: float(np.mean(ql("N3_N2_hidden")[mon == m]) - np.mean(ql("N2_N1_tod")[mon == m])) for m in sorted(set(mon))}
    # the hidden state's own vol content: R² of hidden-only vs ewma-only on test
    res["corr_test"] = {"N5_hidden_only_vs_y": float(np.corrcoef(preds["N5_hidden_only"][m_te], yte)[0, 1]),
                        "N0_vs_y": float(np.corrcoef(preds["N0_ewma60"][m_te], yte)[0, 1]),
                        "N5_vs_N2": float(np.corrcoef(preds["N5_hidden_only"][m_te], preds["N2_N1_tod"][m_te])[0, 1])}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines", default="data/klines"); ap.add_argument("--series", default="spot_BTCUSDC"); ap.add_argument("--hidden", required=True)
    ap.add_argument("--split-day", default="2026-09-21"); ap.add_argument("--es-days", type=int, default=4); ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.klines, a.series, a.hidden, a.out, a.split_day, a.es_days, a.pca, a.seed)
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
