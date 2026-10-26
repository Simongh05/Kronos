"""Market-conditioned head: can Kronos's hidden state improve on the market price when it SEES the price?

Rows: validation decision minutes with an executable market price (assemble output). Features:
    market   logit(mkt_prob_up)
    numeric  (strike_series-close_t)/close_t (bps), log sigma_ewma, h/15, j/15, logit(p_null)
    hidden   Kronos backbone state at t (d_model, from cube hidden/<day>.npy), standardized, optional PCA
Models (all trained on the TRAIN days, early-stopped on the last train days, evaluated on the TEST days):
    A  market only                       (logistic)
    B  market + numeric                  (logistic)
    C  market + numeric + hidden         (MLP, dropout, weight decay)
    D  market + numeric + hidden, linear (ridge logistic) as a sanity check against C
Readouts on TEST days: Brier / log-loss per model, paired day-level bootstrap of Brier(C) - Brier(A) and (C) - (B),
and a backtest with side-consistent market input (ask of the side considered).

    python -m btc_updown.calib.market_head --table data/tables/val_e1b_direct.parquet \
        --hidden data/cloud/cubes/cube_e1b_hidden_val/hidden --train-end 2026-08-25 --out data/calib/market_head
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .fit import EPS, logit, sigmoid
from ..backtest.engine import day_stats, fee


def load_hidden(hdir: str) -> Tuple[np.ndarray, np.ndarray]:
    ts, hs = [], []
    for f in sorted(glob.glob(os.path.join(hdir, "*.t.npy"))):
        day = os.path.basename(f)[:10]
        t = np.load(f); h = np.load(os.path.join(hdir, f"{day}.npy"))
        if len(t) and len(t) == len(h):
            ts.append(t); hs.append(h.astype(np.float32))
    return np.concatenate(ts), np.concatenate(hs)


def numeric_feats(t: pd.DataFrame) -> np.ndarray:
    rel = (t["strike_series"] - t["close_series_t"]) / t["close_series_t"] * 1e4
    return np.stack([rel.to_numpy(), np.log(t["sigma_ewma"].to_numpy()), t["h"].to_numpy() / 15.0, t["j"].to_numpy() / 15.0,
                     logit(t["p_null"].to_numpy())], axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        # skip connection on the market logit so the MLP starts from "trust the market"
        self.market_w = nn.Parameter(torch.ones(1))

    def forward(self, x_market: torch.Tensor, x_rest: torch.Tensor) -> torch.Tensor:
        return self.market_w * x_market + self.net(torch.cat([x_market[:, None], x_rest], dim=1)).squeeze(-1)


def train_mlp(Xm_tr, Xr_tr, y_tr, Xm_es, Xr_es, y_es, epochs=60, lr=1e-3, wd=1e-3, hidden=128, dropout=0.3, seed=0) -> Tuple[MLP, Dict]:
    torch.manual_seed(seed)
    m = MLP(1 + Xr_tr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xm_tr_t, Xr_tr_t, y_tr_t = map(torch.as_tensor, (Xm_tr, Xr_tr, y_tr))
    Xm_es_t, Xr_es_t, y_es_t = map(torch.as_tensor, (Xm_es, Xr_es, y_es))
    best, best_state, bad, hist = np.inf, None, 0, []
    n = len(y_tr); bs = 512
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(m(Xm_tr_t[idx], Xr_tr_t[idx]), y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            es = float(F.binary_cross_entropy_with_logits(m(Xm_es_t, Xr_es_t), y_es_t))
        hist.append(es)
        if es < best - 1e-5:
            best, bad, best_state = es, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    m.load_state_dict(best_state); m.eval()
    return m, {"best_es_logloss": best, "epochs_run": len(hist)}


def brier(p, y): return float(np.mean((p - y) ** 2))
def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired_day_bootstrap(days: np.ndarray, a: np.ndarray, b: np.ndarray, y: np.ndarray, n_boot=2000, seed=0):
    """CI of mean Brier(a) - Brier(b) resampling days."""
    rng = np.random.RandomState(seed); ud = np.unique(days); groups = {d: np.nonzero(days == d)[0] for d in ud}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ud, size=len(ud), replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        diffs.append(brier(a[idx], y[idx]) - brier(b[idx], y[idx]))
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def run(table_path: str, hidden_dir: str, train_end: str, es_days: int, out: str, pca: int, seed: int, venue: str = "polymarket"):
    os.makedirs(out, exist_ok=True)
    t = pd.read_parquet(table_path)
    t = t[(t["venue"] == venue) & t["outcome"].isin([0, 1]) & t["mkt_prob_up"].notna() & t["p_null"].notna()].copy()
    ts, H = load_hidden(hidden_dir)
    pos = pd.Series(np.arange(len(ts)), index=ts)
    idx = pos.reindex(t["t"].to_numpy()).to_numpy()
    ok = ~np.isnan(idx); t = t[ok].copy(); Hrows = H[idx[ok].astype(int)]
    t["day"] = t["label_day"].astype(str)
    days = np.array(sorted(t["day"].unique()))
    train_days = days[days < train_end]; test_days = days[days >= train_end]
    es_set = set(train_days[-es_days:]); fit_set = set(train_days[:-es_days])
    m_fit, m_es, m_te = t["day"].isin(fit_set).to_numpy(), t["day"].isin(es_set).to_numpy(), t["day"].isin(test_days).to_numpy()
    y = t["outcome"].to_numpy().astype(np.float32)
    xm = logit(t["mkt_prob_up"].to_numpy()).astype(np.float32)
    xn = numeric_feats(t)
    # standardize numeric + hidden on the fit set; optional PCA on hidden
    mu_n, sd_n = xn[m_fit].mean(0), xn[m_fit].std(0) + 1e-6; xn = (xn - mu_n) / sd_n
    mu_h, sd_h = Hrows[m_fit].mean(0), Hrows[m_fit].std(0) + 1e-6; xh = (Hrows - mu_h) / sd_h
    if pca and pca < xh.shape[1]:
        U, S, Vt = np.linalg.svd(xh[m_fit], full_matrices=False); comps = Vt[:pca]
        xh = xh @ comps.T; xh = xh / (xh[m_fit].std(0) + 1e-6)
    xh = xh.astype(np.float32)
    res = {"rows": {"fit": int(m_fit.sum()), "early_stop": int(m_es.sum()), "test": int(m_te.sum())},
           "days": {"fit": sorted(fit_set), "early_stop": sorted(es_set), "test": list(test_days)}, "hidden_dim": int(xh.shape[1]),
           "base_rate_test": float(y[m_te].mean())}
    preds: Dict[str, np.ndarray] = {}
    # A / B: logistic regressions (fit on fit+es days)
    tr = m_fit | m_es
    def logreg(X):
        lr = LogisticRegression(C=1.0, max_iter=3000); lr.fit(X[tr], y[tr]); return lr.predict_proba(X)[:, 1], lr
    preds["A_market"], lrA = logreg(xm[:, None])
    preds["B_market_numeric"], lrB = logreg(np.concatenate([xm[:, None], xn], 1))
    preds["D_market_numeric_hidden_linear"], lrD = logreg(np.concatenate([xm[:, None], xn, xh], 1))
    res["A_coef_market"] = float(lrA.coef_[0][0])
    # C: MLP with early stopping on the es days
    Xr = np.concatenate([xn, xh], 1)
    mlp, info = train_mlp(xm[m_fit], Xr[m_fit], y[m_fit], xm[m_es], Xr[m_es], y[m_es], seed=seed)
    with torch.no_grad():
        preds["C_market_numeric_hidden_mlp"] = sigmoid(mlp(torch.as_tensor(xm), torch.as_tensor(Xr)).numpy())
    res["C_train"] = info; res["C_market_skip_weight"] = float(mlp.market_w.item())
    # also an MLP WITHOUT hidden (same capacity) to separate "nonlinearity" from "Kronos"
    mlp0, info0 = train_mlp(xm[m_fit], xn[m_fit], y[m_fit], xm[m_es], xn[m_es], y[m_es], seed=seed)
    with torch.no_grad():
        preds["C0_market_numeric_mlp"] = sigmoid(mlp0(torch.as_tensor(xm), torch.as_tensor(xn)).numpy())
    preds["market_raw"] = t["mkt_prob_up"].to_numpy().astype(np.float32)
    dte = t["day"].to_numpy()[m_te]; yte = y[m_te]
    res["test"] = {k: {"brier": brier(v[m_te], yte), "logloss": logloss(v[m_te], yte)} for k, v in preds.items()}
    res["test"]["null"] = {"brier": brier(t["p_null"].to_numpy()[m_te], yte)}
    res["paired_brier_diff_ci95"] = {
        "C_minus_A": paired_day_bootstrap(dte, preds["C_market_numeric_hidden_mlp"][m_te], preds["A_market"][m_te], yte),
        "C_minus_B": paired_day_bootstrap(dte, preds["C_market_numeric_hidden_mlp"][m_te], preds["B_market_numeric"][m_te], yte),
        "C_minus_C0": paired_day_bootstrap(dte, preds["C_market_numeric_hidden_mlp"][m_te], preds["C0_market_numeric_mlp"][m_te], yte),
        "D_minus_B": paired_day_bootstrap(dte, preds["D_market_numeric_hidden_linear"][m_te], preds["B_market_numeric"][m_te], yte),
        "B_minus_A": paired_day_bootstrap(dte, preds["B_market_numeric"][m_te], preds["A_market"][m_te], yte),
        "A_minus_market_raw": paired_day_bootstrap(dte, preds["A_market"][m_te], preds["market_raw"][m_te], yte),
    }
    # backtest on TEST days with side-consistent market input: re-evaluate C with the side's ask as the market feature
    te = t[m_te].copy(); Xr_te = Xr[m_te]
    def c_prob(mkt):
        with torch.no_grad():
            return sigmoid(mlp(torch.as_tensor(logit(mkt).astype(np.float32)), torch.as_tensor(Xr_te)).numpy())
    te["p_up"] = c_prob(te["exec_price_up"].fillna(0.5).to_numpy())
    te["p_dn"] = 1 - c_prob(1 - te["exec_price_down"].fillna(0.5).to_numpy())
    trades = []
    for mid, g in te.sort_values(["market_id", "t"]).groupby("market_id", sort=False):
        for r in g.itertuples(index=False):
            best = None
            for side, price, size, ps in (("up", r.exec_price_up, r.exec_size_up, r.p_up), ("down", r.exec_price_down, r.exec_size_down, r.p_dn)):
                if not (isinstance(price, float) and np.isfinite(price)) or price <= 0 or price >= 1:
                    continue
                f = float(fee(venue, np.array([price]))[0]); edge = ps - price - f
                if edge > 0 and (best is None or edge > best["edge"]):
                    best = {"side": side, "price": price, "fee": f, "edge": edge, "size": size if np.isfinite(size) else np.inf}
            if best is None:
                continue
            shares = min(100.0 / best["price"], best["size"]); win = (r.outcome == 1) if best["side"] == "up" else (r.outcome == 0)
            pnl_share = (1.0 if win else 0.0) - best["price"] - best["fee"]
            trades.append({"label_day": str(r.label_day), "market_id": mid, "h": r.h, "side": best["side"], "price": best["price"], "fee": best["fee"],
                           "edge": best["edge"], "shares": shares, "notional": shares * best["price"], "win": bool(win), "pnl": pnl_share * shares, "pnl_share": pnl_share})
            break
    tr_df = pd.DataFrame(trades)
    tr_df.to_parquet(os.path.join(out, "trades_test_C.parquet"), index=False)
    res["backtest_test_C"] = day_stats(tr_df, hurdle_per_day=150.0, n_boot=2000, all_days=list(test_days)) if len(tr_df) else {"n_trades": 0}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    torch.save(mlp.state_dict(), os.path.join(out, "head_C.pt"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--hidden", required=True)
    ap.add_argument("--train-end", default="2026-08-25"); ap.add_argument("--es-days", type=int, default=3)
    ap.add_argument("--pca", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.table, a.hidden, a.train_end, a.es_days, a.out, a.pca, a.seed)
    print(json.dumps({k: res[k] for k in ("rows", "hidden_dim", "base_rate_test", "A_coef_market", "C_train", "C_market_skip_weight", "test", "paired_brier_diff_ci95")}, indent=1, default=float))
    bt = res["backtest_test_C"]; print("backtest C (test days):", json.dumps({k: bt[k] for k in bt if k not in ("by_minutes_left", "by_price_bucket")}, default=float))


if __name__ == "__main__":
    main()
