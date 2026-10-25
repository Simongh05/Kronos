"""assemble / calib.fit / backtest.engine on synthetic data."""
import json
import os

import numpy as np
import pandas as pd

from btc_updown.backtest import assemble as A
from btc_updown.backtest import engine as E
from btc_updown.calib import fit as F
from btc_updown.train.dataset import SeriesStore, typical_price


def _windows(store: SeriesStore, n=400, dur=5):
    ts = np.asarray(store.ts)
    tp = typical_price(np.asarray(store.x, dtype=np.float64))
    rows = []
    k = 300
    while len(rows) < n and k + dur < len(ts) - 20:
        ws = int(ts[k]) + 60
        we = ws + 60 * dur
        i0, i1 = store.index_of_bar_closing_at(ws), store.index_of_bar_closing_at(we)
        if i0 >= 0 and i1 >= 0:
            rows.append({"venue": "polymarket", "family": "btc-5m", "market_id": f"m{ws}", "condition_id": "c", "token_up": "u",
                         "token_down": "d", "window_start": ws, "window_end": we, "strike": tp[i0], "final": tp[i1],
                         "outcome": int(tp[i1] >= tp[i0]), "regime": "btc-5m-twap-60", "tape_regime": None, "source": "test",
                         "label_day": pd.Timestamp(ws, unit="s").date()})
        k += dur
    return pd.DataFrame(rows)


def _prices(w: pd.DataFrame, rng, informative=True):
    rows = []
    for r in w.itertuples(index=False):
        for k in range(1, 5):
            t = r.window_start + 60 * k
            p_true = 0.65 if r.outcome == 1 else 0.35
            p = np.clip(p_true + rng.normal(0, 0.15), 0.02, 0.98) if informative else np.clip(0.5 + rng.normal(0, 0.1), 0.02, 0.98)
            rows.append({"market_id": r.market_id, "family": r.family, "window_start": r.window_start, "window_end": r.window_end,
                         "t": t, "minutes_left": 5 - k, "exec_price_up": round(p + 0.01, 2), "exec_size_up": 50.0, "exec_time_up": t + 5.0,
                         "exec_price_down": round(1 - p + 0.01, 2), "exec_size_down": 50.0, "exec_time_down": t + 6.0,
                         "mkt_prob_up": p, "mkt_prob_src": "both", "n_prints_30s": 3})
    return pd.DataFrame(rows)


def test_assemble_fit_and_engine(tmp_path, store_root):
    store = SeriesStore.open(store_root, "synthetic")
    rng = np.random.RandomState(0)
    w = _windows(store)
    assert len(w) > 50
    wp, pp = tmp_path / "w.parquet", tmp_path / "p.parquet"
    w.to_parquet(wp, index=False); _prices(w, rng).to_parquet(pp, index=False)
    t0 = str(pd.Timestamp(int(w.window_start.min()), unit="s").floor("D").date())
    t1 = str((pd.Timestamp(int(w.window_end.max()), unit="s").floor("D") + pd.Timedelta(days=1)).date())
    table = A.build(str(wp), str(pp), None, store_root, "synthetic", t0, t1)
    # one row per (window, minute 1..4), strike_series == strike (labels built from the same series)
    assert len(table) == 4 * len(w)
    assert np.allclose(table["strike_series"], table["strike"])
    assert table["p_null"].between(0, 1).all() and table["mkt_prob_up"].notna().all()
    # a model feature that is pure noise must get a CI containing zero; the market term is informative
    table["noise_feat"] = rng.normal(size=len(table))
    res = F.fit(table, ["noise_feat", "logit_null"], include_market=True, n_boot=50)
    assert res["brier"]["market"] < res["brier"]["constant"]
    assert abs(res["coef"]["noise_feat"]) < 0.25 and res["coef"]["logit_market"] > 0.5
    assert res["n_days"] >= 2 and np.isfinite(res["brier"]["blend_leave_one_day_out"])
    # an oracle feature must dominate
    table["oracle"] = np.where(table["outcome"] == 1, 3.0, -3.0) + rng.normal(0, 0.3, len(table))
    res2 = F.fit(table, ["oracle"], include_market=True, n_boot=20)
    assert res2["brier"]["blend_insample"] < res2["brier"]["market"] * 0.5
    assert res2["bootstrap"]["coef_ci95"]["oracle"][0] > 0
    # engine: with the oracle the backtest makes money; one trade per window at most; no entry with < 1 minute left
    calib_path = tmp_path / "calib.json"
    json.dump(res2, open(calib_path, "w"), default=float)
    trades = E.run_backtest(table, json.load(open(calib_path)), clip_usd=100.0)
    assert len(trades) > 0 and trades["market_id"].is_unique and (trades["h"] >= 1).all()
    assert trades["pnl"].sum() > 0
    stats = E.day_stats(trades, hurdle_per_day=0.0, n_boot=200)
    assert stats["n_trades"] == len(trades) and stats["cents_per_share"] > 0
    # with market-only information (noise feature) the rule should barely trade and not be significantly profitable
    res3 = F.fit(table, ["noise_feat"], include_market=True, n_boot=20)
    trades3 = E.run_backtest(table, res3, clip_usd=100.0)
    assert trades3["pnl"].sum() < trades["pnl"].sum()
    # fee formulas
    assert abs(E.fee_polymarket(np.array([0.5]))[0] - 0.0175) < 1e-9
    assert E.fee_kalshi(np.array([0.5]))[0] >= 0.0175
    bt = E.brier_table(table, res2)
    assert set(bt["by_h"]) == {1, 2, 3, 4} and bt["blend"] < bt["market"]
