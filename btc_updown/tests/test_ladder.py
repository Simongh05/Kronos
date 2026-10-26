import numpy as np
import pandas as pd
import torch

from btc_updown.backtest import ladder as L
from btc_updown.train.dataset import SeriesStore
from btc_updown.train.heads import DirectHead


def _archive(tmp_path, store):
    ts = np.asarray(store.ts); x = np.asarray(store.x)
    mk, cd = [], []
    k0 = 400
    for e in range(3):
        o = int(ts[k0 + 60 * e]) + 60; c = o + 3600                    # open at a bar open; close 60 min later
        i_c = store.index_of_bar_closing_at(c); final = float(x[i_c, :4].mean()) - 14.0
        for s in range(-5, 6):
            strike = round(final / 100) * 100 + 100 * s - 0.01
            tk = f"E{e}-T{strike:.2f}"
            mk.append({"ticker": tk, "event_ticker": f"E{e}", "strike": strike, "strike_type": "greater", "open_time": o, "close_time": c,
                       "hours": 1.0, "expiration_value": final, "result": "yes" if final > strike else "no", "status": "finalized",
                       "volume": 100.0, "open_interest": 0.0, "settlement_ts": None, "label_day": "2020-01-01"})
            p = 0.5 - 0.1 * s
            cd.append({"ticker": tk, "event_ticker": f"E{e}", "strike": strike, "t": o + 60 * 3, "m": 3, "bid": max(p - 0.01, 0.0), "ask": min(p + 0.01, 1.0),
                       "px": p, "volume": 1.0, "open_interest": 0.0})       # one candle at minute 3, then nothing: forward-filled
            cd.append({"ticker": tk, "event_ticker": f"E{e}", "strike": strike, "t": o + 60 * 40, "m": 40, "bid": max(p - 0.02, 0.0), "ask": min(p + 0.02, 1.0),
                       "px": p, "volume": 1.0, "open_interest": 0.0})
    (tmp_path / "markets").mkdir(); (tmp_path / "candles").mkdir()
    pd.DataFrame(mk).to_parquet(tmp_path / "markets" / "2020-01-01.parquet", index=False)
    pd.DataFrame(cd).to_parquet(tmp_path / "candles" / "2020-01-01.parquet", index=False)


def test_ladder_table(tmp_path, store_root):
    store = SeriesStore.open(store_root, "synthetic")
    _archive(tmp_path, store)
    # a random head + a fake hidden dump covering every minute
    head = DirectHead(16, hidden=8, max_h=60); hp = tmp_path / "head.pt"
    torch.save({"d_model": 16, "hidden": 8, "max_h": 60, "dropout": 0.1, "state_dict": head.state_dict()}, hp)
    hd = tmp_path / "hidden"; hd.mkdir()
    ts = np.asarray(store.ts)[300:1000] + 60
    np.save(hd / "2020-01-01.t.npy", ts); np.save(hd / "2020-01-01.npy", np.random.RandomState(0).randn(len(ts), 16).astype(np.float16))
    t = L.build(str(tmp_path), store_root, "synthetic", "2019-01-01", "2021-01-01", max_k=3, hidden=str(hd), head=str(hp))
    assert len(t) > 0 and set(t["m"] if "m" in t else t["j"]) <= set(range(1, 60))
    # minutes before the first candle have no book; minutes 3..39 carry the minute-3 book, 40..59 the minute-40 book
    e0 = t[t.event_ticker == "E0"]
    assert e0["j"].min() == 3 and e0["j"].max() == 59
    r = e0[(e0.k == 0)].set_index("j")
    assert abs(r.loc[10, "ask"] - r.loc[3, "ask"]) < 1e-9 and abs(r.loc[45, "ask"] - r.loc[40, "ask"]) < 1e-9
    assert (t["h"] == 60 - t["j"]).all() and t["outcome"].isin([0, 1]).all()
    assert (t["k"].abs() <= 3).all()
    # the first ladder has no previous settlement -> default basis; later ones use the previous settlement
    assert (t[t.event_ticker == "E0"]["basis_src"] == "default").all() and (t[t.event_ticker == "E2"]["basis_src"] == "prev_settle").all()
    assert t["p_null"].between(0, 1).all() and t["logit_direct"].notna().all()
    assert (t.loc[t.two_sided, "mkt_prob_up"].between(0, 1)).all()
