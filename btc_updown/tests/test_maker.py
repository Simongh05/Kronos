import numpy as np
import pandas as pd

from btc_updown.backtest import maker as M


def _table():
    # one market, minutes 1..59, book 0.40/0.42 throughout; blend says 0.47 (bid side) ; a second market where the blend says 0.35 (ask side)
    rows = []
    for tk, p in [("A", 0.47), ("B", 0.35)]:
        for m in range(1, 60):
            rows.append({"market_id": tk, "label_day": "2020-01-01", "t": 1000 + 60 * m, "j": m, "h": 60 - m, "k": 0, "bid": 0.40, "ask": 0.42,
                         "two_sided": True, "outcome": 1 if tk == "A" else 0, "p": p})
    return pd.DataFrame(rows)


def test_improve_fills_on_taker_sell_at_or_below_bid():
    t = _table()
    # taker sells yes at 0.40 at minute 10 (ts in (t10, t10+60]) -> fills our improving bid at 0.41 (= ask - 1 tick, below p - margin = 0.46)
    fills = pd.DataFrame({"ticker": ["A", "A", "B"], "ts": [1000 + 600 + 30, 1000 + 1200, 1000 + 900 + 5], "yes_price": [0.40, 0.40, 0.42],
                          "count": [50.0, 500.0, 500.0], "taker_side": ["no", "no", "yes"]})
    tr = M.simulate(t, fills, t["p"].to_numpy(), mode="improve", margin_ticks=1, clip_usd=100.0, maker_fee=0.0)
    a = tr[tr.market_id == "A"].iloc[0]
    assert a.side == "yes" and abs(a.yes_price - 0.41) < 1e-9 and a.h == 59 and a.fill_h == 50 and a.shares == 50 and a.win and abs(a.pnl - 50 * 0.59) < 1e-9
    b = tr[tr.market_id == "B"].iloc[0]                                   # blend 0.35 -> ask at max(0.36, bid+1) = 0.41; taker buys at 0.42 >= 0.41
    assert b.side == "no" and abs(b.yes_price - 0.41) < 1e-9 and b.win and abs(b.price - 0.59) < 1e-9
    # join mode: only trades strictly through the price fill; a print at 0.40 does not fill a bid joined at 0.40
    tr2 = M.simulate(t, fills, t["p"].to_numpy(), mode="join", margin_ticks=1)
    assert tr2.empty
    fills3 = pd.concat([fills, pd.DataFrame({"ticker": ["A"], "ts": [1000 + 1800 + 1], "yes_price": [0.39], "count": [10.0], "taker_side": ["no"]})])
    tr3 = M.simulate(t, fills3, t["p"].to_numpy(), mode="join", margin_ticks=1)
    assert len(tr3) == 1 and abs(tr3.iloc[0].yes_price - 0.40) < 1e-9 and tr3.iloc[0].shares == 10
    # maker fee applied
    tr4 = M.simulate(t, fills, t["p"].to_numpy(), mode="improve", margin_ticks=1, maker_fee=0.0175)
    assert tr4.iloc[0].fee > 0 and tr4.iloc[0].pnl < a.pnl


def test_no_post_when_blend_agrees_with_book():
    t = _table(); t["p"] = 0.41                                            # inside the spread with margin 1 tick: no improving price either side
    fills = pd.DataFrame({"ticker": ["A"], "ts": [1000 + 630], "yes_price": [0.40], "count": [50.0], "taker_side": ["no"]})
    assert M.simulate(t, fills, t["p"].to_numpy(), mode="improve", margin_ticks=1).empty
