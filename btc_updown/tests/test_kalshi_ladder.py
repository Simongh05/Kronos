import numpy as np
import pandas as pd

from btc_updown.data import kalshi_ladder as K


def test_parse_candle_and_days():
    c = {"end_period_ts": 1792875660, "yes_bid": {"open_dollars": "0.40", "close_dollars": "0.41"}, "yes_ask": {"close_dollars": "0.43"},
         "price": {"close_dollars": None, "mean_dollars": "0.42"}, "volume_fp": "12.5", "open_interest_fp": "100"}
    r = K.parse_candle(c)
    assert r["t"] == 1792875660 and r["bid"] == 0.41 and r["ask"] == 0.43 and np.isnan(r["px"]) and r["volume"] == 12.5
    assert K.days_between("2026-08-20", "2026-08-23") == ["2026-08-20", "2026-08-21", "2026-08-22"]


def test_atm_strikes_uses_previous_settlement():
    rows = []
    for ev, (o, exp) in {"E1": (0, 100_050.0), "E2": (3600, 100_420.0)}.items():
        for s in range(99_000, 101_100, 100):
            rows.append({"ticker": f"{ev}-T{s}", "event_ticker": ev, "strike": float(s) - 0.01, "open_time": o, "close_time": o + 3600,
                         "hours": 1.0, "expiration_value": exp, "volume": 1000.0 if s == 100_400 else 1.0})
    m = pd.DataFrame(rows)
    sel = K.atm_strikes(m, 2)
    e1 = sel[sel.event_ticker == "E1"]; e2 = sel[sel.event_ticker == "E2"]
    assert len(e1) == 5 and len(e2) == 5
    assert abs(e1.strike.median() - 100_399.99) < 1          # no previous ladder: highest-volume strike
    assert abs(e2.strike.median() - 99_999.99) < 1 or abs(e2.strike.median() - 100_099.99) < 1   # nearest to E1's settlement 100,050
