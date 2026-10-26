"""pm_long: slug -> window parsing (DST-safe), the executable-price rule on public taker orders, label helper."""
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from btc_updown.data import pm_long as P

NY = ZoneInfo("America/New_York")


def _ny(y, m, d, h):
    return int(dt.datetime(y, m, d, h, tzinfo=NY).timestamp())


def test_hourly_slug_parsing_handles_year_dst_and_candle_suffix():
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-may-28-1-am-et-candle", 2025) == (_ny(2025, 5, 28, 1), _ny(2025, 5, 28, 1) + 3600)
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-july-30-4am-et", 2025) == (_ny(2025, 7, 30, 4), _ny(2025, 7, 30, 4) + 3600)
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-october-19-2026-7pm-et", 2026) == (_ny(2026, 10, 19, 19), _ny(2026, 10, 19, 19) + 3600)
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-december-1-2025-12pm-et", 2025)[0] == _ny(2025, 12, 1, 12)
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-december-1-2025-12am-et", 2025)[0] == _ny(2025, 12, 1, 0)
    # EST vs EDT: the same wall-clock hour is 5 h vs 4 h behind UTC
    jan = P.window_from_slug("btc-1h", "bitcoin-up-or-down-january-15-2026-9am-et", 2026)[0]
    jul = P.window_from_slug("btc-1h", "bitcoin-up-or-down-july-15-2026-9am-et", 2026)[0]
    assert jan % 86400 == 14 * 3600 and jul % 86400 == 13 * 3600
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-this-week", 2025) is None
    # a year-less New Year's Eve slug whose gamma endDate falls in the next year resolves to the previous year
    end_hint = int(dt.datetime(2026, 1, 1, 12, tzinfo=dt.timezone.utc).timestamp())
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-december-31-9pm-et", 2026, end_hint)[0] == _ny(2025, 12, 31, 21)
    assert P.window_from_slug("btc-1h", "bitcoin-up-or-down-july-30-4am-et", 2025, int(dt.datetime(2025, 7, 30, 12, tzinfo=dt.timezone.utc).timestamp()))[0] == _ny(2025, 7, 30, 4)
    assert P.window_from_slug("btc-4h", "btc-updown-4h-1760515200") == (1760515200, 1760515200 + 14400)
    ws, we = P.window_from_slug("btc-1d", "bitcoin-up-or-down-on-june-10-2026")
    assert we == _ny(2026, 6, 10, 12) and we - ws == 86400


def _orders(market, rows):
    df = pd.DataFrame(rows, columns=["timestamp", "side", "outcome", "price", "size"])
    df["market_id"] = market; df["condition_id"] = "0x1"; df["tx"] = [f"tx{i}" for i in range(len(df))]; df["wallet"] = "w"
    return df


def test_decision_prices_rule_never_uses_a_print_before_t_plus_3():
    ws = 1_700_000_000 - 1_700_000_000 % 3600
    win = pd.Series({"market_id": "m", "family": "btc-1h", "window_start": ws, "window_end": ws + 3600})
    t1 = ws + 60
    tr = _orders("m", [
        (t1 + 1, "BUY", "Up", 0.40, 10),      # too early (< t+3)
        (t1 + 3, "SELL", "Up", 0.41, 10),     # a taker SELL: not an executable buy
        (t1 + 5, "BUY", "Up", 0.45, 20),      # first Up buy in [t+3, t+30]
        (t1 + 9, "BUY", "Up", 0.47, 5),
        (t1 + 12, "BUY", "Down", 0.60, 7),    # first Down buy
        (t1 + 31, "BUY", "Down", 0.70, 7),    # outside the 30 s window for t1 (inside for t1+... no: t2 = t1+60)
    ])
    p = P.decision_prices(win, tr)
    assert len(p) == 59 and list(p["minutes_left"][:2]) == [59.0, 58.0]
    r = p.iloc[0]
    assert r.exec_price_up == 0.45 and r.exec_size_up == 20 and r.exec_time_up == t1 + 5
    assert r.exec_price_down == 0.60 and r.exec_time_down == t1 + 12
    assert abs(r.mkt_prob_up - (0.45 + 0.40) / 2) < 1e-12 and r.mkt_prob_src == "both"
    assert r.n_prints_30s == 3
    # minute 2 has nothing executable
    r2 = p.iloc[1]
    assert np.isnan(r2.exec_price_up) and np.isnan(r2.exec_price_down) and pd.isna(r2.mkt_prob_src) and r2.n_prints_30s == 0
    assert ((p["exec_time_up"].fillna(1e12) >= p["t"] + 3) & (p["exec_time_down"].fillna(1e12) >= p["t"] + 3)).all()


def test_outcome_from_prices_and_klines_labels():
    assert P._outcome_from_prices({"outcomePrices": '["1", "0"]'}) == 1
    assert P._outcome_from_prices({"outcomePrices": '["0", "1"]'}) == 0
    assert P._outcome_from_prices({"outcomePrices": '["0.5", "0.5"]'}) == -1
    ts = np.arange(1_700_000_000, 1_700_000_000 + 7200, 60, dtype=np.int64)
    x = np.zeros((len(ts), 6), np.float32); x[:, 0] = 100 + np.arange(len(ts)); x[:, 3] = x[:, 0] + 0.5
    ws = int(ts[0]) + 3600
    strike, final = P.klines_strike_final("btc-1h", ws, ws + 3600, ts, x)
    assert strike == x[60, 0] and final == x[119, 3]
    assert np.isnan(P.klines_strike_final("btc-1h", ws + 7200, ws + 10800, ts, x)[0])
