"""Synthetic tests for btc_updown/data/labels.py (no research data; run with the research venv's pytest)."""
import io
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from btc_updown.data import labels as L


# ------------------------------------------------------------------------------------------------ decision times
def test_decision_times_5m_and_15m():
    assert L.decision_times(1000, 1300) == [1060, 1120, 1180, 1240]          # k = 1..4, last has 60 s left
    assert len(L.decision_times(0, 900)) == 14
    assert L.decision_times(0, 900)[-1] == 840


def test_build_decisions_minutes_left():
    w = pd.DataFrame([dict(market_id="m", family="btc-5m", window_start=0, window_end=300)])
    d = L.build_decisions(w)
    assert d["t"].tolist() == [60, 120, 180, 240]
    assert d["minutes_left"].tolist() == [4.0, 3.0, 2.0, 1.0]


# ------------------------------------------------------------------------------------------------ executable price
def _orders(rows):
    cols = ["slug", "tx_hash", "log_index", "tx_index", "block_time", "side", "outcome", "shares", "avg_price"]
    return pd.DataFrame(rows, columns=cols)


def _fills(rows):
    cols = ["tx_hash", "taker_log_index", "log_index", "price", "shares"]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def test_exec_price_never_before_t_plus_3_and_first_after(con):
    t = 1000
    dec = pd.DataFrame([dict(market_id="m", family="btc-5m", window_start=940, window_end=1240, t=t, minutes_left=4.0)])
    orders = _orders([
        ("m", "0xa", 10, 1, t + 2, "BUY", "Up", 10.0, 0.40),     # 2 s after t: must NOT be used
        ("m", "0xb", 20, 1, t + 1, "BUY", "Down", 10.0, 0.50),   # before t+3: must NOT be used
        ("m", "0xc", 30, 1, t + 7, "BUY", "Up", 25.0, 0.55),     # first valid Up buy
        ("m", "0xd", 40, 1, t + 5, "SELL", "Up", 5.0, 0.30),     # a SELL: ignored
        ("m", "0xe", 50, 1, t + 9, "BUY", "Up", 7.0, 0.60),      # later Up buy: ignored for exec, counted in n_prints
        ("m", "0xf", 60, 1, t + 31, "BUY", "Down", 7.0, 0.45),   # beyond t+30: ignored
        ("m", "0xg", 70, 1, t + 12, "BUY", "Down", 12.0, 0.47),  # first valid Down buy
    ])
    fills = _fills([
        ("0xc", 30, 31, 0.56, 10.0), ("0xc", 30, 32, 0.54, 15.0),   # lowest fill price 0.54, first fill 0.56
        ("0xg", 70, 71, 0.47, 12.0),
        ("0xa", 10, 11, 0.40, 10.0), ("0xe", 50, 51, 0.60, 7.0),
    ])
    out = L.pick_exec_prices(con, dec, orders, fills)
    assert len(out) == 1
    r = out.iloc[0]
    assert r.exec_time_up == t + 7 and r.exec_price_up == pytest.approx(0.54) and r.exec_size_up == 25.0
    assert r.exec_time_down == t + 12 and r.exec_price_down == pytest.approx(0.47) and r.exec_size_down == 12.0
    assert r.exec_time_up >= t + 3 and r.exec_time_down >= t + 3
    assert r.n_prints_30s == 3                     # 0xc, 0xe, 0xg
    assert r.mkt_prob_src == "both"
    assert r.mkt_prob_up == pytest.approx((0.56 + (1 - 0.47)) / 2)   # first fill prices, not the lowest
    assert r.fill_missing == 0


def test_exec_price_boundaries_and_ties(con):
    t = 5000
    dec = pd.DataFrame([dict(market_id="m", family="btc-15m", window_start=4000, window_end=4900, t=t, minutes_left=1.0)])
    orders = _orders([
        ("m", "0x1", 5, 3, t + 3, "BUY", "Up", 1.0, 0.70),     # exactly t+3, later tx_index than 0x2
        ("m", "0x2", 9, 1, t + 3, "BUY", "Up", 2.0, 0.71),     # same block, earlier tx_index -> first
        ("m", "0x3", 1, 1, t + 30, "BUY", "Down", 3.0, 0.20),  # exactly t+30: included
    ])
    fills = _fills([("0x1", 5, 6, 0.70, 1.0), ("0x2", 9, 10, 0.71, 2.0), ("0x3", 1, 2, 0.20, 3.0)])
    r = L.pick_exec_prices(con, dec, orders, fills).iloc[0]
    assert r.exec_price_up == pytest.approx(0.71) and r.exec_size_up == 2.0 and r.exec_time_up == t + 3
    assert r.exec_price_down == pytest.approx(0.20) and r.exec_time_down == t + 30


def test_exec_price_none_and_one_sided(con):
    t = 100
    dec = pd.DataFrame([dict(market_id="m", family="btc-5m", window_start=40, window_end=340, t=t, minutes_left=4.0),
                        dict(market_id="n", family="btc-5m", window_start=40, window_end=340, t=t, minutes_left=4.0)])
    orders = _orders([("m", "0x1", 1, 1, t + 10, "BUY", "Down", 4.0, 0.35),
                      ("n", "0x2", 1, 1, t + 1, "BUY", "Up", 4.0, 0.35)])
    fills = _fills([("0x1", 1, 2, 0.35, 4.0), ("0x2", 1, 2, 0.35, 4.0)])
    out = L.pick_exec_prices(con, dec, orders, fills).set_index("market_id")
    m, n = out.loc["m"], out.loc["n"]
    assert pd.isna(m.exec_price_up) and m.exec_price_down == pytest.approx(0.35)
    assert m.mkt_prob_src == "down_only" and m.mkt_prob_up == pytest.approx(0.65)
    assert pd.isna(n.exec_price_up) and pd.isna(n.exec_price_down) and pd.isna(n.mkt_prob_up) and n.n_prints_30s == 0


def test_exec_price_falls_back_to_avg_price_without_fills(con):
    t = 100
    dec = pd.DataFrame([dict(market_id="m", family="btc-5m", window_start=40, window_end=340, t=t, minutes_left=4.0)])
    orders = _orders([("m", "0x1", 1, 1, t + 10, "BUY", "Up", 4.0, 0.42)])
    r = L.pick_exec_prices(con, dec, orders, _fills([])).iloc[0]
    assert r.exec_price_up == pytest.approx(0.42) and r.fill_missing == 1


def test_randomised_exec_price_never_looks_ahead(con):
    rng = np.random.default_rng(0)
    ws = 10_000
    dec = L.build_decisions(pd.DataFrame([dict(market_id="m", family="btc-15m", window_start=ws, window_end=ws + 900)]))
    bt = rng.integers(ws - 60, ws + 900, 3000)
    orders = _orders([("m", f"0x{i}", i, int(rng.integers(0, 50)), int(b), rng.choice(["BUY", "SELL"]),
                       rng.choice(["Up", "Down"]), float(rng.integers(1, 100)), float(rng.uniform(0.01, 0.99)))
                      for i, b in enumerate(bt)])
    fills = _fills([(o.tx_hash, o.log_index, o.log_index + 1, o.avg_price, o.shares) for o in orders.itertuples()])
    out = L.pick_exec_prices(con, dec, orders, fills)
    for side in ("up", "down"):
        et = out[f"exec_time_{side}"].dropna()
        assert (et >= out.loc[et.index, "t"] + 3).all()
        assert (et <= out.loc[et.index, "t"] + 30).all()
    # brute-force check of "first" for every decision
    o = orders[orders.side == "BUY"]
    for r in out.itertuples():
        for side, oc in (("up", "Up"), ("down", "Down")):
            c = o[(o.outcome == oc) & (o.block_time >= r.t + 3) & (o.block_time <= r.t + 30)]
            exp = None if c.empty else c.sort_values(["block_time", "tx_index", "log_index"]).iloc[0]
            got = getattr(r, f"exec_time_{side}")
            if exp is None:
                assert pd.isna(got)
            else:
                assert got == exp.block_time and getattr(r, f"exec_size_{side}") == exp.shares


# ------------------------------------------------------------------------------------------------ mkt prob + outcome
def test_compose_mkt_prob():
    assert L.compose_mkt_prob(0.60, 0.45) == (pytest.approx(0.575), "both")
    assert L.compose_mkt_prob(0.60, None) == (0.60, "up_only")
    assert L.compose_mkt_prob(None, 0.45) == (pytest.approx(0.55), "down_only")
    assert L.compose_mkt_prob(None, None) == (None, None)


def test_outcome_rule_and_mismatch_report():
    assert L.outcome_from(100.0, 100.0) == 1          # final >= strike -> Up
    assert L.outcome_from(99.99, 100.0) == 0
    assert L.outcome_from(None, 100.0) == -1
    w = pd.DataFrame([
        dict(source="a", outcome=1, final=101.0, strike=100.0),
        dict(source="a", outcome=0, final=101.0, strike=100.0),   # mismatch
        dict(source="a", outcome=-1, final=101.0, strike=100.0),  # unknown: not checked
        dict(source="b", outcome=0, final=np.nan, strike=100.0),  # no final: not checked
        dict(source="b", outcome=0, final=99.0, strike=100.0),
    ])
    mm = L.outcome_mismatches(w).set_index("source")
    assert mm.loc["a", "checked"] == 2 and mm.loc["a", "mismatch"] == 1
    assert mm.loc["b", "checked"] == 1 and mm.loc["b", "mismatch"] == 0


# ------------------------------------------------------------------------------------------------ Kalshi meta file
def _zst_file(tmp_path, lines):
    try:
        from compression import zstd
    except ImportError:
        import zstandard as zstd
    p = tmp_path / "kalshi-KXBTC15M-markets-2026-09-07.txt.zst"
    with zstd.open(p, "wb") as f:
        f.write(b"".join(lines))
    return p


def test_kalshi_zst_header_and_rows(tmp_path):
    head = {"day": "2026-09-07", "format": "kalshi-markets/1", "n": 2, "fields": ["ticker"]}
    m1 = {"ticker": "KXBTC15M-26SEP062015-15", "event_ticker": "KXBTC15M-26SEP062015", "open_time": "2026-09-07T00:00:00Z",
          "close_time": "2026-09-07T00:15:00Z", "floor_strike": 80350.34, "expiration_value": "80345.58", "result": "no"}
    m2 = {"ticker": "KXBTC15M-26SEP062030-30", "event_ticker": "KXBTC15M-26SEP062030", "open_time": "2026-09-07T00:15:00Z",
          "close_time": "2026-09-07T00:30:00Z", "floor_strike": 80345.58, "expiration_value": "", "result": ""}
    p = _zst_file(tmp_path, [b"#" + json.dumps(head).encode() + b"\n", json.dumps(m1).encode() + b"\n",
                             json.dumps(m2).encode() + b"\n"])
    h, ms = L.parse_kalshi_meta_zst(p)
    assert h["day"] == "2026-09-07" and len(ms) == 2
    r1 = L.kalshi_window_row(ms[0], "kalshi_meta")
    assert r1["window_start"] == 1788739200 and r1["window_end"] == 1788740100
    assert r1["strike"] == 80350.34 and r1["final"] == 80345.58 and r1["outcome"] == 0
    assert r1["label_day"] == "2026-09-07" and r1["venue"] == "kalshi" and r1["token_up"] is None
    assert L.outcome_from(r1["final"], r1["strike"]) == r1["outcome"]
    r2 = L.kalshi_window_row(ms[1], "kalshi_meta")
    assert r2["final"] is None and r2["outcome"] == -1


def test_kalshi_zst_rejects_missing_header_and_bad_count(tmp_path):
    p = _zst_file(tmp_path, [b'{"ticker": "x"}\n'])
    with pytest.raises(ValueError):
        L.parse_kalshi_meta_zst(p)
    head = {"day": "2026-09-07", "n": 3}
    p = _zst_file(tmp_path, [b"#" + json.dumps(head).encode() + b"\n", b'{"ticker": "x"}\n'])
    with pytest.raises(ValueError):
        L.parse_kalshi_meta_zst(p)


# ------------------------------------------------------------------------------------------------ carried state
def test_state_at_and_first_in():
    ts = np.array([10.0, 12.0, 12.0, 20.0])
    assert L.state_at(ts, 9.9) == -1
    assert L.state_at(ts, 12.0) == 2            # latest record at or before t
    assert L.state_at(ts, 100.0) == 3
    assert L.first_in(ts, 11.0, 15.0) == 1      # first record in [lo, hi]
    assert L.first_in(ts, 13.0, 19.0) == -1
    assert L.first_in(ts, 20.0, 20.0) == 3


def test_compress_days():
    assert L.compress_days(["2026-08-15", "2026-08-14", "2026-08-17"]) == "2026-08-14..2026-08-15, 2026-08-17"
    assert L.compress_days([]) == "(none)"
