# Scope: Kalshi KXBTCD hourly strike ladders (2026-10-26)

Question: the up/down tests (Polymarket 5m/15m/1h, Kalshi 15m) compared one model number against one market price and
found nothing. A strike ladder prices the whole distribution of the hour's close; the test becomes whether Kronos's
conditional distribution (or its direct head evaluated at every strike) adds anything to the ladder's implied
distribution. Everything below comes from the public API (`https://api.elections.kalshi.com/trade-api/v2`, no key); the
research repo is not touched and nothing leaves the Mac except klines.

## What exists

| item | finding |
|---|---|
| Hourly ladder | one event per hour, `KXBTCD-<yy><MON><dd><hh>` (hh = ET hour), **188 markets, strikes $100 apart** ($72,100 … $90,800 on 10-24, i.e. ±$9,300 around spot; the final was inside the range in 315/315 ladders), `strike_type: greater`: Yes iff the 60-second BRTI average before the hour > `floor_strike` (…99.99, so effectively ≥ the round strike). Opens 1 h before the close (created 25 h before, quoting starts at open). |
| Daily ladder | the 21:00 Z (5 pm ET) event: 80 markets, $250 apart, open 25 h. One per day; not for Kronos (horizon), but archivable at no extra cost. |
| Count | 23 hourly + 1 daily ladder per day; the event list runs from 2024-12 (hourly) but **market-level data exists only for the last ~66 days**: earliest settled market with candles/trades now = 2026-08-20 13:00 Z. Older events return no markets at all. The window slides daily. |
| Settlement | `expiration_value` is the BRTI 60-s average; `result == (expiration_value > floor_strike)` in 29,704/29,704 markets (10-18 → 10-24). `settlement_ts` ≈ 2 min after the close. |
| BRTI vs Binance | over 158 hours (10-18 → 10-24): BRTI − Binance BTCUSDC last-bar typical price = **−$14 ± $8** (median −14.3); vs BTCUSDT −$100 ± $16. BTCUSDC is the path proxy; the basis for hour t is the previous ladder's settlement minus the Binance proxy at that time (known at t). Strike spacing $100, so the basis error matters only for the strike nearest the final. |
| Hourly move | settle-to-settle |move| in strikes: median 1.2, p75 2.8, p90 4.3, p99 9.6; 98.7 % of finals within ±10 strikes of the previous settle. |
| Volume | 0.6–2.6 M contracts per hourly ladder (median 1.3–1.7 M in US/Asia hours, 0.6–0.9 M at 07–12 Z); ATM strike 0.25–0.58 M; volume falls to ~10 k by ±8 strikes from the final. Daily ladder 3.6 M. |
| Fills | `GET /markets/trades?ticker=&limit=1000&cursor=`: one row per taker fill (`created_time` µs, `yes_price_dollars`, `count_fp`, `taker_side`); ATM ± 3 strikes of one hour = 18,676 fills, 300–900 per minute after minute 12, median 15.8 contracts; both taker sides balanced. |
| Quotes | `GET /markets/candlesticks?market_tickers=<≤100>&start_ts&end_ts&period_interval=1`: 1-minute candles with `yes_bid`/`yes_ask`/`price` OHLC (`*_dollars` strings), `volume_fp`, `open_interest_fp`; `end_period_ts` on the minute. **Candles exist only for minutes with a change**; after a gap the next candle opens exactly at the previous close (1,040/1,040 checked), so forward-filling the book is exact. ~4,000 candle rows per ladder. |
| Quoted width | after forward-fill, the median number of two-sided strikes (bid > 0, ask < 1) is 12 at minute 1, 9 at minute 30, 5 at minute 45, 1 at minute 58; strikes within ±$350 of the final are two-sided 80 % of the time with a 1 ¢ spread (≤ 2 ¢ 97 %); the ATM strike's spread is 1 ¢ at every minute. |
| Fees | series `fee_type: quadratic, fee_multiplier: 1` → the standard 0.07·C·P·(1−P) per fill, no maker fee. |
| API | no auth, no rate-limit headers, 48 req/s sequential without a 429; the archive will run at ≤ 5 req/s. `/events?series_ticker=KXBTCD&status=settled&limit=200&cursor` lists events; `/markets?series_ticker=KXBTCD&status=settled&min_close_ts&max_close_ts&limit=1000&cursor` lists markets (1,000 per page); batch candlesticks need `start_ts`/`end_ts` and ≤ 100 tickers. |

Samples: `scratchpad/kalshi_ladder_26OCT2411_candles.parquet` (one ladder, 2,723 candles), `kalshi_day_26OCT23_candles.parquet`
(23 ladders, 92,202 candles), `kalshi_events.parquet` (13,085 settled events).

## Archive job (`btc_updown/data/kalshi_ladder.py`, proposed)

Resumable, per label day under `data/kalshi_ladder/` (Mac only, never committed):
1. `markets/<day>.parquet`: every market of every ladder closing that day (ticker, event, strike, open/close, expiration_value, result, volume, OI).
2. `candles/<day>.parquet`: 1-minute candles for **all 188 strikes** of each hourly ladder (2 batch calls per ladder) — ~1,530 ladders × 4,000 rows ≈ 6 M rows, ~3,100 calls, ~12 min.
3. `fills/<day>.parquet`: taker fills for the ATM ± 5 strikes (11 per ladder; ATM = strike nearest the previous settle): ~17 k tickers × ~3 pages ≈ 50 k calls, ~3 h at 5 req/s. Optional: the candles already carry bid/ask per minute, which is the executable quote; fills only serve the print-vs-quote haircut as on Polymarket.
4. Daily ladders: markets + candles, same code path.

First run covers 08-20 → today (~66 days, ~1,530 hourly ladders). The retention window slides one day per day, so each day of delay loses one August day; a weekly rerun of the same command keeps the archive growing. No GPU, no spend.

## Decision table and labels

- Decision minutes t = open + 60·m, m = 1 … 59 (h = 60 − m minutes to the close); book state = the forward-filled candle ending at t (`yes_bid`, `yes_ask` closes), i.e. the book as of t. Executable: buy Yes at `yes_ask`, buy No at 1 − `yes_bid`, size = unknown from candles (use the fills' size distribution; clip $100 as before), haircut assumption = +1 ¢ as the second case.
- Rows: (ladder, t, strike) for the strikes with a two-sided quote at t (≈ 5–12 per minute). Market P(Yes) = mid. Label = `result`.
- Model inputs at t: bars with open ≤ t − 60 s (the same cube timing); strike expressed on Binance levels: K − basis_t.
- Null: EWMA-vol digital P(final > K) from `nulls.null_prob` with the hour's σ (the one used for the hourly Polymarket test).
- Kronos: (a) **direct head at every strike** — the 60-minute heads (`probe_small_1h`, `ft_small_1h`) take (K − close_t)/σ as input, so they price any strike; needs the per-minute hidden state for 08-20 → 10-24 (`cube.py` hidden dump, 66 days × 1,440 × 512 fp16 ≈ 100 MB, ~2 min of GPU, ≈ $0.3), head applied on the Mac. (b) **sampling cube at pred_len 60** — the distribution test proper: N = 64 paths at decision minutes every 5 min (m = 5, 10, …, 55), P(final > K) = path fraction above K − basis_t plus the mean/sd summary; from E0's 25 min/day (1,440 points × 128 paths × 15 bars) scaled by 288/1,440 × 64/128 × 60/15 ≈ 10 min/day → ~11 h for 66 days ≈ $6.5 on an A40. Both need E1b's kline cutoff (07-31), which the ladder dates respect.
- Splits: val = 08-20 → 09-20 (32 days), dev = 09-21 → 10-24 (34 days), dev opened once per frozen pipeline as before.

## Tests (in order; stop at the first negative)

1. **Edge beside the ladder**, the same `calib.fit` as before on (ladder, t, strike) rows: logistic blend of logit(market mid) + logit(model) + logit(null), day-block bootstrap, the model coefficient's CI must exclude 0 on val. Run first with the existing 60-minute head (≈ $0.3), then with the sampler if the head shows anything.
2. **Distribution score**: per (ladder, t), the implied bin probabilities from adjacent two-sided strikes (market), the null's and the model's; log score / Brier of the realised bin; blend vs market by day-block CI. This is the test the up/down markets could not run.
3. **Backtest** (frozen val calibration on dev): one entry per ladder at the first qualifying minute, the strike with the largest blend − price edge after the quadratic fee and a margin; both execution cases (ask, ask + 1 ¢); ¢/contract and $/day with the day as the unit.

## What is different from the up/down tests, and what is not

Different: the market gives P(final > K) at 5–12 strikes per minute with 1 ¢ spreads and 0.3–0.9 M contracts/h at the
money, so a distribution-level comparison is possible and execution is cleaner than Polymarket prints. Not different:
the hourly Polymarket test already showed that the 60-minute head equals the null pricer and that the market beats the
null by only 0.0024 Brier; the ladder's ATM strike is the same bet as the Polymarket hourly market with the strike at the
previous settle rather than the candle open. The new information is in the wings (±2–9 strikes), where a vol/tail view
could in principle differ from the market's. Priors: the ATM result will repeat the hourly result; the wings are the only
place to look.
