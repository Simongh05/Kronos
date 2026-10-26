# Scoping: longer BTC up/down windows (2026-10-26)

Question: after five negative tests at 5/15-minute horizons on Polymarket and Kalshi, is there a longer-window product where a
bar model's 1–60-minute (or longer) conditional distribution could matter, and what data would a test need? Everything below is
from public, read-only APIs and the local Binance klines; nothing was written into the research repo and no key was used.
Probe scripts and samples: the session scratchpad (`pm_*_sample.json`, `kalshi_*_sample.json`).

## 1. What exists

| family | venue / series | window | first market | closed markets | volume $/month (2026) | settlement | labels from local klines |
|---|---|---|---|---|---|---|---|
| **BTC hourly** | Polymarket `btc-up-or-down-hourly` (series 10114), slug `bitcoin-up-or-down-<month>-<d>[-<yyyy>]-<h><am/pm>-et` | the Binance 1 h candle starting at that ET hour | 2025-05-28 | **12,369** (24/day, every day) | 17–180 M | Binance BTCUSDT 1 h candle: close ≥ open → Up | **exact: 600/600 agree** |
| BTC 4h | Polymarket `btc-up-or-down-4h` (10331), slug `btc-updown-4h-<unix start>` | 4 h, 00/04/08/12/16/20 UTC | 2025-10-15 | 2,228 | 2.5–10 M | Chainlink 60-s TWAP at end vs start (as 5m/15m) | proxy: 95.5 % (191/200), like the 15m |
| BTC daily | Polymarket `btc-up-or-down-daily` (41), slug `bitcoin-up-or-down-on-<month>-<d>-<yyyy>` | noon ET → noon ET | 2025-03-15 | 588 | 5–39 M | Binance BTCUSDT 1-min candle close at noon ET vs previous noon ET, via UMA | 99.5 % (199/200) |
| **Kalshi hourly ladder** | `KXBTCD` ("Bitcoin price above/below"), tickers `KXBTCD-<date><hour>-T<strike>` | opens 1 h before, settles on the hour; **188 strikes** $100 apart | API retains settled markets from **2026-08-19** only | 24 ladders/day (≈ 4,500 markets/day) | per hour: 0.6–2.2 M contracts; ATM strike 0.2–0.56 M | BRTI 60-s average at the hour vs fixed strike | BRTI vs Binance basis (as the 15m leg) |
| Kalshi daily ladder | same series, the 21:00 Z (5 pm ET) expiry | opens 25 h before; 80 strikes | same | 1/day | ATM strike ≈ 0.6 M contracts, ladder ≈ 3 M | same | same |
| Polymarket weekly / monthly | dead series since 2025-05 | | | | | | |

The research repo holds nothing for these families (tape meta/fills cover 5m/15m only; HOLD-FREE-MARKETS.md surveyed them
on 2026-10-13 and wrote a recorder patch that was not deployed). All of it is reachable from public endpoints instead.

## 2. Market data available without a recorder

**Polymarket.** `clob.polymarket.com/prices-history?market=<token>&startTs&endTs&fidelity=1` returns the token's price
series for the whole life of every market back to 2025-03 (checked on the first hourly, 4h and daily markets): one point per
minute (fidelity 1; ~20 s on busy recent markets), a mid-type price that sits within 1¢ (4h) / 0.5¢ (daily) of the mean trade
price of the same minute. `data-api.polymarket.com/trades?market=<conditionId>&limit=1000&offset=…` returns every fill with
timestamp, side, outcome, price, size and wallet, also back to 2025-03 — the same content the research tape's on-chain fills
carry, so the existing executable-price rule (first taker BUY of our side ≥ t+3 s, within 30 s) can be reused unchanged.

Density inside the window (recent markets):

| family | trades per window | minutes with ≥ 1 trade | where the volume sits |
|---|---|---|---|
| hourly | 600–3,700 | 56–60 of 60 | spread across the hour; price leaves 0.50 at the candle open |
| 4h | 250–560 | 121–144 of 240 | 27–49 % in the last quarter |
| daily | 1,800–2,700 | 580–833 of 1,440 | 45–56 % in the last 6 h |

So "thinner and staler" holds for the 4h family and for most of the daily window, **not** for the hourly: the hourly book
trades every minute at $20–100k per market. What changes with window length is not the staleness of the price but the
horizon: the market has to *forecast* 1–60 minutes instead of nowcast a 60-second average, which is the regime the plan's
hypothesis was about (vol regime / drift information rather than a faster price feed).

**Kalshi.** `GET /trade-api/v2/markets?series_ticker=KXBTCD&status=settled&min_close_ts&max_close_ts` lists the ladders;
`GET /series/KXBTCD/markets/<ticker>/candlesticks?period_interval=1` gives **1-minute bid/ask/price OHLC and volume for the
whole life** (checked: 60/60 candles, trades in 58 of them, 1¢ spread at the money); `GET /markets/trades?ticker=` gives
every fill (3,628 on one ATM hour). Retention through the public API is only ~68 days (nothing before 2026-08-19), so a
Kalshi test needs an archive job started now and run weekly; the research repo's quarantine (`rec_mirror/kalshi` ≥ 10-18)
concerns the user's recorder files, not the public API, but the registry reserves no KXBTCD ranges either way — to confirm
with the user before any Kalshi labels are used.

The ladder is the better fit for a model that outputs a distribution: 188 strikes per hour price the whole 1-hour-ahead
CDF, so a Kronos sample cube is tested against every strike at once instead of one up/down coin flip, and strikes are known
an hour ahead (no "price to beat" revealed at window open).

## 3. What a test would look like

**Primary: Polymarket hourly, DirectHead first.** Decision minutes 5, 10, …, 55 inside each hour (12 per market, ≥ 5 min
left). Model inputs as before (1-min bars, lookback 256). The DirectHead needs `max_h = 60` and synthetic strikes drawn as
the open of the current hour (j = minutes since the hour), i.e. one config change and a retrain of the head; the sampling
cube needs `pred_len = 60` (4× the per-point cost of the 15-step cube: ≈ 20 min per day on an A40 at a 5-minute stride,
N = 128). Benchmarks exactly as before: null = EWMA-vol digital at 1–55 min; market = prices-history mid at t and the
executable print; edge test = the model coefficient's day-block-bootstrap CI beside the market logit; backtest with the
7 % taker fee and side-consistent asks; Brier by minutes-left bucket.

**Split problem to decide first.** E1b was fine-tuned on klines through 2026-07-31, so the only hourly markets it has not
seen the price path of are 08-01 → 10-26 (≈ 2,000 markets, 87 days): val 08-07 → 09-06, dev 09-07 → 10-14 as before, which
keeps the earlier splits. To use the full 17 months (12k markets) the backbone has to be re-fine-tuned with a cutoff before
2025-05-28 (the pretrained checkpoint itself has seen BTC up to its 2025 pretraining cutoff; the paper does not pin the date,
so the zero-shot model should also be treated as "seen" through mid-2025). Recommendation: run the 87-day test with the
existing E1b first (cost ≈ $2 GPU: head retrain 15 min + direct cube seconds + sampling cube 87 days × 20 min ≈ 30 h —
skip or subsample the sampling cube to 30 days ≈ $6); if the hourly DirectHead shows a coefficient beside the market,
re-fine-tune with the 2025-05 cutoff and open the long history.

**Secondary: Kalshi hourly ladder** (distribution test; needs the archive job first; ≈ 68 days of history by the time the
Polymarket result is in). **Third: daily** (588 markets; model on 15-min or 1-h bars, horizon 24 h; thin per minute but
$170k per market; the hold-free book allows exits).

## 4. Data plan (Mac only, public, ≈ 2 h of API time, ≈ 2 GB)

1. `btc_updown/data/pm_long.py`: Gamma listing of series 10114 / 10331 / 41 (paginate by `end_date_min/max` month windows;
   `offset` is capped at ~2,000) → `data/labels/windows_long.parquet` with window start/end from the slug (ET → UTC, DST-safe),
   outcome from `outcomePrices`, volume, token ids, condition id; outcome cross-checked against the local klines (hourly exact,
   daily exact, 4h proxy).
2. Same script: per market, `prices-history` at fidelity 1 (mid series) and `data-api/trades` (all fills) → the existing
   `prices_*.parquet` layout (executable Up/Down price at every decision minute, mkt_prob_up, n_prints_30s) so `assemble`,
   `fit` and `engine` run unchanged. Volume: ~12k hourly + 2.2k 4h + 0.6k daily markets, ~25k paged calls.
3. `btc_updown/data/kalshi_ladder.py`: weekly archive of settled KXBTCD ladders since 2026-08-19 with 1-min candlesticks for the
   ATM ± 5 strikes and all fills; stored under `data/kalshi_ladder/`. Start now so the retention window does not keep sliding.
4. Head retrain config (`max_h: 60`, strike = hour open) and cube configs for 08-01 → 10-26; `assemble` gains a `family`
   switch for the hour-open strike (no settlement proxy needed for the hourly).

Nothing above touches the research repo, and all of it stays on the Mac except klines.
