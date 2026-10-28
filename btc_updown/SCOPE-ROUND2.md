# Round 2 plan: five remaining tests (2026-10-28)

Context: every Kronos test on BTC up/down markets (Polymarket 5m/15m/1h, Kalshi 15m, Kalshi KXBTCD ladders) and every
non-Kronos follow-up on the ladders (maker, sampler, vol, flow, static arbitrage, ETH/SOL lag) is negative or has no
route to money (REPORT.md). The user wants to try the five remaining ideas with the $81.90 of RunPod credits left.
Expected total spend ≈ $10 of GPU; items 1 and 2 are Mac-only.

Standing rules (unchanged): research repo read-only and guard-checked (none of this plan reads it; everything here is
public data); keys only in `.env`; only public klines go to the cloud; `data/` never committed; pods self-terminate and are
verified gone; dev for the ladders is already open → every ladder result is exploratory; a new family (Polymarket alt 5m,
Kalshi S&P) gets its own val/dev split and dev is opened once with a frozen pipeline; every quote-based backtest is checked
against the public prints before it is believed (the SOL lesson).

Sources checked on 2026-10-28:

| Item | Source | State |
|---|---|---|
| 1 | `data/pm_long/{mid,trades}/btc-1h` (08-01 → 10-26, 87 days), `data/kalshi_ladder/{candles,fills}` (08-20 → 10-25) | on disk; overlap 66 days |
| 2 | Polymarket gamma series 10683 `eth-updown-5m-<unix>`, 10686 `sol-updown-5m-…`, 10685 `xrp-updown-5m-…` (no alt 15m/1h series exist); CLOB `prices-history`, data-api `/trades`; Binance `spot_{ETH,SOL,XRP}USDT` 1m | public; alt 5m volume ≈ $0.5–4 k per window (BTC 1h ≈ $9–26 k) |
| 2 (watch) | Polymarket gold/silver 15m + 1h (series 12967/12968/12976/12977, Chainlink XAU/USD, launched 2026-10-22, $250–900 per 15m window) | too new; revisit in a month |
| 3 | Kalshi `KXINXU` hourly S&P 500 above/below ladders: 329 events 08-20 → 10-27 (7 per trading day, 10:00–16:00 ET), 60 strikes, settle on the index (Google Finance); `KXNASDAQ100U` since 10-15 (53 events, 400 strikes); `KXINX` daily range 48 events. 1-min candles + prints via the same endpoints as KXBTCD; retention slides (~66 days) | public; **archive must start now** |
| 3 | Equity minute bars: Alpaca free data key (IEX feed, SPY/QQQ 1-min since 2016, 10 k bars/request, 200 req/min) | **user step: create a free Alpaca account and put `ALPACA_KEY`/`ALPACA_SECRET` in `.env`** |
| 4 | `pm_long.py` btc-1h windows back to 2025-05-28 (12.4 k markets, already built), prices/trades for 2025-05-28 → 2026-07-31 (not yet pulled, ≈ 8 M taker orders, ≈ 45 min) | public |
| 5 | Kalshi ladder tables and hidden-dump pipeline (`cube_*_hidden_kalshi`), `NeoQuasar/Kronos-base` (102 M) | on disk / HF |

## Order and parallelism

1. **Day 1 (Mac + one pod):** start the `KXINXU` candle archive (item 3, retention slides); run item 1 end to end; launch item 5
   (Kronos-base fine-tune) and item 4 (early-cutoff fine-tune) on one A40 each or sequentially on one pod (both are
   klines-only jobs, ≈ 4.5 h + 1.3 h).
2. **Day 2 (Mac):** item 2 data build + null test; items 4/5 readouts (hidden dumps, calibration, frozen dev); item 3 archive
   check, Alpaca bars pull once the key exists.
3. **Day 3:** item 3 fine-tune + readout; item 2 fine-tune only if its gate passes. Write-up, STATUS, memory, commit after each item.

GPU budget: item 5 ≈ $3.5, item 4 ≈ $2, item 3 ≈ $2, item 2 ≤ $3 (conditional). Total ≤ $11.

## Item 1 — Cross-venue: Kalshi ladder vs Polymarket hourly on the same BTC hour (Mac, $0)

Question: does the Kalshi ladder's implied probability at Polymarket's strike carry information Polymarket's own price
does not (and vice versa), and can it be collected on the liquid venue?

Steps:
1. `backtest/cross_venue.py build`: for every Polymarket btc-1h window in the overlap (08-20 → 10-24, ≈ 1,500 hours) and
   every minute m (1–59): Polymarket mid of the Up token at m (from `mid/btc-1h`, fidelity 1), the executable Up/Down
   buy prices at m+3 s (from `prices_btc-1h.parquet`), the KXBTCD ladder's bid/ask at every strike at m (ladder table),
   and the **Kalshi-implied P(up)** = the ladder's yes-mid interpolated in logit space at K* = Polymarket strike (the
   hour's Binance open) + basis (BRTI − Binance, from the previous ladder's settlement as in `ladder.py`). Also the
   implied P from bids only and asks only (an executable band). Same hour boundaries on both venues (ET hour = UTC hour).
2. Lead-lag (both directions): Δ(pm mid)_{m→m+1} on (kalshi_implied − pm_mid)_m and Δ(kalshi_implied)_{m→m+1} on
   (pm_mid − kalshi_implied)_m, |strike distance| ≤ 2 ladder steps, day-block bootstrap; report R² and the coefficient in
   cents. Control: the same with the EWMA null in place of the other venue.
3. Calibration on val (08-20 → 09-20): P = σ(a + b·logit(pm_mid) + c·logit(kalshi_implied) + d·logit(null)); the test is
   c's CI. Mirror model for the Kalshi side (ATM strike only): logit(ladder mid) + logit(pm_mid).
4. Frozen on dev (09-21 → 10-24): Brier blend vs each venue's own price (paired day-block CI); taker backtest on
   Polymarket at the executable print (`engine.py` rule, $100 clips, Polymarket fee), and with a tick; executability from
   the data-api taker orders (prints at or better than our price in the next 1 and 5 min, fill-size P&L). Kalshi side with
   the existing fills archive.
5. Done = `data/backtests/cross_venue/summary.json`, REPORT subsection, STATUS row. Gate for spending anything further
   on it: c's CI excludes zero on val **and** the fill-size dev P&L is positive.

Risks: Polymarket hourly settles on the Binance candle, Kalshi on BRTI; the basis is handled as in the ladder table, and a
residual mismatch shows up as a worse Brier, not a false positive. The within-minute R² to spot is 0.56 on both venues,
so the residual is expected to be small.

## Item 2 — Polymarket ETH / SOL / XRP 5-minute markets (Mac first; GPU ≤ $3 if the gate passes)

Question: are the thin alt 5m markets mispriced by a free vol model the way Kalshi's SOL ladder was, and this time with
enough flow to collect?

Steps:
1. `pm_long.py` gains families `eth-5m`, `sol-5m`, `xrp-5m` (series 10683/10686/10685, slug `<asset>-updown-5m-<unix
   start>`; settlement per the market description: Chainlink price at end vs start, check whether TWAP-60 applies as for
   BTC 5m since 08-14; strike = the gamma `price to beat` where present, else the Binance 1-min close at the window start
   with the Chainlink basis). Windows from the series' first event (to find), prices/trades for the full life. Binance
   klines: `spot_XRPUSDT` added to `binance_klines.py`; ETH/SOL extended to 2024-01 → 2026-10.
2. Liquidity census first: taker orders per window, share of windows with a print every minute, median size at the first
   decision minute. If the median window has no print in the first 2 minutes, stop here (report only).
3. Null pricer (`nulls.py`, EWMA σ per asset) at every decision minute with ≥ 60 s left; val = first half of the family's
   life, dev = second half (fixed before any fit). Val fit of logit(market) + logit(null); frozen dev Brier vs market;
   taker backtest at the executable print; **prints check** (fill-size P&L over 1 and 5 min).
4. Gate for GPU: the frozen-dev null blend beats the market by a day-block CI excluding zero **and** the fill-size P&L is
   positive. Then: E1b recipe fine-tune on the asset's klines (`ft_small_1m_<asset>`, 30 k × 64, ≈ 77 min ≈ $0.8 each),
   direct-head cube over the family's life, calibration beside market + null, frozen dev, prints check.
5. Done = `data/backtests/pm_alt5m/summary.json`, REPORT subsection, STATUS row.

Risks: ≈ $3 k per window means the fill-size cap binds at a few hundred dollars per window; XRP is near-empty. The honest
outcome is probably "mispriced, uncollectable" again; the prints check is the deliverable.

## Item 3 — Kalshi hourly S&P 500 ladders (KXINXU) with equity minute bars (Mac + ≈ $2)

Question: a different trader population and a different asset class; does the ladder head (and the free null) find anything
beside the mid on the S&P hourly ladders?

Steps:
1. **Now:** `kalshi_ladder.py --series KXINXU` (candles for every strike, 08-20 → today; strike step is a point grid,
   `ladder.py strike_step` handles it) and the prints for ATM ± 5; add a daily cron-free re-run instruction (the retention
   slides, so the archive grows with each run). Optionally `KXNASDAQ100U` (53 events) for a second look later.
2. Minute bars: `data/alpaca_bars.py` pulls SPY and QQQ 1-min bars (IEX) 2016 → today into the `SeriesStore` layout
   (market hours only; the dataset's contiguity mask rejects windows that span a session gap; lookback 256 = 4.3 h so
   windows start ≥ 10:46 ET or use lookback 128 for this family — decide from the valid-start count).
   Settlement proxy: SPX ≈ SPY × ratio, ratio from the previous ladder's settlement (as `basis` in `ladder.py`), checked
   against the archived `expiration_value`s (flip rate per minute, as the BTC proxy check).
3. Table: `ladder.py build --series KXINXU` (decision minutes 1–59, strikes within ± 12 steps of ATM, EWMA null from SPY
   returns, basis as above). Splits: val 08-20 → 09-20, dev 09-21 → end (fixed now).
4. Free tests first: null vs mid by strike bucket and minutes-left (the SOL-style wing check), static arbitrage scan
   (`ladder_arb.py`), volume census from candles (contracts per event; the `volume` field on settled markets is 0 and
   cannot be used).
5. GPU: E1b recipe fine-tune on SPY + QQQ 1-min (`ft_small_1m_eq`, 30 k × 64, ≈ $0.8) + direct head with h ≤ 60, hidden
   dump at every decision minute of the archive (≈ 1 min), head applied at every strike as for KXBTCD; val coef beside
   market + null; frozen dev Brier; taker backtest; prints check.
6. Done = `data/backtests/ladder_inx/summary.json`, REPORT subsection, STATUS row.

Risks: ~7 events per trading day → ~45 dev days of 7 hours; day-level power is lower than on BTC. Kalshi's index ladders
settle on the index print, not SPY, and the ratio drifts intraday with dividends and the futures basis; the proxy flip
rate bounds the damage and is reported first.

## Item 4 — Polymarket hourly BTC with the full 17-month history (≈ $2)

Question: the 85-day hourly test was dead-equal to the market; with 12.4 k markets (2025-05-28 → 2026-10-26) does any
head coefficient appear, or does the zero just tighten?

Steps:
1. `pm_long.py prices --family btc-1h --start 2025-05-28 --end 2026-08-01` (≈ 8 M taker orders, ≈ 45 min at the earlier
   rate; resumable).
2. Fine-tune with the E1b recipe and head grid j + h ≤ 60 on klines **≤ 2025-05-27** only (`ft_small_1h_early`, 30 k ×
   64, 91 min ≈ $0.9; the existing `ft_small_1h` has seen klines through 2026-07-31 and cannot be used on 2025 markets).
3. Direct-head cube at every decision minute of every hour 2025-05-28 → 2026-10-26 (≈ 1 point/min, 517 days; the direct
   path ran 85 days in 99 s, so ≈ 10 min, ≈ $0.1). Hidden dump too (for a market-conditioned head if the linear coef is
   positive).
4. Splits fixed now: val 2025-05-28 → 2026-02-28 (9 months), dev 2026-03-01 → 2026-10-26 (frozen). Calibration beside
   market + null on val (day-block CI of the head coefficient, by minutes-left bucket); frozen dev Brier; taker backtest at
   the executable print and with a tick; prints check.
5. Done = `data/backtests/pm_1h_full/summary.json`, REPORT subsection, STATUS row.

Risks: the Polymarket hourly market's regime changed over 17 months (liquidity, settlement source); report by quarter.
Prior: low; this mostly buys a tighter zero.

## Item 5 — Kronos-base on the KXBTCD ladders (≈ $3.5)

Question: the small model's head adds 0.00024 Brier/day beside the ladder; does the 4× model add more, enough to clear
the 1.3–1.75 ¢ fee + spread?

Steps:
1. `ft_base_1h`: E1b recipe with `NeoQuasar/Kronos-base` (d832, 12 layers), predict 60, head grid j + h ≤ 60, 30 k × 64,
   bf16; small took 77 min on an A40 at batch 64, base ≈ 4–5 h ≈ $3 (batch 64 fits in 48 GB; fall back to 32 with 2×
   steps if not). Checkpoint on the volume.
2. Hidden dump `cube_base_hidden_kalshi` (d832 per minute, 66 days, ≈ 2 min) and the direct head at every strike via
   `ladder.py build --cube`.
3. Same readout as the small model: val coef beside market + null (and beside the small head: does base add to small?),
   frozen dev per-day Brier gain and log-score gain with CIs, taker backtest in the same variants (best small variant was
   −$53/day), maker simulator in improve mode (small: −$38/day).
4. Done = `data/backtests/ladder_base/summary.json`, REPORT subsection, STATUS row. The number that matters: the median
   |blend − mid| at the entry (small: 0.9 ¢ against 1.3–1.75 ¢ fee + 0.5 ¢ half-spread).

Risks: none technical; prior low because the three free tests say the small model's state is already fully extracted
given the ladder's prices.

## Reporting

Each item: a REPORT.md subsection with the gate numbers and the prints check, a STATUS.md row, memory update, commit with
the Co-Authored-By line, push to the fork, pods verified terminated, balance noted. Final: a one-paragraph verdict across
the five and the remaining balance.
