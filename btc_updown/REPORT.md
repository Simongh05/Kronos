# btc_updown — results (validation 2026-08-07 → 09-06, Polymarket BTC 5m/15m, TWAP-60 regimes)

_Label: validation / selection data. Dev (09-07 → 10-14) has not been opened._

## The edge test: does Kronos add information beside the market price?

Logistic blend fitted on validation decision rows with a market price (one row per market window and decision minute, ≥ 60 s to expiry, executable price taken ≥ 3 s after the decision). Day-block bootstrap, 300 draws.

| model | rows | days | Brier market | Brier blend (leave-one-day-out) | Brier null | model coef | 95% CI | Brier gain vs market 95% CI |
|---|---|---|---|---|---|---|---|---|
| Market + EWMA null (benchmark) | 49,131 | 23 | 0.1480 | 0.1481 | 0.1559 | 0.051 | [-0.051, 0.116] | [-0.00001, 0.00036] |
| E0 zero-shot small, sampling (z_model) | 41,867 | 20 | 0.1467 | 0.1469 | 0.1547 | 0.050 | [-0.026, 0.144] | [-0.00000, 0.00056] |
| E1b-probe frozen backbone + head | 49,131 | 23 | 0.1480 | 0.1481 | 0.1559 | 0.008 | [-0.111, 0.160] | [-0.00001, 0.00038] |
| E1b joint fine-tune small + head | 49,131 | 23 | 0.1480 | 0.1481 | 0.1559 | 0.082 | [-0.026, 0.210] | [-0.00001, 0.00044] |

None of the three Kronos readouts has a coefficient whose CI excludes zero beside the market, and no blend improves the market's Brier by more than 0.0005. The market's own slope is 1.0 in every fit (it is well calibrated).

## Standalone (no market term): does Kronos beat the EWMA-vol null pricer?

| model | rows | Brier model-only blend (LODO) | Brier null | gain vs null 95% CI |
|---|---|---|---|---|
| E0 zero-shot sampling | 61,820 | 0.1499 | 0.1503 | [0.00019, 0.00102] |
| E1b-probe | 69,308 | 0.1503 | 0.1512 | [0.00056, 0.00139] |
| E1b joint fine-tune | 69,308 | 0.1500 | 0.1512 | [0.00076, 0.00192] |

Kronos does carry a little information beyond a volatility model (gains of 0.0002–0.002 Brier, CIs above zero), but the market already has all of it and more. The fine-tuned head correlates 0.96 with the null pricer and 0.95 with the market logit.

## E1b head beside the market, by horizon and family

| bucket | rows | Brier market | Brier blend (LODO) | coef | 95% CI |
|---|---|---|---|---|---|
| h = 1 | 6,151 | 0.0525 | 0.0524 | 0.056 | [−0.019, 0.133] |
| h = 2–4 | 21,370 | 0.1428 | 0.1428 | 0.029 | [−0.054, 0.085] |
| h = 5–9 | 10,800 | 0.1509 | 0.1512 | 0.067 | [−0.069, 0.253] |
| h = 10–14 | 10,810 | 0.2097 | 0.2096 | 0.323 | [0.120, 0.614] |
| 5m family | 19,887 | 0.1367 | 0.1367 | 0.127 | [0.022, 0.211] |
| 15m family | 29,244 | 0.1557 | 0.1558 | 0.015 | [−0.080, 0.145] |

Two sub-buckets show a statistically detectable coefficient (early in 15-minute windows, and the 5m family), but the Brier gain is ≤ 0.0001 in both, six buckets were tested, and nothing of that size survives a 1.75¢ taker fee.

## Market-conditioned head (option 2): a head that sees both the Kronos state and the market price

The three tests above fit a linear blend of a scalar Kronos readout and the market logit. A sharper question is whether a small non-linear head that sees the fine-tuned model's full hidden state *and* the market price can beat the market on days it was not trained on. Setup (`btc_updown/calib/market_head.py`): the E1b backbone's last-position hidden state (512-d, float16) was dumped for every validation minute (`cube_e1b_hidden_val`, 31 days, 43 s on an A40 after the earlier cube runs); rows are the validation decision minutes with an executable Polymarket price. Train days 08-08 → 08-20 (21,148 rows), early stopping on 08-21 → 08-24 (7,227), **test days 08-26 → 09-06 (20,756 rows, 9 days, never seen in training)**. Models, all fit on the train days only:

| model | inputs | Brier on test days |
|---|---|---|
| raw market price | — | 0.14989 |
| A | market logit (logistic) | 0.15010 |
| B | market + 5 numeric feats (strike distance, EWMA vol, h, j, null logit) | 0.15018 |
| C0 | B's inputs through the MLP (control for the MLP itself) | 0.15017 |
| **C** | **market + numeric + Kronos hidden state (PCA 64), MLP with a market skip weight** | **0.15021** |
| D | same inputs as C, linear | 0.15154 |

Paired day-level bootstrap of Brier(C) − Brier(A): [−0.00086, +0.00097]; Brier(C) − Brier(B): [−0.00098, +0.00093]. The head's learned market skip weight is 1.02, i.e. it kept trusting the market. Robustness over seeds and PCA widths (same train/test days):

| run | C − A Brier, 95% CI | backtest C on test days, $/day (lb95) |
|---|---|---|
| seed 0, PCA 64 | [−0.0009, +0.0010] | −101 (−240) |
| seed 1, PCA 64 | [−0.0007, +0.0011] | −11 (−133) |
| seed 2, PCA 64 | [−0.0002, +0.0008] | −134 (−286) |
| seed 0, PCA 16 | [+0.0000, +0.0020] | +27 (−176) |
| seed 0, PCA 256 | [+0.0003, +0.0024] | −245 (−442) |
| seed 0, no PCA (512) | [−0.0012, +0.0012] | −118 (−190) |

In no setting does the hidden state improve on the market-only model; where the CI excludes zero it is because the hidden state made the head *worse* (overfitting the 11 training days). The backtest of C on the test days (one entry per window, side-consistent ask, 7% fee) loses money in five of six runs; the one positive run has a bootstrap lower bound of −$176/day. The null pricer on the same rows scores 0.15624. Artifacts: `data/calib/market_head*/summary.json`, `trades_test_C.parquet`, `head_C.pt` (Mac only).

## Kalshi KXBTC15M with the frozen pipeline (dev, exploratory)

User decision 2026-10-26: run the frozen pipeline on Kalshi's 15-minute market. Frozen = the E1b model + DirectHead (`/workspace/runs/ft_small_1m/best`), the validation calibration `data/calib/val_e1b_direct_market.json` (market 0.987, head 0.082, null −0.016, fitted on Polymarket val days), the engine rule (one entry per window, clip $100, margin 0, 7% Kalshi fee, side-consistent ask). The cube (`cube_e1b_direct_kalshi`, 41 days, 53 s on an A40) covers every Kalshi label day: 09-07 → 10-03 and 10-10 → 10-17; executable quotes (recorder BBO at t+3 s) exist for **10-11 → 10-17 only: 7 days, 663 windows, 9,282 decision rows**. This opens the dev range, so everything here is labelled exploratory.

| rows | Brier market (yes mid) | Brier frozen blend | Brier null | blend − market |
|---|---|---|---|---|
| 9,282 (7 days) | 0.14931 | 0.14967 | 0.15620 | +0.0004 (worse) |

By horizon the frozen blend is within ±0.001 of the market at every minutes-left bucket and never better by more than 0.00002 (h = 1). Standalone, with the frozen no-market calibration on all 46,466 label rows (35 days): head blend 0.15663 vs null 0.15681, the same 0.0002 as on validation.

Backtest of the frozen pipeline on the 7 quote days (Kalshi fee `0.07·p·(1−p)` per contract, clip $100):

| execution | trades | P&L total | per quote day | ¢/contract | hit rate | positive days |
|---|---|---|---|---|---|---|
| quoted ask | 656 | −$1,690 | −$241 | −2.09 | 0.794 | 3 of 7 |
| ask + 1 tick | 331 | −$1,358 | −$194 | −3.64 | 0.852 | 3 of 7 |

Losses concentrate in the 0.4–0.8 price range (−$1,844 of the −$1,690 total at the quoted ask), i.e. the head disagrees with the market most where the market is uncertain, and is wrong.

Exploratory refit on the 7 Kalshi days (not frozen; 7 day-blocks only): market slope 1.32 [1.02, 1.58], head coefficient **−0.50 [−0.73, −0.21]**, null +0.27 [0.08, 0.52]; leave-one-day-out blend Brier 0.14927 vs market 0.14931. A market-only refit with the null gives slope 1.18 and null −0.14, the same pattern. The negative head sign is the collinearity signature of a feature that is a noisier copy of the market (corr 0.95 on val): the fit sharpens the market term past 1 and subtracts the copy. Even taken at face value it buys 0.00004 Brier out of sample and nothing after fees.

## Hourly Polymarket markets (btc-1h) with a 60-minute head (2026-10-26)

Scope: `SCOPE-LONGER-WINDOWS.md`. Data: `btc_updown/data/pm_long.py` built every hourly market from public endpoints
(12,338 markets since 2025-05-28, labels exact from the local Binance BTCUSDT 1 h candle, 12,303/12,303 agree) and, for
the clean window 2026-08-01 → 10-26 (after E1b's kline cutoff), every taker order (1.52 M) and the minute mid series:
2,074 markets, 122,366 decision minutes (t = 1 … 59 min into the hour), executable price on 68 % / 66 % of the minutes per
side, some market price on 84 %. Model: the E1b backbone frozen, a DirectHead retrained on the (j, h) grid with j + h ≤ 60
(`probe_small_1h`, 13 min on an A40, grid Brier 0.1612 from step 1,000), cube `cube_e1b_direct_1h` (85 days, 99 s).
Fit on val days 08-07 → 09-06 (31 days, 36,276 rows with a market price), frozen, then applied to dev days 09-07 → 10-24
(48 days, 57,647 rows). Exploratory label: this is the second and last time the dev dates were opened.

| rows | Brier market | Brier null | Brier head raw | corr(head, market) |
|---|---|---|---|---|
| 101,269 (85 days) | 0.1666 | 0.1690 | 0.1691 | 0.968 |

The hourly market is much closer to the null than the 15-minute one (gap 0.0024 vs 0.008), so there was room: at 45–59
minutes to the close the market scores 0.2292 and the null 0.2309.

| fit | coef(head) | 95 % CI (day blocks) | Brier market | Brier blend (LODO) |
|---|---|---|---|---|
| val, market + head + null | 0.079 | [−0.23, 0.46] | 0.16588 | 0.16665 |
| val, head + null, no market | 0.40 | [0.02, 0.87] (null 0.60 [0.12, 1.12]) | — | — |
| dev refit (exploratory) | 0.095 | [−0.19, 0.38] | 0.16740 | 0.16788 |

Frozen val calibration on dev: blend 0.16740 vs market 0.16740 (equal at every minutes-left bucket), null 0.16998.
Backtest on dev (7 % fee, clip $100, side-consistent ask): −$7/day at the quoted print (1,135 trades, 28 of 48 days
positive, −1.2 ¢/share), −$4/day with a tick; bootstrap lower bounds −$23 and −$14. The head's standalone information is
real but it is the null's (coefficient 0.40 beside the null, raw Brier equal to the null's); beside the market it is nothing.

### Joint fine-tune with 60-bar targets (`ft_small_1h`, 2026-10-26, user's go)

The E1b recipe (joint next-token CE + DirectHead BCE from the pretrained small, 30,000 steps × 64, lr 4e-5) with
`predict: 60` and the (j, h) grid up to j + h ≤ 60, so the CE targets cover the full hour: 91 min on an A40 (353 samples/s,
≈ $0.9); val CE 2.594 → 2.576 (a 316-token window, not comparable to E1b's 2.609 at 271); val grid Brier 0.1615 (probe
0.1612, i.e. the joint fine-tune does not improve the synthetic-strike task either). Cube `cube_ft1h_direct_1h` (85 days,
102 s), the same tables, splits and rule as above; dev opened a third time with this frozen pipeline (exploratory).

| rows | Brier market | Brier null | Brier head raw | corr(head, market) | corr(head, probe head) |
|---|---|---|---|---|---|
| 101,269 (85 days) | 0.1666 | 0.1690 | 0.1691 | 0.963 | 0.996 |

| fit | coef(head) | 95 % CI (day blocks) | Brier market | Brier blend (LODO) |
|---|---|---|---|---|
| val, market + head + null | 0.128 | [−0.21, 0.52] | 0.16588 | 0.16665 |
| val, head + null, no market | 0.47 | [0.07, 0.89] (null 0.54 [0.11, 1.04]) | — | — |
| dev refit (exploratory) | 0.122 | [−0.10, 0.36] | 0.16740 | 0.16785 |

Frozen val calibration on dev: blend 0.16739 vs market 0.16740, null 0.16998; the difference is within ±0.0002 in every
minutes-left bucket (the 45–59 bucket: blend 0.22959, market 0.22968). The backtest splits on the execution assumption:
+$33/day at the quoted print (1,142 trades, 34 of 48 days positive, +5.8 ¢/share, bootstrap lb95 +$22, of which +$1,037
of the +$1,584 total comes from entries 45–59 minutes before the close) but −$11/day with a one-tick haircut (798 trades,
lb95 −$25). The probe head on the same rows: −$7 and −$4. With calibrated probabilities equal to the market's at every
horizon, a print-only profit that a one-tick haircut erases is the gap between early, thin prints and the mid, not model
information; neither run passes the $150/day hurdle (t = −15 and −19). The jointly fine-tuned head is the probe head
(correlation 0.996) and the probe head is the null pricer.

Not run for the hourly family: the sampling cube at pred_len 60 (the sampler did not beat the direct head at 15 minutes).

## Kalshi KXBTCD hourly strike ladders (2026-10-26)

Scope: `SCOPE-KALSHI-LADDER.md`. Data: `btc_updown/data/kalshi_ladder.py` archived, from the public API, every market and
every strike's 1-minute bid/ask candles for the retention window 2026-08-20 → 10-25 (1,502 hourly ladders × 188 strikes,
10.7 M candles; the Friday 169 h ladders are skipped; fills were started and stopped when the API began rate-limiting —
not needed here). Table: `btc_updown/backtest/ladder.py`, one row per (ladder, minute 1–59, strike within ±12 of the
previous settlement), book = the forward-filled candle at t (fresh in 99.9 % of the kept rows), strike moved onto Binance
levels with the previous ladder's BRTI–Binance basis (median −$14 in October, −$3 in August; sd $11 across ladders).
Model: the two 60-minute heads (`probe_small_1h` on the E1b backbone, `ft_small_1h`) applied at every ladder strike from
a per-minute hidden-state dump (`cube_*_hidden_kalshi`, 86 s each on an A40, ≈ $0.2; the Mac-side application reproduces
the cube's own logits to 0.001). Rows with a two-sided quote: 858,405 over 67 days. Val = 08-20 → 09-20 (32 days,
424,589 rows), dev = 09-21 → 10-24 (34 days, 433,424 rows); dev opened once with the val calibration frozen.

**Raw scores (two-sided rows, 67 days).** Market 0.09081, null 0.09311, head raw 0.09153 (E1b) / 0.09180 (ft1h);
corr(head, market) 0.978. This is the first product where the head is not the null pricer: at 10–20 minutes to the
close the raw head (0.08514) equals the market (0.08519), and at 8–12 strikes out it beats it (0.03536 vs 0.03543).

| fit (E1b head) | coef(head) | 95 % CI (day blocks) | coef(null) | Brier market | Brier blend (LODO) |
|---|---|---|---|---|---|
| val, market + head + null | **0.363** | **[0.008, 0.721]** | −0.24 [−0.41, −0.06] | 0.09305 | 0.09294 |
| val, h 10–20 only | 0.718 | [0.34, 1.08] | −0.15 | 0.08228 | 0.08187 |
| val, wings |k| ≥ 2 only | 0.252 | [−0.14, 0.77] | −0.26 | 0.06492 | 0.06470 |
| dev refit (exploratory) | 0.352 | [0.13, 0.57] | −0.11 | 0.08854 | 0.08844 |
| ft1h head, val | 0.416 | [0.08, 0.80] | −0.27 | 0.09305 | 0.09293 |
| ft1h head, dev refit | 0.295 | [0.07, 0.52] | −0.09 | 0.08854 | 0.08848 |

**The plan's gate is met for the first time**, narrowly: the head's coefficient beside the market excludes zero on val
and again on dev. Frozen val calibration on dev: blend 0.08835 vs market 0.08854 (null 0.09061); per-day Brier gain
0.00024, CI [0.00002, 0.00047], 21 of 34 days positive (ft1h head: 0.00018 [−0.00005, 0.00040]). The gain sits at
5–30 minutes to the close (0.0003–0.0004) and at the money and far wings; nothing at 45–59 minutes or under 5.

**Distribution score (test 2, dev, frozen).** Bin probabilities between adjacent two-sided strikes (10 strikes per
ladder-minute on average, 40,422 interior points): mean log score market 1.7413, blend 1.7272, null 1.7787; per-day gain
market − blend 0.0142, CI [0.0093, 0.0196], 27 of 34 days. A monotone (isotonic) market curve scores the same as the raw
mids (1.7414), so the gain is not smoothing of non-monotone quotes (1.7 % of ladder-minutes).

**Backtest (test 3, dev, frozen, quadratic fee 0.07·p(1−p), clip $100).** Negative in every variant. The engine's rule
(first qualifying minute per market) fires per strike at 45–59 minutes and loses −$3,163/day over 12,414 trades; with
fresh quotes, h ≤ 30 and one trade per ladder −$361/day (−$158 with a tick); h 10–20 one per ladder −$90/day at the
quote, −$53 with a tick (lb95 −$130), hit rate 0.87 but −0.7 to −1.0 ¢/share. Why: the blend and the mid differ by
0.9 ¢ at the median (p90 2.6 ¢) while the taker fee is 1.3–1.75 ¢ and the half-spread 0.5 ¢; the rows that clear
fee + half-spread (42 %) are the ones where the disagreement is noise, as the 0.3 % Brier gain implies.

**Reading.** On the ladder the fine-tuned head carries information the market has not fully priced: statistically real
on val, dev and the distribution score, about 0.3 % of the market's Brier (0.8 % of its log score), concentrated at
5–30 minutes to the close. It is too small to earn the taker fee and spread; a maker execution (posting at the bid,
no fee on this series' maker side if any, fill risk unmeasured) was the only route by which it could be money. Tested
below with the fills archive and two queue bounds: it loses even under the optimistic bound.

### Sampling cube at 60 bars (`cube_ft1h_sample_kalshi`, 2026-10-27, user's go)

The distribution test proper: the 60-bar joint fine-tune (`ft_small_1h`) sampled free-running, 64 paths per decision
point, decision points every 5 minutes of the retention window (66 days, 288 points/day), lookback 256, T = 1,
top_p = 1; 588 s per day on an A40, 10.8 h, ≈ $6.4. Readout at every ladder strike (`backtest/ladder.py --cube`):
p_samp = smoothed fraction of the sorted typical-price samples at h = 60 − m above the strike, p_gauss = normal tail
from the sample mean and sd. Same val/dev split, same fits, on the 5-minute rows only (80,678 two-sided val rows,
82,218 dev rows).

| val, beside market | coef(head) | coef(sampler) | LODO Brier |
|---|---|---|---|
| A: head + null | 0.416 [0.045, 0.807] | — | 0.09256 |
| B: head + null + sampler | 0.397 [0.004, 0.790] | 0.007 [−0.027, 0.038] | 0.09264 |
| C: sampler + null (no head) | — | 0.013 [−0.018, 0.043] | 0.09277 |
| D: head + null + normal tail | 0.401 [0.002, 0.785] | 0.003 [−0.016, 0.022] | 0.09265 |
| sampled centre as drift, z = (mean − close)/(σ√h) | 0.404 [−0.002, 0.791] | 0.007 [−0.053, 0.064] | 0.09272 |

Market Brier on these rows 0.09271. The sampler's coefficient is zero in every form, with or without the head; the
no-market fit gives the same (−0.004 [−0.038, 0.025]). Frozen on dev the head-only blend A beats the market by
0.00011 per day [−0.00031, 0.00044] on the 5-minute rows (the full-minute result was 0.00024 [0.00002, 0.00047];
one twelfth of the rows) and adding the sampler makes it worse by 0.00006 [−0.00013, 0.00001]; distribution log
score A 1.7124, B 1.7131, market 1.7250 (A − B −0.0008 [−0.0017, −0.0000]). The `ft_small_1h` head gives the same
picture. Dev refit of B: sampler −0.012 [−0.042, 0.023].

Why the sampler has nothing: its raw Brier is 0.257 (the constant 0.25), against market 0.0927 and head 0.0932,
because the free-running distribution is right at one bar and wrong after twenty. Sample sd against realized, in
bps of the close: h = 1 3.2 vs 3.4; h = 5 10.1 vs 12.5; h = 15 28 vs 22; h = 30 26 vs 31; h = 60 19 vs 43. The width
*shrinks* beyond h ≈ 20 as the sampled bars flatten, and the centre drifts down 4–15 bps (half a strike step), so
at the ladder's horizons most strikes sit outside the sampled range and p_samp is near 0 or 1. The zero-shot E0 cube
at 15 bars was already under-dispersed (6.9 vs 22 bps at h = 15); the 60-bar fine-tune did not fix the free-running
path. The sampled centre is uncorrelated with the head's logit (0.003) and with the market's excess over the null
(0.003), so it is not even a noisy version of the same signal.

**Reading.** The full sampled distribution adds nothing to the direct head on the ladders: the autoregressive sampler
is unusable beyond ~20 bars (collapsing width, drifting centre), and what it says at shorter horizons the head already
says better. The ladder result stands as the head's alone: 0.3 % Brier, 0.8 % log score, no money after fee and spread.

### Maker execution on the ladders (fills archive + queue bounds, 2026-10-27, user's go, no GPU)

The one route left for the 0.3 % edge: rest orders instead of crossing the spread. Public inputs only: Kalshi's
`/markets/trades` endpoint (26.3 M taker prints over the 66 retained days, pulled at 3 req/s, `data/kalshi_ladder/fills/`)
and the ladder table. Simulator `backtest/maker.py`: at every minute and strike with a two-sided quote, rest a yes bid
or yes ask priced off the frozen val blend A (bid at the blend minus a tick, ask at the blend plus a tick, only where
that improves on the book), size $100 / price, fill against the taker prints of the following minute, one position per
market, hold to settlement. The public data has no book depth, so queue position is unknown and the truth is bracketed:

- **improve** — post one tick inside the spread, first at a new level; any taker print at or through the price fills
  us. Optimistic: assumes no other maker reacts and that the taker would have taken our level.
- **join** — post at the existing best; only prints strictly through the level fill us (we are behind an unknown queue).
  Pessimistic.

Controls posted the same way: the EWMA null pricer (model-free prices) and the mid (pure market making off the book,
measuring the adverse selection any resting order eats on this book). Maker fee 0 and 0.0175·C·P(1−P) both (the series
endpoint does not say whether KXBTCD charges makers; the fee-schedule page refuses automated reads). Day-level stats as
in the taker backtest (block bootstrap, hurdle $150/day).

| dev (frozen blend A, 34 days, $/day) | improve | join |
|---|---|---|
| blend, 1 tick, no maker fee | −38 (91 fills/day, hit 0.71, −1.3 ¢/share, lb95 −92) | −122 (183 fills/day, hit 0.67, −1.5 ¢, lb95 −237) |
| blend, with maker fee | −47 | −142 |
| blend, 2-tick margin | +8 (54/day, lb95 −39) | −87 |
| blend, h 5–30 only | +37 (60/day, lb95 −18) | +33 (140/day, lb95 −57) |
| null pricer | +9 (113/day, hit 0.39) | −366 (hit 0.38) |
| null pricer, h 5–30 | +50 (lb95 −20) | −144 |
| mid (pure market making) | +3 (1.8 fills/day) | −149 (106/day, −3.0 ¢/share, lb95 −237) |

Val (in-sample for the blend, 31 days): blend improve +49/day (lb95 −52; −1.5 with h 5–30), join −74; null −57 / −292;
mid −5 / −121. Fill delay median 4 s (improve) to 20 s (join) after posting; the blend's resting price differs from
the book at 8,384 of the dev markets.

What the numbers say. (1) Even the optimistic bound loses on dev (−$38/day, lb95 −$92), and the pessimistic one loses
three times as much; the hurdle is +$150. (2) Joining the queue at the mid loses $149/day at −3 ¢/share: the prints
that trade through a level on this book are informed, and that adverse selection is ten times the head's edge
(0.9 ¢ median disagreement, 0.3 % Brier). The blend's prices are far better than the null's (hit 0.71 vs 0.39 at the
same posting rule) and still do not cover it. (3) The only positive cells are the h 5–30 window, where the null
pricer posting the same way does as well or better (+50 vs +37) and val flips sign (−1.5): that is the window, not the
model, and it is noise (lb95 negative everywhere). (4) The maker fee question is moot: it moves the result by $10–20/day
in the wrong direction.

**Reading.** Maker execution does not turn the ladder edge into money. The optimistic fill model already loses, so no
queue model, book recorder or fee ruling can rescue it; the resting-order flow on KXBTCD is informed enough that a 0.3 %
distribution edge is far below the cost of being hit. This closes the last untested route on the ladders.

### Three free tests on the ladders (Mac only, 2026-10-27, user's go)

Before any further GPU spend, the three cheapest questions that could still have changed the ladder answer, all on the
existing hidden-state dumps, tables and the fills archive. All train on the val days and test frozen on the dev days
(dev is open for the ladders → exploratory). Scripts `calib/ladder_head.py`, `calib/vol_test.py`, `calib/flow_test.py`.

**1. Market-conditioned head on the ladders** (`data/calib/ladder_head`). Option 2 from the 5m markets, rebuilt for the
ladder rows: market logit + numeric (k, strike distance, log σ, h, null logit, the frozen head's logit, quote age) +
the ladder's own shape (the mids at k ± 1..5 relative to the row's) + the 512-d hidden state (PCA 64), as a logistic
(D) and as an MLP (C), against the same without the hidden state (B logistic, C0 MLP). 424,589 fit/early-stop rows on
32 val days, 433,424 test rows on 34 dev days. Dev Brier: market 0.08854, frozen blend 0.08835, A (market refit)
0.08841, B 0.08871, D 0.09149, C0 0.09069, **C 0.09552**. Paired day-block differences: C − B +0.0068 [0.0048, 0.0089];
D − B +0.0028 [0.0017, 0.0039]; C − C0 +0.0048 [0.0025, 0.0071]; B − A +0.0003 [−0.0002, 0.0008]. Worse with the
hidden state in every h and |k| bucket, linear or not; the MLP early-stops after 7 epochs at a worse loss than the
no-hidden MLP. Given the ladder's own prices, the hidden state only adds month-to-month drift: it does not transfer
from the val month to the dev month at all.

**2. Realized-vol forecast** (`data/calib/vol_test`). If Kronos knows anything the ladder does not, it should be the vol
regime. Target: log realized variance of the 60 bars after t, every minute of the 66 days (40,320 fit / 5,760
early-stop / 48,900 test minutes). Ridge, alpha picked on the early-stop days. Test MSE of log RV: EWMA-60 alone
0.505; EWMA (4 halflives) + HAR-RV (1 h, 6 h, 1 d, 1 w) 0.436; + hour-of-day and weekday one-hots 0.375 (N2);
**+ hidden 0.370** (N3 − N2 −0.0045 [−0.017, +0.008]); hidden + time of day without the vol features 0.615; hidden
alone 0.679 (R² 0.39 vs 0.54 for EWMA-60 alone; corr 0.73 vs 0.74 with the target, 0.82 with N2). QLIKE looked better
for N3 (0.2345 vs 0.2546, −0.020 [−0.033, −0.008]) but that is a level effect: N3 forecasts higher variance on the
dev regime (test bias +0.087 vs +0.042 in log RV) and QLIKE punishes under-forecasts; with each model's own early-stop
bias removed the difference is −0.009 [−0.020, +0.0007]. Gradient boosting on either set is far worse than ridge
(overfits 32 days). The hidden state is a vol encoder about as good as one EWMA, with nothing beyond EWMA + HAR +
time of day.

**3. Taker flow beside the price** (`data/calib/flow_test`). From the 26.3 M public prints: signed taker count
(yes-buys − no-buys) over total, per ladder over the last 1, 5, 15 minutes and its whole life, at the row's own strike
over 5 minutes, and log volume, all with prints up to t (the same information as the candle's quote at t). Val fit of
the frozen blend's features + flow, day-block bootstrap: every flow coefficient includes zero (imb1 −0.03 [−0.09,
0.02], imb5 −0.03 [−0.19, 0.20], imb15 −0.08 [−0.84, 1.02], own-strike 0.04 [−0.04, 0.12], volume −0.02 [−0.06,
0.08]); leave-one-day-out Brier 0.09313 vs 0.09294 without the flow (worse). Frozen on dev: blend + flow − blend
−0.00001 [−0.00012, +0.00010]; market + flow − market −0.00010 [−0.00031, +0.00011]. Dev refit: the only coefficient
away from zero is log volume (−0.14 [−0.22, −0.06], a calibration-by-activity effect, exploratory). The ladder's quote
absorbs its own taker flow within the minute; the flow is informed (the maker study) and already priced.

**Reading.** None of the three moves the ladder result: the hidden state does not generalize month to month once the
ladder's own prices are in the model, it carries no vol information beyond the textbook set, and the public flow is
in the price. The plan's one remaining GPU step (Kronos-base on the ladders) would be chasing a 0.3 % edge that these
tests say is already fully extracted by the small model; not run.

## Training facts

- E1b-probe: 6,000 steps × 128, frozen pretrained small backbone, 10 min on an A40; val grid Brier plateaued at 0.161 from step 1,000.
- E1b: 30,000 steps × 64, joint CE + BCE, 77 min on an A40 (416 samples/s); val CE 2.671 → 2.609; val grid Brier 0.1609 (no better than the probe). Checkpoint: volume `/workspace/runs/ft_small_1m/best`, Mac copy `data/cloud/runs/ft_small_1m/best`.
- E0: zero-shot small, N = 128 paths, lookback 256, T = 1.0, top_p = 1.0; 25 min per day on an A40; raw path-count Brier 0.169 (worse than the null's 0.150 until calibrated); path dispersion 2.5 bps at h = 1 to 10.6 bps at h = 14.
- Sampling cube at 60 bars: `ft_small_1h` backbone, 64 paths, 288 points/day, 588 s per day on an A40 (no KV cache; 0.39× E0's per-day time for 0.2 × 0.5 × 4.3 the work), 10.8 h for 66 days; output 413 MB (`data/cloud/cubes/cube_ft1h_sample_kalshi`).
- Controls: tokenizer round-trip, settlement-proxy flip rate and the null-pricer benchmark are in STATUS.md.

## Spend

About $19 of GPU time (A40 Secure Cloud at $0.59/h) plus $7/month for the 100 GB volume; balance $81.90 of $100 (RunPod `clientBalance` on 2026-10-27 after the 60-bar sampling cube) after E0 + probe + E1b + the E0 resume pod + the hidden-state dump, a pull pod, the Kalshi cube pod, the hourly probe pod, the hourly fine-tune pod, the Kalshi hidden-dump pod and the 60-bar sampling cube (10.8 h, ≈ $6.4).

## Verdict (validation)

The plan's gate for the next step (Kronos-base, E2) required the model coefficient's CI to exclude zero beside the market on validation. It does not, for the zero-shot model, the frozen-backbone head, or the jointly fine-tuned small model. The hypothesis that Kronos's conditional distribution of the next 1–15 minutes carries information the market has not priced is **not supported** on this data; the research repo's prior (the market embeds a near-exact nowcast and more) stands. The market-conditioned head (option 2) reaches the same conclusion from the other direction: given the market price, the fine-tuned model's full hidden state adds nothing on held-out days. Dev was opened once, for Kalshi KXBTC15M with the frozen pipeline (user decision): the frozen blend is 0.0004 Brier worse than the market's yes-mid on the 7 days with quotes and the backtest loses about $200 per day at $100 clips. The market is the better pricer on both venues; the venue was not the problem. The hourly Polymarket market (60-minute horizon, 85 days, head retrained for h ≤ 60) gives the same answer: the head equals the null pricer and adds nothing beside the market, with the backbone frozen or jointly fine-tuned on 60-bar targets. The Kalshi KXBTCD strike ladders are the one place the head adds anything beside the market: a 0.3 % Brier / 0.8 % log-score gain, real on val and dev, and not enough to pay the taker fee and spread. The sampled 60-bar distribution adds nothing to that head (coefficient zero in every form; the free-running sampler collapses beyond ~20 bars). Maker execution of that edge, simulated against 26 M public taker prints, loses under the optimistic fill bound (−$38/day on dev) and more under the pessimistic one (−$122/day); the resting-order flow on the ladders is informed (joining at the mid loses 3 ¢/share), so there is no route left from the ladder edge to money. Three free follow-ups (a market-conditioned head on the ladder rows, a realized-vol forecast test, taker-flow features beside the price) found nothing more in the hidden state or the public flow.
