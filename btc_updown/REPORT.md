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

### Static arbitrage and cross-asset lag on the ladders (Mac only, 2026-10-27, user's go)

Two non-Kronos ideas from the "what else" list, both on public Kalshi data (`data/kalshi_ladder.py --series`,
`backtest/ladder_arb.py`, `backtest/ladder_lag.py`, `backtest/stale.py`). Dev is open for the ladders → exploratory.

**1. Monotonicity arbitrage on the KXBTCD ladders** (`data/backtests/ladder_arb`). A ladder's yes prices must fall
with the strike; a yes bid at a higher strike above a yes ask at a lower strike is a locked profit (buy the low
strike, sell the high one). Scanned every candle minute of the 1,544 hourly events (94,327 ladder-minutes with
quotes) in three tiers: closing quotes of the minute, the minute's low ask against its high bid (the crossing must
hold for the whole minute), and a forward-filled book. Closing quotes: 27 events, 44 ladder-minutes, 67 strike
pairs cross gross; **18 ladder-minutes net of the two taker fees** (36 pairs, median 4.1 ¢ net, upper quartile
25 ¢), 25 of the 36 on 2026-08-21 (the first archived day), 24 of 36 in the last 9 minutes before settlement, and all
but one at bid levels below 10 ¢ or above 70 ¢ (the wings). Whole-minute tier: **one** ladder-minute, 0.8 ¢ net. The
4 crossed rows that fall inside the fills archive did print both legs at or through the crossed prices, so the
quotes were real, but they are a handful of 1–30 ¢ events in 66 days that last less than a minute. Not a strategy.

**2. Cross-asset lag: ETH and SOL ladders** (`data/backtests/ladder_lag_{ethd,sold}`, `stale_{ethd,sold}`). Kalshi
also lists hourly above/below ladders on ETH (KXETHD, $5 strikes) and SOL (KXSOLD, $0.25 strikes). Archived both
for the same 66 days (ETH 1,483 events, SOL 1,476), built the same ladder tables with the EWMA null, and asked
whether BTC's last 1/5/15-minute moves (z-scored by the EWMA σ) or the asset's own moves predict the next minute's
change in the ladder's mid, and whether a frozen null blend beats the mid. Next-minute regressions on val (|k| ≤ 2,
fresh two-sided quotes): ETH R² 0.0008 (BTC z1 coef 0.15 ¢ [0.03, 0.28] of a 5.9 ¢ sd; dev 0.06 [−0.04, 0.16]), SOL
R² 0.007 (own z1 0.63 ¢ [0.50, 0.77], BTC z1 0.11 [0.01, 0.22]; dev 0.47 and 0.06). The same-minute R² is 0.56
on both, i.e. the ladders track their spot within the minute and leave ≈ nothing for the next one. Blends on val →
frozen on dev (35 days):

| Ladder | Dev Brier: market mid | EWMA null | Frozen null blend − market | + BTC/own returns − null blend | Taker backtest at the close quote |
|---|---|---|---|---|---|
| ETH (KXETHD) | 0.09413 | 0.09687 | −0.00019 [−0.00034, −0.00003] | [−0.00006, +0.00003] | 0 trades (no edge > fee); next-open −$84/day (145 trades) |
| SOL (KXSOLD) | 0.12208 | **0.11394** | **−0.00986 [−0.01392, −0.00617]** | +0.0006 to +0.0016 (worse) | +$371/day (lb95 +13), next-open **+$289/day (lb95 −46, 24/35 days)**, +$173 with a tick |

ETH is as efficient as BTC: the null adds nothing beside the mid in any strike bucket, and the lag coefficients
are too small to trade. SOL is different: the free EWMA null alone beats the market mid on dev, the mispricing
sits in the wings (|k| > 3.5 steps: null 0.069 vs market 0.108) and 20–59 minutes before the close, and the frozen
null blend's taker backtest ($100 clips at the ask/bid, one entry per market at the first minute with
blend − price > fee + 1 ¢) is positive at the closing quote, at the next minute's opening quote (the stale-quote
check: 95 % of entries have a next-open quote, and the quotes barely move between a close and the next open) and
with a tick of slippage. The P&L is skewed: the top 3 days are 84 % of it, longshots at ≤ 10 ¢ give +$14.9 k at a
17 % hit rate and 10–30 ¢ give −$9.5 k; a spread ≤ 2 ¢ filter chosen on val gives +$229/+$264 on dev
(lb95 −15/+88). Adding BTC or SOL returns to the blend makes it worse on dev (collinear with the null's σ).

**Why it does not pay: there is nobody to trade with.** SOL's hourly ladders trade a median 6,265 contracts per
event against 1.42 M for BTC and 43 k for ETH; the median strike trades nothing, and the entry minute's candle
volume at the entry strike is 0 for most entries. Capping each trade's shares at that candle's volume turns
+$289/day into **−$42/day**; capping at 100 shares gives +$70 (lb95 −29). The decisive check is the public print
archive (`fills/`, 551 k SOL prints, 301 k on the dev days, median 8.7 k/day, ATM ± 5 strikes): for each of the
3,361 dev next-open trades, count the taker prints in the market in the minute and the five minutes after the next
open that were at or better than our price on our side (a taker bought yes at ≤ our ask, or sold yes at ≥ our bid).
89 % of the markets printed at some point in their life, but within 5 minutes of the entry only 33 % of trades see
a print at or better than our price, 1.9 % could have been filled in full, and the fillable quantity is 39,520 of
the 784,479 simulated shares (5 %). P&L on the fillable quantity: **−$139/day** (1-minute window −$49; windows
from the decision minute instead of the next open −$19 / −$128). The trades that do print are the ones that lose
(ATM and the 10–70 ¢ prices, −$2.5 k / −$3.5 k capped), and the longshot wins that made the simulated profit
(≤ 10 ¢, +$14.9 k simulated) never print at our price (+$59 capped). The SOL ladder's quoted wings are mispriced
and untradeable: the quotes are stale resting orders at strikes nobody is taking, and what does trade is priced
like BTC's ladder.

**Reading.** No static arbitrage on the BTC ladders beyond a few 1–30 ¢ crossings that vanish within a minute;
ETH's ladder is as efficient as BTC's; SOL's ladder is mispriced on paper by a free vol model and the mispricing
cannot be collected because the market is ~200× thinner than BTC's and the mispriced quotes do not trade.

## Round 2 (SCOPE-ROUND2.md, 2026-10-28, user's go): five remaining tests

### Item 1 — Cross-venue: Kalshi KXBTCD ladder vs Polymarket hourly BTC (Mac only, negative)

`backtest/cross_venue.py`, 66 overlapping days (08-20 → 10-24; val to 09-20, dev from 09-21, both exploratory: dev has
been opened on both families before). For every Polymarket hourly window and minute, the ladder's yes-mid interpolated
in logit space at the Polymarket strike in BRTI terms (USDC/USDT ratio at the open + the ladder basis) gives a
Kalshi-implied P(up); 87 % of the 87 k rows have one (median bracketing gap one $100 step, median ladder width 1 ¢).

**Lead-lag is a sampling artifact.** Polymarket's CLOB price history (fidelity 1 min) is a point 0–60 s before t, the
ladder candle closes at t. Regressing the pm point's next-minute change on the gap to the Kalshi mid gives R² 0.49–0.65
and a 0.3–0.7 coefficient, but the gap to the EWMA null (spot at t) explains the same move, and with the Kalshi candle
one minute older than the pm point the Kalshi coefficient turns negative (−0.41 val, −0.20 dev) while the null's stays
0.66–0.70. The pm history is simply stale relative to spot; nothing about Kalshi. Kalshi's own next-minute change is
unpredictable from the pm gap (R² 0.001).

**Beside the executable Polymarket print, Kalshi adds nothing.** With the market term = the mid of the first Up-buy and
Down-buy prints after t+3 s (the price one can actually trade), val coefficients: logit(market) 1.00, logit(kalshi)
**0.03 [−0.42, 0.48]**, logit(null) 0.00 [−0.30, 0.33]; leave-one-day-out Brier 0.1805 vs market 0.1795 (worse); frozen
on dev 0.18225 vs 0.18235, CI [−0.0004, +0.0002]; zero trades clear fee + 1 ¢ in the backtest, on val or dev.

**The one positive number, and why it is not money.** Fitted against the stale pm history instead (market weight 0.2,
Kalshi 0.6–0.8), the blend is essentially the ladder's probability, and executing it against the next Polymarket print
gives +$43–49/day on dev (lb95 +9 to +16, +$23 with a tick for the kalshi+null blend, +$49 / lb95 +14 with a tick for
Kalshi alone; 22–23 of 34 days positive). Anatomy of those 749 trades: 57 % in the first eight minutes of the hour,
where the pm market still sits near 50 ¢ while the ladder (tracking spot) is 5 ¢ away; the executing prints are odd
lots (median 10 shares, 90th percentile 57), so the $100 clips fill to a median of 10 shares and the whole strategy
deploys **$303 of notional per day**; the next same-side print after ours is 1 ¢ *lower* (the market did not move
toward Kalshi within two minutes); the gain comes from the ≤ 45 ¢ side (hit 32–46 %, 4.6–8.7 ¢/share), the 45–70 ¢
trades earn 0.3 ¢/share. A pick-off of small off-mid prints at the open of the hour, capacity a few hundred dollars a
day, below the $150/day hurdle by construction and gone once the market term is the fresh print.

**Kalshi mirror.** Ladder rows |k| ≤ 2 with the pm mid mapped to the strike through the null's shape: val coef
0.10 [−0.01, 0.21], frozen dev blend − market [−0.0004, +0.0001]; backtest −$42/day (−$109 with a tick); prints check
−$106 to −$123/day on the fillable quantity. Polymarket's price carries nothing for the ladder either.

**Reading.** Both venues track spot within the minute and neither leads the other at the executable price; the cross-venue
gap is sampling staleness plus odd-lot prints at the start of the hour. Item 1 closed, $0 spent.

### Item 4 — Polymarket hourly BTC over the full 17-month history (≈ $1.1 GPU, negative for money, positive for information)

`ft_small_1h_early` (the ft_small_1h recipe with the kline cutoff moved to 2025-05-27, 91 min on an A40) and
`cube_early_direct_1h` (direct logits + hidden state every minute, 2025-05-28 → 2026-10-24, 515 days); public data
from `pm_long.py` extended back to the series' first hour (10,264 more markets, 20.5 M taker orders, 2 h on the Mac).
Table `data/tables/hourly_early.parquet`: 725,877 decision rows in 12,303 markets, 641,754 with an executable print
(labels agree with the Binance candle in 12,303 / 12,303). Splits fixed in the plan: **val 2025-05-28 → 2026-02-28**
(277 days, 6,595 markets), **dev 2026-03-01 → 10-26** (238 days, 5,688 markets; this family's dev had been opened once
before on 08-07 → 10-24, so the overlap is exploratory).

**Val (the gate).** Beside the print mid: logit(head) **0.29 [0.15, 0.43]** with the null (null 0.04 [−0.09, 0.18]),
0.31 [0.20, 0.42] alone; leave-one-day-out Brier 0.16449 vs market 0.16469 (null 0.16590). The 85-day test gave 0.08
[−0.23, 0.46]; three times the days moved the point estimate up and the interval off zero. First time the plan's
coefficient gate is met on Polymarket.

**Frozen on dev.** Blend 0.16176 vs market 0.16199 (null 0.16420), paired day-block CI **[−0.00051, +0.00003]**; by
minutes left: worse at 1–9 (+0.0007), better at 10–59 (−0.0002 to −0.0004). Quarter by quarter (refit coefficient of the
head, and the frozen blend's gain): 2025Q3 0.34 [0.10, 0.59] / −0.0002; Q4 0.22 [−0.03, 0.46] / −0.0003; 2026Q1 0.36
[0.12, 0.64] / −0.0006; Q2 0.24 [−0.01, 0.43] / −0.0003; Q3 0.28 [0.04, 0.52] / −0.0001; Q4 (26 days) −0.04 / +0.0001.
Consistent in sign for 15 of 17 months, 0.1–0.4 % of the market's Brier.

**Backtest (dev, $100 clips at the first print after t+3 s, fee + 1 ¢ margin).** Head + null: +$7.4/day (lb95 −1.8,
4,952 trades, 1.6 ¢/share, 126/238 days positive), **−$1.3/day with a tick** (lb95 −8.6). Head alone +$11.3 (lb95
+2.9) at the print, −$2.2 with a tick. Null alone +$2.3 / +$1.5. The print's own size caps every trade (the prints
check is the engine's own fill by construction: 100 %). The information is the same size as on the Kalshi ladders
(0.0002–0.0003 Brier per day) and, as there, smaller than the spread.

**Reading.** With enough history the hourly head is measurably informative beside Polymarket's price, at the level of
a few tenths of a percent of Brier, stable across quarters, and worth about one cent a share at the print and nothing
after a tick. Same verdict as every other venue: information, no money. $1.1 of GPU.

### Item 5 — Kronos-base on the KXBTCD ladders (≈ $2.5 GPU, negative: the 4× model adds nothing to the small head)

`ft_base_1h` (the ft_small_1h recipe on `NeoQuasar/Kronos-base`, d832 / 12 layers / 102 M, 30 k steps × 64 bf16, 4.0 h
on an A40 at 134 samples/s; val head Brier 0.16148 vs the small model's 0.16124 on the same 2,048 windows, val CE 2.568
vs 2.577) and `cube_base_hidden_kalshi` (832-d hidden state every minute, 08-20 → 10-24, 3 min). Table
`data/tables/ladder_base.parquet` by `ladder.py` with the base head at every strike (same rows as the small-head table:
val 424,589 rows / 32 days, dev 433,424 / 34 days; the archive's 10-25 stub of 392 rows is dropped as before). The
base and small logits correlate at 0.993; raw Brier on the two-sided rows base 0.09212, small 0.09153, market 0.09081.
Readout `data/backtests/ladder_base/summary.json`.

| | coef(head) [95 % CI, day blocks] | frozen dev per-day Brier gain vs market | dev log-score gain vs market |
|---|---|---|---|
| small head (reference, as reported above) | 0.363 [0.011, 0.714] | **0.00024 [0.00002, 0.00047]**, 21/34 days | 0.0142 [0.0093, 0.0196] |
| **base head** | 0.376 [0.103, 0.694] | **0.00011 [−0.00014, 0.00034]**, 18/34 | 0.0136 [0.0092, 0.0186] |
| base + small together | base 0.304 [0.027, 0.618], small 0.116 [−0.227, 0.488] | 0.00015 [−0.00010, 0.00039] | — |
| base − small, paired by day | | **−0.00014 [−0.00026, −0.00001]**, base better on 13/34 days | small − base −0.0006 [−0.0029, 0.0015] |

On val the base head's coefficient is the small head's (h 10–20: 0.52 [0.11, 0.90]; wings: 0.33 [0.00, 0.81]). Frozen on
dev it is slightly *worse* than the small head, significantly so in the paired per-day comparison, and adding it to the
small head does not help (the pair's gain is 0.00015 vs the small head's 0.00024). The distribution log score is the
same for both. Backtests, frozen val calibration on dev, negative in every variant with a tick: per strike −$959/day
(6,786 trades, −1.1 ¢/share); one trade per ladder −$45/day; h ≤ 30 one per ladder −$45 (lb95 −92); h 10–20 one per
ladder −$78 (lb95 −126, −1.8 ¢/share; small head −$53). Maker simulator (`backtest/maker.py`, 26 M public prints):
improve +$22/day (lb95 −56; 3,098 fills; the profit is in h 35–59 where the head has no edge, h 9–20 loses −$457 in
total) vs the small head's −$38; join −$135/day (lb95 −269) vs −$122; h 5–30 improve −$8 (small +$37, and the null
pricer posting the same way earned +$50 there, so that window is the venue, not the model); with the 1.75 ¢ maker fee
+$13 (lb95 −65). Within noise of the small head in every mode, and no mode clears zero at the lower bound.

**Reading.** More capacity does not find more in the bars: the base head reproduces the small head's edge (same
coefficient, same log-score gain) and is marginally worse on the frozen Brier. The ladder edge is what the 1-minute
history contains beside the market, about 0.3 % of the market's Brier, independent of model size. Item 5 closed,
≈ $2.5 of GPU (4.1 h), pod `xrus4qaq2dprtn` terminated and verified gone.

### Item 2 — Polymarket ETH / SOL / XRP 5-minute markets (Mac only, $0, negative: efficient, not thin)

`pm_long.py` gained the three families (gamma series 10683 / 10686 / 10685, slug `<asset>-updown-5m-<unix start>`, Chainlink
TWAP-60 regime; gamma publishes no strike or final for them, so labels are gamma's resolved outcome and the strike proxy is
the Binance 1-min typical price of the bar closing at the window start, which agrees with gamma's outcome on 94–95 % of
windows, vs 89 % for close-vs-close); windows over each family's life (83,753 markets per asset, 2025-12-18 → 10-27),
taker orders and executable prints for 2026-08-01 → 10-26 (25,053 windows × 4 decision minutes per asset; 6.7 M / 3.2 M /
2.2 M orders; data-api paged by day because gamma caps the events offset at 2,000). Table `data/tables/alt5m_<fam>.parquet`
via `assemble.py` with the EWMA null pricer on `spot_<ASSET>USDT`; readout `data/backtests/pm_alt5m/summary.json`.

**Liquidity census.** Not thin at the minute level: median 232 / 113 / 70 taker orders per window (ETH / SOL / XRP), a
print in the first two minutes in 98.7 % of windows for all three, median 193 / 77 / 30 shares printed in the first
decision minute, median window volume $2,531 / $809 / $231. The executable-print coverage at a decision minute is 85 % /
79 % / 67 %. The scope's stop rule (no print in the first 2 minutes) does not fire.

**Null pricer beside the print mid.** Val = the first 43 priced days (08-01 → 09-12), dev = the last 44 (09-13 → 10-26),
fixed before the fit. The null's coefficient beside the market is *negative* for all three (ETH −0.10 [−0.15, −0.06],
SOL −0.05 [−0.08, −0.01], XRP −0.08 [−0.11, −0.04]) with the market's own coefficient 1.10–1.15: given the print mid, the
vol model points the wrong way, slightly. Frozen on dev: ETH blend − market −0.00009 [−0.00028, +0.00009], SOL −0.00009
[−0.00024, +0.00006], XRP **−0.00031 [−0.00047, −0.00014]**. A market-only recalibration (coefficient 1.05–1.06 on val)
gets ETH −0.00001, SOL +0.00005, XRP −0.00011 [−0.00026, +0.00005], so XRP's 0.2 % gain is the null, not the
sharpening of the mid. Taker backtest at the executable print (engine rule, 1 ¢ margin, $100 clips): ETH −$12/day
(lb95 −33), SOL −$14 (lb95 −31), XRP +$1.5/day (lb95 −13; +$2.3 with a tick, 802 trades, 1.5 ¢/share, 21/44 days);
3 ¢ margin: −$9 / −$1 / +$0.05. The prints check is tautological on Polymarket (the executing taker order is inside the
window and the engine already caps the clip by its size), so the fill-size P&L equals the backtest.

**Reading.** The alt 5m markets are not the Kalshi SOL ladder: they are liquid at the minute, priced as well as the BTC
ones, and a free vol model finds nothing beside their print mid (XRP: 0.2 % of the Brier, $1.5/day). The GPU gate
(frozen-dev gain with a CI excluding zero *and* positive fill-size P&L) fails on all three; no fine-tune was run.
Item 2 closed, $0.

### Item 3 — Kalshi hourly S&P 500 ladders (KXINXU) with equity minute bars (≈ $2.7 GPU, negative)

Data: `data/kalshi_ladder.py --series KXINXU --max-hours 72` (329 events 08-21 → 10-27, 7 per trading day, 60 strikes at
$5; the events open the evening before so only their last hour is tabled, minutes re-indexed from the close; 388 k
public prints over 46 days). `data/alpaca_bars.py` (Alpaca Market Data v2, SIP feed, the user's paper key in `.env`)
built `eq_SPY` and `eq_QQQ` 1-minute bars 2016 → 2026-10-27 (2.2 M / 2.1 M bars, market hours only; a session boundary
is a gap for the training mask). `ladder.py --basis-mode mul` moves the ladder's strike onto SPY levels with the
previous event's SPX / SPY ratio (median 10.49, sd 0.02; proxy error of SPX at the close vs SPY × ratio: median 0.7
index points, p90 2.0, against a $5 strike step). Table `data/tables/ladder_inx.parquet`: 357,119 rows, 261 events,
39 days; two-sided rows 38 %; median 142 k contracts per event, 118 k in the last hour, 8 two-sided strikes per minute,
ATM spread 4 ¢ (BTC ladders: 2 ¢).

**Free tests first** (`data/backtests/ladder_inx/free_tests.json`, Mac): null pricer beside the market on val (08-21 →
09-19, 16 days): coef 0.02 [−0.11, 0.26]; frozen on dev (09-22 → 10-27, 21 days) blend = market (−0.00075 → +0.0020);
taker backtest −$504/day; static arbitrage (`ladder_arb.py --max-hours 72`): 1 gross crossing in 15,884 ladder-minutes,
0 net of fees.

**Equity fine-tune** (`ft_small_1m_eq`: the E1b recipe on SPY + QQQ bars, kline cutoff 2026-08-01, 30 k × 64 bf16, 91
min on an A40 at 350 samples/s, val head Brier 0.1623 on SPY 08-07 → 09-07) and `cube_eq_hidden_inx` (hidden state
every SPY minute 08-20 → 10-28, 96 files); head applied at every strike (`ladder_inx_head.parquet`, 208 k rows with a
head, 58 % — the rest fall where the 256-bar lookback crosses a session gap or the minute has no bar). Readout
`data/backtests/ladder_inx/head_tests.json`, val 27,066 rows / 16 days, dev 41,408 / 21 days:

| fit | coef(head) [95 % CI] | coef(null) | coef(market) | Brier blend LODO / market |
|---|---|---|---|---|
| val, market + head + null | **−0.71 [−1.43, −0.19]** | 0.11 [−0.14, 0.62] | 1.71 | 0.1063 / 0.1052 |
| val, market + head | −0.57 [−0.83, −0.14] | | 1.69 | 0.1058 / 0.1052 |
| val h 10–20 | −0.54 [−1.18, −0.10] | | | 0.1023 / 0.1015 |
| dev refit (exploratory) | 0.20 [−0.37, 0.60] | −0.20 | 1.13 | 0.1047 / 0.1034 |

The head's coefficient on val is significantly *negative* beside the market (the head agrees with the null at 0.97
correlation and is worse than it raw: 0.1102 vs 0.1088 vs market 0.1041), and the fit's market coefficient of 1.7 is
the 16-day sample overfitting. Frozen on dev the blend is *worse* than the market by 0.0022 Brier (CI [−0.0001,
+0.0052]; +0.0138 in the last 9 minutes); the dev refit's head coefficient includes zero. Backtest, frozen, dev: engine
rule −$443/day (1,241 trades, −3.2 ¢/share), −$359 with a tick, −$204 at 1 ¢ margin; one fresh trade per ladder −$45/day
(lb95 −71), h ≤ 30 −$85. Prints check against the 388 k archived prints: 36 % of trades see any print in the next
minute, 20 % a print at or better than our price, 2.6 % fully fillable; on the fillable quantity −$32/day (1 min) and
−$130/day (5 min). Gate not met; no Kronos-base run.

**Reading.** On a product where the hourly ladder is thick ($5 steps, 118 k contracts in the last hour) and the
underlying is an equity index with a 1-minute proxy error of 0.7 points, the fine-tuned equity head carries nothing
beside the ladder's prices, and the small sample (16 val days, 21 dev days, 7 events a day) is enough to see that the
sign is wrong. Item 3 closed, ≈ $2.7 of GPU (4.6 h, 2.6 h of which the pod idled waiting for an upload flag the
launcher failed to write), pod `468v6jipmtmd0y` terminated and verified gone.

### Round 2 verdict

Five tests, four venues, ≈ $6.3 of GPU, none of them money. Beside the executable market price, Kronos's head adds a
statistically real but tiny amount on the Kalshi BTC ladders (0.3 % of the Brier; the base model adds nothing to the
small one) and on Polymarket's hourly BTC market over 17 months (coefficient 0.29 on val, frozen dev gain with a CI
touching zero), and nothing on the S&P ladders (sign wrong), nothing across venues at the executable price, and a free
vol model finds nothing in the alt 5m markets. Every taker backtest is negative after a tick; the maker route loses on
the ladders. The round's one positive statistic (the long hourly history) is what the earlier rounds found at smaller
scale: information the market has already priced to within the fee. The remaining credits ($75.39) are not worth
spending on more of the same; a different question (not "does the model beat the market" but e.g. a new venue's first
weeks, where the market is not yet efficient) would need new data, not more GPU.

## Training facts

- E1b-probe: 6,000 steps × 128, frozen pretrained small backbone, 10 min on an A40; val grid Brier plateaued at 0.161 from step 1,000.
- E1b: 30,000 steps × 64, joint CE + BCE, 77 min on an A40 (416 samples/s); val CE 2.671 → 2.609; val grid Brier 0.1609 (no better than the probe). Checkpoint: volume `/workspace/runs/ft_small_1m/best`, Mac copy `data/cloud/runs/ft_small_1m/best`.
- E0: zero-shot small, N = 128 paths, lookback 256, T = 1.0, top_p = 1.0; 25 min per day on an A40; raw path-count Brier 0.169 (worse than the null's 0.150 until calibrated); path dispersion 2.5 bps at h = 1 to 10.6 bps at h = 14.
- Sampling cube at 60 bars: `ft_small_1h` backbone, 64 paths, 288 points/day, 588 s per day on an A40 (no KV cache; 0.39× E0's per-day time for 0.2 × 0.5 × 4.3 the work), 10.8 h for 66 days; output 413 MB (`data/cloud/cubes/cube_ft1h_sample_kalshi`).
- ft_base_1h (Kronos-base, 102 M): 30,000 steps × 64 bf16, 4.0 h on an A40 (134 samples/s, 3.1× the small model's step cost); val CE 2.568 vs 2.577, val head Brier 0.16148 vs 0.16124 (small); the 832-d hidden dump over 66 days took 3 min. Checkpoint `data/cloud/runs/ft_base_1h/best`.
- ft_small_1m_eq (SPY + QQQ 1-min, E1b recipe): 30,000 steps × 64, 91 min on an A40 (350 samples/s); val CE 2.470, val head Brier 0.1623 on SPY. Checkpoint `data/cloud/runs/ft_small_1m_eq/best`.
- Controls: tokenizer round-trip, settlement-proxy flip rate and the null-pricer benchmark are in STATUS.md.

## Spend

Round 2 (2026-10-28): ≈ $6.3 of GPU — `ft_small_1h_early` + cube ≈ $1.1, `ft_base_1h` + cube ≈ $2.5 (4.3 h), `ft_small_1m_eq` + cube ≈ $2.7 (4.6 h incl. 2.6 h idle) — balance **$75.39** (RunPod `clientBalance`, 2026-10-28 after the equity pod); all pods verified gone.

About $19 of GPU time (A40 Secure Cloud at $0.59/h) plus $7/month for the 100 GB volume; balance $81.90 of $100 (RunPod `clientBalance` on 2026-10-27 after the 60-bar sampling cube) after E0 + probe + E1b + the E0 resume pod + the hidden-state dump, a pull pod, the Kalshi cube pod, the hourly probe pod, the hourly fine-tune pod, the Kalshi hidden-dump pod and the 60-bar sampling cube (10.8 h, ≈ $6.4).

## Verdict (validation)

The plan's gate for the next step (Kronos-base, E2) required the model coefficient's CI to exclude zero beside the market on validation. It does not, for the zero-shot model, the frozen-backbone head, or the jointly fine-tuned small model. The hypothesis that Kronos's conditional distribution of the next 1–15 minutes carries information the market has not priced is **not supported** on this data; the research repo's prior (the market embeds a near-exact nowcast and more) stands. The market-conditioned head (option 2) reaches the same conclusion from the other direction: given the market price, the fine-tuned model's full hidden state adds nothing on held-out days. Dev was opened once, for Kalshi KXBTC15M with the frozen pipeline (user decision): the frozen blend is 0.0004 Brier worse than the market's yes-mid on the 7 days with quotes and the backtest loses about $200 per day at $100 clips. The market is the better pricer on both venues; the venue was not the problem. The hourly Polymarket market (60-minute horizon, 85 days, head retrained for h ≤ 60) gives the same answer: the head equals the null pricer and adds nothing beside the market, with the backbone frozen or jointly fine-tuned on 60-bar targets. The Kalshi KXBTCD strike ladders are the one place the head adds anything beside the market: a 0.3 % Brier / 0.8 % log-score gain, real on val and dev, and not enough to pay the taker fee and spread. The sampled 60-bar distribution adds nothing to that head (coefficient zero in every form; the free-running sampler collapses beyond ~20 bars). Maker execution of that edge, simulated against 26 M public taker prints, loses under the optimistic fill bound (−$38/day on dev) and more under the pessimistic one (−$122/day); the resting-order flow on the ladders is informed (joining at the mid loses 3 ¢/share), so there is no route left from the ladder edge to money. Three free follow-ups (a market-conditioned head on the ladder rows, a realized-vol forecast test, taker-flow features beside the price) found nothing more in the hidden state or the public flow. Two non-Kronos checks on the public ladder data (static monotonicity arbitrage on the BTC ladders; cross-asset lag and a null-blend on the ETH and SOL ladders) found no tradeable crossings, an efficient ETH ladder, and a SOL ladder whose mispriced wings are quotes that nobody takes (−$139/day on the quantity that actually printed at our prices).
