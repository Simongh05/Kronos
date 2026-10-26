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

## Training facts

- E1b-probe: 6,000 steps × 128, frozen pretrained small backbone, 10 min on an A40; val grid Brier plateaued at 0.161 from step 1,000.
- E1b: 30,000 steps × 64, joint CE + BCE, 77 min on an A40 (416 samples/s); val CE 2.671 → 2.609; val grid Brier 0.1609 (no better than the probe). Checkpoint: volume `/workspace/runs/ft_small_1m/best`, Mac copy `data/cloud/runs/ft_small_1m/best`.
- E0: zero-shot small, N = 128 paths, lookback 256, T = 1.0, top_p = 1.0; 25 min per day on an A40; raw path-count Brier 0.169 (worse than the null's 0.150 until calibrated); path dispersion 2.5 bps at h = 1 to 10.6 bps at h = 14.
- Controls: tokenizer round-trip, settlement-proxy flip rate and the null-pricer benchmark are in STATUS.md.

## Spend

About $11 of GPU time (A40 Secure Cloud at $0.59/h) plus $7/month for the 100 GB volume; balance ≈ $88 of $100 after E0 + probe + E1b + the E0 resume pod + the hidden-state dump, a pull pod and the Kalshi cube pod.

## Verdict (validation)

The plan's gate for the next step (Kronos-base, E2) required the model coefficient's CI to exclude zero beside the market on validation. It does not, for the zero-shot model, the frozen-backbone head, or the jointly fine-tuned small model. The hypothesis that Kronos's conditional distribution of the next 1–15 minutes carries information the market has not priced is **not supported** on this data; the research repo's prior (the market embeds a near-exact nowcast and more) stands. The market-conditioned head (option 2) reaches the same conclusion from the other direction: given the market price, the fine-tuned model's full hidden state adds nothing on held-out days. Dev was opened once, for Kalshi KXBTC15M with the frozen pipeline (user decision): the frozen blend is 0.0004 Brier worse than the market's yes-mid on the 7 days with quotes and the backtest loses about $200 per day at $100 clips. The market is the better pricer on both venues; the venue was not the problem.
