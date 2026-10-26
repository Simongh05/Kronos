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

## Training facts

- E1b-probe: 6,000 steps × 128, frozen pretrained small backbone, 10 min on an A40; val grid Brier plateaued at 0.161 from step 1,000.
- E1b: 30,000 steps × 64, joint CE + BCE, 77 min on an A40 (416 samples/s); val CE 2.671 → 2.609; val grid Brier 0.1609 (no better than the probe). Checkpoint: volume `/workspace/runs/ft_small_1m/best`, Mac copy `data/cloud/runs/ft_small_1m/best`.
- E0: zero-shot small, N = 128 paths, lookback 256, T = 1.0, top_p = 1.0; 25 min per day on an A40; raw path-count Brier 0.169 (worse than the null's 0.150 until calibrated); path dispersion 2.5 bps at h = 1 to 10.6 bps at h = 14.
- Controls: tokenizer round-trip, settlement-proxy flip rate and the null-pricer benchmark are in STATUS.md.

## Spend

About $10 of GPU time (A40 Secure Cloud at $0.59/h) plus $7/month for the 100 GB volume; balance $91 of $100 after E0 + probe + E1b + the E0 resume pod.

## Verdict (validation)

The plan's gate for the next step (Kronos-base, E2) required the model coefficient's CI to exclude zero beside the market on validation. It does not, for the zero-shot model, the frozen-backbone head, or the jointly fine-tuned small model. The hypothesis that Kronos's conditional distribution of the next 1–15 minutes carries information the market has not priced is **not supported** on this data; the research repo's prior (the market embeds a near-exact nowcast and more) stands. Dev was not opened and no α was spent.
