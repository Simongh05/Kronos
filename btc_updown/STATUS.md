# btc_updown — status (2026-10-26)

Plan: `~/.claude/plans/keen-questing-quill.md` (approved 2026-10-25). Branch `btc-updown`. Nothing committed yet.

## What exists

| Piece | Path | State |
|---|---|---|
| Upstream: explicit `causal` arg on the s2 cross-attention; `return_samples` on inference | `model/module.py`, `model/kronos.py` | done, defaults unchanged, `tests/` regression passes (4/4) |
| Kline downloader → parquet + memmaps | `btc_updown/data/binance_klines.py` | done (other session); data in `data/klines/{spot_BTCUSDC,spot_BTCUSDT,um_BTCUSDT}` 2019-01 → 2026-10-24 |
| Labels + executable prices (guard-checked, read-only) | `btc_updown/data/labels.py` → `data/labels/*.parquet` | done (other session); Polymarket windows 2025-12 → 10-11, prints 07-26 → 10-09, BBO 10-10/11, Kalshi quotes 10-11 → 10-17 |
| Dataset (memmap, lookback-only z-score, contiguity, deterministic sampling) | `btc_updown/train/dataset.py` | done, tested |
| Training loop (CE + DirectHead BCE, bf16 on CUDA, exact resume, best-by-task-metric) | `btc_updown/train/loop.py`, `heads.py`, `validate.py`, `model_io.py`, CLIs `train_predictor.py`, `train_tokenizer.py` | done, tested (resume reproduces losses to 1e-5) |
| Forecast cube (sampling summary + sorted TP samples + DirectHead logits per minute) | `btc_updown/infer/cube.py` | done, tested |
| Tokenizer round-trip control | `btc_updown/infer/roundtrip.py` → `data/controls/tokenizer_roundtrip.json` | done |
| Null pricer (EWMA-vol TWAP-aware digital) | `btc_updown/data/nulls.py` | done |
| Proxy validation (bar proxy vs exact 60-s mean vs recorded final) | `btc_updown/data/proxy_check.py` → `data/controls/proxy_check.json` | done (research venv) |
| Decision table (labels × prices × cube × null) | `btc_updown/backtest/assemble.py` | done, tested |
| Calibration / blend with day-block bootstrap | `btc_updown/calib/fit.py` | done, tested |
| Market-conditioned head (market + numeric + Kronos hidden state, MLP; train/test by day; paired bootstrap; backtest) | `btc_updown/calib/market_head.py` | done, run on val (see REPORT.md) |
| Backtest engine (one entry per window, fees, two execution assumptions, day stats) | `btc_updown/backtest/engine.py` | done, tested |
| RunPod scripts | `btc_updown/cloud/` | written (other session); untested against a live account |
| Configs | `btc_updown/configs/*.yaml` | `ft_small_1m`, `probe_small_1m`, `cube_e0_small_val`, `smoke_local`, `cube_smoke_local` |

Tests: `.venv/bin/python -m pytest btc_updown/tests/test_dataset.py btc_updown/tests/test_model_changes.py btc_updown/tests/test_loop_and_cube.py btc_updown/tests/test_backtest.py tests/` (15 + 4 pass). `test_labels.py` needs the research venv (duckdb); `test_binance_klines.py` / `test_cloud_scripts.py` belong to the other sessions.

## Control results (validation range 2026-08-07 → 09-06 unless stated)

**Tokenizer round-trip (pretrained Kronos-Tokenizer-base, BTCUSDC 1-min, 400 windows, last 16 bars):**

| lookback | TP reconstruction error $ p50 / p90 / p99 | 1-min |close move| $ p50 / p90 | share err > $58 | distinct s1 / full tokens |
|---|---|---|---|---|
| 512 | 4.9 / 24.8 / 76.2 | 13.9 / 57.1 | 2.2% | 358 / 1163 |
| 256 | 3.7 / 15.8 / 42.1 | 13.3 / 54.5 | 0.4% | 384 / 1205 |
| 128 | 2.9 / 13.3 / 42.9 | 13.4 / 58.4 | 0.5% | 392 / 1204 |

Decision: **no tokenizer fine-tune for now; lookback 256 is the default** (cube and training).

**Settlement proxy (Binance 1-s klines 07-31 → 10-10 vs Polymarket labels; outcome = final ≥ strike):**

| series | family | n | basis (final − exchange 60-s mean) | TP-proxy err vs exact mean $ p50/p90 | median |final−strike| $ | flip: exact exchange mean | flip: TP bar proxy | flip: close proxy | flip TP when margin < $20 |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 5m | 10,610 | −$23.3 | 3.0 / 11.3 | 35.9 | 2.2% | 5.1% | 9.2% | 15.2% |
| BTCUSDT | 15m | 3,535 | −$23.3 | 2.7 / 10.5 | 60.9 | 1.4% | 3.1% | 4.2% | 14.7% |
| BTCUSDC | 5m | 8,574 | −$10.3 | 3.1 / 11.7 | 36.9 | 2.6% | 5.1% | 9.5% | 15.7% |
| BTCUSDC | 15m | 2,856 | −$10.3 | 3.0 / 11.1 | 63.3 | 1.5% | 3.1% | 5.1% | 15.6% |

Reading: a bar-based model cannot beat ~95% (5m) / ~97% (15m) outcome agreement even with a perfect forecast of the bar; the typical price is clearly the right readout (the close alone doubles the flip rate). The basis to Chainlink is stable (sd ≈ $4.9) so comparing paths against the series' own strike proxy is sound.

**Null pricer and market on validation (Polymarket, TWAP-60 regimes, decision minutes with ≥ 60 s left):**

| rows | windows | days | Brier market | Brier null | Brier blend(market+null) | Brier constant | coef(logit market) | coef(logit null) 95% CI |
|---|---|---|---|---|---|---|---|---|
| 49,131 | 7,242 | 23 | **0.1480** | 0.1559 | 0.1479 (LODO 0.1481) | 0.2500 | 1.01 [0.93, 1.12] | 0.05 [−0.05, 0.12] |

By family: 5m market 0.1367 / null 0.1469; 15m market 0.1557 / null 0.1621. The market is well calibrated (slope ≈ 1) and the EWMA null adds nothing to it. **This is the bar for Kronos: a model term with a bootstrap CI excluding zero next to the market term.**

Backtest sanity with the market-only blend (null model, side-consistent market input): print execution +$15/day mean over 31 days (bootstrap lower bound −$28), print+tick +$3/day; not significant, as it should be for a null.

Note found while testing: a blend fed the *market mid* (two prints up to 27 s apart) and traded against the *earlier* print showed a fake +2.4¢/share edge. The engine therefore feeds each side's own executable ask as the market input (`engine.run_backtest`). Keep it that way.

## How to run the real experiments (RunPod, after the account exists)

```bash
# on the pod, after cloud/bootstrap.sh
python -m btc_updown.infer.cube --config btc_updown/configs/cube_e0_small_val.yaml          # E0, ~5 h on a 4090
python -m btc_updown.train.train_predictor --config btc_updown/configs/probe_small_1m.yaml  # E1b probe, ~30 min
python -m btc_updown.train.train_predictor --config btc_updown/configs/ft_small_1m.yaml     # E1b, ~3 h (set data.lookback: 256)
# on the Mac, after cloud/sync.sh pulled data/cloud/cubes/<exp>
python -m btc_updown.backtest.assemble --start 2026-08-07 --end 2026-09-07 --cube data/cloud/cubes/cube_e0_small_val --out data/tables/val_e0.parquet
python -m btc_updown.calib.fit --table data/tables/val_e0.parquet --features z_model log_rel_sd --venue polymarket --out data/calib/e0.json
python -m btc_updown.backtest.engine --table data/tables/val_e0.parquet --calib data/calib/e0.json --venue polymarket --out data/backtests/val_e0
```
Dev (09-07 → 10-14) is opened once, with the frozen calib JSON, after selection on val.

## Observations from the Mac smoke runs

- End-to-end chain verified on real data (cube_smoke_local, 10 decision minutes of 2026-08-20, zero-shot small, N=8, CPU): cube → `assemble` → `z_model`, `p_samp` populate and correlate 0.8 with the null pricer.
- **Zero-shot dispersion looks ~3× too wide**: tp_sd ≈ $100–320 at 3–14 bars ahead where realised BTC vol implies ≈ $40–95. The calibration layer absorbs the scale, but expect the selected temperature on val to be **below 1**, not above.
- **MPS hang**: `infer.cube` on `device: auto` (Apple GPU) stalls at 0% CPU inside the autoregressive sampler; CPU works (10 points × 8 samples × 15 steps in 15 s). Training on MPS works. Use `device: cpu` for Mac cube smoke tests; the real runs are CUDA.
- Mac training throughput (Kronos-small, lookback 128, batch 4, MPS): ~10 samples/s. Not a substitute for the cloud.

## RunPod resources (created 2026-10-25)

| Item | Value |
|---|---|
| Account key | `RUNPOD_API_KEY` in `/Users/streetteam/Kronos/.env` (Mac only; also used for the first pods' self-termination test via RunPod's injected pod key) |
| Datacenter | **CA-MTL-1** (Montréal): A40 48 GB at $0.59/h, Medium stock; RTX A5000 24 GB at $0.27/h fallback. RTX 4090s were $0.89/h and Low stock in every datacenter on 2026-10-25 |
| Network volume | `kronos-ws`, id **`stv454bk3l`**, 100 GB, mounted at `/workspace` ($7/month) |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Pod recipe | REST `POST /v1/pods` with `gpuTypeIds ["NVIDIA A40"]`, `cloudType SECURE`, `dataCenterIds ["CA-MTL-1"]`, `networkVolumeId stv454bk3l`, `ports ["22/tcp"]`, `dockerStartCmd` = bootstrap from the public fork + `run_job.sh` |
| Fork | https://github.com/Simongh05/Kronos branch `btc-updown` (public; code only) |

## Cloud results so far (2026-10-25)

| Exp | Pod | Outcome |
|---|---|---|
| Dry run | A40, 50 steps + 2,048-window val in 21 s | pipeline OK; self-termination works via the GraphQL fallback with RunPod's injected pod key (REST DELETE returns 403 for it); no second key needed |
| E1b-probe (frozen small backbone + DirectHead, 6,000 steps × 128) | 10 min, $0.10 | val grid Brier 0.161 (plateau from step 1,000). On real val decision points: raw Brier 0.1503 vs null 0.1512; corr(logit_direct, logit_null) = 0.97; **beside the market: coef 0.008, CI [−0.11, 0.16] → no incremental information** (`data/calib/val_probe_direct_market.json`) |
| E0 (zero-shot small sampling cube, val, N=128, lookback 256) | 12 h (28 days) + resume pod for the last 3 days, ≈ $8 | beside the market: z_model coef 0.05, CI [−0.03, 0.14]; standalone gain vs null 0.0002–0.001 Brier. See REPORT.md |
| E1b (joint CE + BCE fine-tune, small, 30k steps × 64) | 77 min, $0.76 | val CE 2.671 → 2.609; grid Brier 0.1609 (= probe). On real val rows beside the market: coef 0.082, CI [−0.03, 0.21]; standalone beats the null by 0.001 Brier. **Gate for E2 not met.** See REPORT.md |
| Option 2: hidden-state dump (`cube_e1b_hidden_val`, 31 val days, 512-d per minute) + market-conditioned MLP head on the Mac | 43 s dump + pull pod, ≈ $1 | Trained on 08-08 → 08-20, early-stopped 08-21 → 08-24, **tested on 08-26 → 09-06**: Brier C (market+numeric+hidden) 0.1502 vs market-only A 0.1501; paired CI of C − A [−0.0009, +0.0010]; six seed/PCA variants, none better than A; backtest C negative in 5/6. **No information in the hidden state beside the market.** See REPORT.md |

## Open items

- RunPod account, prepaid credits, API keys, network volume (user) → dry run of `cloud/run_job.sh` self-termination.
- Training configs use `lookback: 256` (matches the cube; decided from the round-trip test).
- `backtest/report.py` renders fit + engine JSON to markdown.
- Mac shell is zsh: never put an ssh command in a variable and expand it unquoted (`$SSH host …` fails with "no such file"); write the full command or use `eval`. This silently broke one waiter on 2026-10-26.
- Pods based on the bare `runpod/pytorch` image have no `rsync`; pull with `tar czf - | tar xzf -` over ssh.
- Kalshi leg: quotes exist 10-11 → 10-17 with labels from the recorder meta; not yet exercised through assemble (prices_kalshi has the same columns).
