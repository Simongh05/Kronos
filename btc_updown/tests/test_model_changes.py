"""Upstream touch points: return_samples and the explicit causal flag."""
import numpy as np
import pandas as pd
import torch

from model import KronosPredictor
from btc_updown.train.model_io import backbone_hidden, dual_logits, encode_tokens
from .conftest import make_model, make_tokenizer, synthetic_series


def _inputs(n=80):
    x, _, ts = synthetic_series(n=n, gap_at=None)
    df = pd.DataFrame(x, columns=["open", "high", "low", "close", "volume", "amount"])
    t = pd.to_datetime(ts, unit="s", utc=True)
    return df, pd.Series(t)


def test_return_samples_topk1_matches_mean_path():
    tok, model = make_tokenizer(), make_model()
    pred = KronosPredictor(model, tok, device="cpu", max_context=64)
    df, t = _inputs()
    L, P = 60, 8
    x_df, x_t = df.iloc[:L], t.iloc[:L]
    y_t = t.iloc[L:L + P]
    mean_df = pred.predict(x_df, x_t, y_t, pred_len=P, T=1.0, top_k=1, sample_count=3, verbose=False)
    samples = pred.predict(x_df, x_t, y_t, pred_len=P, T=1.0, top_k=1, sample_count=3, verbose=False, return_samples=True)
    assert samples.shape == (3, P, 6)
    # top_k=1 is greedy, so all sample paths coincide and equal the mean path
    assert np.allclose(samples[0], samples[1], atol=1e-5) and np.allclose(samples[1], samples[2], atol=1e-5)
    assert np.allclose(samples.mean(0), mean_df.values, atol=1e-4)
    # batch API
    out = pred.predict_batch([x_df, x_df], [x_t, x_t], [y_t, y_t], pred_len=P, top_k=1, sample_count=2,
                             verbose=False, return_samples=True)
    assert len(out) == 2 and out[0].shape == (2, P, 6) and np.allclose(out[0][0], samples[0], atol=1e-4)


def test_default_predict_unchanged_shape():
    tok, model = make_tokenizer(), make_model()
    pred = KronosPredictor(model, tok, device="cpu", max_context=64)
    df, t = _inputs()
    out = pred.predict(df.iloc[:60], t.iloc[:60], t.iloc[60:66], pred_len=6, top_k=1, sample_count=2, verbose=False)
    assert isinstance(out, pd.DataFrame) and out.shape == (6, 6)


def test_causal_flag_blocks_leakage_in_eval_and_leaves_decode_s2_untouched():
    tok, model = make_tokenizer(), make_model()
    model.eval()
    x, stamp, _ = synthetic_series(n=40, gap_at=None)
    xt = torch.from_numpy(x[None]).float()
    st = torch.from_numpy(stamp[None]).float()
    xt = (xt - xt.mean(1, keepdim=True)) / (xt.std(1, keepdim=True) + 1e-5)
    s1, s2 = encode_tokens(tok, xt)
    s1_in, s2_in, st_in, s1_t = s1[:, :-1], s2[:, :-1], st[:, :-1], s1[:, 1:]

    # reference single-step decode_s2 output BEFORE any causal validation pass
    with torch.no_grad():
        logits_ref, ctx = model.decode_s1(s1_in, s2_in, st_in)
        s2_ref = model.decode_s2(ctx, s1_t[:, -1:])

    # perturb the LAST input token: the s2 cross-attention at earlier positions must not see the
    # hidden state of that later position when causal=True; with causal=False (eval default) it does
    s1_in_p, s2_in_p = s1_in.clone(), s2_in.clone()
    s1_in_p[:, -1] = (s1_in_p[:, -1] + 1) % (2 ** 4)
    s2_in_p[:, -1] = (s2_in_p[:, -1] + 3) % (2 ** 4)
    with torch.no_grad():
        hid = backbone_hidden(model, s1_in, s2_in, st_in)
        hid_p = backbone_hidden(model, s1_in_p, s2_in_p, st_in)
        assert torch.allclose(hid[:, :-1], hid_p[:, :-1], atol=1e-6)  # self-attention is always causal
        _, a = dual_logits(model, hid, s1_t, teacher_forcing=True, causal=True)
        _, b = dual_logits(model, hid_p, s1_t, teacher_forcing=True, causal=True)
        _, c = dual_logits(model, hid, s1_t, teacher_forcing=True, causal=False)
        _, d = dual_logits(model, hid_p, s1_t, teacher_forcing=True, causal=False)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), "causal=True must block future hidden states"
    assert not torch.allclose(c[:, :-1], d[:, :-1], atol=1e-6), "eval default is non-causal (documents the leak)"

    # decode_s2 behaves exactly as before: the flag is an argument, not module state
    with torch.no_grad():
        logits_after, ctx2 = model.decode_s1(s1_in, s2_in, st_in)
        s2_after = model.decode_s2(ctx2, s1_t[:, -1:])
    assert torch.allclose(logits_ref, logits_after) and torch.allclose(s2_ref, s2_after)
    # and the training-mode default (causal=None) is causal
    model.train()
    with torch.no_grad():
        hid = backbone_hidden(model, s1_in, s2_in, st_in)
        hid_p = backbone_hidden(model, s1_in_p, s2_in_p, st_in)
        _, e = dual_logits(model, hid, s1_t, teacher_forcing=True, causal=None)
        _, f = dual_logits(model, hid_p, s1_t, teacher_forcing=True, causal=None)
    assert torch.allclose(e[:, :-1], f[:, :-1], atol=1e-6)
