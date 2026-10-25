"""Loading/saving Kronos models and the shared forward pass used by training and validation."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Optional, Tuple

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import Kronos, KronosTokenizer  # noqa: E402


def load_tokenizer(path: str, revision: Optional[str] = None) -> KronosTokenizer:
    if os.path.isdir(path):
        return KronosTokenizer.from_pretrained(path)
    return KronosTokenizer.from_pretrained(path, revision=revision)


def load_model(path: str, revision: Optional[str] = None) -> Kronos:
    if os.path.isdir(path):
        return Kronos.from_pretrained(path)
    return Kronos.from_pretrained(path, revision=revision)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, enabled: bool):
    """bf16 autocast on CUDA only; the tokenizer is always run outside of it in fp32."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def encode_tokens(tokenizer: KronosTokenizer, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 tokenization of a normalized window x [B, W, 6] -> (s1 [B, W], s2 [B, W])."""
    s1, s2 = tokenizer.encode(x.float(), half=True)
    return s1, s2


def backbone_hidden(model: Kronos, s1_in: torch.Tensor, s2_in: torch.Tensor, stamp_in: torch.Tensor,
                    padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Hidden states [B, T, d_model] after the transformer stack and final norm (mirrors Kronos.forward)."""
    x = model.embedding([s1_in, s2_in])
    if stamp_in is not None:
        x = x + model.time_emb(stamp_in)
    x = model.token_drop(x)
    for layer in model.transformer:
        x = layer(x, key_padding_mask=padding_mask)
    return model.norm(x)


def dual_logits(model: Kronos, hidden: torch.Tensor, s1_targets: torch.Tensor, teacher_forcing: bool,
                causal: Optional[bool], padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """s1 and s2 logits from hidden states.

    teacher_forcing=True conditions the s2 head on the true s1 tokens (deterministic, used for
    validation); False samples s1 from the model's own distribution as upstream training does.
    """
    s1_logits = model.head(hidden)
    if teacher_forcing:
        sibling = model.embedding.emb_s1(s1_targets)
    else:
        probs = torch.softmax(s1_logits.detach().float(), dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(s1_targets.shape)
        sibling = model.embedding.emb_s1(sampled)
    x2 = model.dep_layer(hidden, sibling, key_padding_mask=padding_mask, causal=causal)
    s2_logits = model.head.cond_forward(x2)
    return s1_logits, s2_logits
