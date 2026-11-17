"""
Activation extraction from transformer residual streams.

Design decisions:
  - last_token: integrates causal context; unaffected by shared prompt prefix
  - mean_pool:  included for ablation; degrades when pairs share a long prefix
  - Caching:    saves recomputation; keyed by (config.cache_key, split_name)
  - batch_size=1: ensures correctness of last-token index across all tokenizers
"""
from __future__ import annotations
import logging
import os
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_activations(
    texts: List[str],
    model,
    tokenizer,
    max_length: int = 256,
    extraction: Literal["last_token", "mean_pool"] = "last_token",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract residual-stream activations for each text in `texts`.

    Returns
    -------
    np.ndarray  shape (N, L+1, d)
        N = number of texts
        L+1 = embedding + L transformer layers
        d = hidden_dim
    """
    all_acts: List[np.ndarray] = []
    it = tqdm(texts, desc="Extracting", disable=not show_progress)

    for text in it:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (1, T, d) tensors, one per layer (incl. embedding)
        hidden_states = outputs.hidden_states  # len = L+1

        if extraction == "last_token":
            # Take position of the last real token (T-1 for non-padded input)
            acts = torch.stack(
                [h[0, -1, :] for h in hidden_states]
            )  # (L+1, d)
        elif extraction == "mean_pool":
            # Average across all token positions
            attention_mask = inputs["attention_mask"][0]  # (T,)
            acts = torch.stack(
                [
                    (h[0] * attention_mask.unsqueeze(-1).float()).sum(0)
                    / attention_mask.float().sum()
                    for h in hidden_states
                ]
            )  # (L+1, d)
        else:
            raise ValueError(f"Unknown extraction mode: {extraction!r}")

        all_acts.append(acts.cpu().float().numpy())

    return np.stack(all_acts)  # (N, L+1, d)


# ── Caching ────────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, cache_key: str, split: str) -> str:
    return os.path.join(cache_dir, f"{cache_key}__{split}.npz")


def save_activations(
    cache_dir: str,
    cache_key: str,
    split: str,
    c_acts: np.ndarray,
    nc_acts: np.ndarray,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, cache_key, split)
    np.savez_compressed(path, c_acts=c_acts, nc_acts=nc_acts)
    logger.info("Saved activation cache → %s", path)


def load_activations(
    cache_dir: str,
    cache_key: str,
    split: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = _cache_path(cache_dir, cache_key, split)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    logger.info("Loaded activation cache ← %s", path)
    return data["c_acts"], data["nc_acts"]


# ── Paired extraction with caching ────────────────────────────────────────────

def get_paired_activations(
    c_texts: List[str],
    nc_texts: List[str],
    model,
    tokenizer,
    max_length: int = 256,
    extraction: Literal["last_token", "mean_pool"] = "last_token",
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    split: str = "unknown",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and optionally cache paired activations.

    Returns
    -------
    c_acts:  (N, L+1, d)
    nc_acts: (N, L+1, d)
    """
    if cache_dir and cache_key:
        cached = load_activations(cache_dir, cache_key, split)
        if cached is not None:
            return cached

    logger.info(
        "Extracting %d compliant + %d non-compliant activations [%s]",
        len(c_texts), len(nc_texts), split,
    )
    c_acts = extract_activations(
        c_texts, model, tokenizer, max_length, extraction
    )
    nc_acts = extract_activations(
        nc_texts, model, tokenizer, max_length, extraction
    )

    if cache_dir and cache_key:
        save_activations(cache_dir, cache_key, split, c_acts, nc_acts)

    return c_acts, nc_acts
