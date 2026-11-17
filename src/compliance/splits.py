"""
Shared calibration-split utility.

Every Probe experiment makes three independent modelling decisions that must
each see disjoint data, plus a held-out test set that is touched exactly
once for final reporting:

    train       -> fit the compliance direction (mean-difference / PCA / probe)
    val_layer   -> select the anchor layer (argmax AUROC over layers)
    val_thresh  -> calibrate the decision threshold (used for F1)
    test        -> report final AUROC / pairwise accuracy / F1, once

Reusing one split for two of these decisions is a double-dipping bug:
picking the anchor layer AND the threshold from the same slice inflates
the reported F1, and evaluating on
data that informed any calibration decision inflates AUROC / pairwise
accuracy. Every script that needs a split should go through this module
rather than rolling its own train_test_split calls, so there is exactly
one place to get the split logic right.
"""
from __future__ import annotations
import random
from typing import List, Sequence, Tuple, TypeVar

T = TypeVar("T")

# Splits data into four disjoint groups (40/20/20/20) so that a genuinely
# held-out test slice exists alongside the three calibration groups above.
DEFAULT_TRAIN_FRAC = 0.40
DEFAULT_VAL_LAYER_FRAC = 0.20
DEFAULT_VAL_THRESH_FRAC = 0.20
# test_frac is the remainder (default 0.20)


def three_way_split(
    n: int,
    seed: int,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_layer_frac: float = DEFAULT_VAL_LAYER_FRAC,
    val_thresh_frac: float = DEFAULT_VAL_THRESH_FRAC,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Deterministically shuffle range(n) and split off train, then split the
    remainder three ways: (train_idx, val_layer_idx, val_thresh_idx, test_idx).

    test_frac is implied as 1 - train_frac - val_layer_frac - val_thresh_frac.
    Some callers already have an external, disjoint test pool (e.g. a
    guardrail script that samples its test cases from a separate held-out
    pool outside the direction-building set) — those callers can simply
    ignore the returned test_idx and only consume train/val_layer/val_thresh.
    """
    if train_frac + val_layer_frac + val_thresh_frac >= 1.0:
        raise ValueError(
            "train_frac + val_layer_frac + val_thresh_frac must leave room "
            "for a nonzero test_frac"
        )

    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(n * train_frac)
    n_val_layer = int(n * val_layer_frac)
    n_val_thresh = int(n * val_thresh_frac)

    train_idx = idx[:n_train]
    val_layer_idx = idx[n_train : n_train + n_val_layer]
    val_thresh_idx = idx[n_train + n_val_layer : n_train + n_val_layer + n_val_thresh]
    test_idx = idx[n_train + n_val_layer + n_val_thresh :]

    return train_idx, val_layer_idx, val_thresh_idx, test_idx


def three_way_split_items(
    items: Sequence[T],
    seed: int,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_layer_frac: float = DEFAULT_VAL_LAYER_FRAC,
    val_thresh_frac: float = DEFAULT_VAL_THRESH_FRAC,
) -> Tuple[List[T], List[T], List[T], List[T]]:
    """Same split as three_way_split, applied directly to a list of items."""
    train_idx, val_layer_idx, val_thresh_idx, test_idx = three_way_split(
        len(items), seed, train_frac, val_layer_frac, val_thresh_frac
    )
    return (
        [items[i] for i in train_idx],
        [items[i] for i in val_layer_idx],
        [items[i] for i in val_thresh_idx],
        [items[i] for i in test_idx],
    )
