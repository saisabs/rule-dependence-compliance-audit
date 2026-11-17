"""
Evaluation metrics for Probe experiments.

All metrics operate on parallel arrays of compliant and
non-compliant Probe scores from the test split.
"""
from __future__ import annotations
import logging
from typing import Dict, Tuple

import numpy as np
from scipy.stats import wilcoxon, binomtest
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def calibrate_threshold(probe_c: np.ndarray, probe_nc: np.ndarray) -> float:
    """
    Calibrate a decision threshold as the midpoint of the two class means.

    Callers MUST pass a val_thresh split here, never the test split — the
    whole point of a separate val_thresh slice (see splits.py) is that the
    threshold is picked before test is ever touched. Passing test scores in
    here computes a threshold "on the answer key" and inflates F1 (Bug 1).
    """
    return float((np.asarray(probe_c).mean() + np.asarray(probe_nc).mean()) / 2)


def compute_metrics(
    probe_c: np.ndarray,
    probe_nc: np.ndarray,
    threshold: float,
) -> Dict:
    """
    Compute all paper metrics from paired Probe scores.

    Parameters
    ----------
    probe_c:  (N,) Probe scores for compliant inputs (test split)
    probe_nc: (N,) Probe scores for non-compliant inputs (test split)
    threshold: classification threshold, calibrated beforehand on a
        held-out val_thresh split via calibrate_threshold() — never
        derived from probe_c/probe_nc themselves, or F1 leaks test info.

    Returns
    -------
    dict with AUROC, pairwise_accuracy, F1, mean/std/gap, threshold
    """
    assert len(probe_c) == len(probe_nc), "Paired arrays must have equal length"
    n = len(probe_c)

    scores = np.concatenate([probe_c, probe_nc])
    labels = np.array([1] * n + [0] * n)

    auroc = float(roc_auc_score(labels, scores))
    pair_acc = float(np.mean(probe_c > probe_nc))

    preds = (scores >= threshold).astype(int)
    f1 = float(f1_score(labels, preds))

    return {
        "auroc": auroc,
        "pairwise_accuracy": pair_acc,
        "f1": f1,
        "threshold": threshold,
        "mean_probe_compliant": float(probe_c.mean()),
        "mean_probe_non_compliant": float(probe_nc.mean()),
        "probe_gap": float(probe_c.mean() - probe_nc.mean()),
        "std_probe_compliant": float(probe_c.std()),
        "std_probe_non_compliant": float(probe_nc.std()),
        "n_pairs": n,
    }


def compute_layerwise_auroc(
    c_acts: np.ndarray,
    nc_acts: np.ndarray,
    direction,
) -> np.ndarray:
    """
    Compute AUROC at every layer using the fitted direction estimator.

    Parameters
    ----------
    c_acts:  (N, L+1, d)
    nc_acts: (N, L+1, d)
    direction: fitted BaseDirection instance

    Returns
    -------
    aurocs: (L+1,) — AUROC per layer
    """
    n_layers = c_acts.shape[1]
    aurocs = np.zeros(n_layers)
    labels = np.array([1] * len(c_acts) + [0] * len(nc_acts))

    for l in range(n_layers):
        probe_c = c_acts[:, l] @ direction.direction_[l]
        probe_nc = nc_acts[:, l] @ direction.direction_[l]
        scores = np.concatenate([probe_c, probe_nc])
        try:
            aurocs[l] = roc_auc_score(labels, scores)
        except ValueError:
            aurocs[l] = 0.5  # degenerate case
    return aurocs


def select_anchor_layer(
    c_acts_val: np.ndarray,
    nc_acts_val: np.ndarray,
    direction,
) -> int:
    """
    Select the layer with highest AUROC on the validation set.

    Parameters
    ----------
    c_acts_val, nc_acts_val: (N_val, L+1, d)
    direction: fitted BaseDirection

    Returns
    -------
    int: selected anchor layer index
    """
    aurocs = compute_layerwise_auroc(c_acts_val, nc_acts_val, direction)
    l_star = int(np.argmax(aurocs))
    logger.info(
        "Anchor layer selection — l* = %d (val AUROC = %.4f)",
        l_star, aurocs[l_star],
    )
    return l_star


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Returns (statistic, p_value).  Used in V3 surface confound test.
    """
    stat, p = wilcoxon(scores_a, scores_b)
    return float(stat), float(p)


def pairwise_accuracy(probe_a: np.ndarray, probe_b: np.ndarray) -> float:
    """
    Fraction of pairs where probe_a[i] > probe_b[i].
    """
    return float(np.mean(probe_a > probe_b))


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
) -> Tuple[int, int, float]:
    """
    Exact McNemar's test for paired binary correct/incorrect outcomes
    (e.g. first-candidate-correct vs Probe-guided-candidate-correct on the
    same cases) — the right test for paired binary outcomes, unlike
    Wilcoxon's signed-rank test which assumes a continuous, ranked signal.

    Only the discordant pairs (where a and b disagree) carry information;
    concordant pairs (both right or both wrong) are uninformative and
    dropped. The exact test is a two-sided binomial test on the
    discordant-pair counts against p=0.5 — no statsmodels dependency
    needed for this.

    Returns
    -------
    n_a_only: cases correct under a but not b ("a wins")
    n_b_only: cases correct under b but not a ("b wins")
    p_value: two-sided exact McNemar p-value
    """
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)
    n_a_only = int(np.sum(correct_a & ~correct_b))
    n_b_only = int(np.sum(correct_b & ~correct_a))
    n_discordant = n_a_only + n_b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(n_b_only, n_discordant, p=0.5, alternative="two-sided").pvalue
    return n_a_only, n_b_only, float(p_value)
