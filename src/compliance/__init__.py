"""
src.compliance — activation-probe research package for the counterfactual
rule-dependence experiments in this repository.

Key entry points:
    MeanDifferenceDirection — the normalised difference-of-means probe (Eq. 1
                               in the paper)
    three_way_split          — train/val_layer/val_thresh/test split used by
                               every experiment
"""
from .direction import (
    MeanDifferenceDirection,
    PCADirection,
    LinearProbeDirection,
    get_direction_estimator,
)
from .metrics import (
    compute_metrics,
    compute_layerwise_auroc,
    select_anchor_layer,
    calibrate_threshold,
    mcnemar_test,
)
from .splits import three_way_split, three_way_split_items

__all__ = [
    "MeanDifferenceDirection",
    "PCADirection",
    "LinearProbeDirection",
    "get_direction_estimator",
    "compute_metrics",
    "compute_layerwise_auroc",
    "select_anchor_layer",
    "calibrate_threshold",
    "mcnemar_test",
    "three_way_split",
    "three_way_split_items",
]
