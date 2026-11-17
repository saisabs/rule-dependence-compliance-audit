"""
Direction estimators for Activation Probe Scores.

Three estimators are provided:
  MeanDifferenceDirection  — default Probe; no hyperparameters, most interpretable
  PCADirection             — first principal component of the pooled dataset
  LinearProbeDirection     — L2-normalised logistic regression weight vector

All estimators share the same interface:
    fit(c_acts, nc_acts)  →  self
    score(acts)           →  (N, L+1) Probe matrix
    direction_            →  (L+1, d) normalised direction per layer
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseDirection(ABC):
    direction_: np.ndarray  # (L+1, d)

    @abstractmethod
    def fit(self, c_acts: np.ndarray, nc_acts: np.ndarray) -> "BaseDirection":
        """
        Estimate compliance direction from calibration activations.

        Parameters
        ----------
        c_acts:  (N_train, L+1, d)
        nc_acts: (N_train, L+1, d)
        """

    def score(self, acts: np.ndarray) -> np.ndarray:
        """
        Project activations onto the compliance direction.

        Parameters
        ----------
        acts: (N, L+1, d)

        Returns
        -------
        (N, L+1)  — Probe score per sample per layer
        """
        return np.einsum("nld,ld->nl", acts, self.direction_)

    def score_at_layer(self, acts: np.ndarray, layer: int) -> np.ndarray:
        """Return (N,) Probe scores at a specific layer."""
        return acts[:, layer, :] @ self.direction_[layer]


class MeanDifferenceDirection(BaseDirection):
    """
    Normalised mean-difference estimator (default for Probe).

    direction[l] = normalise(mean(c_acts[:, l]) - mean(nc_acts[:, l]))

    Properties:
    - Parameter-free; no risk of overfitting on small calibration sets.
    - Interpretable: each layer direction points from the mean non-compliant
      prototype to the mean compliant prototype in that layer's subspace.
    """

    def fit(self, c_acts: np.ndarray, nc_acts: np.ndarray) -> "MeanDifferenceDirection":
        mean_c = c_acts.mean(axis=0)   # (L+1, d)
        mean_nc = nc_acts.mean(axis=0) # (L+1, d)
        diff = mean_c - mean_nc        # (L+1, d)
        norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
        self.direction_ = diff / norms
        logger.info(
            "MeanDifferenceDirection: mean direction norm across layers = %.4f",
            np.linalg.norm(diff, axis=1).mean(),
        )
        return self


class PCADirection(BaseDirection):
    """
    First principal component of the pooled (compliant + non-compliant) set.

    The sign is chosen so that compliant samples score higher than
    non-compliant on average (otherwise the direction is flipped).
    Included as an ablation baseline.
    """

    def fit(self, c_acts: np.ndarray, nc_acts: np.ndarray) -> "PCADirection":
        from sklearn.decomposition import PCA

        n_layers = c_acts.shape[1]
        directions = []
        for l in range(n_layers):
            X = np.concatenate([c_acts[:, l], nc_acts[:, l]], axis=0)  # (2N, d)
            pca = PCA(n_components=1, random_state=0)
            pca.fit(X)
            d = pca.components_[0]  # (d,)
            # Align: ensure compliant mean > non-compliant mean
            mean_c = (c_acts[:, l] @ d).mean()
            mean_nc = (nc_acts[:, l] @ d).mean()
            if mean_c < mean_nc:
                d = -d
            directions.append(d / (np.linalg.norm(d) + 1e-8))

        self.direction_ = np.stack(directions)  # (L+1, d)
        return self


class LinearProbeDirection(BaseDirection):
    """
    L2-normalised weight vector of a logistic regression probe.

    Trained per layer. The normalised weight vector serves as the
    compliance direction. Included as an ablation; requires enough
    calibration data to avoid overfitting.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, c_acts: np.ndarray, nc_acts: np.ndarray) -> "LinearProbeDirection":
        from sklearn.linear_model import LogisticRegression

        n_layers = c_acts.shape[1]
        directions = []
        labels = np.array([1] * len(c_acts) + [0] * len(nc_acts))

        for l in range(n_layers):
            X = np.concatenate([c_acts[:, l], nc_acts[:, l]], axis=0)
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=0)
            clf.fit(X, labels)
            d = clf.coef_[0]
            directions.append(d / (np.linalg.norm(d) + 1e-8))

        self.direction_ = np.stack(directions)
        return self


_ESTIMATORS = {
    "mean_diff": MeanDifferenceDirection,
    "pca": PCADirection,
    "linear_probe": LinearProbeDirection,
}


def get_direction_estimator(method: str, **kwargs) -> BaseDirection:
    if method not in _ESTIMATORS:
        raise ValueError(f"Unknown direction method '{method}'. Choose from {list(_ESTIMATORS)}")
    return _ESTIMATORS[method](**kwargs)
