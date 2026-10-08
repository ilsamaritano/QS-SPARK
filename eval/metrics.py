"""
Evaluation metrics matching PLASyR definitions.

Metrics
-------
AUROC           : standard ROC area under curve
EER             : equal error rate (FPR = FNR crossing point)
TAR@FAR=1%      : true-acceptance rate at 1% false-acceptance (OvR task)
TRR@FAR=1%      : true-replay-rejection rate at 1% false-alarm on genuine (replay task)

GaussianMahalanobis
-------------------
Fit a multivariate Gaussian on genuine train embeddings; score query embeddings
by squared Mahalanobis distance. Higher score = more anomalous = more replay-like.

Per PLASyR replay_eval.py: features are standardised (zero-mean, unit-std) before
fitting the covariance matrix for numerical stability with high-dimensional embeddings.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def tar_at_far(y_true: np.ndarray, scores: np.ndarray,
               far_target: float = 0.01) -> float:
    """True Acceptance Rate at FAR ≤ far_target (for OvR authentication)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Find highest TPR where FPR ≤ far_target
    mask = fpr <= far_target
    if not mask.any():
        return 0.0
    return float(tpr[mask][-1])


def trr_at_far(genuine_scores: np.ndarray, replay_scores: np.ndarray,
               far_target: float = 0.01) -> tuple:
    """
    True Replay Rejection rate at FAR = far_target.

    The threshold θ is set from the genuine score distribution:
    reject (flag as replay) when Mahalanobis ≥ θ, where θ is the
    (1 - far_target) percentile of genuine scores.

    Returns (trr, theta).
    """
    theta = float(np.percentile(genuine_scores, 100.0 * (1.0 - far_target)))
    trr   = float(np.mean(replay_scores >= theta))
    return trr, theta


# ── Mahalanobis anomaly detector ──────────────────────────────────────────────

class GaussianMahalanobis:
    """
    Fit a Gaussian on D-dimensional embeddings; score queries by squared
    Mahalanobis distance from the fitted distribution.

    When N (genuine train samples) ≤ D (embedding dim), the sample covariance
    is rank-deficient. We automatically apply PCA to reduce to min(N-1, max_pca)
    principal components before fitting. This is the standard approach for small-N
    anomaly detection in high-dimensional spaces.

    Attributes saved after fit():
        mu_f, sd_f : per-feature mean / std (for standardisation)
        mu         : standardised centroid in PCA space
        inv_cov    : inverse covariance in PCA space
        pca        : sklearn PCA (or None if not applied)
    """

    def __init__(self, reg: float = 1e-6, max_pca: int = 64):
        self.reg     = reg
        self.max_pca = max_pca  # max PCA components; None disables auto-PCA
        self.mu_f    = None
        self.sd_f    = None
        self.mu      = None
        self.inv_cov = None
        self.pca     = None

    def fit(self, embeddings: np.ndarray) -> "GaussianMahalanobis":
        """embeddings: (N, D) float array of genuine training embeddings."""
        self.mu_f = embeddings.mean(axis=0)
        self.sd_f = embeddings.std(axis=0) + 1e-8
        z = (embeddings - self.mu_f) / self.sd_f

        # Apply PCA if N ≤ D (covariance would be singular without it)
        n, d = z.shape
        if self.max_pca and n <= d:
            from sklearn.decomposition import PCA
            n_components = min(self.max_pca, n - 1)
            self.pca = PCA(n_components=n_components, random_state=0)
            z = self.pca.fit_transform(z)

        self.mu  = z.mean(axis=0)
        cov = np.cov(z.T)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov += self.reg * np.eye(len(self.mu))
        self.inv_cov = np.linalg.inv(cov)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return squared Mahalanobis distances (higher = more anomalous)."""
        z = (embeddings - self.mu_f) / self.sd_f
        if self.pca is not None:
            z = self.pca.transform(z)
        d = z - self.mu
        return np.einsum("bi,ij,bj->b", d, self.inv_cov, d)
