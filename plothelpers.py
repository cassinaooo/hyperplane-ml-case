"""Small plotting helpers for evaluating binary classifiers.

- `KS`: Kolmogorov-Smirnov statistic between the score distributions of the
  positive and negative classes, plus a plot of both cumulative distributions.
- `ScoreDistribution`: overlaid histograms of the scores for each class.
- `PSI`: Population Stability Index between two score samples.
"""

import matplotlib.pyplot as plt
import numpy as np


def _as_arrays(scores, y):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y).reshape(-1).astype(int)
    if scores.shape != y.shape:
        raise ValueError(f"scores and y must have the same length ({scores.shape} != {y.shape})")
    return scores, y


def _ecdf(values, grid):
    values = np.sort(values)
    return np.searchsorted(values, grid, side="right") / max(len(values), 1)


class KS:
    """Kolmogorov-Smirnov separation between positive (y == 1) and negative (y == 0) scores."""

    def __init__(self, scores, y):
        self.scores, self.y = _as_arrays(scores, y)
        self.grid = np.unique(self.scores)
        self.cdf_pos = _ecdf(self.scores[self.y == 1], self.grid)
        self.cdf_neg = _ecdf(self.scores[self.y == 0], self.grid)
        gaps = np.abs(self.cdf_neg - self.cdf_pos)
        self.argmax = int(np.argmax(gaps))
        self.value = float(gaps[self.argmax])
        self.threshold = float(self.grid[self.argmax])

    def plot(self, title=None, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.grid, self.cdf_neg, label="negative (y = 0)")
        ax.plot(self.grid, self.cdf_pos, label="positive (y = 1)")
        ax.vlines(
            self.threshold,
            self.cdf_pos[self.argmax],
            self.cdf_neg[self.argmax],
            colors="black",
            linestyles="dashed",
            label=f"KS = {self.value:.3f} @ {self.threshold:.3f}",
        )
        ax.set_xlabel("score")
        ax.set_ylabel("cumulative fraction")
        ax.set_title(title if title is not None else f"KS: {self.value:.3f}")
        ax.legend(loc="lower right")
        return ax


class ScoreDistribution:
    """Histogram of scores split by class."""

    def __init__(self, scores, y, bins=20):
        self.scores, self.y = _as_arrays(scores, y)
        self.bins = np.linspace(self.scores.min(), self.scores.max(), bins + 1)

    def plot(self, title=None, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        for label, name in ((0, "negative (y = 0)"), (1, "positive (y = 1)")):
            ax.hist(self.scores[self.y == label], bins=self.bins, alpha=0.5, label=name)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        return ax


class PSI:
    """Population Stability Index between an expected (reference) and an actual score sample."""

    def __init__(self, expected, actual, bins=10, eps=1e-6):
        expected = np.asarray(expected, dtype=float).reshape(-1)
        actual = np.asarray(actual, dtype=float).reshape(-1)
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        self.edges = edges
        self.expected_pct = np.histogram(expected, edges)[0] / len(expected) + eps
        self.actual_pct = np.histogram(actual, edges)[0] / len(actual) + eps
        self.contributions = (self.actual_pct - self.expected_pct) * np.log(self.actual_pct / self.expected_pct)
        self.value = float(self.contributions.sum())

    def plot(self, title=None, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(self.expected_pct))
        ax.bar(x - 0.2, self.expected_pct, width=0.4, label="expected")
        ax.bar(x + 0.2, self.actual_pct, width=0.4, label="actual")
        ax.set_xlabel("bucket")
        ax.set_ylabel("fraction")
        ax.set_title(title if title is not None else f"PSI: {self.value:.4f}")
        ax.legend()
        return ax
