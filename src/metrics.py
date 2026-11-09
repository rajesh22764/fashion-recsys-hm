"""Top-k ranking metrics. `recs` is (n_users, k) item ids; `truth` is a list of sets."""
import numpy as np


def recall_at_k(recs: np.ndarray, truth: list[set]) -> float:
    k = recs.shape[1]
    scores = [len(set(r[:k]) & t) / len(t) for r, t in zip(recs, truth)]
    return float(np.mean(scores))


def ndcg_at_k(recs: np.ndarray, truth: list[set]) -> float:
    k = recs.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for r, t in zip(recs, truth):
        gains = np.array([item in t for item in r], dtype=float)
        ideal = discounts[: min(len(t), k)].sum()
        scores.append((gains * discounts).sum() / ideal)
    return float(np.mean(scores))


def coverage(recs: np.ndarray, n_items: int) -> float:
    """Fraction of the catalog that appears in at least one user's list."""
    return len(np.unique(recs)) / n_items
