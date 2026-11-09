"""Non-learned baselines. Each model scores all items for a batch of users."""
import numpy as np
from scipy import sparse


class Popularity:
    """Recommend the globally most-interacted items."""

    def fit(self, train: sparse.csr_matrix):
        self.train = train
        self.pop = np.asarray(train.sum(axis=0)).ravel()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.pop, (len(users), 1))


class ItemCooccurrence:
    """Item-to-item cosine similarity on co-purchases; score = sum over a user's history."""

    def fit(self, train: sparse.csr_matrix):
        self.train = train
        cooc = (train.T @ train).tocsr()
        cooc.setdiag(0)
        cooc.eliminate_zeros()
        norm = np.sqrt(np.asarray(train.sum(axis=0)).ravel()) + 1e-9
        self.sim = sparse.diags(1 / norm) @ cooc @ sparse.diags(1 / norm)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.train[users] @ self.sim).toarray()


def recommend(model, users: np.ndarray, k: int = 10, batch: int = 1024) -> np.ndarray:
    """Top-k items per user, excluding items already seen in train."""
    out = []
    for s in range(0, len(users), batch):
        u = users[s : s + batch]
        scores = model.score(u).astype(np.float64)
        scores[model.train[u].nonzero()] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
        out.append(np.take_along_axis(top, order, 1))
    return np.vstack(out)


class RecencyCooccurrence(ItemCooccurrence):
    """Co-occurrence similarity, but a user's history is weighted by how recent each purchase is.

    `train` stays binary (used to hide already-bought items); `hist` carries the decayed weights.
    """

    def fit(self, train: sparse.csr_matrix, hist: sparse.csr_matrix | None = None):
        super().fit(train)
        self.hist = hist if hist is not None else train
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.hist[users] @ self.sim).toarray()


class RecentPopularity(Popularity):
    """Popularity counted only over a recent window (pass the windowed item counts)."""

    def fit(self, train: sparse.csr_matrix, recent_counts: np.ndarray | None = None):
        super().fit(train)
        if recent_counts is not None:
            self.pop = recent_counts
        return self
