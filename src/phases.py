"""Rolling evaluation phases for the H&M pipeline.

validation: history = train,       labels = validation week  (used for tuning and for ranker training)
test:       history = train + val, labels = test week        (reported)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import Split, to_csr, to_csr_weighted, warm_only

DAY = 24 * 3600 * 1000


@dataclass
class Phase:
    name: str
    hist: pd.DataFrame     # interactions before the cutoff
    labels: pd.DataFrame   # next-week purchases of warm users on warm items
    n_users: int
    n_items: int

    def binary(self) -> sparse.csr_matrix:
        return to_csr(self.hist, self.n_users, self.n_items)

    def decayed(self, tau_days: float | None) -> sparse.csr_matrix:
        return to_csr_weighted(self.hist, self.n_users, self.n_items, tau_days)

    def recent_counts(self, days: int) -> np.ndarray:
        """Item purchase counts in the last `days` days of the history."""
        recent = self.hist[self.hist["timestamp"] > self.hist["timestamp"].max() - days * DAY]
        return np.bincount(recent["i"].values, minlength=self.n_items).astype(np.float32)

    def truth(self):
        """(users, list of sets of true items) for the label week."""
        by_user = self.labels.groupby("u")["i"].apply(set)
        return by_user.index.values, list(by_user.values)


def make_phases(split: Split) -> tuple[Phase, Phase]:
    val = Phase("val", split.train, warm_only(split.val, split.train), split.n_users, split.n_items)
    hist = pd.concat([split.train, split.val], ignore_index=True)
    test = Phase("test", hist, warm_only(split.test, hist), split.n_users, split.n_items)
    return val, test
