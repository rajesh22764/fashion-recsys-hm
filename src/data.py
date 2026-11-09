"""Load Amazon Reviews 2023 (Amazon_Fashion) interactions and split them by time."""
import gzip
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


def load_reviews(path: str) -> pd.DataFrame:
    """Stream a .jsonl / .jsonl.gz review file; keep only the columns we need.

    Streaming avoids holding the (large) review text in memory.
    """
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["user_id"], r["parent_asin"], r["rating"], r["timestamp"]))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return df.drop_duplicates(["user_id", "item_id"])


def k_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Repeatedly drop users and items with fewer than k interactions."""
    while True:
        before = len(df)
        df = df[df.groupby("user_id")["item_id"].transform("size") >= k]
        df = df[df.groupby("item_id")["user_id"].transform("size") >= k]
        if len(df) == before:
            return df.reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    item_ids: np.ndarray  # item_ids[i] is the original id of the item coded i


def _index(df: pd.DataFrame):
    """Sort by time and add integer user/item codes shared by every split."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["u"] = df["user_id"].astype("category").cat.codes
    item_cat = df["item_id"].astype("category")
    df["i"] = item_cat.cat.codes
    return df, item_cat.cat.categories.to_numpy()


def _make_split(df: pd.DataFrame, item_ids: np.ndarray, t1: int, t2: int) -> Split:
    return Split(df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:], df["u"].nunique(), len(item_ids), item_ids)


def time_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1) -> Split:
    """Global time split: oldest interactions train, newest test.

    Ids are re-indexed from the full data so train/val/test share one id space.
    Validation/test rows for items never seen in train are kept (use
    `warm_only` when scoring models that cannot rank unseen items).
    """
    df, item_ids = _index(df)
    n = len(df)
    return _make_split(df, item_ids, int(n * (1 - val_frac - test_frac)), int(n * (1 - test_frac)))


def last_days_split(df: pd.DataFrame, val_days: int = 7, test_days: int = 7, end: int | None = None) -> Split:
    """Calendar split: the final `test_days` are test, the `val_days` before that are validation.

    Matches the Kaggle H&M protocol (predict the next week of purchases).
    `end` overrides the last timestamp (used when rows were filtered after the calendar was fixed).
    """
    df, item_ids = _index(df)
    day = 24 * 3600 * 1000
    end = df["timestamp"].max() if end is None else end
    t2 = int(df["timestamp"].searchsorted(end - test_days * day, side="right"))
    t1 = int(df["timestamp"].searchsorted(end - (test_days + val_days) * day, side="right"))
    return _make_split(df, item_ids, t1, t2)


def warm_only(eval_df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Keep eval rows whose user and item both appear in train."""
    return eval_df[eval_df["u"].isin(train["u"].unique()) & eval_df["i"].isin(train["i"].unique())]


def to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    """Binary user x item interaction matrix."""
    data = np.ones(len(df), dtype=np.float32)
    return sparse.csr_matrix((data, (df["u"].values, df["i"].values)), shape=(n_users, n_items))


def to_csr_weighted(df: pd.DataFrame, n_users: int, n_items: int, tau_days: float | None) -> sparse.csr_matrix:
    """User x item matrix with exponentially decayed weights exp(-age/tau), age measured from the newest row.

    tau_days=None gives the plain binary matrix. Timestamps are in milliseconds.
    """
    if tau_days is None:
        return to_csr(df, n_users, n_items)
    age_days = (df["timestamp"].max() - df["timestamp"].values) / (24 * 3600 * 1000)
    data = np.exp(-age_days / tau_days).astype(np.float32)
    return sparse.csr_matrix((data, (df["u"].values, df["i"].values)), shape=(n_users, n_items))


def last_days_split_kcore(df: pd.DataFrame, core: int = 5, val_days: int = 7, test_days: int = 7) -> Split:
    """Calendar split with a leak-free k-core.

    The k-core is computed on the training period only (everything before the validation week). Validation and
    test rows are then restricted to the users and items that survived, so the eligible population is frozen
    at the training cutoff and future purchases can never make a user or item eligible.
    """
    day = 24 * 3600 * 1000
    end = int(df["timestamp"].max())
    train_cut = end - (test_days + val_days) * day
    train = k_core(df[df["timestamp"] <= train_cut], core)
    later = df[df["timestamp"] > train_cut]
    later = later[later["user_id"].isin(set(train["user_id"])) & later["item_id"].isin(set(train["item_id"]))]
    return last_days_split(pd.concat([train, later], ignore_index=True), val_days, test_days, end=end)
