import numpy as np
import pandas as pd

from src.baselines import ItemCooccurrence, Popularity, recommend
from src.content import ContentKNN
from src.data import k_core, last_days_split, last_days_split_kcore, time_split, to_csr
from src.metrics import coverage, ndcg_at_k, recall_at_k


def test_metrics_perfect_and_miss():
    recs = np.array([[1, 2], [3, 4]])
    assert recall_at_k(recs, [{1, 2}, {3}]) == 1.0
    assert ndcg_at_k(recs, [{1, 2}, {3}]) == 1.0
    assert recall_at_k(recs, [{9}, {9}]) == 0.0
    assert coverage(recs, 8) == 0.5


def test_k_core_removes_sparse_rows():
    df = pd.DataFrame({"user_id": list("aaabbc"), "item_id": list("xyzxyx"), "timestamp": range(6)})
    out = k_core(df, 2)
    assert set(out.user_id) == {"a", "b"} and set(out.item_id) == {"x", "y"}


def test_last_days_split_uses_calendar_windows():
    day = 24 * 3600 * 1000
    df = pd.DataFrame({
        "user_id": ["a"] * 30, "item_id": [str(i) for i in range(30)],
        "timestamp": np.arange(30) * day,
    })
    split = last_days_split(df, val_days=7, test_days=7)
    assert len(split.test) == 7 and len(split.val) == 7 and len(split.train) == 16
    assert split.train.timestamp.max() < split.val.timestamp.min() < split.test.timestamp.min()


def test_content_knn_ranks_similar_item_first_and_excludes_seen():
    emb = np.array([[1, 0], [0.99, 0.1], [0, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    train = to_csr(pd.DataFrame({"u": [0], "i": [0]}), 1, 3)  # user 0 bought item 0
    recs = recommend(ContentKNN(emb).fit(train), np.array([0]), k=2)
    assert list(recs[0]) == [1, 2]  # item 0 excluded; near-duplicate item 1 outranks item 2


def test_recommend_excludes_seen_items_and_split_is_chronological():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "user_id": rng.integers(0, 30, 600).astype(str),
        "item_id": rng.integers(0, 40, 600).astype(str),
        "timestamp": np.arange(600),
    }).drop_duplicates(["user_id", "item_id"])
    split = time_split(df)
    assert split.train.timestamp.max() < split.test.timestamp.min()
    train = to_csr(split.train, split.n_users, split.n_items)
    users = np.arange(split.n_users)
    for model in (Popularity(), ItemCooccurrence()):
        recs = recommend(model.fit(train), users, k=5)
        for u, r in zip(users, recs):
            assert not set(r) & set(train[u].nonzero()[1])


def test_kcore_on_training_period_does_not_use_future_purchases():
    day = 24 * 3600 * 1000
    rows = []
    for u in ("u1", "u2", "u3", "u4"):          # everyone buys a, c, d before the validation week
        rows += [(u, i, 1 * day) for i in "acd"]
    rows += [("u1", "b", 2 * day), ("u2", "b", 3 * day)]                 # item b: only 2 purchases in training
    rows += [("u3", "b", 28 * day), ("u4", "b", 29 * day), ("u1", "b", 30 * day)]  # ...and 3 more in the future
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    assert "b" in set(k_core(df, 3).item_id)                              # the global filter lets the future in
    assert "b" not in set(last_days_split_kcore(df, core=3).item_ids)     # the leak-free filter does not
