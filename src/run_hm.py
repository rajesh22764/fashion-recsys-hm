"""Stage 1: tuned baselines on H&M (validation for tuning, test for reporting).

    python -m src.run_hm data/hm --k 12
"""
import argparse
import os

from src.baselines import ItemCooccurrence, Popularity, RecencyCooccurrence, RecentPopularity, recommend
from src.content import ContentKNN, build_item_embeddings
from src.data import k_core, last_days_split, last_days_split_kcore
from src.hm import load_hm
from src.metrics import coverage, ndcg_at_k, recall_at_k
from src.phases import make_phases


def score_model(model, phase, k):
    users, truth = phase.truth()
    recs = recommend(model, users, k)
    return {"recall": recall_at_k(recs, truth), "ndcg": ndcg_at_k(recs, truth),
            "coverage": coverage(recs, phase.n_items), "users": len(users)}


DAY_MS = 24 * 3600 * 1000


def cache_dir(data_dir, kcore="train", shift_weeks=0):
    """Separate cache per protocol so legacy (global k-core) results are never mixed with leak-free ones."""
    name = "cache" if (kcore == "global" and shift_weeks == 0) else f"cache_{kcore}_s{shift_weeks}"
    path = os.path.join(data_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def load_phases(data_dir, weeks=26, core=5, kcore="train", shift_weeks=0):
    """kcore='train': k-core on the training period only (leak-free). 'global': legacy, k-core on all 26 weeks.

    shift_weeks=n moves the whole window n weeks earlier (rolling-origin evaluation).
    """
    df = load_hm(os.path.join(data_dir, "transactions_train.csv"), weeks + shift_weeks)
    if shift_weeks:
        df = df[df["timestamp"] <= df["timestamp"].max() - shift_weeks * 7 * DAY_MS]
    split = last_days_split(k_core(df, core)) if kcore == "global" else last_days_split_kcore(df, core)
    return split, *make_phases(split)


def add_protocol_args(parser):
    parser.add_argument("--kcore", choices=["train", "global"], default="train",
                        help="train: leak-free k-core on the training period (default). global: legacy.")
    parser.add_argument("--shift-weeks", type=int, default=0, help="move the evaluation window n weeks earlier")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--k", type=int, default=12)
    add_protocol_args(p)
    args = p.parse_args()

    split, val, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    print(f"{split.n_users} users, {split.n_items} items; val users {len(val.truth()[0])}, test users {len(test.truth()[0])}")

    def build(phase, kind, param=None):
        b = phase.binary()
        if kind == "pop":
            return Popularity().fit(b)
        if kind == "recent-pop":
            return RecentPopularity().fit(b, phase.recent_counts(param))
        if kind == "cooc":
            return ItemCooccurrence().fit(b)
        if kind == "recency-cooc":
            return RecencyCooccurrence().fit(b, phase.decayed(param))
        if kind == "content":
            emb = build_item_embeddings(os.path.join(args.data_dir, "articles.csv"), split.item_ids)
            return ContentKNN(emb).fit(phase.decayed(param))

    grids = {"pop": [None], "recent-pop": [3, 7, 14, 28], "cooc": [None],
             "recency-cooc": [3, 7, 14, 30, 60], "content": [7, 30, 120]}
    for kind, params in grids.items():
        best = max(params, key=lambda q: score_model(build(val, kind, q), val, args.k)["recall"])
        res = score_model(build(test, kind, best), test, args.k)
        print(f"{kind:13s} param={best}  " + "  ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in res.items()),
              flush=True)


if __name__ == "__main__":
    main()
