"""Print recent purchases and ranker recommendations for a few customers.

Uses the ranker without the two-tower feature, so it needs lightgbm but not torch.

    python -m src.demo data/hm --users 5
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.content import build_item_embeddings, load_item_meta
from src.ranker import CandidateBuilder
from src.run_hm import add_protocol_args, cache_dir, load_phases
from src.run_ranker import FEATURES_EXCLUDE, to_recs


def describe(arts: pd.DataFrame, item_id: str) -> str:
    row = arts.loc[int(item_id)]
    return f"{row['prod_name']} ({row['colour_group_name']} {row['product_type_name']}, {row['department_name']})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--users", type=int, default=5)
    p.add_argument("--k", type=int, default=12)
    add_protocol_args(p)
    args = p.parse_args()

    split, _, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    articles = os.path.join(args.data_dir, "articles.csv")
    arts = pd.read_csv(articles).set_index("article_id")
    emb = build_item_embeddings(articles, split.item_ids)
    meta = load_item_meta(articles, split.item_ids)
    ranker = pickle.load(open(os.path.join(cache_dir(args.data_dir, args.kcore, args.shift_weeks), "ranker.pkl"), "rb"))

    users = np.random.default_rng(1).choice(test.truth()[0], size=args.users, replace=False)
    df = CandidateBuilder(test, meta, emb).build(users)
    feats = [c for c in df.columns if c not in FEATURES_EXCLUDE]
    ok_users, recs = to_recs(df, ranker.predict(df[feats]), args.k)
    truth = test.labels.groupby("u")["i"].apply(set)

    for u, r in zip(ok_users, recs):
        hist = test.hist[test.hist["u"] == u].sort_values("timestamp").tail(5)
        print(f"\n=== customer #{u} ===\nrecently bought:")
        for i in hist["i"]:
            print("   -", describe(arts, split.item_ids[i]))
        print(f"recommended (top {args.k}); * = actually bought the next week:")
        for i in r:
            mark = "*" if i in truth.get(u, set()) else " "
            print(f"  {mark}", describe(arts, split.item_ids[i]))


if __name__ == "__main__":
    main()
