"""Evaluate baselines on a time split.

    python -m src.run_baselines data/Amazon_Fashion.jsonl.gz --k 10
"""
import argparse

import numpy as np

from src.baselines import ItemCooccurrence, Popularity, recommend
from src.content import ContentKNN, build_item_embeddings
from src.data import k_core, last_days_split, load_reviews, time_split, to_csr, warm_only
from src.hm import load_hm
from src.metrics import coverage, ndcg_at_k, recall_at_k


def evaluate(model, split, eval_df, k):
    eval_df = warm_only(eval_df, split.train)
    truth_by_user = eval_df.groupby("u")["i"].apply(set)
    users = truth_by_user.index.values
    recs = recommend(model, users, k)
    truth = list(truth_by_user.values)
    return {
        f"recall@{k}": recall_at_k(recs, truth),
        f"ndcg@{k}": ndcg_at_k(recs, truth),
        "coverage": coverage(recs, split.n_items),
        "users": len(users),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--core", type=int, default=5)
    p.add_argument("--dataset", choices=["amazon", "hm"], default="amazon")
    p.add_argument("--weeks", type=int, default=26, help="hm only: keep the last N weeks")
    p.add_argument("--split", choices=["fraction", "week"], default="fraction",
                   help="week: last 7 days test, 7 days before that validation (Kaggle protocol)")
    p.add_argument("--articles", help="hm only: articles.csv, enables the content-based model")
    args = p.parse_args()

    raw = load_hm(args.path, args.weeks) if args.dataset == "hm" else load_reviews(args.path)
    df = k_core(raw, args.core)
    split = last_days_split(df) if args.split == "week" else time_split(df)
    train = to_csr(split.train, split.n_users, split.n_items)
    print(f"{len(df)} interactions, {split.n_users} users, {split.n_items} items")

    models = [("popularity", Popularity()), ("co-occurrence", ItemCooccurrence())]
    if args.articles:
        models.append(("content-knn", ContentKNN(build_item_embeddings(args.articles, split.item_ids))))

    for name, model in models:
        res = evaluate(model.fit(train), split, split.test, args.k)
        print(name, {key: round(v, 4) if isinstance(v, float) else v for key, v in res.items()})


if __name__ == "__main__":
    main()
