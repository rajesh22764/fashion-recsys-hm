"""How much does the ranker's test recall move just from training randomness?

Varies the held-out early-stopping customers and LightGBM's bagging seed; the test set, candidates and features are fixed.
Gives the noise floor that differences between ranker variants must exceed to mean anything.

    python -m src.seed_variance data/hm
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.analyze import K, per_user_metrics
from src.multisnap import load_table
from src.run_hm import add_protocol_args, cache_dir, load_phases
from src.run_ranker import FEATURES_EXCLUDE, fit_ranker


def summarize(name, recalls, ndcgs):
    r, n = np.array(recalls), np.array(ndcgs)
    print(f"| {name} | {len(r)} | {r.mean():.4f} +/- {r.std(ddof=1):.4f} | {r.min():.4f} to {r.max():.4f} | {n.mean():.4f} +/- {n.std(ddof=1):.4f} |", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--snapshot-seeds", type=int, default=3)
    add_protocol_args(p)
    args = p.parse_args()
    cache = cache_dir(args.data_dir, args.kcore, args.shift_weeks)
    articles = os.path.join(args.data_dir, "articles.csv")
    _, _, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    n_true = test.labels.groupby("u").size()
    load = lambda n: pickle.load(open(os.path.join(cache, n), "rb"))

    print("| Ranker | seeds | test recall@12 (mean +/- sd) | range | test NDCG@12 (mean +/- sd) |\n|---|---|---|---|---|")
    for name, (tr, te) in {"4 generators (original codes)": ("cand_val.pkl", "cand_test.pkl"),
                           "5 generators (+ two-tower)": ("cand_val_tt.pkl", "cand_test_tt.pkl")}.items():
        train_df, test_df = load(tr), load(te)
        feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE]
        rec, ndcg = [], []
        for seed in range(args.seeds):
            m = fit_ranker(train_df, feats, K, np.random.default_rng(seed), seed=seed)
            pu = per_user_metrics(test_df, m.predict(test_df[feats]), n_true)
            rec.append(pu["recall"].mean()); ndcg.append(pu["ndcg"].mean())
        summarize(name, rec, ndcg)

    snaps = {s: load_table(args.data_dir, s, "cand_val.pkl", articles)[0] for s in range(3)}
    test_df, _ = load_table(args.data_dir, 0, "cand_test.pkl", articles)
    feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE]
    for name, which in {"4 generators, 1 snapshot (stable codes)": [0], "4 generators, 3 snapshots": [0, 1, 2]}.items():
        train_df = pd.concat([snaps[s] for s in which], ignore_index=True)
        rec, ndcg = [], []
        for seed in range(args.snapshot_seeds):
            m = fit_ranker(train_df, feats, K, np.random.default_rng(seed), seed=seed)
            pu = per_user_metrics(test_df, m.predict(test_df[feats]), n_true)
            rec.append(pu["recall"].mean()); ndcg.append(pu["ndcg"].mean())
        summarize(name, rec, ndcg)


if __name__ == "__main__":
    main()
