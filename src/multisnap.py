"""Train the ranker on several earlier weekly snapshots instead of one, and compare on the same test week.

Snapshot k = candidates and features built from history before week W-k, labels = purchases in week W-k.
Every training snapshot ends before the test week begins. Reuses the cached candidate tables of the rolling folds
(4-generator pool); category codes are re-derived from the full articles table so they agree across folds.

    python -m src.multisnap data/hm
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.analyze import K, fmt, paired_boot_diff, per_user_metrics
from src.content import META_COLUMNS, stable_meta
from src.run_hm import cache_dir, load_phases
from src.run_ranker import FEATURES_EXCLUDE, fit_ranker


def load_table(data_dir, shift, name, articles):
    split, val, test = load_phases(data_dir, kcore="train", shift_weeks=shift)
    df = pickle.load(open(os.path.join(cache_dir(data_dir, "train", shift), name), "rb"))
    meta = stable_meta(articles, split.item_ids)
    for col in META_COLUMNS:
        df[f"cat_{col}"] = meta[col].values[df["i"].values]
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df["u"] = shift * 10_000_000 + df["u"]      # unique ranking-group ids across snapshots
    return df, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    args = p.parse_args()
    articles = os.path.join(args.data_dir, "articles.csv")

    # test week = main final week (fold 0); training snapshots = the validation weeks of folds 0..3 (W-1 .. W-4)
    snaps = {}
    for s in range(4):
        snaps[s], _ = load_table(args.data_dir, s, "cand_val.pkl", articles)
        print(f"snapshot W-{s + 1}: {len(snaps[s])} rows, positives {int(snaps[s]['y'].sum())}", flush=True)
    test_df, test = load_table(args.data_dir, 0, "cand_test.pkl", articles)
    n_true = test.labels.groupby("u").size()
    feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE]

    configs = {
        "1 snapshot (W-1), current setup": [0],
        "3 snapshots (W-1 .. W-3)": [0, 1, 2],
        "4 snapshots (W-1 .. W-4)": [0, 1, 2, 3],
    }
    results = {}
    for name, which in configs.items():
        train_df = pd.concat([snaps[s] for s in which], ignore_index=True)
        model = fit_ranker(train_df, feats, K, np.random.default_rng(0))
        results[name] = per_user_metrics(test_df, model.predict(test_df[feats]), n_true)
        print(f"trained: {name} ({len(train_df)} rows, best iteration {model.best_iteration_})", flush=True)

    users = sorted(set.intersection(*[set(v.index) for v in results.values()]))
    base = results["1 snapshot (W-1), current setup"].loc[users]
    print(f"\n## Ranker trained on several earlier weeks (test = final week, {len(users)} customers, 4-generator pool)")
    print("| Training data | recall@12 | NDCG@12 | recall@12 minus single snapshot [95% CI] |\n|---|---|---|---|")
    rng = np.random.default_rng(1)
    for name, m in results.items():
        m = m.loc[users]
        diff = "-" if name.startswith("1 ") else fmt(paired_boot_diff(m["recall"].values, base["recall"].values, rng))
        print(f"| {name} | {m['recall'].mean():.4f} | {m['ndcg'].mean():.4f} | {diff} |")


if __name__ == "__main__":
    main()
