"""Rolling-origin results with several training seeds per fold (4-generator pool, no two-tower).

Each fold trains its own ranker on its own validation week and is tested on the following week; the seeds change the
held-out early-stopping customers and LightGBM's bagging. Reports seed-mean recall per fold and the spread across folds.

    python -m src.rolling_seeds data/hm
"""
import argparse
import os
import pickle

import numpy as np

from src.analyze import GENERATORS, K, per_user_metrics
from src.run_hm import cache_dir, load_phases
from src.run_ranker import FEATURES_EXCLUDE, fit_ranker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    print("| fold (test week ends) | best generator | ranker recall@12 (mean +/- sd over seeds) | ranker NDCG@12 | relative gain |\n|---|---|---|---|---|")
    fold_rec, fold_ndcg, fold_best = [], [], []
    for s in range(args.folds):
        _, _, test = load_phases(args.data_dir, kcore="train", shift_weeks=s)
        cache = cache_dir(args.data_dir, "train", s)
        load = lambda n: pickle.load(open(os.path.join(cache, n), "rb"))
        train_df, test_df = load("cand_val.pkl"), load("cand_test.pkl")
        n_true = test.labels.groupby("u").size()
        feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE]
        gens = {g: per_user_metrics(test_df, test_df[g].values, n_true)["recall"].mean() for g in GENERATORS if g in test_df.columns}
        best_name = max(gens, key=gens.get)
        rec, ndcg = [], []
        for seed in range(args.seeds):
            m = fit_ranker(train_df, feats, K, np.random.default_rng(seed), seed=seed)
            pu = per_user_metrics(test_df, m.predict(test_df[feats]), n_true)
            rec.append(pu["recall"].mean()); ndcg.append(pu["ndcg"].mean())
        fold_rec.append(np.mean(rec)); fold_ndcg.append(np.mean(ndcg)); fold_best.append(gens[best_name])
        print(f"| shift {s} | {best_name.replace('_score', '')} {gens[best_name]:.4f} | {np.mean(rec):.4f} +/- {np.std(rec, ddof=1):.4f} | "
              f"{np.mean(ndcg):.4f} | {np.mean(rec) / gens[best_name] - 1:+.1%} |", flush=True)
    r, n, b = np.array(fold_rec), np.array(fold_ndcg), np.array(fold_best)
    print(f"| **mean +/- std across folds** | {b.mean():.4f} +/- {b.std(ddof=1):.4f} | {r.mean():.4f} +/- {r.std(ddof=1):.4f} | "
          f"{n.mean():.4f} +/- {n.std(ddof=1):.4f} | {(r / b - 1).mean():+.1%} |")


if __name__ == "__main__":
    main()
