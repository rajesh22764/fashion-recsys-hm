"""2x2 experiment: is the two-tower model useful as a candidate source, as a ranking feature, or both?

Needs src.add_tt_feature to have run first (adds two-tower scores to the 4-generator pool). LightGBM only.

    python -m src.two_by_two data/hm
"""
import argparse
import os
import pickle

import numpy as np

from src.analyze import K, fmt, paired_boot_diff, per_user_metrics
from src.run_hm import add_protocol_args, cache_dir, load_phases
from src.run_ranker import FEATURES_EXCLUDE, fit_ranker

TT = ["tt_score", "tt_rel"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--seeds", type=int, default=3)
    add_protocol_args(p)
    args = p.parse_args()
    cache = cache_dir(args.data_dir, args.kcore, args.shift_weeks)
    _, val, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    n_true = test.labels.groupby("u").size()
    load = lambda n: pickle.load(open(os.path.join(cache, n), "rb"))
    pool4 = (load("cand_val_ttfeat.pkl"), load("cand_test_ttfeat.pkl"))   # candidates from 4 generators (+ tt score columns)
    pool5 = (load("cand_val_tt.pkl"), load("cand_test_tt.pkl"))           # candidates from 5 generators (tt is one of them)
    configs = {
        "no candidates / no feature (baseline)": (pool4, False),
        "no candidates / feature only": (pool4, True),
        "candidates only / no feature": (pool5, False),
        "candidates and feature (full)": (pool5, True),
    }
    results, per_seed = {}, {}
    for name, ((train_df, test_df), use_tt) in configs.items():
        feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE and (use_tt or c not in TT)]
        runs = []
        for seed in range(args.seeds):
            m = fit_ranker(train_df, feats, K, np.random.default_rng(seed), seed=seed)
            runs.append(per_user_metrics(test_df, m.predict(test_df[feats]), n_true))
        per_seed[name] = [(r["recall"].mean(), r["ndcg"].mean()) for r in runs]
        results[name] = sum(runs) / len(runs)          # per-customer metrics averaged over seeds
        print(f"trained: {name}", flush=True)

    users = sorted(set.intersection(*[set(v.index) for v in results.values()]))
    base = results["no candidates / no feature (baseline)"].loc[users]
    print(f"\n## Two-tower: candidate source vs ranking feature ({len(users)} test customers, k={K}, mean of {args.seeds} seeds)")
    print("| Two-tower used as | recall@12 (mean +/- sd over seeds) | NDCG@12 | recall@12 minus baseline [95% CI, customer sampling] |\n|---|---|---|---|")
    rng = np.random.default_rng(1)
    for name, m in results.items():
        m = m.loc[users]
        r, n = np.array(per_seed[name]).T
        diff = "-" if name.startswith("no candidates / no feature") else fmt(paired_boot_diff(m["recall"].values, base["recall"].values, rng))
        print(f"| {name} | {r.mean():.4f} +/- {r.std(ddof=1):.4f} | {n.mean():.4f} +/- {n.std(ddof=1):.4f} | {diff} |")


if __name__ == "__main__":
    main()
