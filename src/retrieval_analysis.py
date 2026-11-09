"""Retrieval analysis: what does each generator contribute, and does a bigger candidate pool help?

Builds candidate tables at several pool sizes (4 generators, no two-tower), reports per-generator contribution and overlap on the
default pool, then trains the ranker on each pool size (several seeds) and evaluates on the test week.
LightGBM only; no torch.

    python -m src.retrieval_analysis data/hm
"""
import argparse
import os
import pickle

import numpy as np

from src.analyze import K, per_user_metrics
from src.content import build_item_embeddings, load_item_meta
from src.ranker import GEN_N, CandidateBuilder
from src.run_hm import add_protocol_args, cache_dir, load_phases
from src.run_ranker import FEATURES_EXCLUDE, fit_ranker


def cached(cache, name, make):
    path = os.path.join(cache, name)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    df = make()
    pickle.dump(df, open(path, "wb"))
    return df


def pool_stats(df, n_true):
    hit = df.groupby("u")["y"].sum()
    n = n_true.reindex(hit.index)
    return len(df) / df["u"].nunique(), float((hit / n).mean()), float((hit > 0).mean())


def contributions(df, n_true):
    flags = [c for c in df.columns if c.startswith("src_")]
    users = df["u"].unique()
    n = n_true.reindex(users)

    def recall(mask):
        hits = df.loc[mask].groupby("u")["y"].sum().reindex(users, fill_value=0)
        return float((hits / n).mean())

    everything = np.ones(len(df), dtype=bool)
    print(f"\n## Generator contributions (test week, default pool, {len(users)} customers)")
    print(f"pool: {len(df) / len(users):.0f} candidates per customer, candidate recall {recall(everything):.4f}, "
          f"mean sources per candidate {df[flags].sum(axis=1).mean():.2f}")
    print("| Generator | candidates per customer | recall of its own candidates | unique candidates per customer | candidate recall without it |")
    print("|---|---|---|---|---|")
    for f in flags:
        others = df[[g for g in flags if g != f]].any(axis=1).values
        unique = df[f].values & ~others
        print(f"| {f[4:]} | {df[f].sum() / len(users):.0f} | {recall(df[f].values):.4f} | "
              f"{unique.sum() / len(users):.1f} | {recall(others):.4f} |")
    print("\nJaccard overlap of candidate sets (pooled over customers):")
    print("| | " + " | ".join(f[4:] for f in flags) + " |\n|---|" + "---|" * len(flags))
    for a in flags:
        row = []
        for b in flags:
            inter, union = (df[a] & df[b]).sum(), (df[a] | df[b]).sum()
            row.append(f"{inter / union:.2f}")
        print(f"| {a[4:]} | " + " | ".join(row) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--train-users", type=int, default=40000)
    add_protocol_args(p)
    args = p.parse_args()
    cache = cache_dir(args.data_dir, args.kcore, args.shift_weeks)
    articles = os.path.join(args.data_dir, "articles.csv")

    split, val, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    emb = build_item_embeddings(articles, split.item_ids)
    meta = load_item_meta(articles, split.item_ids)
    n_true = test.labels.groupby("u").size()
    val_users = np.random.default_rng(0).permutation(val.truth()[0])[: args.train_users]
    builder_val, builder_test = CandidateBuilder(val, meta, emb), CandidateBuilder(test, meta, emb)

    rows, default_test = [], None
    for scale in args.scales:
        builder_val.gen_n = builder_test.gen_n = {k: max(1, int(round(v * scale))) for k, v in GEN_N.items()}
        tag = f"b{scale:g}"
        train_df = cached(cache, f"ret_val_{tag}.pkl", lambda: builder_val.build(val_users))
        test_df = cached(cache, f"ret_test_{tag}.pkl", lambda: builder_test.build(test.truth()[0]))
        if scale == 1.0:
            default_test = test_df
        per_cust, cand_recall, hit_rate = pool_stats(test_df, n_true)
        feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE]
        seeds = args.seeds if scale < 3 else max(2, args.seeds - 1)
        rec, ndcg = [], []
        for seed in range(seeds):
            m = fit_ranker(train_df, feats, K, np.random.default_rng(seed), seed=seed)
            pu = per_user_metrics(test_df, m.predict(test_df[feats]), n_true)
            rec.append(pu["recall"].mean()); ndcg.append(pu["ndcg"].mean())
        rows.append((scale, per_cust, cand_recall, hit_rate, np.mean(rec), np.std(rec, ddof=1), np.mean(ndcg), seeds))
        print(f"done scale {scale:g}: {per_cust:.0f} candidates, candidate recall {cand_recall:.4f}, ranker recall {np.mean(rec):.4f}", flush=True)
        del train_df, test_df

    if default_test is not None:
        contributions(default_test, n_true)
    print("\n## Candidate budget vs final quality (4 generators, test week)")
    print("| pool scale | candidates per customer | candidate recall | candidate hit rate | ranker recall@12 (mean +/- sd) | ranker NDCG@12 | seeds |")
    print("|---|---|---|---|---|---|---|")
    for scale, pc, cr, hr, r, sd, nd, sds in rows:
        print(f"| x{scale:g} | {pc:.0f} | {cr:.4f} | {hr:.3f} | {r:.4f} +/- {sd:.4f} | {nd:.4f} | {sds} |")


if __name__ == "__main__":
    main()
