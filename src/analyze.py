"""Diagnostics on the cached candidate tables: candidate quality, score fusion, bootstrap CIs, ablations.

Run after `run_ranker --stage train` (uses the cached tables and the saved ranker). LightGBM only; no torch.

    python -m src.analyze data/hm --twotower
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.run_hm import add_protocol_args, cache_dir, load_phases
from src.run_ranker import FEATURES_EXCLUDE, fit_binary, fit_ranker

K = 12
GENERATORS = ["rcooc_score", "cooc_score", "rpop_score", "content_score", "tt_score"]
GROUPS = {
    "recency co-occurrence": ["rcooc_score", "rcooc_rel"],
    "co-occurrence": ["cooc_score"],
    "content kNN": ["content_score", "content_rel"],
    "recent popularity": ["rpop_score"],
    "two-tower": ["tt_score", "tt_rel"],
    "item stats": ["item_pop", "item_pop7", "item_pop28", "item_price", "item_age"],
    "customer stats": ["user_n", "user_n28", "user_recency", "price_ratio"],
}


def per_user_metrics(df: pd.DataFrame, scores: np.ndarray, n_true: pd.Series, k: int = K) -> pd.DataFrame:
    """recall@k and NDCG@k for every customer, from candidate scores (y marks true next-week articles)."""
    d = pd.DataFrame({"u": df["u"].values, "y": df["y"].values, "s": scores})
    d = d.sort_values(["u", "s"], ascending=[True, False], kind="stable")
    d["r"] = d.groupby("u").cumcount()
    enough = d.groupby("u")["r"].transform("size") >= k
    d = d[enough & (d["r"] < k)]
    hits = d.groupby("u")["y"].sum()
    gain = (d["y"] / np.log2(d["r"] + 2)).groupby(d["u"]).sum()
    n = n_true.reindex(hits.index)
    idcg = np.concatenate([[1.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])  # idcg[m] for m relevant items
    ideal = idcg[np.minimum(n.values, k)]
    return pd.DataFrame({"recall": hits / n, "ndcg": gain / ideal}, index=hits.index)


def boot_ci(x: np.ndarray, rng, b: int = 1000):
    n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(b)])
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def paired_boot_diff(a: np.ndarray, b_: np.ndarray, rng, b: int = 1000):
    d = a - b_
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(b)])
    return d.mean(), *np.percentile(means, [2.5, 97.5])


def fmt(m):
    return f"{m[0]:.4f} [{m[1]:.4f}, {m[2]:.4f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--twotower", action="store_true")
    p.add_argument("--no-ablation", action="store_true")
    add_protocol_args(p)
    args = p.parse_args()
    cache = cache_dir(args.data_dir, args.kcore, args.shift_weeks)
    tag = "_tt" if args.twotower else ""
    rng = np.random.default_rng(0)

    _, val, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    load = lambda name: pickle.load(open(os.path.join(cache, name), "rb"))
    val_df, test_df = load(f"cand_val{tag}.pkl"), load(f"cand_test{tag}.pkl")
    n_true = test.labels.groupby("u").size()
    feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE]
    ranker = load(f"ranker{tag}.pkl")

    print(f"## Candidate quality (test; {test_df['u'].nunique()} customers, ~{len(test_df) / test_df['u'].nunique():.0f} candidates each)")
    hit = test_df.groupby("u")["y"].sum()
    print(f"- candidate recall (mean per-customer share of true articles present): {(hit / n_true.reindex(hit.index)).mean():.4f}")
    print(f"- candidate hit rate (customers with at least one true article present): {(hit > 0).mean():.4f}")

    gens = [g for g in GENERATORS if g in test_df.columns]
    scores = {g: test_df[g].values for g in gens}
    rr = np.zeros(len(test_df))
    for g in gens:  # reciprocal-rank fusion across generators
        rr += 1.0 / (60 + test_df.groupby("u")[g].rank(ascending=False, method="first").values)
    scores["fusion (reciprocal-rank)"] = rr
    scores["ranker"] = ranker.predict(test_df[feats])
    pu = {name: per_user_metrics(test_df, s, n_true) for name, s in scores.items()}
    users = sorted(set.intersection(*[set(v.index) for v in pu.values()]))
    pu = {k: v.loc[users] for k, v in pu.items()}

    conditional = hit.reindex(users) > 0
    print(f"\n## Test results with 95% bootstrap CIs over {len(users)} customers (k={K})")
    print("| Model | recall@12 [95% CI] | NDCG@12 [95% CI] |\n|---|---|---|")
    for name, m in pu.items():
        print(f"| {name} | {fmt(boot_ci(m['recall'].values, rng))} | {fmt(boot_ci(m['ndcg'].values, rng))} |")
    print(f"\nranker recall@12 among customers whose candidates contain a true article: "
          f"{pu['ranker']['recall'][conditional.values].mean():.4f} (n={int(conditional.sum())})")

    print("\n## Paired differences (recall@12, 95% CI)")
    base = max(gens, key=lambda g: pu[g]["recall"].mean())
    for a, b_ in [("ranker", base), ("ranker", "fusion (reciprocal-rank)")]:
        print(f"- {a} minus {b_}: {fmt(paired_boot_diff(pu[a]['recall'].values, pu[b_]['recall'].values, rng))}")

    plain_df_path, plain_ranker_path = os.path.join(cache, "cand_test.pkl"), os.path.join(cache, "ranker.pkl")
    if args.twotower and os.path.exists(plain_df_path) and os.path.exists(plain_ranker_path):
        # ranker built without the two-tower generator (its own, smaller candidate pool), same customers
        plain_df, plain_ranker = load("cand_test.pkl"), load("ranker.pkl")
        plain_feats = [c for c in plain_df.columns if c not in FEATURES_EXCLUDE]
        plain = per_user_metrics(plain_df, plain_ranker.predict(plain_df[plain_feats]), n_true)
        common = pu["ranker"].index.intersection(plain.index)
        for metric in ("recall", "ndcg"):
            a, b_ = pu["ranker"].loc[common, metric].values, plain.loc[common, metric].values
            print(f"- ranker with two-tower generator minus without it, {metric}@12 ({len(common)} customers): "
                  f"{fmt(paired_boot_diff(a, b_, rng))}  (without: {b_.mean():.4f}, with: {a.mean():.4f})")


    print("\n## Scorers on the same candidate pool (trained on the validation phase, evaluated on the test week)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gen_cols = [g for g in gens]
    norm = np.zeros(len(test_df))
    for g in gen_cols:  # per-customer max-normalised scores, equal weights
        mx = test_df.groupby("u")[g].transform("max").clip(lower=1e-9).values
        norm += test_df[g].values / mx
    numeric = [c for c in feats if not c.startswith("cat_")]
    scaler = StandardScaler().fit(val_df[numeric])
    logit = LogisticRegression(max_iter=300).fit(scaler.transform(val_df[numeric]), val_df["y"])
    binary = fit_binary(val_df, feats, np.random.default_rng(0))
    same_pool = {
        "reciprocal-rank fusion": pu["fusion (reciprocal-rank)"],
        "normalised-score sum (equal weights)": per_user_metrics(test_df, norm, n_true).loc[users],
        "logistic regression (numeric features)": per_user_metrics(
            test_df, logit.decision_function(scaler.transform(test_df[numeric])), n_true).loc[users],
        "LightGBM, binary objective": per_user_metrics(test_df, binary.predict_proba(test_df[feats])[:, 1], n_true).loc[users],
        "LightGBM, LambdaRank": pu["ranker"],
    }
    print("| Scorer | recall@12 [95% CI] | NDCG@12 [95% CI] |\n|---|---|---|")
    for name, m in same_pool.items():
        print(f"| {name} | {fmt(boot_ci(m['recall'].values, rng))} | {fmt(boot_ci(m['ndcg'].values, rng))} |", flush=True)
    for name in list(same_pool)[:-1]:
        print(f"- LambdaRank minus {name}: {fmt(paired_boot_diff(same_pool['LightGBM, LambdaRank']['recall'].values, same_pool[name]['recall'].values, rng))}")

    if args.no_ablation:
        return
    print("\n## Feature-group ablation (candidates fixed; retrain ranker without one feature group)")
    groups = {k: [c for c in v if c in feats] for k, v in GROUPS.items()}
    groups["customer-article affinity"] = [c for c in feats if c.startswith("aff_")]
    groups["article attributes"] = [c for c in feats if c.startswith("cat_")]
    print("| Ranker | recall@12 | NDCG@12 |\n|---|---|---|")
    settings = dict(n_estimators=400, learning_rate=0.08)
    runs = {"all features": []}
    runs.update({f"without {k}": v for k, v in groups.items() if v})
    for name, drop in runs.items():
        f = [c for c in feats if c not in drop]
        model = fit_ranker(val_df, f, K, np.random.default_rng(0), **settings)
        m = per_user_metrics(test_df, model.predict(test_df[f]), n_true).loc[users]
        print(f"| {name} | {m['recall'].mean():.4f} | {m['ndcg'].mean():.4f} |", flush=True)


if __name__ == "__main__":
    main()
