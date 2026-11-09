"""Stage 2: LightGBM ranker over multi-generator candidates.

    python -m src.run_ranker data/hm
"""
import argparse
import os
import pickle

import numpy as np

from src.content import build_item_embeddings, load_item_meta
from src.metrics import coverage, ndcg_at_k, recall_at_k
from src.ranker import CandidateBuilder, candidate_recall
from src.run_hm import add_protocol_args, cache_dir, load_phases

FEATURES_EXCLUDE = {"u", "i", "y", "src_rcooc", "src_cooc", "src_rpop", "src_content", "src_tt"}  # src_* = candidate provenance
CATEGORICAL = ["cat_product_type_no", "cat_colour_group_code", "cat_department_no", "cat_index_group_no",
               "cat_garment_group_no", "cat_graphical_appearance_no"]


def candidates(cache, name, builder_fn, phase, users):
    path = os.path.join(cache, f"cand_{name}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    df = builder_fn().build(users)
    pickle.dump(df, open(path, "wb"))
    return df


def fit_ranker(train_df, feats, k, rng, n_estimators=600, learning_rate=0.05, seed=None):
    """LambdaRank fit with 20% of customers held out for early stopping (each customer is one ranking group)."""
    import lightgbm as lgb

    users = train_df["u"].unique()
    holdout = set(rng.choice(users, size=len(users) // 5, replace=False))
    is_hold = train_df["u"].isin(holdout).values
    fit, hold = train_df[~is_hold], train_df[is_hold]
    group = lambda d: d.groupby("u", sort=False).size().values
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, lambdarank_truncation_level=k, random_state=seed, verbose=-1)
    model.fit(fit[feats], fit["y"], group=group(fit), eval_set=[(hold[feats], hold["y"])], eval_group=[group(hold)],
              eval_at=[k], categorical_feature=[c for c in CATEGORICAL if c in feats],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    return model


def fit_binary(train_df, feats, rng, n_estimators=600, learning_rate=0.05):
    """Same model and settings as fit_ranker but a plain binary-classification objective (for comparison)."""
    import lightgbm as lgb

    users = train_df["u"].unique()
    holdout = set(rng.choice(users, size=len(users) // 5, replace=False))
    is_hold = train_df["u"].isin(holdout).values
    fit, hold = train_df[~is_hold], train_df[is_hold]
    model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=63,
                               min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    model.fit(fit[feats], fit["y"], eval_set=[(hold[feats], hold["y"])], eval_metric="auc",
              categorical_feature=[c for c in CATEGORICAL if c in feats],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    return model


def to_recs(df, scores, k):
    """Top-k items per user by model score. Returns (users, recs) with users that have >= k candidates."""
    d = df[["u", "i"]].copy()
    d["s"] = scores
    d = d.sort_values(["u", "s"], ascending=[True, False])
    d["r"] = d.groupby("u").cumcount()
    d = d[d["r"] < k]
    counts = d.groupby("u").size()
    d = d[d["u"].isin(counts[counts == k].index)]
    users = d["u"].unique()
    return users, d["i"].values.reshape(len(users), k)


def evaluate(df, scores, phase, k):
    users, recs = to_recs(df, scores, k)
    by_user = phase.labels.groupby("u")["i"].apply(set)
    truth_all = by_user.reindex(users)
    return {"recall": recall_at_k(recs, list(truth_all.values)), "ndcg": ndcg_at_k(recs, list(truth_all.values)),
            "coverage": coverage(recs, phase.n_items)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--train-users", type=int, default=40000)
    p.add_argument("--stage", choices=["candidates", "train"], required=True,
                   help="candidates: build+cache candidate tables (may use torch). train: fit/evaluate LightGBM. "
                        "Separate processes because torch and LightGBM crash together (duplicate OpenMP).")
    p.add_argument("--twotower", action="store_true", help="add the two-tower model as a candidate generator and feature")
    add_protocol_args(p)
    args = p.parse_args()
    cache = cache_dir(args.data_dir, args.kcore, args.shift_weeks)

    split, val, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    emb = build_item_embeddings(os.path.join(args.data_dir, "articles.csv"), split.item_ids)
    meta = load_item_meta(os.path.join(args.data_dir, "articles.csv"), split.item_ids)
    rng = np.random.default_rng(0)
    tag = "_tt" if args.twotower else ""

    def extra(phase):
        if not args.twotower:
            return None
        import torch
        from src.twotower import TwoTower, TwoTowerGen, last_items
        model = TwoTower(split.n_items, emb)
        model.load_state_dict(torch.load(os.path.join(cache, "twotower.pt")))
        return {"tt": TwoTowerGen(model.to("mps" if torch.backends.mps.is_available() else "cpu").eval(),
                                  last_items(phase.hist, split.n_users, split.n_items))}

    val_users = rng.permutation(val.truth()[0])[: args.train_users]
    train_df = candidates(cache, "val" + tag, lambda: CandidateBuilder(val, meta, emb, extra(val)), val, val_users)
    print(f"val candidates {len(train_df)} rows, positives {train_df.y.sum()}, "
          f"candidate recall {candidate_recall(train_df, val):.4f}", flush=True)
    test_df = candidates(cache, "test" + tag, lambda: CandidateBuilder(test, meta, emb, extra(test)), test, test.truth()[0])
    print(f"test candidates {len(test_df)} rows, candidate recall {candidate_recall(test_df, test):.4f}", flush=True)

    if args.stage == "candidates":
        return
    feats = [c for c in train_df.columns if c not in FEATURES_EXCLUDE]
    model = fit_ranker(train_df, feats, args.k, rng)
    print("best iteration", model.best_iteration_, flush=True)

    print("\nTEST (k=%d):" % args.k)
    for name in ["rcooc_score", "rpop_score", "content_score", "cooc_score"] + (["tt_score"] if args.twotower else []):
        print(f"  generator {name:14s}", {a: round(b, 4) for a, b in evaluate(test_df, test_df[name].values, test, args.k).items()})
    res = evaluate(test_df, model.predict(test_df[feats]), test, args.k)
    print("  ranker               ", {a: round(b, 4) for a, b in res.items()})
    imp = sorted(zip(model.booster_.feature_importance("gain"), feats), reverse=True)[:10]
    print("top features:", [(f, int(g)) for g, f in imp])
    pickle.dump(model, open(os.path.join(cache, f"ranker{tag}.pkl"), "wb"))


if __name__ == "__main__":
    main()
