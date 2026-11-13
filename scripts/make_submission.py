"""Create a Kaggle submission for the H&M competition, and an offline estimate of its MAP@12.

The Kaggle task: predict 12 articles for every one of the 1.37M customers for the 7 days after the last day of data, scored by MAP@12 with
REPEAT PURCHASES counting. Our model recommends only articles a customer has not bought before (new-to-customer discovery) and covers only the
eligible customers (at least 5 purchases in the training period), so the submission uses the model for eligible customers and "what sold in
the last 7 days" for everybody else. Expect a score well below the leaderboard's best, which exploit repeat purchases.

modes
  offline  pretend the last week is the Kaggle week: history ends the day before it, the ranker is trained on the validation week only, a random
           sample of customers is scored against their real purchases that week (repeats included). No files are written.
  final    history through the last day of data, ranker trained on the validation AND test weeks (two weekly snapshots);
           writes data/hm/submission/submission.csv for all customers.

    python scripts/make_submission.py data/hm --mode offline
    python scripts/make_submission.py data/hm --mode final

LightGBM only (no torch). Needs the cached candidate tables from src.retrieval_analysis (ret_val_b1.pkl, ret_test_b1.pkl).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.content import build_item_embeddings, load_item_meta  # noqa: E402
from src.phases import Phase  # noqa: E402
from src.ranker import CandidateBuilder  # noqa: E402
from src.run_hm import cache_dir, load_phases  # noqa: E402
from src.run_ranker import FEATURES_EXCLUDE, fit_ranker  # noqa: E402

K = 12
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def aid(item_id) -> str:
    return str(int(item_id)).zfill(10)


def load_window(path: str, start: str, end: str) -> pd.DataFrame:
    """Raw (uncollapsed) purchases with start <= date <= end, as customer_id / article_id (10-digit strings)."""
    parts = []
    for chunk in pd.read_csv(path, usecols=["t_dat", "customer_id", "article_id"], dtype={"article_id": str}, chunksize=2_000_000):
        parts.append(chunk[(chunk["t_dat"] >= start) & (chunk["t_dat"] <= end)][["customer_id", "article_id"]])
    return pd.concat(parts, ignore_index=True)


def popular(window: pd.DataFrame, k: int = K) -> list:
    return window["article_id"].value_counts().index[:k].tolist()


def ap12(pred: list, truth: set) -> float:
    hits, total = 0, 0.0
    for rank, a in enumerate(pred[:K], start=1):
        if a in truth:
            hits += 1
            total += hits / rank
    return total / min(len(truth), K)


def recommend_eligible(phase: Phase, ranker, feats, meta, emb, codes: np.ndarray, split, chunk: int = 20000) -> dict:
    """Top-12 article ids for each customer code, from candidates scored by the ranker."""
    builder = CandidateBuilder(phase, meta, emb)
    tick(f"candidate builder ready for phase '{phase.name}'")
    out = {}
    for s in range(0, len(codes), chunk):
        df = builder.build(codes[s:s + chunk])
        df["score"] = ranker.predict(df[feats])
        top = df.sort_values(["u", "score"], ascending=[True, False]).groupby("u", sort=False).head(K)
        for u, items in top.groupby("u", sort=False)["i"]:
            out[int(u)] = [aid(split.item_ids[int(i)]) for i in items]
        tick(f"  recommended for {min(s + chunk, len(codes)):,} / {len(codes):,} eligible customers")
    return out


def complete(recs: list, fallback: list) -> list:
    """Exactly 12 distinct ids: our recommendations first, topped up with the fallback list."""
    seen, out = set(), []
    for a in list(recs) + list(fallback):
        if a not in seen:
            seen.add(a)
            out.append(a)
        if len(out) == K:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--mode", choices=["offline", "final"], required=True)
    p.add_argument("--sample", type=int, default=100000, help="offline: number of random customers to score")
    args = p.parse_args()
    data = args.data_dir
    csv = os.path.join(data, "transactions_train.csv")
    cache = cache_dir(data, "train", 0)

    split, val, test = load_phases(data, kcore="train", shift_weeks=0)
    tick("phases loaded")
    last = pd.to_datetime(split.test.timestamp.max(), unit="ms").normalize()
    day = lambda d: d.strftime("%Y-%m-%d")
    articles = os.path.join(data, "articles.csv")
    emb = build_item_embeddings(articles, split.item_ids)
    meta = load_item_meta(articles, split.item_ids)
    all_customers = pd.read_csv(os.path.join(data, "customers.csv"), usecols=["customer_id"])["customer_id"]
    code_of = pd.Series(np.arange(len(split.user_ids)), index=split.user_ids)
    valid_articles = set(pd.read_csv(articles, usecols=["article_id"], dtype=str)["article_id"])

    load = lambda name: pickle.load(open(os.path.join(cache, name), "rb"))
    train_val = load("ret_val_b1.pkl")
    if args.mode == "final":
        tbl_test = load("ret_test_b1.pkl")
        tbl_test["u"] = tbl_test["u"] + 10_000_000          # keep the two weekly snapshots as separate ranking groups
        train_df = pd.concat([train_val, tbl_test], ignore_index=True)
        tick("training the ranker on two weekly snapshots (validation week + test week)")
    else:
        train_df = train_val
        tick("training the ranker on the validation week only")
    feats = [c for c in train_df.columns if c not in FEATURES_EXCLUDE]
    ranker = fit_ranker(train_df, feats, K, np.random.default_rng(0), seed=0)
    tick(f"ranker trained: {ranker.best_iteration_} trees")
    del train_df

    # ------------------------------------------------------------------ offline estimate
    if args.mode == "offline":
        week_start, week_end = day(last - pd.Timedelta(days=6)), day(last)
        prev_start, prev_end = day(last - pd.Timedelta(days=13)), day(last - pd.Timedelta(days=7))
        truth_df = load_window(csv, week_start, week_end)
        fallback = popular(load_window(csv, prev_start, prev_end))
        tick(f"pretend Kaggle week {week_start} to {week_end}: {len(truth_df):,} purchases; fallback = top-12 of {prev_start} to {prev_end}")
        sample = all_customers.sample(min(args.sample, len(all_customers)), random_state=0).tolist()
        elig = [c for c in sample if c in code_of.index]
        codes = code_of[elig].values.astype(int)
        tick(f"sample {len(sample):,} customers, of which {len(elig):,} eligible ({len(elig) / len(sample):.0%})")
        model_recs = recommend_eligible(test, ranker, feats, meta, emb, codes, split)
        truth = truth_df[truth_df["customer_id"].isin(set(sample))].groupby("customer_id")["article_id"].apply(set).to_dict()

        # diagnostic baseline, NOT our model: each eligible customer's 12 most recent distinct purchases
        hist = test.hist[test.hist["u"].isin(codes)].sort_values("timestamp")
        recent = hist.groupby("u")["i"].apply(lambda s: list(dict.fromkeys(s.tolist()[::-1]))[:K])
        recent_ids = {split.user_ids[u]: [aid(split.item_ids[i]) for i in items] for u, items in recent.items()}

        preds = {"popularity only (everybody)": {c: fallback for c in sample},
                 "our model (eligible) + popularity (others)": {c: complete([aid_ for aid_ in model_recs.get(int(code_of[c]), [])] if c in code_of.index else [], fallback) for c in sample},
                 "diagnostic: own 12 most recent purchases + popularity": {c: complete(recent_ids.get(c, []), fallback) for c in sample}}
        elig_set = set(elig)
        print(f"\nOffline estimate on {len(sample):,} random customers; Kaggle week = {week_start} to {week_end}; repeat purchases count.")
        print("| strategy | MAP@12, customers with a purchase | MAP@12, all customers (zeros included) | MAP@12, eligible customers with a purchase |\n|---|---|---|---|")
        for name, pr in preds.items():
            scores = {c: ap12(pr[c], truth[c]) for c in truth}
            with_p = np.mean(list(scores.values()))
            all_c = sum(scores.values()) / len(sample)
            elig_p = np.mean([v for c, v in scores.items() if c in elig_set])
            print(f"| {name} | {with_p:.4f} | {all_c:.5f} | {elig_p:.4f} |")
        # how much of the buying is repeat buying?
        bought_before = test.hist[test.hist["u"].isin(codes)].groupby("u")["i"].apply(lambda s: {aid(split.item_ids[i]) for i in s})
        hist_ids = {split.user_ids[u]: items for u, items in bought_before.items()}
        tot = rep = 0
        for c in elig_set:
            for a in truth.get(c, set()):
                tot += 1
                rep += a in hist_ids.get(c, set())
        print(f"\nOf the (customer, article) purchases made by eligible customers in that week, {rep / max(tot, 1):.0%} were articles the customer had bought before (repeat purchases),")
        print("which a new-to-customer model can never recommend.")
        return

    # ------------------------------------------------------------------ final submission
    start_week = day(last - pd.Timedelta(days=6))
    fallback = popular(load_window(csv, start_week, day(last)))
    tick(f"history ends {day(last)}; fallback top-12 over {start_week} to {day(last)}: {fallback[:3]}...")
    final_hist = pd.concat([split.train, split.val, split.test], ignore_index=True)
    final = Phase("final", final_hist, pd.DataFrame({"u": np.array([], dtype=np.int64), "i": np.array([], dtype=np.int64)}),
                  split.n_users, split.n_items)
    model_recs = recommend_eligible(final, ranker, feats, meta, emb, np.arange(split.n_users), split)

    rows, from_model = [], 0
    for c in all_customers:
        mine = model_recs.get(int(code_of[c]), []) if c in code_of.index else []
        from_model += bool(mine)
        rows.append(" ".join(complete(mine, fallback)))
    sub = pd.DataFrame({"customer_id": all_customers, "prediction": rows})

    bad = [r for r in rows if len(r.split()) != K or len(set(r.split())) != K or not set(r.split()) <= valid_articles]
    assert not bad, f"{len(bad)} invalid prediction rows, e.g. {bad[0]}"
    assert len(sub) == len(all_customers) == sub["customer_id"].nunique()
    out_dir = os.path.join(data, "submission")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "submission.csv")
    sub.to_csv(path, index=False)
    summary = {"customers": len(sub), "from_model": from_model, "from_popularity_fallback": len(sub) - from_model,
               "history_ends": day(last), "predicts": f"{day(last + pd.Timedelta(days=1))} to {day(last + pd.Timedelta(days=7))}",
               "ranker_trees": int(ranker.best_iteration_)}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    tick(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
