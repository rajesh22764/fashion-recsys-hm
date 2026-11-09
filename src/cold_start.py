"""Stage 4: cold-start experiment.

Hide a random 10% of the articles bought in the test week: delete every one of their purchases from
the history, retrain the two-tower model without them, then rank *only these new articles* for users
who bought one. Models that need purchase history cannot score them and fall back to random order.

    python -m src.cold_start data/hm
"""
import argparse
import os

import numpy as np

from src.baselines import ItemCooccurrence, Popularity
from src.content import ContentKNN, build_item_embeddings
from src.metrics import ndcg_at_k, recall_at_k
from src.phases import Phase
from src.run_hm import add_protocol_args, load_phases

import torch  # noqa: E402  (after sklearn/lightgbm imports above)
from src.twotower import TwoTowerGen, last_items, train_two_tower  # noqa: E402


def rank_in_pool(model, users, pool, k, rng, batch=1024):
    """Top-k of `pool` items per user; exact ties (e.g. all-zero scores) are broken randomly."""
    out = []
    for s in range(0, len(users), batch):
        scores = model.score(users[s:s + batch])[:, pool].astype(np.float64)
        scores = scores + rng.random(scores.shape) * 1e-9
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
        out.append(pool[np.take_along_axis(top, order, 1)])
    return np.vstack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--cold-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-users", type=int, default=20000)
    add_protocol_args(p)
    args = p.parse_args()
    rng = np.random.default_rng(0)

    split, _, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    emb = build_item_embeddings(os.path.join(args.data_dir, "articles.csv"), split.item_ids)

    bought = test.labels["i"].unique()
    cold = np.sort(rng.choice(bought, size=int(len(bought) * args.cold_frac), replace=False))
    is_cold = np.zeros(split.n_items, dtype=bool)
    is_cold[cold] = True
    print(f"{len(cold)} cold items out of {split.n_items} (bought in test week: {len(bought)})", flush=True)

    hist = test.hist[~is_cold[test.hist["i"].values]]
    labels = test.labels[is_cold[test.labels["i"].values]]
    phase = Phase("cold", hist, labels, test.n_users, test.n_items)
    by_user = labels.groupby("u")["i"].apply(set)
    users = rng.permutation(by_user.index.values)[: args.max_users]
    truth = [by_user[u] for u in users]
    print(f"{len(users)} users bought at least one cold item; mean {np.mean([len(t) for t in truth]):.2f} each", flush=True)

    train_no_cold = split.train[~is_cold[split.train["i"].values]]
    model = train_two_tower(train_no_cold, split.n_items, emb, epochs=args.epochs, log=lambda m: print(m, flush=True))

    seen = phase.binary()
    models = {
        "random": type("Rand", (), {"score": lambda self, u: np.zeros((len(u), split.n_items))})(),
        "popularity": Popularity().fit(seen),
        "co-occurrence": ItemCooccurrence().fit(seen),
        "content-knn": ContentKNN(emb).fit(phase.decayed(30)),
        "two-tower (content only for new items)": TwoTowerGen(
            model, last_items(hist, split.n_users, split.n_items), no_id_items=cold).fit(seen),
    }
    print(f"\nrank only the {len(cold)} new items, k={args.k} (random recall ~ {args.k / len(cold):.4f}):")
    for name, m in models.items():
        recs = rank_in_pool(m, users, cold, args.k, rng)
        print(f"  {name:40s} recall@{args.k}={recall_at_k(recs, truth):.4f}  ndcg@{args.k}={ndcg_at_k(recs, truth):.4f}", flush=True)


if __name__ == "__main__":
    main()
