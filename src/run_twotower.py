"""Stage 3: train the two-tower model on the training weeks and evaluate it as a standalone retriever.

    python -m src.run_twotower data/hm --epochs 3
"""
import argparse
import os

import numpy as np
import sklearn  # noqa: F401  (before torch)
import torch

from src.content import build_item_embeddings
from src.run_hm import add_protocol_args, cache_dir, load_phases, score_model
from src.twotower import TwoTowerGen, last_items, train_two_tower


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--k", type=int, default=12)
    add_protocol_args(p)
    args = p.parse_args()

    split, val, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    emb = build_item_embeddings(os.path.join(args.data_dir, "articles.csv"), split.item_ids)
    model = train_two_tower(split.train, split.n_items, emb, epochs=args.epochs, log=lambda m: print(m, flush=True))
    cache = cache_dir(args.data_dir, args.kcore, args.shift_weeks)
    torch.save(model.state_dict(), os.path.join(cache, "twotower.pt"))

    for phase in (val, test):
        gen = TwoTowerGen(model, last_items(phase.hist, split.n_users, split.n_items)).fit(phase.binary())
        res = score_model(gen, phase, args.k)
        print(f"two-tower [{phase.name}]", {a: round(b, 4) if isinstance(b, float) else b for a, b in res.items()}, flush=True)


if __name__ == "__main__":
    main()
