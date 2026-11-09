"""Add two-tower scores as features to the candidate tables of the 4-generator pool (candidates unchanged).

Needed for the 2x2 experiment: two-tower as candidate source vs as ranking feature. Uses torch, so it runs
as its own process; the LightGBM step is src/two_by_two.py.

    python -m src.add_tt_feature data/hm
"""
import argparse
import os
import pickle

import numpy as np
import sklearn  # noqa: F401  (before torch)
import torch

from src.content import build_item_embeddings
from src.run_hm import add_protocol_args, cache_dir, load_phases
from src.twotower import TwoTower, TwoTowerGen, last_items, pick_device


def add_scores(df, gen, batch: int = 1024):
    """Fill tt_score / tt_rel for each (customer, article) row exactly as CandidateBuilder does for its generators."""
    order = np.argsort(df["u"].values, kind="stable")
    u_sorted = df["u"].values[order]
    users = np.unique(u_sorted)
    starts = np.searchsorted(u_sorted, users)
    ends = np.append(starts[1:], len(df))
    score = np.zeros(len(df), np.float32)
    rel = np.zeros(len(df), np.float32)
    seen = gen.train
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        S = gen.score(ub).astype(np.float32)
        r, c = seen[ub].nonzero()
        S[r, c] = -np.inf                              # same masking as the candidate builder
        idx = order[starts[s]:ends[s + len(ub) - 1]]
        local = np.searchsorted(ub, df["u"].values[idx])
        vals = S[local, df["i"].values[idx]]
        score[idx] = vals
        rel[idx] = vals / np.maximum(S.max(axis=1)[local], 1e-9)
    out = df.copy()
    out["tt_score"], out["tt_rel"] = score, rel
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    add_protocol_args(p)
    args = p.parse_args()
    cache = cache_dir(args.data_dir, args.kcore, args.shift_weeks)
    split, val, test = load_phases(args.data_dir, kcore=args.kcore, shift_weeks=args.shift_weeks)
    emb = build_item_embeddings(os.path.join(args.data_dir, "articles.csv"), split.item_ids)
    model = TwoTower(split.n_items, emb)
    model.load_state_dict(torch.load(os.path.join(cache, "twotower.pt")))
    model = model.to(pick_device()).eval()
    for phase in (val, test):
        gen = TwoTowerGen(model, last_items(phase.hist, split.n_users, split.n_items)).fit(phase.binary())
        df = pickle.load(open(os.path.join(cache, f"cand_{phase.name}.pkl"), "rb"))
        out = add_scores(df, gen)
        pickle.dump(out, open(os.path.join(cache, f"cand_{phase.name}_ttfeat.pkl"), "wb"))
        print(phase.name, len(out), "rows; tt_score mean", float(out["tt_score"].mean()), flush=True)


if __name__ == "__main__":
    main()
