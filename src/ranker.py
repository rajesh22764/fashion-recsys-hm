"""Candidate generation + feature building + LightGBM ranker for H&M."""
import numpy as np
import pandas as pd
from scipy import sparse

from src.baselines import ItemCooccurrence, RecencyCooccurrence, RecentPopularity
from src.content import ContentKNN
from src.phases import DAY, Phase

# how many candidates each generator contributes per user
GEN_N = {"rcooc": 60, "cooc": 20, "rpop": 30, "content": 20, "tt": 40}
AFFINITY_COLS = ["product_type_no", "colour_group_code", "department_no", "index_group_no"]


def _onehot(codes: np.ndarray) -> sparse.csr_matrix:
    n = len(codes)
    return sparse.csr_matrix((np.ones(n, np.float32), (np.arange(n), codes)), shape=(n, codes.max() + 1))


class CandidateBuilder:
    """Builds (user, item) candidate rows with features from one Phase's history."""

    def __init__(self, phase: Phase, meta: pd.DataFrame, item_emb: np.ndarray, extra_gens: dict | None = None, scale: float = 1.0):
        self.phase, self.meta = phase, meta
        self.gen_n = {k: max(1, int(round(v * scale))) for k, v in GEN_N.items()}  # candidates per generator
        self.seen = phase.binary()
        decayed30 = phase.decayed(30)
        rcooc = RecencyCooccurrence().fit(self.seen, decayed30)
        cooc = ItemCooccurrence()  # same similarity matrix, unweighted history
        cooc.train, cooc.sim = self.seen, rcooc.sim
        self.gens = {
            "rcooc": rcooc,
            "cooc": cooc,
            "rpop": RecentPopularity().fit(self.seen, phase.recent_counts(14)),
            "content": ContentKNN(item_emb).fit(phase.decayed(7)),
        }
        for name, gen in (extra_gens or {}).items():
            self.gens[name] = gen.fit(self.seen)
        self.decayed30 = decayed30
        self._item_features()
        self._user_features()
        self.onehots = {c: _onehot(meta[c].values) for c in AFFINITY_COLS}

    def _item_features(self):
        h, n = self.phase.hist, self.phase.n_items
        end = h["timestamp"].max()
        self.item_pop = np.bincount(h["i"].values, minlength=n).astype(np.float32)
        self.item_pop7 = self.phase.recent_counts(7)
        self.item_pop28 = self.phase.recent_counts(28)
        price = h.groupby("i")["price"].mean().reindex(range(n))
        self.item_price = price.fillna(price.median()).values.astype(np.float32)
        first = h.groupby("i")["timestamp"].min().reindex(range(n))
        self.item_age = ((end - first.fillna(end)) / DAY).values.astype(np.float32)
        self.end = end

    def _user_features(self):
        h, n = self.phase.hist, self.phase.n_users
        g = h.groupby("u")
        self.user_n = g.size().reindex(range(n)).fillna(0).values.astype(np.float32)
        last = g["timestamp"].max().reindex(range(n))
        self.user_recency = ((self.end - last) / DAY).fillna(999).values.astype(np.float32)
        self.user_price = g["price"].mean().reindex(range(n)).fillna(self.item_price.mean()).values.astype(np.float32)
        recent = h[h["timestamp"] > self.end - 28 * DAY]
        self.user_n28 = np.bincount(recent["u"].values, minlength=n).astype(np.float32)

    def batch(self, users: np.ndarray) -> pd.DataFrame:
        scores = {name: g.score(users).astype(np.float32) for name, g in self.gens.items()}
        rows, cols = self.seen[users].nonzero()
        tops = {}
        for name, s in scores.items():
            s[rows, cols] = -np.inf
            n = self.gen_n[name]
            tops[name] = np.argpartition(-s, n, axis=1)[:, :n]
        cand = np.sort(np.concatenate(list(tops.values()), axis=1), axis=1)
        keep = np.ones_like(cand, dtype=bool)
        keep[:, 1:] = cand[:, 1:] != cand[:, :-1]
        r, c = np.nonzero(keep)
        item = cand[r, c]
        local_user = r
        out = {"u": users[local_user], "i": item}
        for name, top in tops.items():  # which generator(s) proposed each candidate (analysis only, not a ranker feature)
            member = np.zeros(scores[name].shape, dtype=bool)
            np.put_along_axis(member, top, True, axis=1)
            out[f"src_{name}"] = member[local_user, item]
        for name, s in scores.items():
            vals = s[local_user, item]
            out[f"{name}_score"] = vals
            if name in ("rcooc", "content", "tt"):
                out[f"{name}_rel"] = vals / np.maximum(s.max(axis=1)[local_user], 1e-9)
        out["item_pop"] = self.item_pop[item]
        out["item_pop7"] = self.item_pop7[item]
        out["item_pop28"] = self.item_pop28[item]
        out["item_price"] = self.item_price[item]
        out["item_age"] = self.item_age[item]
        u = users[local_user]
        out["user_n"] = self.user_n[u]
        out["user_n28"] = self.user_n28[u]
        out["user_recency"] = self.user_recency[u]
        out["price_ratio"] = self.item_price[item] / np.maximum(self.user_price[u], 1e-6)
        hist = self.decayed30[users]
        total = np.maximum(np.asarray(hist.sum(axis=1)).ravel(), 1e-9)
        for col in AFFINITY_COLS:
            aff = (hist @ self.onehots[col]).toarray() / total[:, None]
            out[f"aff_{col}"] = aff[local_user, self.meta[col].values[item]]
        for col in self.meta.columns:
            out[f"cat_{col}"] = self.meta[col].values[item]
        return pd.DataFrame(out)

    def build(self, users: np.ndarray, batch: int = 1024) -> pd.DataFrame:
        parts = [self.batch(users[s:s + batch]) for s in range(0, len(users), batch)]
        df = pd.concat(parts, ignore_index=True)
        n = self.phase.n_items
        keys = self.phase.labels["u"].values.astype(np.int64) * n + self.phase.labels["i"].values
        df["y"] = np.isin(df["u"].values.astype(np.int64) * n + df["i"].values, keys).astype(np.int8)
        return df


def candidate_recall(df: pd.DataFrame, phase: Phase) -> float:
    """Ceiling for any ranker: average share of a user's true items present among the candidates."""
    hit = df.groupby("u")["y"].sum()
    total = phase.labels.groupby("u").size().reindex(hit.index)
    return float((hit / total).mean())
