"""Two-tower retrieval: user tower encodes recent purchase history, item tower = id + content embedding.

The user tower has no user-id embedding, so brand-new users with any history can be served.
The item tower mixes a learned id embedding with the TF-IDF/SVD content embedding; id embeddings are
randomly dropped in training so the model also learns to rank items from content alone (cold start).
"""
import numpy as np
import pandas as pd
import sklearn  # noqa: F401  import before torch: avoids the duplicate-libomp crash with this conda env
import torch
import torch.nn.functional as F
from torch import nn

HIST_LEN = 20


def last_items(hist: pd.DataFrame, n_users: int, n_items: int, length: int = HIST_LEN) -> np.ndarray:
    """(n_users, length) most recent items per user, oldest first, left-padded with n_items."""
    h = hist.sort_values(["u", "timestamp"], kind="stable")
    from_end = h.groupby("u").cumcount(ascending=False).values
    sel = from_end < length
    out = np.full((n_users, length), n_items, dtype=np.int32)
    out[h["u"].values[sel], length - 1 - from_end[sel]] = h["i"].values[sel]
    return out


def training_examples(train: pd.DataFrame, n_items: int, length: int = HIST_LEN):
    """(history, target) pairs: every purchase is a target, its history is the user's earlier purchases."""
    t = train.sort_values(["u", "timestamp"], kind="stable")
    users, items = t["u"].values, t["i"].values.astype(np.int32)
    hist = np.full((len(t), length), n_items, dtype=np.int32)
    for o in range(1, length + 1):
        same = users[o:] == users[:-o]
        hist[o:, length - o] = np.where(same, items[:-o], n_items)
    has_history = hist[:, -1] != n_items
    return hist[has_history], items[has_history]


class TwoTower(nn.Module):
    def __init__(self, n_items: int, content: np.ndarray, dim: int = 64):
        super().__init__()
        self.n_items = n_items
        self.register_buffer("content", torch.from_numpy(content))
        self.id_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        nn.init.normal_(self.id_emb.weight, std=0.1)
        self.content_proj = nn.Linear(content.shape[1], dim)
        self.item_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.user_mlp = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(14.0))))

    def item_vec(self, idx: torch.Tensor, id_gate: torch.Tensor | None = None) -> torch.Tensor:
        """id_gate: float tensor broadcastable to idx (1 = use id embedding, 0 = content only)."""
        x = self.content_proj(self.content[idx.clamp(max=self.n_items - 1)])
        ids = self.id_emb(idx)
        x = x + (ids if id_gate is None else ids * id_gate.unsqueeze(-1))
        return F.normalize(self.item_out(x), dim=-1)

    def user_vec(self, hist: torch.Tensor) -> torch.Tensor:
        v = self.item_vec(hist)                       # (B, L, d); pad rows are masked out below
        mask = (hist != self.n_items).unsqueeze(-1).float()
        mean = (v * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.user_mlp(torch.cat([mean, v[:, -1]], dim=-1)), dim=-1)


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_two_tower(train: pd.DataFrame, n_items: int, content: np.ndarray, epochs: int = 3, batch: int = 4096,
                    lr: float = 2e-3, id_drop: float = 0.3, seed: int = 0, log=print) -> TwoTower:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hist, target = training_examples(train, n_items)
    log(f"two-tower: {len(target)} training examples")
    device = pick_device()
    model = TwoTower(n_items, content).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    freq = np.bincount(target, minlength=n_items).astype(np.float64) + 1
    log_q = torch.tensor(np.log(freq / freq.sum()), dtype=torch.float32, device=device)  # sampling-bias correction

    steps = len(target) // batch
    for epoch in range(epochs):
        order = rng.permutation(len(target))
        total = 0.0
        for s in range(steps):
            idx = order[s * batch:(s + 1) * batch]
            h = torch.from_numpy(hist[idx]).to(device).long()
            t = torch.from_numpy(target[idx]).to(device).long()
            gate = (torch.rand(len(idx), device=device) > id_drop).float()
            u = model.user_vec(h)
            i = model.item_vec(t, gate)
            logits = (u @ i.T) * model.log_scale.exp() - log_q[t][None, :]
            same = t[:, None] == t[None, :]
            logits = logits.masked_fill(same & ~torch.eye(len(idx), dtype=torch.bool, device=device), -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(idx), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            if (s + 1) % 400 == 0:
                log(f"  epoch {epoch + 1} step {s + 1}/{steps} loss {total / 400:.3f}")
                total = 0.0
    return model.eval()


class TwoTowerGen:
    """Scorer with the same interface as the other models (fit / score), for use as a candidate generator."""

    def __init__(self, model: TwoTower, user_hist: np.ndarray, no_id_items: np.ndarray | None = None):
        self.model, self.user_hist = model, user_hist
        self.device = next(model.parameters()).device
        with torch.no_grad():
            idx = torch.arange(model.n_items, device=self.device)
            gate = None
            if no_id_items is not None:
                gate = torch.ones(model.n_items, device=self.device)
                gate[torch.from_numpy(no_id_items).to(self.device)] = 0
            self.items = model.item_vec(idx, gate)

    def fit(self, train):
        self.train = train
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            h = torch.from_numpy(self.user_hist[users]).to(self.device).long()
            return (self.model.user_vec(h) @ self.items.T).cpu().numpy()
