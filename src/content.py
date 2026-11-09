"""Content-based item embeddings from H&M article metadata, and a retrieval model on top."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TEXT_COLUMNS = [
    "prod_name", "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name",
    "department_name", "index_name", "index_group_name", "section_name",
    "garment_group_name", "detail_desc",
]


def build_item_embeddings(articles_path: str, item_ids: np.ndarray, dim: int = 128) -> np.ndarray:
    """L2-normalised (n_items, dim) matrix aligned with `item_ids` (the split's item coding).

    TF-IDF over the concatenated name, attribute and description text, reduced with SVD.
    """
    arts = pd.read_csv(articles_path, dtype={"article_id": str})
    arts["item_id"] = arts["article_id"].astype(int).astype(str)  # loader strips leading zeros
    arts = arts.set_index("item_id").reindex(item_ids)
    text = arts[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000)
    x = tfidf.fit_transform(text)
    z = TruncatedSVD(n_components=dim, random_state=0).fit_transform(x)
    return normalize(z).astype(np.float32)


class ContentKNN:
    """Score items by similarity to the mean embedding of the user's training history."""

    def __init__(self, item_emb: np.ndarray):
        self.emb = item_emb

    def fit(self, train: sparse.csr_matrix):
        self.train = train
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        hist = self.train[users]
        counts = np.asarray(hist.sum(axis=1)).ravel()
        profile = (hist @ self.emb) / np.maximum(counts, 1)[:, None]
        return normalize(profile) @ self.emb.T


META_COLUMNS = ["product_type_no", "colour_group_code", "department_no", "index_group_no",
                "garment_group_no", "graphical_appearance_no"]


def load_item_meta(articles_path: str, item_ids: np.ndarray) -> pd.DataFrame:
    """Integer category codes per item, row-aligned with `item_ids` (the split's item coding)."""
    arts = pd.read_csv(articles_path, dtype={"article_id": str})
    arts["item_id"] = arts["article_id"].astype(int).astype(str)
    arts = arts.set_index("item_id").reindex(item_ids)
    meta = pd.DataFrame(index=range(len(item_ids)))
    for col in META_COLUMNS:
        meta[col] = arts[col].astype("category").cat.codes.values
    return meta


def stable_meta(articles_path: str, item_ids: np.ndarray) -> pd.DataFrame:
    """Like load_item_meta, but category codes come from the whole articles table.

    The codes are therefore identical in every rolling fold, which is required before pooling
    ranker training data from several weeks.
    """
    arts = pd.read_csv(articles_path, dtype={"article_id": str})
    arts["item_id"] = arts["article_id"].astype(int).astype(str)
    arts = arts.set_index("item_id")
    meta = pd.DataFrame(index=range(len(item_ids)))
    for col in META_COLUMNS:
        codes = pd.Series(pd.Categorical(arts[col]).codes, index=arts.index)
        meta[col] = codes.reindex(item_ids).values
    return meta
