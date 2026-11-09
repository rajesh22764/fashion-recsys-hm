"""Loader for the H&M Personalized Fashion Recommendations data (Kaggle).

Expects data/hm/transactions_train.csv with columns
t_dat, customer_id, article_id, price, sales_channel_id.
"""
import pandas as pd


def load_hm(path: str, last_weeks: int | None = 26) -> pd.DataFrame:
    """Return user_id / item_id / timestamp / price rows, optionally only the last N weeks.

    The file is read in chunks and filtered by date as it goes, so the full 31M-row
    table (with 64-char customer ids) never sits in memory.
    Repeat purchases of the same article by the same customer are collapsed to the
    first one, so the interaction matrix is binary like the other loaders.
    """
    cutoff = None
    if last_weeks:
        dates = pd.read_csv(path, usecols=["t_dat"], parse_dates=["t_dat"])["t_dat"]
        cutoff = dates.max() - pd.Timedelta(weeks=last_weeks)

    parts = []
    reader = pd.read_csv(
        path,
        usecols=["t_dat", "customer_id", "article_id", "price"],
        dtype={"article_id": "int32", "price": "float32"},
        parse_dates=["t_dat"],
        chunksize=2_000_000,
    )
    for chunk in reader:
        if cutoff is not None:
            chunk = chunk[chunk["t_dat"] >= cutoff]
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)

    df = df.rename(columns={"customer_id": "user_id", "article_id": "item_id"})
    df["item_id"] = df["item_id"].astype(str)
    # pandas may parse at s/ms/us/ns resolution; force ms to match the Amazon data
    df["timestamp"] = df["t_dat"].astype("datetime64[ms]").astype("int64")
    return df.drop(columns="t_dat").drop_duplicates(["user_id", "item_id"])
