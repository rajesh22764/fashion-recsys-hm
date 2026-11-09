"""Build the precomputed data and thumbnails for the Streamlit demo.

Trains the ranker on the validation week, recommends 12 articles for a random sample of customers from the held-out final week,
and stores their recent purchases, recommendations and next-week purchases. Product thumbnails are fetched from the competition
data through the Kaggle API. LightGBM only (no torch). Requires the cached candidate tables from src.retrieval_analysis.

    python scripts/build_demo.py data/hm --customers 60
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_hm import cache_dir, load_phases  # noqa: E402
from src.run_ranker import FEATURES_EXCLUDE, fit_ranker  # noqa: E402

COMPETITION = "h-and-m-personalized-fashion-recommendations"
SOURCES = {"src_rcooc": "recency co-occurrence", "src_cooc": "co-occurrence",
           "src_rpop": "recent popularity", "src_content": "content similarity"}
K = 12


def article_id(item_id: str) -> str:
    return str(int(item_id)).zfill(10)


DELAY = 0.0


def thumbnail_from_zip(zf, aid: str, out_dir: str) -> bool:
    """Create the thumbnail from a locally downloaded images.zip (members look like '010/0108775015.jpg')."""
    from PIL import Image

    target = os.path.join(out_dir, f"{aid}.jpg")
    if os.path.exists(target):
        return True
    member = f"{aid[:3]}/{aid}.jpg"
    try:
        with zf.open(member) as fh, Image.open(BytesIO(fh.read())) as im:
            im = im.convert("RGB")
            im.thumbnail((200, 260))
            im.save(target, "JPEG", quality=82, optimize=True)
        return True
    except Exception as exc:
        print("not in zip / unreadable", aid, type(exc).__name__, flush=True)
        return False


def fetch_thumbnail(aid: str, out_dir: str) -> bool:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from PIL import Image

    import shutil
    import time

    target = os.path.join(out_dir, f"{aid}.jpg")
    if os.path.exists(target):
        return True
    time.sleep(DELAY)
    tmp = os.path.join(os.path.dirname(out_dir.rstrip("/")) , "_download_tmp", aid)
    for attempt in range(6):  # Kaggle rate-limits (HTTP 429), so back off for a long time and retry
        try:
            api = KaggleApi()
            api.authenticate()
            os.makedirs(tmp, exist_ok=True)
            api.competition_download_file(COMPETITION, f"images/{aid[:3]}/{aid}.jpg", path=tmp, quiet=True)
            with Image.open(os.path.join(tmp, f"{aid}.jpg")) as im:
                im = im.convert("RGB")
                im.thumbnail((200, 260))
                im.save(target, "JPEG", quality=82, optimize=True)
            shutil.rmtree(tmp, ignore_errors=True)
            return True
        except Exception as exc:
            if attempt == 5:  # a missing image just becomes a placeholder in the app
                print("image failed", aid, type(exc).__name__, flush=True)
            else:
                time.sleep(15 * (attempt + 1) if "429" in str(exc) else 2 ** attempt)
    shutil.rmtree(tmp, ignore_errors=True)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--customers", type=int, default=60)
    p.add_argument("--history", type=int, default=6)
    p.add_argument("--out", default="demo")
    p.add_argument("--sample-seed", type=int, default=7)
    p.add_argument("--images-only", action="store_true", help="only (re)fetch missing thumbnails for the existing demo data")
    p.add_argument("--images-zip", help="with --images-only: read thumbnails from this local images.zip instead of the Kaggle API")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--delay", type=float, default=1.5, help="seconds to wait before each image request")
    args = p.parse_args()
    global DELAY
    DELAY = args.delay

    if args.images_only:
        data = json.load(open(os.path.join(args.out, "data", "customers.json")))["customers"]
        needed = sorted({c_["article"] for c in data for c_ in c["history"] + c["recs"]} | {a for c in data for a in c["bought_next_week"]})
        img_dir = os.path.join(args.out, "images")
        if args.images_zip:
            import zipfile

            with zipfile.ZipFile(args.images_zip) as zf:
                ok = [thumbnail_from_zip(zf, a, img_dir) for a in needed]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                ok = list(ex.map(lambda a: fetch_thumbnail(a, img_dir), needed))
        print(f"{sum(ok)}/{len(needed)} thumbnails available", flush=True)
        return

    split, val, test = load_phases(args.data_dir, kcore="train", shift_weeks=0)
    cache = cache_dir(args.data_dir, "train", 0)
    load = lambda n: pickle.load(open(os.path.join(cache, n), "rb"))
    train_df, test_df = load("ret_val_b1.pkl"), load("ret_test_b1.pkl")
    feats = [c for c in test_df.columns if c not in FEATURES_EXCLUDE]
    model = fit_ranker(train_df, feats, K, np.random.default_rng(0), seed=0)
    print("ranker trained, best iteration", model.best_iteration_, flush=True)

    users = np.random.default_rng(args.sample_seed).choice(test_df["u"].unique(), size=args.customers, replace=False)
    sub = test_df[test_df["u"].isin(users)].copy()
    sub["score"] = model.predict(sub[feats])
    sub = sub.sort_values(["u", "score"], ascending=[True, False])
    sub["rank"] = sub.groupby("u").cumcount() + 1
    top = sub[sub["rank"] <= K]

    hist = test.hist[test.hist["u"].isin(users)].sort_values(["u", "timestamp"])
    n_hist = hist.groupby("u").size()
    hist = hist.groupby("u").tail(args.history)
    labels = test.labels[test.labels["u"].isin(users)].sort_values(["u", "timestamp"])

    def aid_of(code):
        return article_id(split.item_ids[code])

    customers = []
    for n, u in enumerate(users, start=1):
        recs = []
        for _, r in top[top["u"] == u].iterrows():
            recs.append({"article": aid_of(int(r["i"])), "rank": int(r["rank"]), "bought": bool(r["y"]),
                         "sources": [name for flag, name in SOURCES.items() if flag in r and bool(r[flag])]})
        bought = [aid_of(int(i)) for i in labels[labels["u"] == u]["i"].tolist()][:6]
        past = [{"article": aid_of(int(r["i"])), "date": pd.to_datetime(int(r["timestamp"]), unit="ms").strftime("%Y-%m-%d")}
                for _, r in hist[hist["u"] == u].iterrows()]
        customers.append({"id": n, "n_purchases": int(n_hist.get(u, 0)), "history": past, "recs": recs,
                          "bought_next_week": bought, "hits": int(sum(r["bought"] for r in recs)),
                          "n_true": int((labels["u"] == u).sum())})

    needed = sorted({c_["article"] for c in customers for c_ in c["history"] + c["recs"]} |
                    {a for c in customers for a in c["bought_next_week"]})
    arts = pd.read_csv(os.path.join(args.data_dir, "articles.csv"), dtype={"article_id": str})
    arts["aid"] = arts["article_id"].str.zfill(10)
    arts = arts[arts["aid"].isin(needed)].set_index("aid")
    meta = {a: {"name": str(r["prod_name"]), "type": str(r["product_type_name"]), "colour": str(r["colour_group_name"]),
                "department": str(r["department_name"]), "section": str(r["section_name"]),
                "description": str(r["detail_desc"]) if isinstance(r["detail_desc"], str) else ""}
            for a, r in arts.iterrows()}

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ok = list(ex.map(lambda a: fetch_thumbnail(a, img_dir), needed))
    print(f"{sum(ok)}/{len(needed)} thumbnails available", flush=True)

    payload = {"n_customers": len(customers), "k": K, "customers": customers}
    json.dump(payload, open(os.path.join(args.out, "data", "customers.json"), "w"))
    json.dump(meta, open(os.path.join(args.out, "data", "articles.json"), "w"))
    hit_rate = np.mean([c["hits"] > 0 for c in customers])
    recall = np.mean([c["hits"] / c["n_true"] for c in customers])
    print(f"sample of {len(customers)} customers: share with at least one hit {hit_rate:.2f}, mean recall@12 {recall:.4f}")


if __name__ == "__main__":
    main()
