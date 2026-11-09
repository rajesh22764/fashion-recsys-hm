"""Streamlit demo for the H&M fashion recommender.

Shows precomputed recommendations for a random sample of held-out customers (built by scripts/build_demo.py), plus the
evaluation results. It does not run the model live: the full pipeline needs several GB of transactions and minutes of compute.

    streamlit run demo/app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
REPO = "https://github.com/rajesh22764/fashion-recsys-hm"

st.set_page_config(page_title="Fashion recommender demo", layout="wide")


@st.cache_data
def load():
    customers = json.loads((ROOT / "data" / "customers.json").read_text())
    articles = json.loads((ROOT / "data" / "articles.json").read_text())
    return customers, articles


def show_image(aid: str):
    path = ROOT / "images" / f"{aid}.jpg"
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        st.caption("(image unavailable)")


def article_caption(articles: dict, aid: str) -> str:
    a = articles.get(aid, {})
    return f"**{a.get('name', aid)}**  \n{a.get('colour', '')} {a.get('type', '').lower()}"


customers, articles = load()
all_customers = customers["customers"]

st.title("Fashion recommender: next-week purchase prediction")
st.info(
    "**Precomputed demo, not live inference.** These are recommendations my ranker produced for a random sample of "
    f"{customers['n_customers']} customers from the held-out final week of the H&M dataset. "
    "The full pipeline needs several GB of transactions, so it is not run in the browser. "
    f"[Code, method and honest limitations on GitHub]({REPO})."
)

tab_demo, tab_results, tab_how, tab_limits = st.tabs(["Recommendations", "Results", "How it works", "Limits and data"])

with tab_demo:
    hit_rate = sum(c["hits"] > 0 for c in all_customers) / len(all_customers)
    st.markdown(
        f"In this sample, **{hit_rate:.0%} of customers** had at least one of the 12 recommended articles among what they actually bought "
        "the following week. Most recommendations are misses: next-week buying is hard to predict, and the sample is small and noisy "
        "(the full evaluation reports recall@12 of about 0.055)."
    )
    only_hits = st.checkbox("Only show customers with at least one correct recommendation", value=False)
    pool = [c for c in all_customers if c["hits"] > 0] if only_hits else all_customers
    choice = st.selectbox("Customer", pool, format_func=lambda c: f"Customer {c['id']}  ({c['n_purchases']} purchases before the test week, {c['hits']} correct)")

    st.subheader("Recently bought")
    cols = st.columns(max(len(choice["history"]), 1))
    for col, item in zip(cols, choice["history"]):
        with col:
            show_image(item["article"])
            st.markdown(article_caption(articles, item["article"]))
            st.caption(item["date"])

    st.subheader("Top 12 recommendations for the following week")
    st.caption("Green = the customer actually bought it that week. 'Proposed by' shows which candidate generator surfaced it before the ranker ordered everything.")
    recs = choice["recs"]
    for start in range(0, len(recs), 6):
        cols = st.columns(6)
        for col, rec in zip(cols, recs[start:start + 6]):
            with col:
                show_image(rec["article"])
                st.markdown(f"#{rec['rank']}  " + article_caption(articles, rec["article"]))
                st.caption("Proposed by: " + (", ".join(rec["sources"]) or "n/a"))
                if rec["bought"]:
                    st.markdown(":green[**Bought that week**]")

    st.subheader("What the customer actually bought that week")
    if choice["bought_next_week"]:
        cols = st.columns(max(len(choice["bought_next_week"]), 1))
        for col, aid in zip(cols, choice["bought_next_week"]):
            with col:
                show_image(aid)
                st.markdown(article_caption(articles, aid))
        if choice["n_true"] > len(choice["bought_next_week"]):
            st.caption(f"Showing {len(choice['bought_next_week'])} of {choice['n_true']} articles.")
    else:
        st.caption("No purchases recorded.")

with tab_results:
    st.markdown("All numbers are recall@12 on the held-out final week unless stated. Full tables, confidence intervals and caveats are in the [README](" + REPO + ").")
    st.subheader("Models on the main test week (40,188 customers)")
    st.dataframe(pd.DataFrame([
        ("Popularity", 0.0072, 0.0043),
        ("Item co-occurrence", 0.0210, 0.0123),
        ("Content similarity (article text)", 0.0122, 0.0074),
        ("Recent popularity (last 14 days)", 0.0284, 0.0170),
        ("Two-tower neural retriever", 0.0276, 0.0170),
        ("Recency-weighted co-occurrence (best single generator)", 0.0323, 0.0199),
        ("Rank fusion of generators (no learning)", 0.0398, 0.0248),
        ("LightGBM ranker, 4 generators (mean of 5 seeds)", 0.0551, 0.0349),
        ("LightGBM ranker, 5 generators (mean of 5 seeds)", 0.0559, 0.0350),
    ], columns=["Model", "recall@12", "NDCG@12"]), hide_index=True, width="stretch")
    st.subheader("Four consecutive test weeks (ranker vs best single generator)")
    st.dataframe(pd.DataFrame([
        ("Final week", 0.0323, 0.0552, "+71%"), ("1 week earlier", 0.0333, 0.0522, "+57%"),
        ("2 weeks earlier", 0.0316, 0.0544, "+72%"), ("3 weeks earlier", 0.0285, 0.0500, "+76%"),
    ], columns=["Test week", "Best generator", "Ranker", "Relative gain"]), hide_index=True, width="stretch")
    st.subheader("New articles with no purchase history (simulation)")
    st.dataframe(pd.DataFrame([
        ("Random", 0.0073), ("Popularity", 0.0094), ("Item co-occurrence", 0.0090),
        ("Content similarity", 0.0532), ("Two-tower (content only for new items)", 0.0673),
    ], columns=["Model", "recall@12"]), hide_index=True, width="stretch")
    st.caption("A single run in which 10% of the articles bought in the test week were hidden from the model; not a real product-launch study.")

with tab_how:
    st.markdown(
        "**Task:** given everything a customer bought before week W, rank articles they will buy for the first time in week W and show 12.\n\n"
        "**Evaluation:** strictly time-based. Eligibility (at least 5 purchases) is computed on the training period only, so future purchases "
        "can never make a customer or article eligible. Results are averaged over several training seeds with bootstrap intervals."
    )
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; node [shape=box, style=rounded];
          history [label="Customer purchase\\nhistory"];
          g1 [label="Recency-weighted\\nco-occurrence"]; g2 [label="Recent\\npopularity"];
          g3 [label="Co-occurrence"]; g4 [label="Content\\nsimilarity"];
          pool [label="~112 candidates\\nper customer"]; rank [label="LightGBM ranker"]; top [label="Top 12"];
          history -> g1; history -> g2; history -> g3; history -> g4;
          g1 -> pool; g2 -> pool; g3 -> pool; g4 -> pool; pool -> rank -> top;
        }
        """
    )
    st.markdown(
        "The ranker is trained on the previous week's real purchases, using the generator scores plus article popularity, price, article age, "
        "customer activity and category affinity. On the same candidates it beats simple rank fusion and a linear model by a wide margin."
    )

with tab_limits:
    st.markdown(
        "- **Only customers who bought in the test week are scored** (about 10% of eligible customers), because recall is undefined for the rest.\n"
        "- **New-to-customer discovery:** articles a customer already bought are never recommended, so this is narrower than the original Kaggle task and not comparable to its leaderboard.\n"
        "- **Retrieval limits the system:** only about a quarter of customers have even one relevant article among the candidates, and a bigger candidate pool stops helping after roughly double the size.\n"
        "- **The two-tower neural model is not part of the headline gain.** It performs about as well as recent popularity alone and I could not show a reliable improvement from adding it; its value is on brand-new articles.\n"
        "- **This page is precomputed** for a random sample; it does not demonstrate serving, latency, or an A/B test.\n\n"
        "**Data:** [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (Kaggle). "
        f"Product images are small thumbnails from that dataset. Source code: [{REPO}]({REPO})."
    )
