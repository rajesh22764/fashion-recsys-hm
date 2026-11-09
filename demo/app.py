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
KAGGLE = "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations"
SOURCE_BADGE = {  # generator name in the data -> (short label, badge colour)
    "recency co-occurrence": ("recency co-occ", "blue"),
    "co-occurrence": ("co-occ", "violet"),
    "recent popularity": ("trending", "orange"),
    "content similarity": ("content", "gray"),
}
IMG_W, CARD_H, CARD_H_SHORT, PER_ROW = 150, 450, 390, 6

st.set_page_config(page_title="Fashion recommender demo", layout="wide")


@st.cache_data
def load():
    customers = json.loads((ROOT / "data" / "customers.json").read_text())
    articles = json.loads((ROOT / "data" / "articles.json").read_text())
    return customers, articles


customers, articles = load()
all_customers = customers["customers"]


def pool() -> list:
    return [c for c in all_customers if c["hits"] > 0] if st.session_state.get("only_hits") else all_customers


def step(delta: int):
    ids = [c["id"] for c in pool()]
    cur = st.session_state.get("customer_id")
    i = ids.index(cur) if cur in ids else 0
    st.session_state["customer_id"] = ids[(i + delta) % len(ids)]


def show_image(aid: str):
    path = ROOT / "images" / f"{aid}.jpg"
    if path.exists():
        st.image(str(path), width=IMG_W)
    else:
        st.caption("(no image in the dataset)")


def card(aid: str, rank: int | None = None, sources: list | None = None, bought: bool = False, date: str | None = None):
    a = articles.get(aid, {})
    with st.container(border=True, height=CARD_H if sources is not None else CARD_H_SHORT):
        if bought:
            st.success("Bought that week")
        show_image(aid)
        st.markdown(f"**{'#%d  ' % rank if rank else ''}{a.get('name', aid)}**")
        st.caption(f"{a.get('colour', '')} {a.get('type', '').lower()}")
        if date:
            st.caption(date)
        if sources:
            st.markdown(" ".join(f":{SOURCE_BADGE[s][1]}-badge[{SOURCE_BADGE[s][0]}]" for s in sources if s in SOURCE_BADGE))


def row_of_cards(items: list, **kwargs):
    cols = st.columns(PER_ROW)
    for col, item in zip(cols, items):
        with col:
            card(**{**kwargs, **item})


# ---------------------------------------------------------------- header
st.title("Fashion recommender")
st.caption("Predicting what H&M customers will buy next week: candidate retrieval, then a LightGBM ranker. Evaluated on a held-out week.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recall@12, ranker", "0.055", "+71% vs best single generator", delta_color="off", delta_arrow="off")
m2.metric("Best single generator", "0.032", "recency-weighted co-occurrence", delta_color="off", delta_arrow="off")
m3.metric("Customers evaluated", "40,188", "9.7% of eligible customers", delta_color="off", delta_arrow="off")
m4.metric("New articles, recall@12", "0.067", "random: 0.007", delta_color="off", delta_arrow="off")
st.info(
    "**Precomputed demo, not live inference.** Recommendations from my ranker for a random sample of "
    f"{customers['n_customers']} held-out customers; the full pipeline needs several GB of data, so it is not run in the browser. "
    f"[Code, method and limitations on GitHub]({REPO})."
)

tab_demo, tab_results, tab_how, tab_limits = st.tabs(["Recommendations", "Results", "How it works", "Limits and data"])

# ---------------------------------------------------------------- recommendations
with tab_demo:
    ids = [c["id"] for c in pool()]
    if st.session_state.get("customer_id") not in ids:
        st.session_state["customer_id"] = ids[0]
    c_prev, c_next, c_pick, c_filter = st.columns([1, 1, 5, 4], vertical_alignment="bottom")
    c_prev.button("Previous", on_click=step, args=(-1,), width="stretch")
    c_next.button("Next", on_click=step, args=(1,), width="stretch")
    by_id = {c["id"]: c for c in all_customers}
    c_pick.selectbox("Customer", ids, key="customer_id", format_func=lambda i: f"Customer {i}  ({by_id[i]['hits']} correct)")
    c_filter.checkbox("Only customers with a correct recommendation", key="only_hits")
    ch = by_id[st.session_state["customer_id"]]

    hit_rate = sum(c["hits"] > 0 for c in all_customers) / len(all_customers)
    verdict = f":green[**{ch['hits']} of 12 recommendations correct**]" if ch["hits"] else "**0 of 12 recommendations correct**"
    st.markdown(f"**Customer {ch['id']}**  ·  {ch['n_purchases']} purchases before the test week  ·  bought {ch['n_true']} article(s) that week  ·  {verdict}")
    st.caption(
        f"In this sample, {hit_rate:.0%} of customers have at least one correct recommendation. Most recommendations miss: next-week buying is hard to "
        "predict, and the full evaluation reports recall@12 of about 0.055."
    )

    st.subheader("Recently bought")
    row_of_cards([{"aid": h["article"], "date": h["date"]} for h in ch["history"]])

    st.subheader("Top 12 recommendations for the following week")
    st.caption("Badges show which candidate generator proposed each article before the ranker ordered them: "
               ":blue-badge[recency co-occ] customers with similar recent purchases, :orange-badge[trending] sold a lot in the last 14 days, "
               ":violet-badge[co-occ] bought together, :gray-badge[content] similar description.")
    recs = [{"aid": r["article"], "rank": r["rank"], "sources": r["sources"], "bought": r["bought"]} for r in ch["recs"]]
    for start in range(0, len(recs), PER_ROW):
        row_of_cards(recs[start:start + PER_ROW])

    st.subheader("What the customer actually bought that week")
    if ch["bought_next_week"]:
        row_of_cards([{"aid": a} for a in ch["bought_next_week"]])
        if ch["n_true"] > len(ch["bought_next_week"]):
            st.caption(f"Showing {len(ch['bought_next_week'])} of {ch['n_true']} articles.")
    else:
        st.caption("No purchases recorded.")

# ---------------------------------------------------------------- results
with tab_results:
    st.markdown(f"All numbers are recall@12 on the held-out final week unless stated. Full tables, confidence intervals and caveats are in the [README]({REPO}).")
    bar = st.column_config.ProgressColumn("recall@12", format="%.4f", min_value=0.0, max_value=0.07)
    num = st.column_config.NumberColumn("NDCG@12", format="%.4f")

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
    ], columns=["Model", "recall@12", "NDCG@12"]), hide_index=True, width="stretch",
        column_config={"recall@12": bar, "NDCG@12": num})

    st.subheader("Four consecutive test weeks: ranker vs best single generator")
    st.dataframe(pd.DataFrame([
        ("Final week", 0.0323, 0.0552, "+71%"), ("1 week earlier", 0.0333, 0.0522, "+57%"),
        ("2 weeks earlier", 0.0316, 0.0544, "+72%"), ("3 weeks earlier", 0.0285, 0.0500, "+76%"),
    ], columns=["Test week", "Best generator", "Ranker", "Relative gain"]), hide_index=True, width="stretch",
        column_config={"Best generator": st.column_config.NumberColumn(format="%.4f"), "Ranker": st.column_config.NumberColumn(format="%.4f")})
    st.caption("A separate ranker is trained for each week on earlier data only. Gains are relative, not percentage points.")

    st.subheader("New articles with no purchase history (simulation)")
    st.dataframe(pd.DataFrame([
        ("Random", 0.0073), ("Popularity", 0.0094), ("Item co-occurrence", 0.0090),
        ("Content similarity", 0.0532), ("Two-tower (content only for new items)", 0.0673),
    ], columns=["Model", "recall@12"]), hide_index=True, width="stretch", column_config={"recall@12": bar})
    st.caption("A single run in which 10% of the articles bought in the test week were hidden from the model; not a real product-launch study.")

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown(
        "**Task:** given everything a customer bought before week W, rank the articles they will buy for the first time in week W, and show 12.\n\n"
        "**Evaluation:** strictly time-based. A customer or article is eligible only if it has at least 5 purchases in the training period, "
        "so future purchases can never decide eligibility. Results are averaged over several training seeds, with bootstrap intervals."
    )
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eef2f6", color="#9aa5b1", fontsize=11];
          edge [color="#9aa5b1"];
          history [label="Customer purchase\\nhistory", fillcolor="#fff3d6", color="#e0b34a"];
          g1 [label="Recency-weighted\\nco-occurrence"]; g2 [label="Recent popularity\\n(last 14 days)"];
          g3 [label="Co-occurrence"]; g4 [label="Content similarity"];
          pool [label="~112 candidates\\nper customer"];
          rank [label="LightGBM ranker", fillcolor="#c9efe7", color="#0f766e"];
          top [label="Top 12", fillcolor="#c9efe7", color="#0f766e"];
          history -> g1; history -> g2; history -> g3; history -> g4;
          g1 -> pool; g2 -> pool; g3 -> pool; g4 -> pool; pool -> rank -> top;
        }
        """
    )
    a, b, c = st.columns(3)
    with a.container(border=True, height=170):
        st.markdown("**1. Retrieve**")
        st.caption("Four simple generators each propose articles. They overlap very little, so together they cover more than any one of them.")
    with b.container(border=True, height=170):
        st.markdown("**2. Rank**")
        st.caption("A LightGBM model, trained on the previous week's real purchases, orders the candidates using generator scores plus article popularity, price, age and the customer's category history.")
    with c.container(border=True, height=170):
        st.markdown("**3. Check honestly**")
        st.caption("Each week is predicted only from earlier weeks. On identical candidates the ranker beats simple rank fusion and a linear model by a wide margin.")

# ---------------------------------------------------------------- limits
with tab_limits:
    st.warning("Read this before quoting any number: these are the main limits of the evaluation.")
    st.markdown(
        "- **Only customers who bought in the test week are scored** (about 10% of eligible customers), because recall is undefined for the rest.\n"
        "- **New-to-customer discovery:** articles a customer already bought are never recommended, so this is narrower than the original Kaggle task and not comparable to its leaderboard.\n"
        "- **Retrieval limits the system:** only about a quarter of customers have even one relevant article among the candidates, and a bigger candidate pool stops helping after roughly double the size.\n"
        "- **The two-tower neural model is not part of the headline gain.** It performs about as well as recent popularity alone, and I could not show a reliable improvement from adding it; its value is on brand-new articles.\n"
        "- **This page is precomputed** for a random sample; it does not demonstrate serving, latency, or an A/B test."
    )
    st.markdown(f"**Data:** [H&M Personalized Fashion Recommendations]({KAGGLE}) on Kaggle. Product images are small thumbnails from that dataset. Source code: [{REPO}]({REPO}).")

st.divider()
st.caption("Built by Rajesh. Data: H&M Personalized Fashion Recommendations (Kaggle).")
