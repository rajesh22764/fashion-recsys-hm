# fashion-recsys

A two-stage fashion recommender (candidate retrieval + learned ranking) on the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
data. Evaluated with a leak-free, time-based protocol, bootstrap confidence intervals, repeated training seeds, ablations, four rolling
test weeks, a chronology audit and a cold-start experiment.

**Headline results (predict the next week's purchases, recall@12, 40,188 test customers):**

- A LightGBM ranker over candidate generators reaches **0.0551 +/- 0.0004** (4 generators) and **0.0559 +/- 0.0004** (5 generators, mean +/- sd over
  5 training seeds). The best single generator reaches 0.0323 and popularity 0.0072. Uncertainty from which customers are in the test set is about
  +/-0.0018 (95% bootstrap), roughly four times the seed noise.
- Over four consecutive test weeks the ranker beats the best single generator every week, by 57% to 76% **relative** (mean 69%; not percentage points); mean recall@12 = 0.0529 +/- 0.0023 (std across weeks; this uses 3 seeds per fold, so the final-week value 0.0552 differs slightly from the 5-seed 0.0551 above).
- Learned ranking beats simple score combination on the same candidates by a wide margin: about +0.016 recall@12 over reciprocal-rank fusion
  and +0.014 over a normalised-score sum, far beyond the noise.
- Training the ranker on three earlier weeks instead of one adds a further +0.0024 (0.0576 vs 0.0552; one test week, 3 seeds each).
- For brand-new articles with no purchase history, content-based models reach recall@12 of 0.053 to 0.067, about 7x to 9x random. Popularity and
  co-occurrence are no better than random there.
- Honest negative result: the two-tower model is a decent retriever on its own (0.0276) and raises candidate recall, but I could not demonstrate a reliable gain in the final ranker (+0.0006, 95% CI includes zero). It is an experimental retriever with cold-start value, not part of the claimed improvement.

## Setup

```bash
pip install -r requirements.txt            # lightgbm and torch are also needed for the ranker and two-tower stages
# download transactions_train.csv, articles.csv (and customers.csv, unused) into data/hm/
PY=python bash scripts/run_pipeline.sh     # whole pipeline, main test week (about 30 min on a 10-core Mac)
PY=python bash scripts/run_pipeline.sh 2   # same, evaluation window moved 2 weeks earlier (no two-tower)
python -m src.audit data/hm                # chronology audit of all four folds
pytest
```

Candidate building (uses torch) and ranker training (LightGBM) run as separate processes on purpose: the two libraries bundle different OpenMP
runtimes and segfault when loaded together on macOS.

## Demo

`demo/` is a Streamlit app that shows the ranker's recommendations for a random sample of 60 customers from the held-out final week (their recent
purchases, the 12 recommended articles with the generator that proposed each, and what they actually bought), plus the results tables.
It runs on **precomputed** output, not live inference: the full pipeline needs several GB of transactions and minutes of compute. In the sample, most
customers have no correct recommendation, which matches the roughly 5% recall@12 reported below.

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
python scripts/build_demo.py data/hm      # rebuild the sample (needs the cached candidate tables and Kaggle credentials)
```

Product images are small thumbnails of the H&M dataset images (H&M Personalized Fashion Recommendations on Kaggle).

## Data and protocol

- Last 26 weeks of transactions; repeat purchases of the same article by the same customer are collapsed to the first.
- Rolling, time-based phases:
  - **Validation:** history = everything before the last two weeks; labels = the week before the last. Used for tuning the generators and for training the ranker.
  - **Test:** history = everything before the final week; labels = the final week.
- **Leak-free k-core:** a customer or article is eligible only if it has at least 5 purchases in the *training period* (everything before the validation week),
  applied iteratively. Eligibility is frozen at that cutoff, so future purchases can never make a customer or article eligible.
  Result: 414,131 customers and 33,801 articles. (An earlier version applied the filter to all 26 weeks before splitting; a regression test now guards against that.)
- Test customers: those who bought at least one eligible article in the final week (40,188, i.e. 9.7% of eligible customers). Every model is scored on the same customers.
- Metrics: recall@12, NDCG@12, catalog coverage. Articles a customer already bought are never recommended.
- Generator parameters (decay windows) were tuned on the validation week, not on the test week.

### Chronology audit (`python -m src.audit data/hm`)

Each rolling fold has its own k-core, candidates, features and ranker, trained only on information before its own test week. The audit asserts
that history always ends before the labels begin.

| Fold | Ranker trained on labels from | Test week | History for the test ends | Eligible customers | Customers with a test label |
|---|---|---|---|---|---|
| main | 9-15 Sep 2020 | 16-22 Sep | 15 Sep | 414,131 | 40,188 (9.7%) |
| 1 week earlier | 2-8 Sep | 9-15 Sep | 8 Sep | 410,823 | 41,714 (10.2%) |
| 2 weeks earlier | 26 Aug-1 Sep | 2-8 Sep | 1 Sep | 408,309 | 44,449 (10.9%) |
| 3 weeks earlier | 19-25 Aug | 26 Aug-1 Sep | 25 Aug | 409,580 | 47,225 (11.5%) |

## Models

1. **Baselines:** popularity, recent popularity (14 days), item-item co-occurrence (cosine), recency-weighted co-occurrence
   (exponential decay, 30 days, on the customer's history), content kNN (TF-IDF + SVD-128 over article name, attributes and description).
2. **Two-tower retrieval (PyTorch):** user tower = last 20 purchases (mean of item vectors + last item, then an MLP; no user-id embedding).
   Item tower = id embedding + content embedding. In-batch sampled softmax with log-Q correction, id-embedding dropout (0.3), 3 epochs, trained on training weeks only.
3. **LightGBM ranker (lambdarank):** about 112 candidates per customer from four generators (136 with the two-tower generator added). One ranking group per customer; trained on 40,000 validation-phase
   customers using the next week's purchases as labels, with 20% of customers held out for early stopping. Features: generator scores, article popularity
   (all-time, 7d, 28d), price, article age, customer activity and recency, price ratio, affinity of the customer's history to the article's category, article attributes.
4. **Score-combination baselines** on the same candidates: reciprocal-rank fusion, normalised-score sum, logistic regression.

## Results: main test week (k=12)

Single trained models; bootstrap intervals reflect which customers are in the test set, not training randomness.

| Model | recall@12 [95% CI] | NDCG@12 [95% CI] | coverage |
|---|---|---|---|
| Popularity (all time) | 0.0072 | 0.0043 | 0.06% |
| Item co-occurrence | 0.0210 [0.0198, 0.0222] | 0.0123 [0.0116, 0.0131] | 59.7% |
| Content kNN | 0.0122 [0.0113, 0.0131] | 0.0074 [0.0068, 0.0080] | 88.7% |
| Recent popularity (14 days) | 0.0284 [0.0270, 0.0297] | 0.0170 [0.0162, 0.0178] | 0.05% |
| Two-tower | 0.0276 [0.0264, 0.0290] | 0.0170 [0.0161, 0.0177] | 31.5% |
| Recency-weighted co-occurrence | 0.0323 [0.0307, 0.0336] | 0.0199 [0.0190, 0.0208] | 76.9% |
| Reciprocal-rank fusion (no learning) | 0.0398 [0.0384, 0.0414] | 0.0248 [0.0237, 0.0258] | n/a |
| Ranker, 4 generators (5-seed mean) | 0.0551 +/- 0.0004 | 0.0349 +/- 0.0003 | about 16% |
| **Ranker, 5 generators (5-seed mean)** | **0.0559 +/- 0.0004** | **0.0350 +/- 0.0002** | about 17% |

A note on noise: one early single run of the 4-generator ranker scored 0.0532 because early stopping halted after 41 trees (typical runs use about 120);
it lies below the lowest of five seeds (0.0546). Repeated seeds are therefore used for every ranker number that is compared. A difference is only claimed
when its paired customer-level bootstrap interval excludes zero and it exceeds the seed spread; overlapping separate intervals are never read as a significance test.

What these show:

- **Recency is the biggest single lever.** Weighting a customer's history by recency lifts co-occurrence from 0.0210 to 0.0323; "what sold in the last 14 days" alone beats plain co-occurrence.
- **Retrieval limits the ranker.** The five generators put at least one true article in the candidates for 29.4% of customers (hit rate), so about 70.6% of
  evaluated customers have no relevant article in the pool at all, and they recover on average 15.3% of each customer's true articles (candidate recall). Among customers where retrieval succeeds, the ranker reaches recall@12 of about 0.19. Better candidates are the next improvement, not a bigger ranker.
- **The ranker trades coverage for accuracy**: its lists concentrate on recent and popular articles (about 17% of the catalog) versus 77% for recency co-occurrence.

## Scorers on the same candidate pool

All five scorers rank the same 5-generator candidates; each learned model is trained on the validation phase and evaluated on the test week (single runs).

| Scorer | recall@12 [95% CI] | NDCG@12 [95% CI] |
|---|---|---|
| Reciprocal-rank fusion | 0.0398 [0.0381, 0.0416] | 0.0248 [0.0237, 0.0258] |
| Normalised-score sum (equal weights) | 0.0421 [0.0404, 0.0438] | 0.0256 [0.0246, 0.0267] |
| Logistic regression (numeric features only) | 0.0501 [0.0484, 0.0519] | 0.0310 [0.0299, 0.0321] |
| LightGBM, binary objective | 0.0570 [0.0552, 0.0588] | 0.0351 [0.0339, 0.0363] |
| LightGBM, LambdaRank | 0.0567 [0.0549, 0.0587] | 0.0351 [0.0338, 0.0362] |

- The learned gradient-boosted model beats simple combination clearly (LambdaRank minus normalised-score sum: +0.0146 [0.0131, 0.0162]).
- **LambdaRank is not better than a plain binary objective here** (difference -0.0003 [-0.0015, 0.0009]; neither is a winner). I used LambdaRank because the task is to order
  candidates per customer, but I do not claim it is empirically superior. The gain comes from the boosted-tree model on rich features, not the ranking loss.
- The logistic regression omits the categorical article attributes, so its gap to LightGBM mixes "nonlinear" with "more features".

## Retrieval analysis: what each generator contributes, and does a bigger pool help? (4 generators, test week)

Generator contributions on the default pool (112 candidates per customer, candidate recall 0.1305, on average 1.16 generators propose each candidate):

| Generator | Candidates per customer | Recall of its own candidates | Unique candidates per customer | Candidate recall without it |
|---|---|---|---|---|
| Recency co-occurrence | 60 | 0.0740 | 43.7 | 0.0941 |
| Co-occurrence | 20 | 0.0295 | 7.7 | 0.1268 |
| Recent popularity (14 days) | 30 | 0.0573 | 28.1 | 0.0827 |
| Content kNN | 20 | 0.0169 | 16.3 | 0.1258 |

Overlap between candidate sets (Jaccard) is low: 0.02 between recency co-occurrence and recent popularity, at most 0.18 anywhere. The generators are complementary.
Recent popularity is the most valuable single source even though it is not personalised: removing it cuts candidate recall from 0.1305 to 0.0827 (fashion demand
follows what is trending now). These are candidate-recall effects only; the ranker was not retrained without each generator.

Candidate budget versus final quality (same ranking procedure, candidate quotas scaled together; mean over seeds):

| Pool scale | Candidates per customer | Candidate recall | Customers with at least one hit | Ranker recall@12 | Ranker NDCG@12 | Seeds |
|---|---|---|---|---|---|---|
| x0.5 | 57 | 0.0905 | 18.1% | 0.0528 +/- 0.0005 | 0.0334 | 3 |
| x1 (default) | 112 | 0.1305 | 25.6% | 0.0552 +/- 0.0005 | 0.0350 | 3 |
| x2 | 221 | 0.1877 | 35.0% | 0.0565 +/- 0.0002 | 0.0353 | 3 |
| x3 | 328 | 0.2285 | 41.1% | 0.0563 +/- 0.0012 | 0.0348 | 2 |

Candidate recall keeps growing with the pool, but final recall flattens: doubling the pool from the default adds +0.0013 (about 2% relative) and a third
copy adds nothing, so beyond roughly x2 the ranker, not retrieval volume, is the bottleneck. Even a 328-candidate pool contains no relevant article for 59% of
customers, so a large share of next week's purchases is not predictable from purchase history alone. Pool sizes up to x2 are within the 112-221 range an
ordinary production system can afford; x3 uses only two seeds.

## Results: four rolling test weeks (ranker over 4 generators, no two-tower, mean of 3 seeds per fold)

Generator parameters were fixed from the main validation week.

| Fold (test week) | Best single generator | Ranker recall@12 | Ranker NDCG@12 | Relative gain |
|---|---|---|---|---|
| final week | recency co-occurrence 0.0323 | 0.0552 +/- 0.0005 | 0.0350 | +71% |
| 1 week earlier | recency co-occurrence 0.0333 | 0.0522 +/- 0.0002 | 0.0329 | +57% |
| 2 weeks earlier | recent popularity 0.0316 | 0.0544 +/- 0.0007 | 0.0338 | +72% |
| 3 weeks earlier | recency co-occurrence 0.0285 | 0.0500 +/- 0.0006 | 0.0309 | +76% |
| **mean +/- std across weeks** | 0.0314 +/- 0.0021 | **0.0529 +/- 0.0023** | 0.0331 +/- 0.0017 | **+69%** |

The best single generator changes between weeks (recent popularity wins in one), which is a reason to learn a combination rather than hand-pick one rule.
The four weeks overlap in history and share market conditions, so they are not four independent experiments.

## Is the two-tower model useful in the final ranker? (2x2, mean of 3 seeds)

| Two-tower used as | recall@12 | NDCG@12 | recall@12 minus baseline [95% CI] |
|---|---|---|---|
| Nowhere (4-generator pool) | 0.0552 +/- 0.0005 | 0.0350 | - |
| Ranking feature only | 0.0552 +/- 0.0013 | 0.0344 | -0.0000 [-0.0009, 0.0008] |
| Candidate source only | 0.0557 +/- 0.0023 | 0.0351 | +0.0005 [-0.0003, 0.0012] |
| Candidate source and feature | 0.0558 +/- 0.0004 | 0.0349 | +0.0006 [-0.0003, 0.0015] |

Although the two-tower generator raises candidate recall from 0.130 to 0.153, the final ranker's gain is at most about +0.0006 and indistinguishable
from zero. An earlier single-run comparison suggested +0.0035; that came from an unlucky baseline run and is withdrawn. The two-tower model's
real contribution here is cold start.

## Training the ranker on several earlier weeks (4-generator pool; test = final week)

| Training data | recall@12 | NDCG@12 |
|---|---|---|
| 1 snapshot (W-1), current setup | 0.0552 +/- 0.0005 | 0.0350 +/- 0.0003 |
| 3 snapshots (W-1 to W-3) | **0.0576 +/- 0.0002** | **0.0365 +/- 0.0001** |
| 4 snapshots (W-1 to W-4), single run | 0.0569 | 0.0365 |

Every snapshot's features use only history before its own label week, and all training weeks precede the test week. Snapshots come from different
folds (each with its own k-core population and 40,000 sampled customers); the same customer can appear in several snapshots as separate query groups,
so training rows are not independent, and early stopping (random held-out customers) is slightly optimistic as a result (test results are unaffected).
Category codes come from a fixed vocabulary over the whole articles table (static attributes, identical in every fold). Three snapshots add +0.0024 over
one (about five times the seed noise); the 4-snapshot figure is a single run, so no conclusion about a fourth week can be drawn. One test week only;
the main pipeline still uses one snapshot.

## Ablation: which feature groups matter (main week, single run, retrained without one group)

Candidate pool fixed (five generators); faster ranker settings. Given the seed noise above, only large differences are meaningful.

| Ranker | recall@12 | NDCG@12 |
|---|---|---|
| all features | 0.0564 | 0.0350 |
| without item stats (popularity, price, age) | 0.0497 | 0.0309 |
| without customer stats | 0.0535 | 0.0332 |
| without article attributes | 0.0536 | 0.0333 |
| without content kNN | 0.0548 | 0.0346 |
| without recent popularity | 0.0552 | 0.0343 |
| without co-occurrence | 0.0553 | 0.0345 |
| without recency co-occurrence | 0.0554 | 0.0351 |
| without two-tower | 0.0554 | 0.0346 |
| without customer-article affinity | 0.0557 | 0.0344 |

Article-level statistics matter most (-0.0067). Individual generator scores cost only about 0.001 each when removed, because the others cover for them.

## Cold start: recommending articles with no purchase history

Hide a random 10% of the articles bought in the test week (1,346 of 13,460), delete every purchase of them from the history, retrain the two-tower model
without them, then rank only those 1,346 new articles for the 8,760 customers who bought one.

| Model | recall@12 | NDCG@12 |
|---|---|---|
| Random | 0.0073 | 0.0032 |
| Popularity | 0.0094 | 0.0043 |
| Item co-occurrence | 0.0090 | 0.0039 |
| Content kNN | 0.0532 | 0.0268 |
| Two-tower (id embedding switched off for new items) | **0.0673** | **0.0344** |

Models that depend on purchase history cannot score a new article and fall back to random. Content features carry the signal. How the two-tower model scores a
new article: the item vector is LayerNorm and a linear layer applied to (id embedding + projected content embedding). During training the id term is randomly
zeroed for 30% of examples, so the content-only path is trained, not just a fallback; for the hidden articles the id term is switched off at inference. This is a simulation of unseen article ids (one run, one random choice of hidden articles, no confidence
intervals; expected random recall is about 0.0089), not a real new-product-launch study, and it is evaluated only on customers who bought a hidden article.

## Effect of the k-core fix

Under the original (leaky) filter the single-run ranker scored 0.0496 versus 0.0299 for the best generator (1.66x). Under the leak-free filter it scores
0.0559 versus 0.0323 (1.73x, seed-averaged). Absolute numbers are not comparable across the two protocols: leak-free eligible customers all have at least
5 training purchases, so they are heavier shoppers and easier to predict, and the test set changed (49,088 -> 40,188 customers). The relative conclusion did not change.

## What I tried and what did not work

- **Amazon Reviews 2023 Fashion first.** Too sparse for personalisation (median 1 review per user; 5-core leaves 0 rows). Switched to H&M.
- **Content kNN as a warm-customer recommender** (0.0122) loses to behaviour-based models; its value is cold start and as ranker features.
- **Two-tower** needed id dropout and log-Q correction to be competitive and still only ties recent popularity on its own; it adds no reliable gain to the ranker.
- **LambdaRank vs a binary objective:** no measurable difference on this data.
- **Single-run comparisons misled me once** (the early-stopped baseline above), which is why seeds are now reported.
- **Bugs caught along the way:** a time-unit mistake (microseconds read as milliseconds) emptied the training set in the weekly split; a torch/LightGBM OpenMP segfault
  forced the two-process design; the k-core leak was found in review and fixed with a regression test.

## Limitations

- Warm evaluation only, and only customers who bought in the test week (about 10% of eligible customers each week; recall is undefined for the others).
  All models share the same customers, so comparisons are fair, but absolute numbers are not population-level.
- Not comparable to the Kaggle leaderboard (MAP@12, repeat purchases allowed, all customers scored).
- Repeat purchases are collapsed and already-bought articles excluded: this is a new-to-customer product discovery task, narrower than general purchase prediction.
- Seeds vary the early-stopping holdout and LightGBM bagging only; the test set, candidates and features are fixed. The bootstrap intervals cover only test-customer sampling.
- Rolling weeks, ablations and the multi-week training result are limited (rolling weeks exclude the two-tower generator; ablations and the cold-start experiment are single runs; multi-week training is evaluated on one test week).
- Ranker hyperparameters were not searched; generator decay parameters were tuned on one week.
- `customers.csv` and article images are not used.
- Sample recommendations (`src.demo`) look sensible, but at about 5% recall@12 most customers buy none of the 12 suggestions the next week.

## Layout

```
src/data.py        splits, leak-free k-core, sparse matrices      src/ranker.py      candidates + features
src/hm.py          H&M transaction loader                         src/run_ranker.py  ranker training (LambdaRank, binary)
src/baselines.py   popularity, co-occurrence, recency variants     src/twotower.py    two-tower model
src/content.py     TF-IDF/SVD article embeddings, stable codes     src/cold_start.py  cold-start experiment
src/phases.py      validation/test phases                         src/demo.py        readable recommendations
src/audit.py       chronology audit                               src/analyze.py     CIs, fusion/scorer baselines, ablations
src/two_by_two.py  two-tower candidate vs feature                  src/multisnap.py   multi-week ranker training
src/retrieval_analysis.py   generator contributions and candidate-budget curve
src/seed_variance.py, src/rolling_seeds.py   noise floor and seeded rolling weeks
src/add_tt_feature.py   scripts/run_pipeline.sh   scripts/build_demo.py   demo/ (Streamlit app)   tests/
```
