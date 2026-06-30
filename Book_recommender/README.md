# Book Recommender — *"I like Lord of the Rings, what else should I read?"*

> **Note — this was a take-home task for a job application.** It comes in two parts: a **Data Scientist** exercise (build a book recommender from open data) and a **bonus Machine Learning Engineer** exercise (design how the recommender would be productionalized as an app). Both are addressed here. In the spirit of the task, the emphasis is on the reasoning and the journey — *own, transparent solutions over black-box libraries* — rather than on squeezing out a leaderboard score.

---

## What it does

Given a **single book title** the user likes — no user account, no history, just the title — return a list of other books they are likely to enjoy. This is an **item-seeded** task (the seed is a *book*), which is different from classic personalised user-history recommendation, and shapes every design choice below.

```python
from hybrid_model import load_recommender
rec = load_recommender()
rec("Lord of the Rings")        # -> DataFrame[ISBN, Title]
```

---

## Repository layout

```text
Book_recommender/
│
├── Data_Analysis_Book_Recommendation.ipynb  # TASK 1 (part A): data cleaning, EDA & enrichment
├── Exercise.ipynb                           # TASK 1 (part B): the 5 base models + hybrid + evaluation
├── hybrid_model.py                          # the FINAL model, packaged for serving (importable)
│
├── Architecture_APP/                        # TASK 2 (bonus): the productionalization design
│   ├── architecture.drawio                  #   editable source
│   └── architecture.svg                     #   rendered diagram
│
├── requirements.txt                         # pinned dependencies
│
├── archive/                                 # raw Book-Crossing dataset (Kaggle) + diagram assets
│   ├── Books.csv  Users.csv  Ratings.csv
│   └── *.png                                # taxonomy/illustration images used in the notebooks
│
├── cleaned_data/                            # output of the data-analysis notebook (model input)
│   ├── Books.csv    (~50k books, enriched with Subjects/Language/Year via OpenLibrary)
│   ├── Ratings.csv  (~284k explicit ratings, 1–10)
│   └── Users.csv
│
├── models/                                  # saved artifacts loaded by hybrid_model.py
│   ├── lgbm_recommender.pkl                 #   trained LightGBM model
│   ├── lgbm_artifacts.joblib                #   SVD factors, TF-IDF matrix, encoders, etc.
│   └── lgbm_best_params.json                #   tuned LightGBM hyperparameters
│
├── support/                                 # intermediate enrichment checkpoints
│   ├── enriched_books_checkpoint.csv        #   OpenLibrary metadata pull (resumable)
│   └── language_progress.csv               #   language-detection progress
│
└── img/                                     # pipeline diagrams embedded in Exercise.ipynb
    ├── reco_pipeline_new.svg
    └── filters_pipeline.svg
```

The work is split across **two notebooks purely for readability**: one for the data, one for the modelling. Together they constitute the Task 1 solution; the final model distilled from them lives in `hybrid_model.py`.

---

## Data

The [Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) (Kaggle): books, users, and ~1.1M explicit ratings (1–10). After cleaning and enrichment: **~50k books and ~284k ratings**.

---

## Task 1 — the recommender

### Part A — `Data_Analysis_Book_Recommendation.ipynb`

Loads, cleans, and **enriches** the raw data:

- **Books**: drop image columns; fix 3 rows where columns were shifted; repair mojibake encoding with `ftfy`; clean `Year` (drop `Year=0` / impossible years); drop books with `TotalRating < 2` and books with no ratings.
- **Enrichment via the OpenLibrary API**: pull `Subjects`, `Language`, `Publisher`, and corrected `Year` per ISBN (cached in `support/` so the slow network pull is resumable).
- **Users**: drop `Location`; keep only users who actually rated something.
  - An interesting detour: ~45% of users have missing/implausible **Age**. Rather than drop it, the notebook tries to *recover* age from rating behaviour (median age per author → median over a user's authors). It is then **validated against a "predict the global median age" baseline and rejected** — authors turn out to be poor age predictors (popular authors are read across all ages). A small, honest negative result, kept in for the reasoning.
- **Ratings**: drop duplicates, drop implicit `Rating=0` (missing, not a real score), drop orphans, compute `TotalRating` per book.
- **Language**: the `lingua` detector is tried but found unreliable (esp. Spanish/German), so OpenLibrary's language is preferred.

Output: the three files in `cleaned_data/`.

### Part B — `Exercise.ipynb` — five base models + a hybrid

A bare title gives no user profile, so the problem is bridged to the item-seeded task **two ways**: *item-to-item similarity* and *fan-based collaborative filtering* (find users who rated the query highly — its "fans" — and predict what else they'd like).

| # | Method | Idea |
|---|--------|------|
| 1 | **Item-Based CF** (cosine KNN) | Books co-rated by the same users. Adds **edition pooling** (merge all ISBNs of one work into a single work-level vector) and **significance weighting / shrinkage** (down-weight similarities backed by few shared raters). |
| 2 | **Content-Based** (TF-IDF) | Cosine similarity over Title + Author + Subjects + Publisher + Year. |
| 3 | **SVD latent factors** | Truncated SVD of the mean-centered rating matrix; reconstruct the query's fans' predicted ratings and recommend the highest. `K` chosen by held-out RMSE. |
| 4 | **User-Based CF** (KNNWithMeans, mean-centered cosine ≈ Pearson) | Per-fan nearest neighbours — preserves the diversity of different fans' tastes instead of collapsing them into one profile. |
| 5 | **LightGBM** (feature-based) | Gradient-boosted trees on SVD factors + light user stats. **Item popularity/mean deliberately excluded** to avoid the "predict ≈ item mean → recommend globally popular books to everyone" trap. Serving uses a fast **prototype-fan centroid** so it predicts once instead of per-fan. |
| 6 | **Hybrid — Reciprocal Rank Fusion (RRF)** | Combines all five. |

**Why RRF and not a weighted score blend?** The five methods produce nearly **disjoint** candidate lists on **incompatible score scales** (SVD ≈ 8, User-CF ≈ 12, Item-CF ≈ 1, Content ≈ 0.2). A weighted sum just interleaves separate lists. RRF scores each item by its **rank** in each list:

```
RRF(i) = Σ_method  w_method · 1 / (k + rank_method(i))
```

It is **scale-invariant** and **rewards agreement** — a book ranked mid-list by *two* methods beats a book ranked #1 by only one, so genuine cross-signal picks rise to the top.

**Output quality filters** (in `hybrid_model.py`): `filter_same_work` removes near-identical editions of the seed; `cap_per_family` diversifies by capping each series/author "family" (e.g. all of Harry Potter → 2), while keeping an author's *distinct* works (e.g. *The Hobbit* vs *The Silmarillion*). **Language scoping**: models train on the full multilingual catalogue (maximum signal) but recommendations default to **English** so a German edition is never returned for an English query.

---

## A note on the evaluation cells (and the unused weights)

The lower part of `Exercise.ipynb` contains a substantial **evaluation and optimization block** — hold-out RMSE, Precision@10 / Recall@10 / **NDCG@10**, K tuning for SVD / User-CF / Item-CF, and two *learned* hybrids (a LightGBM stacker and a logistic-regression stacker whose coefficients are read off as data-driven weights).

**These are for analysis/reference and are intentionally NOT wired into the final model.** Two reasons:

1. **The metric optimises the wrong thing.** RMSE and the rating-≥8 ranking metrics measure *rating prediction*, but the exercise's actual goal is *good recommendations*. The weights those procedures produce overfit to NDCG and don't obviously give better book suggestions. `hybrid_model.py` therefore ships **human-chosen weights** (`w_item=0.1, w_content=0.33, w_svd=0.22, w_user=0.10, w_lgbm=0.25`); the metric-optimal weights are left in the code as a commented alternative.
2. **The data is too sparse for rating prediction.** With a median of ~1 rating/user, once a mean-fallback gives full coverage every method collapses to ≈1.6 RMSE — *the predict-the-user's-mean baseline*. That null result is itself a key finding, and the reason the problem is framed as **ranking**, not rating prediction.

So: treat the bottom of the notebook as the "showing my work" section, not as the configuration of the shipped recommender.

---

## `hybrid_model.py` — the final, servable model

A self-contained module. On import it loads the saved artifacts (`models/`) and data (`cleaned_data/`), **rebuilds the cheap structures** (sparse matrices, KNN index, content list) **without retraining** SVD / TF-IDF / LightGBM, and exposes the recommender:

```python
from hybrid_model import load_recommender
rec = load_recommender()
rec("Lord of the Rings")      # -> DataFrame[ISBN, Title]
```

This is the seed of the production serving layer — a FastAPI app would simply `import` it and expose `rec(title)` over HTTP. (Run it from inside the `Book_recommender/` folder, since it loads `cleaned_data/` and `models/` via relative paths.)

---

## Task 2 (bonus) — productionalization architecture

`Architecture_APP/architecture.svg` (editable `.drawio` source alongside) is the answer to the ML-Engineer bonus: *how would this become a real app?* The design separates an **ONLINE** always-on serving zone from an **OFFLINE** scheduled pipeline:

- **Online**: API Gateway (auth, rate-limit, TLS) → optional A/B canary router → **FastAPI** service (`hybrid_model.py` loaded once at startup) backed by a **Redis recs cache** (`book_id → top-N`, write-back on miss, durable copy in S3), with Postgres for book metadata and event logging. Monitoring via CloudWatch / Prometheus + Grafana.
- **Offline** (Airflow / Databricks, nightly): ingest new books/ratings → retrain the 5 base models + tune weights → evaluate on a frozen test set (RMSE + NDCG@10) → **register in MLflow with a promote-only-if-better gate** → batch-score top-N for every book → atomically rebuild the Redis recs store.
- **Two independent update paths**: a **CI/CD code path** (rebuild & redeploy the container image only when *code* changes) kept separate from the **model-update path** (new artifacts → S3 + rebuilt Redis, **no redeploy**).
- **Future extensions** (dashed in the diagram): user profile store, online feature store, and a personalized user-seeded inference path; plus a Kafka/Kinesis event queue and a data warehouse for analytics.

It explicitly addresses the brief's "moving parts" prompts — *new books/ratings arrive* (offline ingest + nightly rebuild) and *a better model is trained* (MLflow gate + atomic recs-store swap, no API redeploy).

---

## Running it

```bash
pip install -r requirements.txt

# 1. (optional) regenerate cleaned_data/ — run Data_Analysis_Book_Recommendation.ipynb
# 2. (optional) retrain + re-save models/ — run Exercise.ipynb
# 3. serve / query the saved model:
python -c "from hybrid_model import load_recommender; print(load_recommender()('Lord of the Rings'))"
```

**Technologies:** Python, pandas, NumPy, scikit-learn, SciPy, LightGBM, ftfy, lingua, matplotlib, OpenLibrary API; FastAPI / Redis / S3 / MLflow / Airflow (architecture).
