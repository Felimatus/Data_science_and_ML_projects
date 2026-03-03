"""
Train model and predict next grocery order categories.

Unlike score.py this script has no test set and always shows and saves the predicted results.
"""

import sys
import time
from pathlib import Path

# -- add project root to Python's import search path so 'import config' works
#    regardless of which directory the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from ensure_data_files import ensure_data_files

import numpy as np                                         
import pandas as pd                                         
import xgboost as xgb                                       # -- ML algorith: gradient-boosted decision tree classifier
from scipy.stats import randint, uniform                    # -- probability distributions for random hyperparameter search
from sklearn.metrics import f1_score, make_scorer          # -- f1_score for evaluation; make_scorer wraps it for use in CV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit  # -- random search over hyperparameters; time-series-aware CV (no future leakage)
from sklearn.multioutput import MultiOutputClassifier       # -- trains one binary classifier per category


def main(val_size: float = 0.2, # fraction of data used for validation (default: 0.2)
         verbosity: int = 0,    # XGBoost verbosity: 0=silent, 1=warnings (default: 0)
         n_to_show: int = None, # number of categories to show; auto if omitted (default: None)
         update: bool = False,  # if True, regenerate training files even if they already exist (default: False)
         delete_not_found: bool = False): # if True, drop order rows for products not found in the API instead of substituting a placeholder category (default: False)
    _results_dir = config.DATA_DIR / "results"
    _results_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    # validate orders folder and (re)generate training CSVs; also checks for new product categories if update=True
    ensure_data_files(update=update, delete_not_found=delete_not_found) 

    X_all  = pd.read_csv(config.TRAINING_DIR / "orders_rohlik.csv")
    basket = pd.read_csv(config.TRAINING_DIR / "basket.csv", index_col=0)

    # ── 1. One date per order, sorted chronologically ─────────────────────────
    order_dates   = X_all.groupby("order_id")["date"].first().sort_index()
    basket_sorted = basket.loc[order_dates.index]
    dates         = pd.to_datetime(order_dates)

    # ── 2. Time features ───────────────────────────────────────────────────────
    time_feat = pd.DataFrame({
        "month":                 dates.dt.month,
        "day_of_week":           dates.dt.dayofweek,
        "days_since_last_order": dates.diff().dt.days.fillna(0).astype(int),
    }, index=basket_sorted.index)

    # ── 3. Lag features: amounts bought 1 and 2 orders ago ────────────────────
    lag1 = basket_sorted.shift(1).fillna(0).astype(int)
    lag1.columns = [f"{c}_lag1" for c in basket_sorted.columns]

    lag2 = basket_sorted.shift(2).fillna(0).astype(int)
    lag2.columns = [f"{c}_lag2" for c in basket_sorted.columns]

    # ── 4. Feature matrix and binary target ───────────────────────────────────
    all_features  = pd.concat([time_feat, lag1, lag2], axis=1)
    target_binary = (basket_sorted.shift(-1) > 0).astype(int)

    valid = basket_sorted.index[2:-1]
    X_ml  = all_features.loc[valid]
    y_ml  = target_binary.loc[valid]

    print(f"Feature matrix : {X_ml.shape}  (orders × features)")
    print(f"Target matrix  : {y_ml.shape}  (orders × categories)")

    # ── 5. Train / val split (no test set) ────────────────────────────────────
    N_VAL = max(1, int(len(X_ml) * val_size)) # fraction of all data; score.py uses (len - N_TEST) because it has a test set

    X_val_s   = X_ml.iloc[-N_VAL:].values
    y_val_s   = y_ml.iloc[-N_VAL:].values
    X_train_s = X_ml.iloc[:-N_VAL].values
    y_train_s = y_ml.iloc[:-N_VAL].values
    print(f"\nvalidation orders = {N_VAL} | training orders = {len(X_train_s)}")

    # ── Stage 1: RandomizedSearchCV on top-10% most purchased categories ─────────
    TOP_K     = max(1, round(y_ml.shape[1] * 0.10))  # ~10% of all categories
    top_K_idx = y_ml.sum().nlargest(TOP_K).index
    top_K_pos = [list(y_ml.columns).index(c) for c in top_K_idx]

    y_train_topK = y_train_s[:, top_K_pos]

    param_dist = {
        "estimator__max_depth":        randint(3, 8),
        "estimator__learning_rate":    uniform(0.01, 0.2),
        "estimator__subsample":        uniform(0.6, 0.4),
        "estimator__colsample_bytree": uniform(0.5, 0.5),
        "estimator__min_child_weight": randint(1, 6),
    }
    f1_scorer2 = make_scorer(f1_score, average="weighted", zero_division=0)
    search = RandomizedSearchCV(
        MultiOutputClassifier(
            xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              tree_method="hist", random_state=42, n_jobs=-1)
            # `tree_method="hist"` is added to avoid "XGBoostError: [14] Out of memory".
            # This error crashes the program completely, so it's better to use a faster,
            # more memory-efficient tree method (at the cost of slightly worse performance).
        ),
        param_distributions=param_dist,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=5),
        scoring=f1_scorer2,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train_s, y_train_topK)

    best_params = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
    best_params.pop("n_estimators", None)   # Stage 2 uses 1000 + early stopping
    print(f"\nBest hyperparameters: {best_params}")

    # ── Stage 2: Train all categories with best params + early stopping ────────
    N_CATS = y_ml.shape[1]
    models = []

    for i in range(N_CATS):
        clf = xgb.XGBClassifier(
            **best_params,
            n_estimators=1000,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            early_stopping_rounds=20,
            verbosity=verbosity,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(
            X_train_s, y_train_s[:, i],
            eval_set=[(X_val_s, y_val_s[:, i])],
            verbose=False,  # -- suppress per-round logloss output to stdout
        )
        models.append(clf)

    print(f"Trained {N_CATS} models.")

    # ── Threshold tuning on val set ────────────────────────────────────────────
    proba_val = np.column_stack([m.predict_proba(X_val_s)[:, 1] for m in models])

    avg_bought = y_val_s.sum(axis=1).mean()

    best_thresh, best_diff = 0.5, float("inf") # initialize with default threshold of 0.5
    for t in np.arange(0.05, 0.50, 0.01):
        avg_pred = (proba_val >= t).sum(axis=1).mean()
        diff = abs(avg_pred - avg_bought)
        if diff < best_diff:
            best_diff, best_thresh = diff, t 
        # update best threshold if this one is closer to the average number of categories actually bought

    print(f"Avg categories actually bought (val) : {avg_bought:.1f}")
    print(f"Best threshold : {best_thresh:.2f}  (avg predicted = {(proba_val >= best_thresh).sum(axis=1).mean():.1f})")

    # ── Predict next order ─────────────────────────────────────────────────────
    df_pc = pd.read_csv(config.CAT_DIR / "product_categories.csv")

    today           = pd.Timestamp.today(tz="UTC").normalize()
    last_order_date = dates.iloc[-1]
    days_since      = max(0, (today - last_order_date).days)
    # for the next order prediction, we use the same features as for training: 
    # month, day_of_week, days_since_last_order, and the amounts bought in the last 2 orders (lag1 and lag2)
    X_next = np.array([[ 
        today.month,
        today.dayofweek,
        days_since,
        *basket_sorted.iloc[-1].values,
        *basket_sorted.iloc[-2].values,
    ]])
    # predict probabilities for all categories, then apply the best threshold to get the predicted categories for the next order
    proba_next = np.array([m.predict_proba(X_next)[0, 1] for m in models]) 
    # predict_proba returns an array of shape (n_samples, 2) with probabilities for classes [0, 1].
    # We take the probability of class 1 (bought) for each category
    pred_mask  = proba_next >= best_thresh
    pred_cats  = basket_sorted.columns[pred_mask]
    pred_proba = proba_next[pred_mask]

    order             = np.argsort(pred_proba)[::-1]
    pred_cats_sorted  = pred_cats[order]
    pred_proba_sorted = pred_proba[order]

    if n_to_show is None:
        n_to_show = int(round(basket_sorted.gt(0).sum(axis=1).mean()))
    # clean product categories data for display: drop rows with missing mainCategoryId and convert
    df_pc_clean = (
        df_pc
        .dropna(subset=["mainCategoryId"])
        .assign(mainCategoryId=lambda d: d["mainCategoryId"].astype(int))
    ) 
    cat_products    = df_pc_clean.groupby("mainCategoryId")["name"].apply(list).to_dict()
    cat_product_ids = df_pc_clean.groupby("mainCategoryId")["id"].apply(list).to_dict()

    header_lines = [
        f"\nLast order : {last_order_date.date()}  ({days_since} days ago)",
        f"Showing top {n_to_show} categories (your avg per order) out of {len(pred_cats_sorted)} predicted\n",
        f"{'Prob':>5}  I should buy",
        "─" * 72,
    ]
    for line in header_lines:
        print(line)

    rows_txt = []
    rows_csv = []
    for cat, prob in zip(pred_cats_sorted[:n_to_show], pred_proba_sorted[:n_to_show]):
        cat_int = int(cat)
        names   = (cat_products.get(cat_int,    []) + [""] * 3)[:3]
        ids     = (cat_product_ids.get(cat_int, []) + [None] * 3)[:3]

        # Terminal: names only, no IDs
        print(f" {prob:.2f}  {' / '.join(n for n in names if n)}")

        # Files: names + IDs
        parts_with_ids = [f"{n} [{i}]" if i is not None else n for n, i in zip(names, ids) if n]
        rows_txt.append(f" {prob:.2f}  {' / '.join(parts_with_ids)}")
        rows_csv.append({
            "mainCategoryId": cat_int,
            "example_1": names[0], "example_2": names[1], "example_3": names[2],
            "id_product_1": ids[0], "id_product_2": ids[1], "id_product_3": ids[2],
        })

    # ── Save products_to_buy.txt ───────────────────────────────────────────────
    with open(_results_dir / "products_to_buy.txt", "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        for line in rows_txt:
            f.write(line + "\n")
    print(f"\nSaved products_to_buy.txt → {_results_dir}")

    # ── Save products_to_buy.csv ───────────────────────────────────────────────
    pd.DataFrame(rows_csv).to_csv(_results_dir / "products_to_buy.csv", index=False)
    print(f"Saved products_to_buy.csv  → {_results_dir}")

    end = time.perf_counter()
    print(f"\nElapsed: {end - start:.3f} seconds")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict next grocery order categories.")
    parser.add_argument("--val-size",  type=float, default=0.2,
                        help="Fraction of data used for validation (default: 0.2)")
    parser.add_argument("--verbosity", type=int,   default=0,
                        help="XGBoost verbosity: 0=silent, 1=warnings (default: 0)")
    parser.add_argument("--n-to-show", type=int,   default=None,
                        help="Number of categories to show; auto if omitted")
    parser.add_argument("--update",          action="store_true", default=False,
                        help="Regenerate training files even if they already exist (default: False)")
    parser.add_argument("--delete-not-found", action="store_true", default=False,
                        help="Drop order rows for products not found in the API instead of substituting a placeholder category (default: False)")
    args = parser.parse_args()
    main(
        val_size=args.val_size,
        verbosity=args.verbosity,
        n_to_show=args.n_to_show,
        update=args.update,
        delete_not_found=args.delete_not_found,
    )
