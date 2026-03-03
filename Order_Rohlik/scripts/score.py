import sys
import time
from pathlib import Path

# -- add project root to Python's import search path so 'import config' works
#    regardless of which directory the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from ensure_data_files import ensure_data_files

import numpy as np                                          # -- numerical arrays; used for stacking model predictions
import pandas as pd                                         # -- DataFrames for orders and basket matrix
import xgboost as xgb                                       # -- gradient-boosted decision tree classifier
from scipy.stats import randint, uniform                    # -- probability distributions for random hyperparameter search
from sklearn.metrics import f1_score, make_scorer          # -- f1_score for evaluation; make_scorer wraps it for use in CV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit  # -- random search over hyperparameters; time-series-aware CV (no future leakage)
from sklearn.multioutput import MultiOutputClassifier       # -- trains one binary classifier per category

class _Tee: # similar to Unix tee command. "_" in _TEE means "private to this module, not meant to be imported".
    """Write to both the real stdout/stderr and a log file simultaneously."""
    class _Stream:
        """Forwards one stream (stdout or stderr) to both its original destination and the shared log file.
        If stderr is not needed, delete_Stream and modify _Tee to only wrap stdout."""
        def __init__(self, original, file):
            self._original = original
            self._file     = file
        def write(self, data):  # write to both the terminal and the file
            self._original.write(data)
            self._file.write(data)
        def flush(self): # ensure that output is written to both places immediately
            self._original.flush() # push buffered data to the terminal
            self._file.flush()     # push buffered data to score_details.txt

    def __init__(self, path: Path): # path to the log file (e.g., score_details.txt)
        self._file        = open(path, "w", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self._Stream(self._orig_stdout, self._file)
        sys.stderr = self._Stream(self._orig_stderr, self._file)

    def close(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._file.close()


def main(show_results: bool = False,
         test_size: float = 0.1,
         val_size: float = 0.2,
         verbosity: int = 0,
         n_to_show: int = None,
         update: bool = False,
         delete_not_found: bool = False):
    _score_dir = config.DATA_DIR / "results" / "score"
    _score_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(_score_dir / "score_details.txt")
    start = time.perf_counter()

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

    # ── 5. Splits: train | val | test  (always chronological) ─────────────────
    N_UNIQUE_DATES = dates.dt.date.nunique()
    N_TEST         = int(N_UNIQUE_DATES * test_size)
    N_VAL          = max(1, int((len(X_ml) - N_TEST) * val_size)) 

    X_test_s  = X_ml.iloc[-N_TEST:].values
    y_test_s  = y_ml.iloc[-N_TEST:].values

    X_trainval = X_ml.iloc[:-N_TEST].values
    y_trainval = y_ml.iloc[:-N_TEST].values

    X_val_s   = X_trainval[-N_VAL:]
    y_val_s   = y_trainval[-N_VAL:]
    X_train_s = X_trainval[:-N_VAL]
    y_train_s = y_trainval[:-N_VAL]

    print(f"\nUnique dates : {N_UNIQUE_DATES}  →  test orders = {N_TEST} | validation orders = {N_VAL} | training orders = {len(X_train_s)}")

    # ── Stage 1: RandomSearchCV on top-10% most purchased categories ────────────
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
        n_iter=20, #number of candidates
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
            verbose=False,
        )
        models.append(clf)

    print(f"Trained {N_CATS} models.")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = np.column_stack([m.predict(X_test_s) for m in models])

    f1_samples = f1_score(y_test_s, y_pred, average="samples", zero_division=0)

    intersection = (y_pred & y_test_s).sum(axis=1)
    union        = (y_pred | y_test_s).sum(axis=1)
    jaccard      = np.where(union > 0, intersection / union, 1.0) #Jaccard index. min = 0, max = 1.

    print(f"\nSample-averaged F1  : {f1_samples:.3f}")
    print(f"Mean Jaccard        : {jaccard.mean():.3f}")
    print(f"\nCategories predicted per order (avg) : {y_pred.sum(axis=1).mean():.1f}")
    print(f"Categories actually bought (avg)     : {y_test_s.sum(axis=1).mean():.1f}")

    # ── Threshold tuning ───────────────────────────────────────────────────────
    proba_val  = np.column_stack([m.predict_proba(X_val_s)[:, 1] for m in models])
    proba_test = np.column_stack([m.predict_proba(X_test_s)[:, 1] for m in models])

    avg_bought = y_val_s.sum(axis=1).mean()

    best_thresh, best_diff = 0.5, float("inf")
    for t in np.arange(0.05, 0.50, 0.01):
        avg_pred = (proba_val >= t).sum(axis=1).mean()
        diff = abs(avg_pred - avg_bought)
        if diff < best_diff:
            best_diff, best_thresh = diff, t

    print(f"Avg categories actually bought (val) : {avg_bought:.1f}")
    print(f"Best threshold : {best_thresh:.2f}  (avg predicted = {(proba_val >= best_thresh).sum(axis=1).mean():.1f})")

    # ── Re-evaluate on test with tuned threshold ───────────────────────────────
    y_pred_tuned = (proba_test >= best_thresh).astype(int)

    f1_tuned = f1_score(y_test_s, y_pred_tuned, average="samples", zero_division=0)

    intersection = (y_pred_tuned & y_test_s).sum(axis=1)
    union        = (y_pred_tuned | y_test_s).sum(axis=1)
    jaccard      = np.where(union > 0, intersection / union, 1.0) #Jaccard index. min = 0, max = 1. 
    #Orders with no categories bought or predicted are counted as perfect matches (1.0) rather than being excluded 
    # or causing division by zero. This is a common approach for multi-label Jaccard when empty sets can occur. 
    # It rewards the model for correctly predicting that no categories will be bought in those cases, 
    # which is better than treating them as undefined or as mismatches. 
    # The average Jaccard across all orders then reflects the overall quality of the predictions, 
    # including the ability to predict "no purchase" when appropriate.

    print(f"\nAfter threshold tuning:")
    print(f"  Sample-averaged F1  : {f1_tuned:.3f}")
    print(f"  Mean Jaccard        : {jaccard.mean():.3f}")
    print(f"\n  Categories predicted per order (avg) : {y_pred_tuned.sum(axis=1).mean():.1f}")
    print(f"  Categories actually bought (avg)     : {y_test_s.sum(axis=1).mean():.1f}")

    # ── Show predicted next order (optional) ──────────────────────────────────
    if show_results:
        df_pc = pd.read_csv(config.CAT_DIR / "product_categories.csv")

        today           = pd.Timestamp.today(tz="UTC").normalize()
        last_order_date = dates.iloc[-1]
        days_since      = max(0, (today - last_order_date).days)

        X_next = np.array([[
            today.month,
            today.dayofweek,
            days_since,
            *basket_sorted.iloc[-1].values,
            *basket_sorted.iloc[-2].values,
        ]])

        proba_next = np.array([m.predict_proba(X_next)[0, 1] for m in models])
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

        _results_dir = config.DATA_DIR / "results" / "score"
        _results_dir.mkdir(parents=True, exist_ok=True)

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

            # Terminal (and score_details.txt): names only, no IDs
            print(f" {prob:.2f}  {' / '.join(n for n in names if n)}")

            # Files: names + IDs
            parts_with_ids = [f"{n} [{i}]" if i is not None else n for n, i in zip(names, ids) if n]
            rows_txt.append(f" {prob:.2f}  {' / '.join(parts_with_ids)}")
            rows_csv.append({
                "mainCategoryId": cat_int,
                "example_1": names[0], "example_2": names[1], "example_3": names[2],
                "id_product_1": ids[0], "id_product_2": ids[1], "id_product_3": ids[2],
            })

        # ── Save products_to_buy.txt ───────────────────────────────────────────
        with open(_results_dir / "products_to_buy.txt", "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            for line in rows_txt:
                f.write(line + "\n")
        print(f"\nSaved products_to_buy.txt → {_results_dir}")

        # ── Save products_to_buy.csv ───────────────────────────────────────────
        pd.DataFrame(rows_csv).to_csv(_results_dir / "products_to_buy.csv", index=False)
        print(f"Saved products_to_buy.csv  → {_results_dir}")

    end = time.perf_counter()
    print(f"\nElapsed: {end - start:.3f} seconds")
    tee.close()
    print(f"Log saved → {_score_dir / 'score_details.txt'}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score the grocery prediction model.")
    parser.add_argument("--show-results", action="store_true", default=False,
                        help="Also predict and save the next order (default: False)")
    parser.add_argument("--test-size",  type=float, default=0.1,
                        help="Fraction of unique dates used for test (default: 0.1)")
    parser.add_argument("--val-size",   type=float, default=0.2,
                        help="Fraction of train+val used for validation (default: 0.2)")
    parser.add_argument("--verbosity",  type=int,   default=0,
                        help="XGBoost verbosity: 0=silent, 1=warnings (default: 0)")
    parser.add_argument("--n-to-show",  type=int,   default=None,
                        help="Number of categories to show; auto if omitted")
    parser.add_argument("--update",          action="store_true", default=False,
                        help="Regenerate training files even if they already exist (default: False)")
    parser.add_argument("--delete-not-found", action="store_true", default=False,
                        help="Drop order rows for products not found in the API instead of substituting a placeholder category (default: False)")
    args = parser.parse_args()
    main(
        show_results=args.show_results,
        test_size=args.test_size,
        val_size=args.val_size,
        verbosity=args.verbosity,
        n_to_show=args.n_to_show,
        update=args.update,
        delete_not_found=args.delete_not_found,
    )
