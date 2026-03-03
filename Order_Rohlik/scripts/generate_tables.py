"""
Build X_all and basket from Rohlik order JSON files.

Saves:
  data/training/orders_rohlik.csv  – one row per order-line  (X_all)
  data/training/basket.csv         – pivot: order_id × mainCategoryId totals
"""

import glob          # -- find all *.json files matching a wildcard pattern
import json
import subprocess    # -- launch fetch_categories.py as a child process
import sys
from pathlib import Path

# -- add project root to Python's import search path so 'import config' works
#    regardless of which directory the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import pandas as pd


# ---------------------------------------------------------------------------
def load_orders(orders_dir: Path = config.ORDERS_DIR) -> pd.DataFrame:
    """Read all order JSON files and return a flat DataFrame (X_all)."""
    all_X = []

    for fp in glob.glob(str(orders_dir / "*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        orders = data if isinstance(data, list) else [data]  # support both single-order and array files
        for order in orders:
            if "items" not in order:
                continue

            meta = [k for k, v in order.items() if not isinstance(v, (list, dict))]
            items_df = pd.json_normalize(
                order,
                record_path="items",
                meta=meta,
                meta_prefix="order_",
                sep="_",
                errors="ignore",
            )

            for c in config.COLUMNS_TO_KEEP:
                if c not in items_df.columns:
                    items_df[c] = pd.NA

            X = items_df[config.COLUMNS_TO_KEEP].copy()
            X["date"] = pd.to_datetime(
                items_df["order_orderTime"],
                format="%Y-%m-%dT%H:%M:%S.%f%z",
                errors="coerce",
                utc=True,
            ).dt.floor("D")

            col = X.pop("date")
            X.insert(1, "date", col)

            all_X.append(X)

    if all_X:
        return pd.concat(all_X, ignore_index=True)
    return pd.DataFrame(columns=["date"] + config.COLUMNS_TO_KEEP)


# ---------------------------------------------------------------------------
def ensure_categories(
    cat_dir: Path = config.CAT_DIR,
    fetch_script: Path = config.FETCH_SCRIPT,
    *,
    needs_fetch: bool = False,
    needs_update: bool = False,
) -> pd.DataFrame: 
    """
    Return df_categories from cat_dir/product_categories.csv.

    Behaviour:
      - needs_fetch=True or folder missing/empty → full re-fetch via fetch_categories.py
      - needs_update=True and folder exists      → fetch only new product IDs not yet in the CSV
      - otherwise                                → load and return the existing CSV
    """
    folder_empty = not cat_dir.exists() or not any(cat_dir.iterdir())

    if needs_fetch or folder_empty:
        print(f"Running {fetch_script.name} → {cat_dir} ...")
        result = subprocess.run(
            [sys.executable, str(fetch_script), "--output-dir", str(cat_dir)],
            stdout=subprocess.PIPE,
            stderr=None,   # let stderr flow directly to the terminal (progress counter)
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"fetch_categories.py failed (exit {result.returncode})"
            )

    elif needs_update:
        output_csv  = cat_dir / "product_categories.csv"
        output_json = cat_dir / "product_categories.json"

        df_existing  = pd.read_csv(output_csv)
        df_existing["id"] = df_existing["id"].astype(int)
        existing_ids = set(df_existing["id"])

        # Collect all product IDs currently in orders
        all_ids: set[int] = set()
        for fp in glob.glob(str(config.ORDERS_DIR / "*.json")):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            orders = data if isinstance(data, list) else [data]  # support both single-order and array files
            for order in orders:
                for item in order.get("items", []):
                    if "id" in item:
                        all_ids.add(item["id"])

        new_ids = all_ids - existing_ids

        if not new_ids:
            print("No new product IDs found — product_categories unchanged.")
        else:
            print(f"Found {len(new_ids)} new product ID(s) — fetching from API ...")
            from fetch_categories import fetch_category
            from concurrent.futures import ThreadPoolExecutor, as_completed  # -- parallel API requests for new IDs only

            new_results: dict = {}
            errors = 0
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_category, pid): pid for pid in new_ids}
                for future in as_completed(futures):
                    result = future.result()
                    new_results[result["id"]] = result
                    if result["status"] != "ok":
                        errors += 1

            # Append to CSV
            df_new = pd.DataFrame(new_results.values())
            pd.concat([df_existing, df_new], ignore_index=True).to_csv(output_csv, index=False)

            # Append to JSON (if it exists)
            if output_json.exists():
                with open(output_json, encoding="utf-8") as f:
                    existing_json = json.load(f)
                existing_json.update({str(k): v for k, v in new_results.items()})
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(existing_json, f, ensure_ascii=False, indent=2)

            ok = sum(1 for r in new_results.values() if r["status"] == "ok")
            print(f"Added {ok} new product(s) to product_categories, {errors} failed.")

    else:
        print(f"Using existing product_categories at {cat_dir}")

    df = pd.read_csv(cat_dir / "product_categories.csv")
    df["id"] = df["id"].astype(int)
    return df


# ---------------------------------------------------------------------------
def apply_categories(
    X_all: pd.DataFrame,
    df_categories: pd.DataFrame,
    *,
    delete_not_found_category: bool = False,
) -> pd.DataFrame:
    """
    Enrich X_all with mainCategoryId from df_categories.

    Products with status 'error' (API call failed) are always dropped — their
    real category is unknown and assigning a placeholder would be misleading.

    Products with status 'not_found' (product discontinued):
      delete_not_found_category=False (default) → substitute mainCategoryId with 300112985 (medicine / pharmacy)
      delete_not_found_category=True            → drop rows from X_all
    """
    STATUS_NOTE = {
        "not_found": "404 – product no longer exists in the catalogue",
        "error":     "API call failed after all retries (network / rate-limit)",
    }

    not_ok = df_categories[df_categories["status"] != "ok"]
    if not not_ok.empty:
        id_to_name = X_all.drop_duplicates("id").set_index("id")["name"]
        print(f"{len(not_ok)} product(s) with status != 'ok':")
        for s, group in not_ok.groupby("status"):
            print(f"  [{s}] {STATUS_NOTE.get(s, s)}")
            display = group[["id", "status"]].copy()
            display["name"] = display["id"].map(id_to_name)
            print(display[["id", "name", "status"]].to_string(index=False))
            print()
    else:
        print("All products have status 'ok'.")

    # Always drop products where the API call failed — their category is unknown
    error_ids = df_categories.loc[df_categories["status"] == "error", "id"]
    if not error_ids.empty:
        before = len(X_all)
        X_all = X_all[~X_all["id"].isin(error_ids)].reset_index(drop=True)
        print(f"Dropped {before - len(X_all)} row(s) with status 'error'.")

    # Handle discontinued products (not_found)
    not_found_ids = df_categories.loc[df_categories["status"] == "not_found", "id"]
    if not not_found_ids.empty:
        if delete_not_found_category:
            before = len(X_all)
            X_all = X_all[~X_all["id"].isin(not_found_ids)].reset_index(drop=True)
            print(f"Dropped {before - len(X_all)} row(s) with status 'not_found'.")
        else:
            df_categories = df_categories.copy()
            df_categories.loc[
                df_categories["status"] == "not_found", "mainCategoryId"
            ] = 300112985  # category id for medicine / pharmacy products, aka léky.
            print("Substituted not_found mainCategoryId with 300112985.")

    X_all["mainCategoryId"] = X_all["id"].map(
        df_categories.set_index("id")["mainCategoryId"]
    )
    return X_all


# ---------------------------------------------------------------------------
def build_basket(X_all: pd.DataFrame) -> pd.DataFrame:
    """Pivot X_all into a basket matrix: order_id × mainCategoryId → total amount."""
    basket = (
        X_all.groupby(["order_id", "mainCategoryId"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .astype(int)
    )
    basket.columns = basket.columns.astype(int)
    return basket


# ---------------------------------------------------------------------------
def main(*, needs_fetch: bool = False, delete_not_found_category: bool = False, needs_update: bool = False):
    """End-to-end pipeline: load orders → categories → basket → save CSVs."""
    print(f"Loading orders from {config.ORDERS_DIR} ...")
    X_all = load_orders()
    print(f"Loaded {len(X_all):,} order lines.\n")

    df_categories = ensure_categories(needs_fetch=needs_fetch, needs_update=needs_update)
    print()

    X_all = apply_categories(
        X_all, df_categories, delete_not_found_category=delete_not_found_category
    )
    print()

    basket = build_basket(X_all)

    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    out_xall   = config.TRAINING_DIR / "orders_rohlik.csv"
    out_basket = config.TRAINING_DIR / "basket.csv"

    X_all.drop(columns=["name"], errors="ignore").to_csv(out_xall, index=False) # name used for display only
    basket.to_csv(out_basket)

    print(f"Saved X_all  → {out_xall}  ({len(X_all):,} rows)")
    print(f"Saved basket → {out_basket}  {basket.shape}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build training CSVs from order JSON files.")
    parser.add_argument("--update-categories", action="store_true", default=False,
                        help="Fetch categories for new product IDs not yet in product_categories.csv")
    parser.add_argument("--delete-not-found",  action="store_true", default=False,
                        help="Drop order rows for products not found in the API instead of substituting a placeholder category")
    args = parser.parse_args()
    main(needs_update=args.update_categories, delete_not_found_category=args.delete_not_found)
