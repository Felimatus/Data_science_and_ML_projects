"""
Fetches mainCategoryId for all unique product IDs found in order JSON files.
Results are saved to product_categories.json and product_categories.csv.

Usage:
    python fetch_categories.py [--output-dir PATH]

Default output directory: data/product_categories (from project root).
"""

import argparse # for command-line argument parsing
import json
import glob # for file pattern matching
import time
import csv
import sys
from pathlib import Path
from urllib.request import urlopen, Request # for making HTTP requests
from urllib.error import HTTPError, URLError# for handling HTTP errors
from concurrent.futures import ThreadPoolExecutor, as_completed # for parallel API requests
# -- add project root to Python's import search path so 'import config' works
#    regardless of which directory the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def collect_ids() -> set[int]: # Collect unique product IDs from all order JSON files in the orders directory.
    ids = set()
    for path in glob.glob(str(config.ORDERS_DIR / "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        orders = data if isinstance(data, list) else [data]  # support both single-order and array files
        for order in orders:
            for item in order.get("items", []):
                if "id" in item:
                    ids.add(item["id"])
    return ids


def fetch_category(product_id: int) -> dict: # Fetch the mainCategoryId and name for a given product ID from the Rohlik API, with retries on failure.
    url = config.API_URL.format(product_id)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    for attempt in range(config.MAX_RETRIES):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return {
                "id": product_id,
                "mainCategoryId": data.get("mainCategoryId"),
                "name": data.get("name"),
                "status": "ok",
            }
        except HTTPError as e: 
            # 404 means product ID not found in API; other HTTP errors are retried
            if e.code == 404:
                return {"id": product_id, "mainCategoryId": None, "name": None, "status": "not_found"}
            # non-404 HTTP error: wait and retry; on the last attempt fall through to the error return below
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1)) # pause program execution in seconds
        except URLError: 
            # network error: wait and retry; on the last attempt fall through to the error return below
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
    # all retries exhausted
    return {"id": product_id, "mainCategoryId": None, "name": None, "status": "error"}


def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "product_categories.json"
    output_csv  = output_dir / "product_categories.csv"

    print("Collecting product IDs from order files...")
    ids = collect_ids()
    print(f"Found {len(ids)} unique product IDs across all orders.\n")

    results = {}
    errors  = 0
    total   = len(ids)

    print(f"Fetching mainCategoryId from API (up to {config.MAX_WORKERS} parallel requests)...")
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor: 
        # submit fetch_category tasks for all product IDs and store futures in a dictionary
        futures = {executor.submit(fetch_category, pid): pid for pid in ids} 
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result["id"]] = result
            if result["status"] != "ok":
                errors += 1
            # progress to terminal only (stderr bypasses _Tee and subprocess pipes)
        print(f"  Fetched {total} ({total - errors} ok, {errors} failed).", file=sys.stderr)
    print(file=sys.stderr)  # newline after the last \r

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON → {output_json}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mainCategoryId", "name", "status"])
        writer.writeheader()
        writer.writerows(results.values())
    print(f"Saved CSV  → {output_csv}")

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\nDone. {ok} products fetched successfully, {errors} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch product categories from Rohlik API.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.CAT_DIR,
        help="Directory where product_categories.csv/.json will be saved.",
    )
    args = parser.parse_args()
    main(args.output_dir.resolve())
