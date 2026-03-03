# Order Rohlik

Predicts which grocery categories you are likely to buy in your next Rohlik order, based on your personal order history. Examples belonging to a category are also shown.

## How it works

1. Your past orders (exported as JSON from Rohlik) are loaded and flattened into a feature matrix.
2. For each category, an XGBoost binary classifier is trained to predict whether you will buy from that category in the next order.
3. Features used: month, day of week, days since last order, and quantities bought per category in the previous two orders (lag-1 and lag-2).
4. A probability threshold is tuned on a validation set to match your average number of categories per order.
5. The predicted categories are ranked by probability and saved to `data/results/`.

## Project structure

```
Order_Rohlik/
├── config.py                  # Central paths and constants
├── data/
│   └── orders/                # Raw order JSON files (one per order or all in an array format [see `sample.json`])
└── scripts/
    ├── ensure_data_files.py   # Validates orders folder and triggers CSV generation; used by both entry-point scripts
    ├── fetch_categories.py    # Fetches mainCategoryId for each product from the Rohlik API
    ├── generate_tables.py     # Builds orders_rohlik.csv and basket.csv from the JSON files
    ├── get_products.py        # Trains models and predicts your next order (**main script**)
    └── score.py               # Same pipeline with train/val/test split for F1 evaluation
```

Generated folders (git-ignored):

```
data/
├── product_categories/        # product_categories.csv/.json (fetched from API)
├── training/                  # orders_rohlik.csv, basket.csv
└── results/
    ├── products_to_buy.txt    # Predicted categories (get_products.py)
    ├── products_to_buy.csv
    └── score/                 # Only if score.py is run.
        ├── score_details.txt  # Full log of the scoring run
        ├── products_to_buy.txt  # Optional: only if `--show-results` is passed
        └── products_to_buy.csv
```

## Setup

```bash
pip install -r requirements.txt
```

Place your Rohlik order JSON files in `data/orders/`. One JSON file per order, named arbitrarily (e.g. `order-123456.json`), or one JSON file in an array format (see `sample.json`).


## Usage

### Predict your next order

```bash
python3 scripts/get_products.py
```
Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--n-to-show` | bought product average quantity | Number of categories to display |
| `--val-size` | 0.2 | Fraction of data used for validation |
| `--verbosity` | 0 | XGBoost verbosity (0 = silent) |
| `--update` | off | Regenerate training files and fetch categories for any new products |
| `--delete-not-found` | off | Drop order rows for products not found in the API instead of substituting a placeholder category |

Results are shown and also saved in `data/results/`, including examples of products belonging to a given category and the
Rohlik id of the product.

### Evaluate model performance (F1 score/ Jaccard index) 

```bash
python3 scripts/score.py #results saved in data/results/score
python3 scripts/score.py --show-results   # also predict next order
```

### Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--show-results` | off | Also predict and save the next order |
| `--test-size` | 0.1 | Fraction of unique dates used for test |
| `--val-size` | 0.2 | Fraction of train+val used for validation |
| `--verbosity` | 0 | XGBoost verbosity (0 = silent) |
| `--n-to-show` | bought product average quantity | Number of categories to display |
| `--update` | off | Regenerate training files and fetch categories for any new products |
| `--delete-not-found` | off | Drop order rows for products not found in the API instead of substituting a placeholder category |

### Specification of optional arguments.
#### Only available for *score.py*:

- `--show-results`: If True also predicts next order and results are saved in `data/results/score/`. Since *score.py* uses also a test data set, results may vary from those of *get_products.py*.
- `--test-size`: see above.

#### Available for *score.py* and *get_products.py*:

- `--val-size`: See above.
- `--verbosity`: See above.
- `--n-to-show`: The program obtains the average number of products bought per order. The same number of predictions is shown and saved.
- `--delete-not-found`: Certain products are no longer available at Rohlik, those with status `not_found` (see below) and their category may not be fetched online. During testing, those cases were caused by medicine / pharmacy products. Therefore, their category is replaced by 300112985 (the proper category). If `--delete-not-found` is passed, the category of such products is not substituted and they are dropped from the data set.
- `--update`: The first time running, data sets for training (orders_rohlik.csv, basket.csv) are created from JSON files and their category fetched from the API and saved to product_categories.csv. If all those files exist, the program will not call ensure_data_files() and the same data will be used for training. If `--update` is passed, the program updates training data files and adds possible new categories.

The full chain when running with --update:
```bash
score.py / get_products.py  --update
  └─ ensure_data_files(update=True)
       └─ generate_tables.py --update-categories
            └─ ensure_categories(needs_update=True)
                 ├─ load existing product_categories.csv
                 ├─ collect all product IDs from data/orders/*.json
                 ├─ diff → new_ids not yet in CSV
                 ├─ if new_ids: fetch from API, append to CSV and JSON
                 └─ if none: print "unchanged"
```

### Refresh product categories from the API

```bash
python3 scripts/fetch_categories.py
```

Run this if you have new products in your orders that are not yet in `data/product_categories/`.

`fetch_categories.py` queries the Rohlik API for each product and records one of three statuses:

| Status | Meaning | How it is handled |
|---|---|---|
| `ok` | API returned successfully — `mainCategoryId` is populated | kept as-is |
| `not_found` | HTTP 404 — the product no longer exists in the catalogue | controlled by `--delete-not-found` (see above) |
| `error` | All retries exhausted — network failure or rate-limit | always dropped from the dataset |

`error` products are always removed from the training data regardless of `--delete-not-found`, because their real category is unknown and assigning a placeholder would be misleading. `not_found` products are handled by the `--delete-not-found` flag.

### Regenerate training tables

```bash
python3 scripts/generate_tables.py
```

Run this after adding new order JSON files to `data/orders/`. Both `get_products.py` and `score.py` run this automatically if the training files are missing.

## Example

A sample of results and outputs can be found in folder `Sample` containing a notebook, which runs `get_products.py` using `sample.json` data stored in `data/orders/`. If not needed, **delete `sample.json`** and **`Sample/`** folder.

## Possible Improvements

- More advanced feature engineering (get the categorical attribute such as jogurts, cheese, chocolates, etc.)

## Author

Felipe Matus — [LinkedIn](https://www.linkedin.com/in/felipe-matus-3a5790285/)

