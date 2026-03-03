from pathlib import Path  # -- OS-independent path handling; works on Windows, Linux and macOS

# -- Anchor all paths to the project root (where this file lives),
#    regardless of which directory the scripts are launched from.
ROOT_DIR     = Path(__file__).parent.resolve()
DATA_DIR     = ROOT_DIR / "data"
ORDERS_DIR   = DATA_DIR / "orders"
CAT_DIR      = DATA_DIR / "product_categories"
TRAINING_DIR = DATA_DIR / "training"
SCRIPTS_DIR  = ROOT_DIR / "scripts"
FETCH_SCRIPT = SCRIPTS_DIR / "fetch_categories.py"

# generate_tables
COLUMNS_TO_KEEP = ["order_id", "order_orderTime", "id", "name", "amount"] # name used for display only

# fetch_categories
API_URL     = "https://www.rohlik.cz/api/v1/products/{}"
MAX_WORKERS = 5   # parallel API requests
RETRY_DELAY = 2   # seconds between retries (multiplied by attempt number for back-off)
MAX_RETRIES = 3   # number of attempts before giving up on a product ID (1 initial try + 2 retries)
