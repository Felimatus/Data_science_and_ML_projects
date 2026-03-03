"""
Validate orders folder and (re)generate training CSVs.

Used by score.py and get_products.py.
"""

import subprocess  # -- launch generate_tables.py as a child process and capture its output
import sys
from pathlib import Path

# -- add project root to Python's import search path so 'import config' works
#    regardless of which directory the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def ensure_data_files(update: bool = False, delete_not_found: bool = False):
    """
    Validate the orders folder and ensure training CSVs exist.

    Steps:
      1. Raise FileNotFoundError if data/orders/ does not exist.
      2. Raise FileNotFoundError if data/orders/ contains no JSON files.
      3. If update=True, always run generate_tables.py to regenerate the CSVs.
         If update=False (default), run it only when the CSVs are missing.
      delete_not_found: passed to generate_tables.py; drops order rows for
         products not found in the API instead of substituting a placeholder.
    """
    # ── 1. Orders folder must exist ───────────────────────────────────────────
    if not config.ORDERS_DIR.exists():
        raise FileNotFoundError(
            f"Orders folder not found: {config.ORDERS_DIR}\n"
            "Create it and add your Rohlik order JSON files."
        )

    # ── 2. Orders folder must contain at least one JSON file ──────────────────
    json_files = list(config.ORDERS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {config.ORDERS_DIR}\n"
            "Add your Rohlik order JSON files and try again."
        )

    # ── 3. Generate (or regenerate) training CSVs ─────────────────────────────
    needed = [
        config.TRAINING_DIR / "orders_rohlik.csv",
        config.TRAINING_DIR / "basket.csv",
    ]

    if update or not all(f.exists() for f in needed):
        generate_script = config.SCRIPTS_DIR / "generate_tables.py"
        if not generate_script.exists():
            raise FileNotFoundError(
                f"generate_tables.py not found at {generate_script}"
            )
        if update:
            print("update=True — regenerating training files and checking for new product categories ...")
        else:
            print("Training files not found — running generate_tables.py ...")
        # build the command; --update-categories tells generate_tables.py to
        # fetch API data for any product IDs not yet in product_categories.csv
        cmd = [sys.executable, str(generate_script)]
        if update:
            cmd.append("--update-categories")
        if delete_not_found:
            cmd.append("--delete-not-found")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # -- merge stderr into stdout so both are captured and printed together
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"generate_tables.py failed (exit {result.returncode})"
            )
