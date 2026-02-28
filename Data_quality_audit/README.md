# Data Quality Audit

## Overview

This project is based on a real data quality audit assignment completed as part of a job application process for a data analyst position.

The objective was to identify and document at least 8 errors, inconsistencies, and anomalies in a multi-table retail sales dataset within a 3-hour time limit, using Python and Pandas rather than Excel — to demonstrate scalability and reproducibility.

## Dataset

Four CSV files representing a retail sales system (anonymised):

| File | Rows | Description |
|------|------|-------------|
| `zones.csv` | 2 | Price zones (Hypermarket, Supermarket) |
| `sites.csv` | 5 | Store locations, each assigned to a zone |
| `articles.csv` | 15 | Product catalogue |
| `sales.csv` | 17 997 | Daily sales aggregations over ~13 months (Jul 2018 – Jul 2019) |

The dataset was provided by the company for evaluation purposes.

## Methodology

The analysis followed four main areas, in order of investigation:

1. **Referential integrity** — cross-table key validation (ArticleId, SiteId, ZoneId)
2. **Missing values** — identifying NaN patterns and their temporal scope
3. **Internal consistency** — business rule checks (margins, duplicates, zone pricing, negative quantities)
4. **Anomalies** — dtype mismatches and statistical outliers

Every finding was pinned to a specific date range and a specific site or zone, making it actionable for a data engineer tracing root causes in a production system.

## Results

9 data quality issues were identified (exceeding the required minimum of 8), grouped below by likely root cause:

**Referential integrity**
- `SiteId = 9` appears in 253 records (Nov 2018) — site does not exist in `sites.csv`
- Those same records carry a mismatched `ZoneId`, placing stores in the wrong pricing group

**Missing values — data pipeline failures**
- 27 records with missing `CostPrice` at SiteId 5, March 1–15, 2019
- 13 records with missing `Price` at SiteId 4, April 1–15, 2019
- Both gaps span exactly 14 days, suggesting a recurring pipeline failure pattern

**Business logic violations**
- 358 records with negative margin (May 13–20, 2019) — cost price exceeds selling price
- 492 duplicate records for SiteId 8 during April 2019
- 820 records (~4.6%) with negative quantities (Feb 1–18, 2019) — likely untagged returns
- 788 combinations of (ArticleId, Date, ZoneId) where sites in the same zone charged different prices — a direct violation of the zone pricing rule

**Anomalies**
- `Quantity` stored as `float64` and `Date` stored as `object` instead of their correct types
- Extreme outlier quantities requiring business validation (e.g., 432 units of a single product in one day)

Full analysis, code, and commentary are in `Data_quality_audit.ipynb`.

## How to Run

```bash
# Clone the repo
git clone https://github.com/felipematus/Data_quality_audit.git
cd Data_quality_audit

# Install dependencies
pip install -r requirements.txt

# Launch the notebook (requires Jupyter — install once with: pip install jupyter)
jupyter notebook Data_quality_audit.ipynb
```

## Tools and Technologies

- Python 3
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Files

- `Data_quality_audit.ipynb` — Main analysis notebook (all findings, code, and commentary)
- `Data_analysis_assignment.pdf` — Original written report submitted during the assignment
- `Data/` — Source CSV files (`zones.csv`, `sites.csv`, `articles.csv`, `sales.csv`)

## Future Improvements

- Build parameterised validation functions (e.g., `check_nulls_by_period(df, column, threshold)`) that can be re-run on each new data batch
- Integrate checks into an ETL pipeline to catch issues at ingestion rather than retrospectively
- Flag returns separately from sales so negative quantities do not distort demand forecasting

## Disclaimer

All company-related information has been anonymised.

---

## Author

Felipe Matus — [LinkedIn](https://www.linkedin.com/in/felipe-matus-3a5790285/)
