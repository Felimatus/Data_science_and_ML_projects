# ROGII Wellbore Geology Prediction

Kaggle competition: **ROGII Wellbore Geology Prediction**

**Public Score: 11.750 RMSE**

---

## Competition Overview

The goal of this competition is to predict the **True Vertical Thickness (TVT)** for horizontal wellbore survey points where TVT is unknown, given a calibration zone where TVT measurements are available along with nearby geological formation surfaces and a typewell (reference well with known TVT-GR relationship).

Each well consists of:

- A **horizontal well** file with survey points (MD, X, Y, Z), formation top surfaces (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA), gamma ray (GR) readings, and a calibration zone where TVT is known (`TVT_input`)
- A **typewell** file with a reference TVT-GR curve

The dataset contains **773 training wells** and **3 test wells**, with **14,151 test survey points** to predict.

### Evaluation

- **Metric**: Root Mean Squared Error (RMSE) between predicted and true TVT values
- **Competition type**: Code competition -- Kaggle re-runs the submitted notebook with hidden test data

---

## Solution Architecture

This solution combines the best techniques from different high-scoring reference pipelines, including:

- DTW alignment, 7-beam search, multi-seed particle filters, bucketed GBDT training, per-bucket Optuna post-processing, cal-zone augmentation, test-well online training and exact train-coordinate overlap blend

### Pipeline Stages

```text
Raw CSVs
    |
    v
[1] Setup & GPU detection
    |
    v
[2] Numba JIT kernel compilation (beam search, DTW, particle filters)
    |
    v
[3-4] Physics-based signal extraction + spatial imputers
    |
    v
[5] Feature engineering (223 features per row)
    |
    v
[6] Cal-zone augmentation (train data +34%)
    |
    v
[7] Online test-well training
    |
    v
[8] CV splits (augmentation-aware) + bucket assignment (easy/hard)
    |
    v
[9] GBDT model training: 18 models (LGB x3 + CB x3 + XGB x3) x 2 buckets
    |
    v
[10] Hill climbing ensemble (per bucket)
    |
    v
[11] Optuna post-processing (per bucket)
    |
    v
[12] Test inference + exact-overlap blend
    |
    v
[13] submission.csv
```

---

> **About this repository:** This folder contains the submitted Kaggle notebook and the source Python file (`rogii_v2_kaggle.py`). The competition dataset (train/test CSVs) is not included -- it is loaded automatically from Kaggle's environment when the notebook runs.

---

> **Technical details below.** The following sections document the internals of `rogii_v2_kaggle.py` -- the algorithms, feature engineering, model training, and post-processing steps. They are intended as a reference for anyone looking to understand or extend the pipeline.

---

## Physics-Based Feature Extraction

Before any ML model is trained, the pipeline extracts rich features from each well using domain-specific algorithms. All physics kernels are JIT-compiled with **Numba** for performance.

### Beam Search (7 configurations)

Aligns the horizontal well's GR log against the typewell's GR log to estimate TVT. A beam search algorithm tracks the top-K candidate index mappings through the typewell, scored by GR residual plus a movement penalty. Seven configurations trade off beam width, movement cost, GR error scaling, and GR smoothing:

| Tag    | Beam Width | Movement Cost | Error Scale | Smooth |
|:------:|:----------:|:-------------:|:-----------:|:------:|
| cons   |     10     |     20.0      |    144.0    |   2    |
| loose  |     10     |      8.0      |     64.0    |   2    |
| vcons  |      8     |     35.0      |    220.0    |   1    |
| sm5    |     10     |     14.0      |     90.0    |   5    |
| vloose |     20     |      4.0      |     36.0    |   3    |
| mid    |     12     |     12.0      |    100.0    |   3    |
| stiff  |     15     |     25.0      |    180.0    |   2    |

### Dynamic Time Warping (DTW)

Two DTW variants align the well's GR to the typewell GR:

- **Deterministic multi-scale DTW**: Runs DTW with Sakoe-Chiba band constraint at 4 radii (20, 50, 100, 200). Each radius produces a TVT prediction and a warp-path slope. The ensemble is an inverse-cost weighted average.
- **Stochastic DTW**: Adds Gumbel noise to the cost matrix and traces K=8 random warp paths. Produces mean, std, and coefficient of variation features that capture alignment uncertainty.

### Particle Filters (2 types, multi-seed)

Sequential Monte Carlo methods that track TVT along the well bore:

- **ANCC Particle Filter** (600 particles, 2 seeds): Models TVT as a hidden state that evolves with measured depth. Particles represent hypothesized TVT positions; they're weighted by how well the typewell's GR at the hypothesized TVT matches the observed GR. Uses systematic resampling with roughening. Outputs: mean TVT estimate + uncertainty (std).

- **Z-aware Particle Filter** (600 particles, 2 seeds): Additionally models the vertical velocity of TVT as a linear function of the Z-derivative (dZ/dMD). Fits beta and intercept from the known zone using O(h^2) `np.gradient` derivatives. Uses both raw and smoothed GR for likelihood weighting.

### Multi-Scale Normalized Cross-Correlation (NCC)

Slides windows of 3 sizes (half-widths 8, 15, 25) from the known zone across the evaluation zone's GR, computing normalized cross-correlation against every possible offset in the typewell. The best-matching offset gives a TVT estimate. Scores are softmax-weighted to blend the three scales.

### Spatial Imputers

- **FormationPlaneKNN**: For each survey point, finds the K=10 nearest training wells by (X, Y) coordinates, fits a local weighted linear plane (ax + by + c) to each formation surface, and predicts the formation depth at the query location. Leave-one-out for training wells.

- **DenseANCCImputer**: Similar to FormationPlaneKNN but operates on the ANCC formation specifically, using up to 60 sample points per well for denser spatial coverage. Returns imputed ANCC value, spatial uncertainty (std), and nearest-neighbor distance.

---

## Feature Engineering

The pipeline constructs **223 features** per survey point, organized into categories:

| Category           | Features | Description |
|:------------------:|:----------:|:---------------------|
| Particle filter    |     ~8     | PF-ANCC and PF-Z mean, std, delta from last known, mutual difference |
| Beam search        |    ~12     | 7 beam configs (delta from last known), mean/std/median across configs |
| NCC                |    ~10     | 3 scale TVT estimates + scores, consensus, ensemble, trust factor |
| DTW                |    ~18     | Multi-scale ensemble, stochastic mean/std/CV, per-radius, slopes, costs |
| Formation surfaces |    ~55     | 6 formations x (full/weighted/last-50/early/mid bias + RMSE), mean/std/range |
| Dense ANCC         |     ~8     | Imputed ANCC, std, distance, bias-corrected TVT variants, RMSE/bias stats |
| GR statistics      |    ~30     | Rolling mean/std (4 windows), lags/leads (4 offsets), derivatives, envelope, energy |
| GR residual offsets|    ~52     | GR minus typewell-GR at various TVT offsets (anchor, beam, NCC, PF, DTW based) |
| Cross-signal       |     ~8     | Pairwise differences between PF, beam, NCC, DTW, spatial, dense signals |
| Calibration        |    ~10     | Affine cal parameters, prefix RMSE, known/eval lengths, TVT slopes |
| Positional         |    ~12     | MD since last known, fractional position, Z, dX/dY/dZ, trajectory derivatives |
| Typewell stats     |      2     | Typewell TVT range, GR mean |

---

## Data Augmentation

### Cal-Zone Augmentation

Each training well's calibration zone (where TVT is known) is split at intermediate points. For each split, the first portion acts as the "known" calibration, and the rest becomes pseudo-evaluation data with known targets. This creates additional training examples that mimic the train/eval boundary.

- Generates +34% additional training rows
- Augmented rows are excluded from validation folds (augmentation-aware CV)

### Online Test-Well Training

Test wells also have a calibration zone. The pipeline applies the same cal-zone augmentation to test wells, adding their pseudo-evaluation rows to the training set. This helps the models learn patterns specific to the test wells' geology.

---

## Model Training

### Bucketed Training

Wells are split into **easy** and **hard** buckets based on the median prefix RMSE (how well the typewell GR matches the known zone GR). Separate models are trained for each bucket:

- Easy wells: typewell is a good match, predictions are more straightforward
- Hard wells: poor typewell match, models must rely more on spatial/formation features

### GBDT Models (18 total)

For each bucket, 3 variants of each algorithm are trained with different learning rates (0.025, 0.020, 0.030) and random seeds (42, 7, 123):

| Model            | Max Estimators | Key Hyperparameters                               |
|:----------------:|:--------------:|:-------------------------------------------------:|
| **LightGBM** x3  | 8,000          | 255 leaves, subsample=0.8, colsample=0.8, L2=3.5  |
| **CatBoost** x3  | 8,000          | depth=7, L2=3.0, min_leaf=20, border=254          |
| **XGBoost** x3   | 8,000          | max_depth=8, min_child=6, subsample=0.8, L2=3.0   |

All models use early stopping with patience of 400 rounds. GPU acceleration is used when available (T4 x2 on Kaggle).

**Note on hyperparameters:** No fine-tuning of model hyperparameters (e.g., num_leaves, depth, regularization) was performed. The Kaggle code competition enforces a 12-hour runtime limit, and each Optuna trial would require a full K-fold CV across 3 frameworks and 2 buckets -- making automated tuning infeasible within the time budget. Instead, hyperparameters were manually selected based on common defaults for large tabular datasets, and model diversity is achieved through varying learning rates and random seeds.

### Cross-Validation

- **5-fold** stratified group K-fold (grouped by well, stratified by prefix RMSE)
- Augmented rows are only used for training, never for validation
- Each model's OOF predictions are averaged across folds for test inference

---

## Ensemble & Post-Processing

### Hill Climbing Stacker (Section 10)

A greedy ensemble algorithm that combines all 9 models per bucket:

1. Starts with the single best model
2. At each step, tries adding a small weight (step size) to each model and picks the one that lowers RMSE the most
3. Step sizes decrease progressively: 0.5, 0.25, 0.1, ..., 0.0005
4. Produces optimal weights for blending all models

### Optuna Post-Processing (Section 11)

Per-bucket optimization of 3 post-processing parameters using 300 TPE trials:

- **alpha** (0.85-1.05): Global scaling of the predicted delta
- **tau** (30-300): Exponential ramp-up distance -- predictions near the calibration boundary are dampened
- **w_pf** (0.0-0.30): Blend weight between the ensemble prediction and the particle filter prediction

### Exact Train-Coordinate Overlap Blend (Section 12)

Some test survey points share exact (X, Y, Z) coordinates with training points where TVT is known. For these points, the pipeline blends the model prediction with the known training TVT using weight 0.28.

---

## Training Results (OOF, 4-fold run)

### Individual Model RMSE

| Model | Easy Bucket OOF | Hard Bucket OOF |
|:-----:|:---------------:|:---------------:|
| LGB-1 |     12.9266     |     10.5389     |
| LGB-2 |     12.9521     |     10.6022     |
| LGB-3 |     12.8760     |     10.4889     |
| CB-1  |     12.8730     |     10.4416     |
| CB-2  |     12.9019     |     10.4788     |
| CB-3  |     12.8408     |     10.3999     |
| XGB-1 |     12.9756     |     10.4803     |
| XGB-2 |     12.9875     |     10.5177     |
| XGB-3 |     12.9635     |     10.4478     |

*Note: Easy bucket OOF is inflated due to one problematic fold (fold 3 at ~24.8 RMSE). This is a known artifact of unlucky well-to-fold assignment with fewer folds. With 5 folds this is expected to improve.*

### Ensemble Results

| Stage                        |  RMSE   |
|:----------------------------:|:-------:|
| Hill Climbing (global)       | ~10.93  |
| Post-processed OOF (global)  | 10.9746 |

### Post-Processing Parameters

| Bucket | Alpha | Tau | w_pf  | Bucket RMSE |
|:------:|:-----:|:---:|:-----:|:-----------:|
| Easy   | 0.870 |  30 | 0.000 |    12.52    |
| Hard   | 0.970 |  55 | 0.030 |    10.23    |

---

## Runtime

| Environment       | N_SPLITS | Runtime              |
|:-----------------:|:--------:|:--------------------:|
| Kaggle T4 x2 GPU  | 4        | ~7.38h               |
| Kaggle T4 x2 GPU  | 5        | ~9.2h (estimated)    |
| Kaggle CPU-only   | 5        | >12h (exceeds limit) |

The Kaggle time limit is 12 hours. GPU acceleration is required.

---

## Repository Contents

```text
rogii-wellbore-geology-prediction/
|-- README.md                                      # This file
|-- rogii_v2_kaggle.py                             # Main pipeline source (Kaggle-optimized, ~2030 lines)
|-- rogii-wellbore-geology-prediction-v2.ipynb     # Submitted Kaggle notebook
```

The competition dataset (773 training wells, 3 test wells, sample_submission.csv) is not included in this repository. It is automatically available at `/kaggle/input/competitions/rogii-wellbore-geology-prediction` when running on Kaggle.

### Data Schema

**horizontal_well.csv**: `MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input`

- MD: Measured Depth
- X, Y, Z: 3D coordinates of the survey point
- ANCC..BUDA: Six geological formation top surfaces (depths)
- TVT: True Vertical Thickness (target, available in train only)
- GR: Gamma Ray reading
- TVT_input: Known TVT values in the calibration zone (NaN in the evaluation zone)

**typewell.csv**: `TVT, GR`

- Reference well with known TVT-to-GR mapping

---

## Key Design Decisions

1. **Why three GBDT frameworks?** LightGBM, CatBoost, and XGBoost have different tree-building algorithms and regularization strategies. Ensembling all three improves robustness -- the hill climbing stacker assigns optimal weights based on OOF performance.

2. **Why bucketed training?** Wells where the typewell is a poor GR match (high prefix RMSE) need fundamentally different prediction strategies than wells with good matches. Separate models per bucket prevent the easy wells from dominating the loss landscape.

3. **Why 7 beam search configs?** Each configuration captures a different prior on how TVT evolves along the wellbore. Conservative configs (high movement cost) assume smooth TVT changes; loose configs allow rapid jumps. The GBDT models learn which configs to trust for each well.

4. **Why two particle filter types?** The ANCC particle filter tracks absolute TVT position using GR observations. The Z-aware filter additionally models the relationship between vertical trajectory changes and TVT changes -- useful when the well's vertical path is informative.

5. **Why DTW in addition to beam search?** DTW provides a global alignment between the well GR and typewell GR, while beam search is a local step-by-step matching. They capture different aspects of the GR-to-TVT relationship and their disagreement is itself an informative feature.

6. **Why cal-zone augmentation?** The calibration/evaluation boundary is where prediction difficulty spikes. By creating artificial boundaries within the calibration zone, the models see more examples of this transition and learn to handle it better.

---

## Requirements

- Python 3.10+
- NumPy, Pandas, SciPy, scikit-learn
- LightGBM (GPU build)
- CatBoost (GPU support)
- XGBoost (GPU support)
- Numba (JIT compilation)
- Optuna (hyperparameter optimization)
- joblib (parallel processing)

This code was written for a specific Kaggle competition and the notebook was designed to be run directly on Kaggle using their GPU environment, where all packages are pre-installed (set accelerator to **GPU T4 x2**). If you would like to run `rogii_v2_kaggle.py` locally, see `requirements.txt` for the full list of dependencies.

---

## How to Run

### On Kaggle (recommended)

1. Create a new Kaggle notebook
2. Add the competition dataset via "Add Data" -> "Competition Data"
3. Set accelerator to **GPU T4 x2**
4. Paste the contents of `rogii_v2_kaggle.py` into a single code cell
5. Set `N_SPLITS = 5` (line 125) for best accuracy
6. Click **"Save Version"** -> **"Save & Run All (Commit)"**
7. Once completed, submit the notebook version

### Locally

```bash
export ROGII_DATA_DIR=/path/to/rogii-wellbore-geology-prediction
python rogii_v2_kaggle.py
```

Requires a CUDA-capable GPU for reasonable runtime. A GTX 1050 Ti (4GB) may run into memory issues with CatBoost/XGBoost GPU modes.

---

## Possible Improvements

1. **Remove bucketed training**: The easy/hard split is based on the median prefix RMSE from the training set. If the test wells' prefix RMSE distribution differs, the bucket threshold may not generalize well, potentially hurting predictions.

2. **Increase Optuna post-processing trials**: Currently set to 300 (`PP_OPTUNA_TRIALS`). Increasing to 500+ could find better alpha/tau/w_pf parameters, especially for the hard bucket where the search space is more sensitive.

3. **Tune the exact-overlap blend weight**: The current `EXACT_OVERLAP_WEIGHT = 0.28` was inherited from a reference pipeline. This value may be too aggressive (overriding good model predictions) or too conservative (not leveraging known TVT values enough) for the hidden test wells.

4. **Hyperparameter tuning for GBDT models**: Run Optuna tuning offline (in a separate session) for key parameters like `num_leaves`, `depth`, `reg_lambda`, and `learning_rate`, then hardcode the best values into the submission notebook.

5. **Increase particle filter seeds**: Currently `PF_NUM_SEEDS = 2`. More seeds (4-8) would produce smoother particle filter estimates at the cost of longer feature engineering time.

6. **Add more beam search configurations**: The current 7 configs may not cover the full diversity of well behaviors. Additional configs with extreme parameters could help the ensemble on unusual wells.
