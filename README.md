# Data Science & Machine Learning Portfolio

## Overview

This repository contains selected data science and machine learning projects developed for learning, practice, and professional portfolio purposes.

The projects demonstrate my ability to:

- Clean and analyze real-world datasets
- Perform exploratory data analysis (EDA)
- Engineer meaningful features
- Build, tune, and evaluate machine learning models
- Communicate results clearly through code and documentation

These projects are intended to showcase practical skills to recruiters and hiring managers.

---

## Repository Structure

```text
Data_science_and_ML_projects/
│
├── Energy_demand/
│ └── 14-day electricity demand forecasting for Czech Republic (LSTM, GRU, WaveNet, TFT)
│
├── Kaggle_Titanic_survival_prediction/
│ └── Survival prediction using classification models
│
├── Data_quality_audit/
│ └── Real-world data cleaning and quality assessment project
│
├── Order_Rohlik/
│ └── Grocery category prediction from personal Rohlik order history
│
├── Kaggle_ROGII_Wellbore_Geology_Prediction/
│ └── TVT prediction for horizontal wellbores (beam search, DTW, particle filters, GBDT ensemble)
│
└── README.md
```

## Projects

### 1. Electricity Demand Forecasting — Czech Republic

**Folder:** `Energy_demand`

- Forecasts daily electricity demand 14 days ahead using deep learning
- Compares 4 architectures: Bidirectional LSTM, Bidirectional GRU, WaveNet, and Temporal Fusion Transformer (TFT)
- Multi-framework: TensorFlow/Keras (LSTM, GRU, WaveNet) + PyTorch (TFT)
- 54 features: weather from 5 regions (population-weighted), temporal, energy crisis flag, lagged load
- Best model: TFT with MAE of 135 MW (1.92% MAPE)

Technologies: Python, TensorFlow, PyTorch, pytorch-forecasting, Pandas, ENTSO-E API, Open-Meteo API

---

### 2. Kaggle Titanic Survival Prediction

**Folder:** `Kaggle_Titanic_survival_prediction`

- Binary classification problem
- End-to-end machine learning pipeline
- Feature engineering and model comparison
- Cross-validation and Kaggle submission

Technologies: Python, Pandas, Scikit-learn, XGBoost, Matplotlib

---

### 3. Data Quality Audit

**Folder:** `Data_quality_audit`

- Real-world data auditing task
- Error detection and data validation
- Data cleaning and reporting
- Documentation of findings

Technologies: Python, Pandas, Jupyter Notebook

---

### 4. Order Rohlik

**Folder:** `Order_Rohlik`

- Predicts which grocery categories you are likely to buy in your next Rohlik order, based on your personal order history
- Multi-label binary classification: one XGBoost model per category
- Features: month, day of week, days since last order, and per-category quantities from the previous two orders (lag-1, lag-2)
- Probability threshold tuned on a validation set to match your average number of categories per order
- Fetches product category data from the Rohlik API

Technologies: Python, XGBoost, Pandas, Rohlik API

---

### 5. Kaggle ROGII Wellbore Geology Prediction

**Folder:** `Kaggle_ROGII_Wellbore_Geology_Prediction`

- Predicts True Vertical Thickness (TVT) for horizontal wellbore survey points in a Kaggle code competition
- Physics-based feature engineering: 7-configuration beam search, multi-scale DTW alignment, dual particle filters (ANCC + Z-aware), normalized cross-correlation, spatial imputers
- 223 engineered features per survey point combining geophysical signals, GR statistics, formation surfaces, and cross-signal consensus
- Bucketed GBDT ensemble: 18 models (LightGBM + CatBoost + XGBoost) with hill climbing stacker and Optuna post-processing
- Cal-zone augmentation and online test-well training for data efficiency
- Public Score: 11.750 RMSE

Technologies: Python, LightGBM, CatBoost, XGBoost, Numba, Optuna, SciPy, scikit-learn, Pandas

---

## Skills Demonstrated

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning modeling (classical and deep learning)
- Deep learning with TensorFlow/Keras and PyTorch
- Time series forecasting
- Hyperparameter tuning
- Model evaluation and comparison
- Technical documentation

---

## About Me

I am a Physicist (Ph.D.) transitioning into Data Science, developing my skills through hands-on projects and continuous learning.

This repository reflects my practical experience and learning journey.

---

## Contact

Felipe Matus

LinkedIn: [Felipe Matus](https://www.linkedin.com/in/felipe-matus-3a5790285/)
