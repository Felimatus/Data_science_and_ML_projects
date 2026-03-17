# Titanic Survival Prediction

## Project Overview

This project explores the classic Kaggle *Titanic: Machine Learning from Disaster* dataset to predict passenger survival using supervised machine learning techniques.

The objective is to build and compare multiple classification models based on gender, social, and travel-related features, and evaluate their performance using cross-validation and Kaggle submission scores.

---

## Dataset

The dataset consists of two files:

- `train.csv`: Contains labeled training data, including the target variable `Survived`
- `test.csv`: Contains unlabeled data for final prediction

Main features include:

- Passenger class (`Pclass`)
- Sex and age
- Family relationships (`SibSp`, `Parch`)
- Fare
- Embarkation port
- Ticket information

---

## Project Workflow

The project follows a complete machine learning pipeline:

### 1. Exploratory Data Analysis (EDA)

- Analysis of survival rates by gender, class, and age
- Visualization of feature distributions
- Identification of important patterns and correlations

### 2. Feature Engineering

- Creation of `FamilySize` from `SibSp` and `Parch`
- Extraction of titles from passenger names
- Creation of `TicketGroupSize` from ticket frequency
- Creation of `Deck` from `Cabin`
- Handling of missing values in `Age`, `Fare`, and `Embarked`
- Removal of irrelevant or noisy features

### 3. Preprocessing

- Categorical encoding
- Feature scaling
- Use of transformation pipelines to prevent data leakage

### 4. Modeling

Multiple classification models were trained and optimized using `RandomizedSearchCV`:

- SGDClassifier
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- Extreme Gradient Boosting

A Voting and a Stacking Classifier were also evaluated.

### 5. Evaluation

- Cross-validation using accuracy
- Model comparison based on mean performance and stability
- Final model selection based on validation and CV results

---

## Results

| Model                         | Public score |
|-------------------------------|--------------|
| SGDClassifier                 | 0.77511      |
| Logistic Regression           | 0.77272      |
| Random Forest                 | 0.77272      |
| Extra Trees                   | **0.77990**  |
| Gradient Boosting             | 0.76076      |
| Extreme Gradient Boosting     | 0.76076      |
| Voting Classifier             | 0.77033      |
| Stacking Classifier           | 0.76315      |

---

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook

---

## Repository Structure

```text
├── Titanic_survival_prediction.ipynb
├── README.md
├── Data
    ├── train.csv
    ├── test.csv
    └── submission_titanic.csv
```

## Key Learning Outcomes

- End-to-end machine learning pipeline development
- Feature engineering for structured data
- Handling missing data and categorical variables
- Model selection and hyperparameter tuning
- Cross-validation and performance analysis
- Kaggle competition workflow

---

## Possible Improvements

- More advanced feature engineering
- Ensemble stacking
- Automated feature selection

---

## Setup

In case you want to run the notebook:

```bash
pip install -r requirements.txt
```

## Author

Felipe Matus

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/felipe-matus-3a5790285/)
