"""
Train a Conv1D + GRU hybrid model for electricity demand forecasting.

Architecture:
    Dilated causal Conv1D stack (multi-scale feature extraction)
    → Bidirectional GRU (temporal modelling on compressed representations)
    → Dense output (14-day × 3-target forecast)

Predicts: load_min, load_mean, load_max for the next 14 days.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent))
from config import SEED, AHEAD, DATA_CSV
from scripts.data_to_training import load_training_data
from scripts.fit_and_evaluate import fit_and_evaluate

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = load_training_data()
train_ds = data["train_ds"]
valid_ds = data["valid_ds"]
n_features = data["n_features"]
n_targets = 3  # load_min, load_mean, load_max

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
conv_gru_model = tf.keras.Sequential([
    # --- Dilated causal Conv1D stack ---
    tf.keras.Input(shape=(None, n_features)),
    tf.keras.layers.Conv1D(32, kernel_size=2, padding="causal",
                           activation="relu", dilation_rate=1),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Conv1D(64, kernel_size=2, padding="causal",
                           activation="relu", dilation_rate=2),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Conv1D(64, kernel_size=2, padding="causal",
                           activation="relu", dilation_rate=4),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Conv1D(64, kernel_size=2, padding="causal",
                           activation="relu", dilation_rate=8),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dropout(0.2),

    # --- Bidirectional GRU on extracted features ---
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),

    # --- Output: 14 days × 3 targets ---
    tf.keras.layers.Dense(AHEAD * n_targets),
    tf.keras.layers.Reshape([-1, AHEAD, n_targets]),
])

conv_gru_model.summary()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
results = fit_and_evaluate(conv_gru_model, train_ds, valid_ds,
                           learning_rate=0.5, model_name="conv_gru_hybrid")

# ---------------------------------------------------------------------------
# Predict next 14 days
# ---------------------------------------------------------------------------
print(f"\n--- Forecasting next {AHEAD} days ---")

# Use the last sequence from validation set
for X_batch, _ in valid_ds:
    last_X = X_batch[-1:]  # shape (1, seq_length, n_features)

Y_pred = conv_gru_model.predict(last_X)
# Take only the last time step's prediction
y_pred_14 = Y_pred[0, -1]  # shape (AHEAD, n_targets)

# Inverse-transform the targets (first 3 columns)
target_mean = data["scaler_mean"][:n_targets]
target_std  = data["scaler_std"][:n_targets]
y_pred_original = y_pred_14 * target_std + target_mean

# Compute forecast start date from the dataset
last_date = pd.read_csv(DATA_CSV, usecols=["date"], parse_dates=["date"]
                        )["date"].max()
forecast_start = last_date + pd.Timedelta(days=1)

print(f"\nPredicted electricity load (MW) for next {AHEAD} days "
      f"(starting from {forecast_start.date()}):")
print(f"{'Date':>12} {'Min':>10} {'Mean':>10} {'Max':>10}")
print("-" * 46)
for day in range(AHEAD):
    date = forecast_start + pd.Timedelta(days=day)
    print(f"{str(date.date()):>12} {y_pred_original[day, 0]:>10.1f} "
          f"{y_pred_original[day, 1]:>10.1f} "
          f"{y_pred_original[day, 2]:>10.1f}")
