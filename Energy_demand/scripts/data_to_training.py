"""
Prepare training data from the processed CSV.

Run once:  python scripts/data_to_training.py
           → saves normalised arrays + scaler to data/processed/data_to_training.npz

Then training scripts call load_training_data() which loads the .npz
and builds tf.data.Datasets on the fly (fast — no CSV re-reading).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Allow imports from the project root
sys.path.append(str(Path(__file__).parent.parent))
from config import (DATA_CSV, TRAINING_NPZ, SEQ_LENGTH, AHEAD, BATCH_SIZE,
                    SEED, VALID_MONTHS)


# ---------------------------------------------------------------------------
# Helper: sliding windows
# ---------------------------------------------------------------------------
def to_windows(dataset, length):
    """Create sliding windows of a given length from a tf.data.Dataset."""
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))


def to_seq2seq_dataset(series, seq_length=SEQ_LENGTH, ahead=AHEAD,
                       target_cols=(0, 1, 2), batch_size=BATCH_SIZE,
                       shuffle=False, seed=None):
    """
    Convert a 2-D numpy array into a seq2seq tf.data.Dataset.

    Returns tf.data.Dataset yielding (X, Y) where
        X has shape (batch, seq_length, n_features)
        Y has shape (batch, seq_length, n_targets)
    """
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = to_windows(ds, seq_length)
    ds = ds.map(lambda S: (S[:, 0], tf.gather(S[:, 1:], target_cols, axis=-1)))
    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)
    return ds.batch(batch_size)


# ---------------------------------------------------------------------------
# Prepare and save (run once)
# ---------------------------------------------------------------------------
def prepare_and_save():
    """
    Load data.csv, split, normalise, and save arrays to .npz.
    """
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # One-hot encode day_type (W/A/U) → day_W, day_A, day_U
    df = pd.get_dummies(df, columns=["day_type"], prefix="day", dtype=float)

    target_cols = ["load_min", "load_mean", "load_max"]
    feature_cols = [c for c in df.columns if c != "date"]
    other_cols = [c for c in feature_cols if c not in target_cols]
    feature_cols = target_cols + other_cols

    # Chronological split
    total_days = len(df)
    valid_size = VALID_MONTHS * 30
    train_size = total_days - valid_size

    train_df = df.iloc[:train_size]
    valid_df = df.iloc[train_size:]

    print(f"Split: train={len(train_df)}, valid={len(valid_df)} days")
    print(f"Train: {train_df['date'].iloc[0].date()} → "
          f"{train_df['date'].iloc[-1].date()}")
    print(f"Valid: {valid_df['date'].iloc[0].date()} → "
          f"{valid_df['date'].iloc[-1].date()}")

    # Normalise (fit on train only)
    train_values = train_df[feature_cols].values.astype(np.float32)
    scaler_mean = train_values.mean(axis=0)
    scaler_std  = train_values.std(axis=0)
    scaler_std[scaler_std == 0] = 1.0

    train_norm = (train_values - scaler_mean) / scaler_std
    valid_norm = (valid_df[feature_cols].values.astype(np.float32)
                  - scaler_mean) / scaler_std

    # Save
    TRAINING_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TRAINING_NPZ,
             train_norm=train_norm,
             valid_norm=valid_norm,
             scaler_mean=scaler_mean,
             scaler_std=scaler_std,
             feature_columns=np.array(feature_cols))

    print(f"\nSaved to {TRAINING_NPZ}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")


# ---------------------------------------------------------------------------
# Load from .npz (used by training scripts)
# ---------------------------------------------------------------------------
def load_training_data(seq_length=SEQ_LENGTH):
    """
    Load pre-saved arrays and build tf.data.Datasets.

    Parameters
    ----------
    seq_length : int
        Input window length. Default is SEQ_LENGTH from config.
        Conv/WaveNet scripts can pass a longer value.

    Returns
    -------
    dict with keys:
        "train_ds", "valid_ds"        — tf.data.Dataset objects
        "train_norm", "valid_norm"     — normalised numpy arrays
        "scaler_mean", "scaler_std"    — for inverse-transform
        "feature_columns"              — list of column names
        "n_features"                   — number of input features
    """
    d = np.load(TRAINING_NPZ, allow_pickle=True)
    train_norm = d["train_norm"]
    valid_norm = d["valid_norm"]
    scaler_mean = d["scaler_mean"]
    scaler_std = d["scaler_std"]
    feature_columns = d["feature_columns"].tolist()

    train_ds = to_seq2seq_dataset(train_norm, seq_length=seq_length,
                                  shuffle=True, seed=SEED)
    valid_ds = to_seq2seq_dataset(valid_norm, seq_length=seq_length)

    return {
        "train_ds": train_ds,
        "valid_ds": valid_ds,
        "train_norm": train_norm,
        "valid_norm": valid_norm,
        "scaler_mean": scaler_mean,
        "scaler_std":  scaler_std,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
    }


# ---------------------------------------------------------------------------
# Stand-alone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prepare_and_save()

    # Quick verification
    data = load_training_data()
    print(f"\nVerification — Features ({data['n_features']}):")
    for name in ("train_ds", "valid_ds"):
        for X, Y in data[name].take(1):
            print(f"  {name}: X shape={X.shape}, Y shape={Y.shape}")
