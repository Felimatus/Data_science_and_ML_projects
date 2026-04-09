"""
Train a Temporal Fusion Transformer (TFT) for electricity demand forecasting.

Uses pytorch-forecasting library. The TFT natively handles:
    - Multi-horizon forecasting
    - Variable selection (learns which weather stations/features matter most)
    - Static vs. time-varying features
    - Interpretable attention weights

Predicts: load_min, load_mean, load_max for the next 14 days.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    GroupNormalizer,
    MultiNormalizer,
)
from pytorch_forecasting.metrics import MAE, MultiLoss

sys.path.append(str(Path(__file__).parent.parent))
from config import (DATA_CSV, SEED, AHEAD, MAX_EPOCHS, PATIENCE, MODELS_DIR,
                    VALID_MONTHS, SEQ_LENGTH)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
pl.seed_everything(SEED)

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# One-hot encode day_type (W/A/U) → day_W, day_A, day_U
df = pd.get_dummies(df, columns=["day_type"], prefix="day", dtype=float)

# TFT requires a "time_idx" (integer time index) and a "group" column
df["time_idx"] = np.arange(len(df))
df["group"] = "CZ"  # single group (country-level data)

# --- Chronological split --------------------------------------------------
total_days = len(df)
valid_size = VALID_MONTHS * 30
train_size = total_days - valid_size

# pytorch-forecasting uses a training cutoff index
training_cutoff = train_size - 1

# --- Identify feature columns --------------------------------------------
targets = ["load_min", "load_mean", "load_max"]

# Time-varying known features (available in the future via weather forecast)
REGION_PREFIXES = ("CB_", "NB_", "SB_", "SM_", "NM_")
time_varying_known = [c for c in df.columns
                      if c.startswith((*REGION_PREFIXES,
                                       "day_W", "day_A", "day_U",
                                       "season_", "is_holiday"))]

# Time-varying unknown features (only known up to present)
time_varying_unknown = [c for c in df.columns
                        if c not in ["date", "time_idx", "group"] + targets
                        and c not in time_varying_known]

print(f"Targets: {targets}")
print(f"Time-varying known features: {time_varying_known}")
print(f"Time-varying unknown features ({len(time_varying_unknown)}): "
      f"{time_varying_unknown[:5]}...")

# ---------------------------------------------------------------------------
# Create TimeSeriesDataSet
# ---------------------------------------------------------------------------
max_encoder_length = SEQ_LENGTH  # input window from config
max_prediction_length = AHEAD

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=targets,
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=time_varying_known if time_varying_known else None,
    time_varying_unknown_reals=time_varying_unknown,
    target_normalizer=MultiNormalizer(
        [GroupNormalizer(groups=["group"]) for _ in targets]
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    df[df.time_idx > training_cutoff],
    stop_randomization=True,
)

# DataLoaders
train_dl = training.to_dataloader(train=True, batch_size=32, num_workers=0)
valid_dl = validation.to_dataloader(train=False, batch_size=32, num_workers=0)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=64, # number of hidden units in LSTM layers
    attention_head_size=4,
    dropout=0.2,
    hidden_continuous_size=32, # size of hidden layers (embeddings) for continuous variables
    loss=MultiLoss([MAE()] * len(targets)),
    reduce_on_plateau_patience=20, # built-in learning rate scheduler patience
    reduce_on_plateau_reduction=2.0 # factor to reduce learning rate when metric plateaus
)

print(f"\nTFT parameters: {tft.size() / 1e3:.1f}k")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)
save_path = MODELS_DIR / "tft_model.ckpt"

early_stop = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, mode="min")

best_ckpt = pl.callbacks.ModelCheckpoint(
    dirpath=MODELS_DIR,
    filename="tft_best",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="cpu",
    gradient_clip_val=0.2,
    log_every_n_steps=38,
    callbacks=[early_stop, best_ckpt],
    enable_progress_bar=True,
)

trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=valid_dl)

# ---------------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------------
import shutil
shutil.copy(best_ckpt.best_model_path, save_path)
print(f"\nBest model (val_loss={best_ckpt.best_model_score:.2f}) saved to {save_path}")

# ---------------------------------------------------------------------------
# Evaluate on validation set
# ---------------------------------------------------------------------------
val_results = trainer.validate(tft, dataloaders=valid_dl)
print(f"\nValidation results: {val_results}")

# ---------------------------------------------------------------------------
# Predict next 14 days and show variable importance
# ---------------------------------------------------------------------------
print(f"\n--- Forecasting next {AHEAD} days ---")
predict_kwargs = dict(trainer_kwargs=dict(accelerator="cpu"))
predictions = tft.predict(valid_dl, **predict_kwargs)

# Multi-target: predictions is a list of tensors, one per target
last_preds = [p[-1].numpy() for p in predictions]  # [load_min, load_mean, load_max]

print(f"\nPredicted electricity load (MW) for next {AHEAD} days:")
print(f"{'Day':>4} {'Min':>10} {'Mean':>10} {'Max':>10}")
print("-" * 38)
for day in range(AHEAD):
    print(f"{day+1:>4} {last_preds[0][day]:>10.1f} "
          f"{last_preds[1][day]:>10.1f} {last_preds[2][day]:>10.1f}")

# --- Variable importance (TFT's key interpretability feature) -------------
# Run forward pass on each batch, interpret, and accumulate
tft.eval()
interpretations = []
with torch.no_grad():
    for batch in valid_dl:
        x, y = batch
        raw_out = tft(x)
        interp = tft.interpret_output(raw_out, reduction="mean")
        interpretations.append(interp)

# Aggregate across batches
interpretation = {
    k: sum(b[k] for b in interpretations)
    for k in interpretations[0]
}

print("\n--- Variable Importance (encoder) ---")
enc_vars = interpretation["encoder_variables"].numpy()
enc_vars = enc_vars / enc_vars.sum()  # normalize to 0-1
synthetic = {"encoder_length", "relative_time_idx",
             "load_min_scale", "load_mean_scale", "load_max_scale",
             "load_min_center", "load_mean_center", "load_max_center"}
importance = {k: v for k, v in zip(training.reals, enc_vars)
              if k not in synthetic}
for var, imp in sorted(importance.items(), key=lambda x: -x[1])[:20]:
    print(f"  {var:.<45} {imp:.4f}")
