"""
Shared training utility for all TensorFlow/Keras models.

Provides fit_and_evaluate() which compiles, trains with early stopping,
evaluates, and returns the training history and validation metrics.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path

import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent))
from config import MAX_EPOCHS, PATIENCE, MODELS_DIR


def fit_and_evaluate(model, train_ds, valid_ds, learning_rate,
                     epochs=MAX_EPOCHS, patience=PATIENCE,
                     model_name=None):
    """
    Compile, train, and evaluate a Keras model.

    Parameters
    ----------
    model : tf.keras.Model
    train_ds : tf.data.Dataset
    valid_ds : tf.data.Dataset
    learning_rate : float
    epochs : int
    patience : int
        Early stopping patience (monitors val_mae).
    model_name : str or None
        If provided, saves the model to models/<model_name>.keras

    Returns
    -------
    dict with keys:
        "history"   — tf.keras.callbacks.History object
        "val_loss"  — final validation loss (Huber)
        "val_mae"   — final validation MAE
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=patience, restore_best_weights=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae", factor=0.5, patience=20,
        min_lr=1e-5, verbose=1)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.95)
    #optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    history = model.fit(train_ds, validation_data=valid_ds,
                        epochs=epochs,
                        callbacks=[early_stopping, reduce_lr])

    val_loss, val_mae = model.evaluate(valid_ds)

    print(f"\nValidation Loss (Huber): {val_loss:.6f}")
    print(f"Validation MAE:          {val_mae:.6f}")

    if model_name:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MODELS_DIR / f"{model_name}.keras"
        model.save(save_path)
        print(f"Model saved at models/{model_name}.keras")

    return {
        "history": history,
        "val_loss": val_loss,
        "val_mae": val_mae,
    }
