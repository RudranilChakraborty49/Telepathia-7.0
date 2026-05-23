"""
train_lstm.py
-------------
Trains the LSTM model on the preprocessed data.

Pipeline:
  1. Load X_train, X_test, y_train, y_test (.npy files)
  2. Build the LSTM architecture
  3. Compile with Adam + MSE
  4. Train for N epochs
  5. Save the model and training history
  6. Plot the loss curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ───── PATHS ─────
HERE = os.path.dirname(__file__)
ARRAYS_DIR = os.path.join(HERE, "saved_models", "arrays")
MODEL_PATH = os.path.join(HERE, "saved_models", "lstm_model.keras")
HISTORY_PATH = os.path.join(HERE, "saved_models", "lstm_history.npy")
PLOTS_DIR = os.path.join(HERE, "..", "research", "paper", "figures")

# ───── HYPERPARAMETERS ─────
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
DENSE_UNITS = 25
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def build_lstm_model(input_shape):
    """
    Build the LSTM architecture.

    Args:
        input_shape: tuple (timesteps, features) → e.g., (60, 5)

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First LSTM layer — returns sequence (so next LSTM can read it)
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),

        # Second LSTM layer — returns only the final output
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),

        # Dense hidden layer
        Dense(DENSE_UNITS, activation="relu"),

        # Output layer — 1 neuron for the predicted close price
        Dense(1),
    ])

    # Compile: Adam optimizer, Mean Squared Error loss
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    return model


def plot_history(history, save_path):
    """Plot training & validation loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history.history["loss"], label="Train Loss", color="#3b82f6")
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="#ef4444")
    axes[0].set_title("Model Loss (MSE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE curve
    axes[1].plot(history.history["mean_absolute_error"], label="Train MAE", color="#10b981")
    axes[1].plot(history.history["val_mean_absolute_error"], label="Val MAE", color="#f59e0b")
    axes[1].set_title("Mean Absolute Error", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"📊 Loss curve saved → {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("LSTM TRAINING — TELEPATHIA 7.0")
    print("=" * 60)

    # ── Load arrays ──
    print("\n📥 Loading preprocessed data...")
    X_train = np.load(os.path.join(ARRAYS_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(ARRAYS_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(ARRAYS_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(ARRAYS_DIR, "y_test.npy"))

    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test:  {y_test.shape}")

    # ── Build model ──
    input_shape = (X_train.shape[1], X_train.shape[2])    # (60, 5)
    print(f"\n🧠 Building LSTM model with input shape {input_shape}...")
    model = build_lstm_model(input_shape)
    model.summary()

    # ── Callbacks ──
    # EarlyStopping: stop training if validation loss doesn't improve for 5 epochs
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # ModelCheckpoint: save the best model during training
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # ── Train ──
    print(f"\n🚀 Training for {EPOCHS} epochs, batch size {BATCH_SIZE}...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    # ── Save history ──
    np.save(HISTORY_PATH, history.history)
    print(f"\n💾 Training history saved → {HISTORY_PATH}")

    # ── Evaluate ──
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test MSE: {test_loss:.6f}")
    print(f"   Test MAE: {test_mae:.6f}")

    # ── Plot ──
    plot_history(history, os.path.join(PLOTS_DIR, "lstm_loss_curves.png"))

    print(f"\n✅ Trained model saved → {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()