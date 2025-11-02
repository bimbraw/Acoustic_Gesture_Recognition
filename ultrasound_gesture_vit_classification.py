#Author - Keshav Bimbraw

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Progress bar (console "trackbar")
from tqdm.auto import tqdm

# ============================
# Helpers
# ============================
def set_seed(seed: int):
    """Make runs reproducible across numpy and TF."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(p):
    """Create parent directory for a file path if it does not exist."""
    if p:
        os.makedirs(os.path.dirname(p), exist_ok=True)

def save_confusion_matrix_png(y_true, y_pred, path):
    """Save a simple confusion matrix figure to PNG."""
    if not path:
        return
    ensure_dir(path)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

def dump_json(obj, path):
    """Dump a JSON file with nice indentation."""
    if not path:
        return
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ============================
# Keras Callback: tqdm progress bars
# ============================
class TqdmProgress(keras.callbacks.Callback):
    """
    Two-level progress:
      - Outer bar: epochs
      - Inner bar: batches within each epoch (the "trackbar" you asked for)
    """
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        if not self.enable:
            return
        total_epochs = self.params.get("epochs", None)
        self.epoch_bar = tqdm(total=total_epochs, desc="Epochs", position=0, leave=True)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.enable:
            return
        # Create/refresh the per-epoch batch bar
        total_steps = self.params.get("steps", None)  # number of batches in an epoch
        # Note: steps can be None if TF infers it; in practice with numpy arrays it’s len(x_train)//batch_size
        self.batch_bar = tqdm(
            total=total_steps, desc=f"Epoch {epoch+1}/{self.params.get('epochs','?')}",
            position=1, leave=False
        )

    def on_train_batch_end(self, batch, logs=None):
        if not self.enable or self.batch_bar is None:
            return
        self.batch_bar.update(1)
        # Optionally show batch-level loss/acc in the bar postfix
        if logs:
            self.batch_bar.set_postfix({
                "loss": f"{logs.get('loss', 0):.4f}",
                "acc": f"{logs.get('accuracy', 0):.4f}"
            })

    def on_epoch_end(self, epoch, logs=None):
        if not self.enable:
            return
        # Close the inner bar and write a one-line summary with val metrics
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None
        if logs:
            tqdm.write(
                f"Epoch {epoch+1} done | "
                f"loss={logs.get('loss', 0):.4f} "
                f"acc={logs.get('accuracy', 0):.4f} "
                f"val_loss={logs.get('val_loss', 0):.4f} "
                f"val_acc={logs.get('val_accuracy', 0):.4f}"
            )
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        if not self.enable:
            return
        if self.batch_bar is not None:
            self.batch_bar.close()
        if self.epoch_bar is not None:
            self.epoch_bar.close()

# ============================
# ViT building blocks
# ============================
class Patches(keras.layers.Layer):
    """Split the input image into non-overlapping patches."""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        batch = tf.shape(patches)[0]
        return tf.reshape(patches, [batch, -1, patches.shape[-1]])

class PatchEncoder(keras.layers.Layer):
    """Linear projection + learnable positional embeddings."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

def mlp(x, hidden_units, dropout_rate):
    """Transformer MLP block."""
    for units in hidden_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def build_vit(input_shape, num_classes,
              patch_size=32, projection_dim=64, num_heads=8,
              transformer_layers=6, transformer_units=(128, 64),
              mlp_head_units=(512, 256)):
    """Build a compact ViT classifier (no CLS token; flatten + MLP head)."""
    h, w, _ = input_shape
    num_patches = (h // patch_size) * (w // patch_size)

    inputs = keras.layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = keras.layers.Add()([attn, encoded])
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded = keras.layers.Add()([x3, x2])

    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = keras.layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model

# ============================
# Data loading
# ============================
def load_subject_arrays(root, mode, subject, image_size):
    """
    Load four arrays for a subject:
      X_m_train.npy, X_m_test.npy, y_m_train.npy, y_m_test.npy
    Then: add channel dim if needed, resize to (image_size, image_size), normalize to [0,1].
    """
    d = os.path.join(root, mode, subject)
    x_train = np.load(os.path.join(d, "X_m_train.npy"))
    x_test  = np.load(os.path.join(d, "X_m_test.npy"))
    y_train = np.load(os.path.join(d, "y_m_train.npy"))
    y_test  = np.load(os.path.join(d, "y_m_test.npy"))

    y_train = y_train.astype(np.int64).ravel()
    y_test  = y_test.astype(np.int64).ravel()

    # Add channel dim if needed: (N,H,W) -> (N,H,W,1)
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
    if x_test.ndim == 3:
        x_test = x_test[..., np.newaxis]

    # Resize to ViT input size
    x_train = tf.image.resize(tf.convert_to_tensor(x_train), (image_size, image_size)).numpy()
    x_test  = tf.image.resize(tf.convert_to_tensor(x_test ), (image_size, image_size)).numpy()

    # Normalize to [0,1]
    if x_train.dtype != np.float32:
        x_train = x_train.astype("float32")
        x_test  = x_test.astype("float32")
    maxv = max(float(x_train.max()), 1.0)
    x_train /= maxv
    x_test  /= maxv

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    return (x_train, y_train), (x_test, y_test), num_classes

# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser(description="Run ViT on Subject_1 ultrasound data with progress bars.")
    # Paths / data
    parser.add_argument("--root", type=str,
        default=r"C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload",
        help="Root folder containing 'mirror' and 'perp'.")
    parser.add_argument("--mode", type=str, choices=["mirror", "perp"], default="mirror",
        help="Dataset mode: mirror or perp.")
    parser.add_argument("--subject", type=str, default="Subject_1",
        help="Subject folder name.")
    parser.add_argument("--image-size", type=int, default=320,
        help="Model input size (pixels).")
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split from training set.")
    parser.add_argument("--progress", type=str, choices=["tqdm", "none"], default="tqdm",
        help="Use tqdm progress bars (tqdm) or Keras logging only (none).")
    # Save / load
    parser.add_argument("--load-model", type=str, default="", help="Path to an existing .keras model to load (skip training if provided).")
    parser.add_argument("--save-model", type=str, default="", help="Path to save trained model, e.g., results/vit_mirror_subject1.keras")
    parser.add_argument("--out", type=str, default="", help="Path to save metrics JSON, e.g., results/subject1_vit.json")
    parser.add_argument("--cm", type=str, default="", help="Path to save confusion matrix PNG, e.g., results/figs/subject1_vit_cm.png")

    args = parser.parse_args()
    set_seed(args.seed)

    # Load data
    (x_train, y_train), (x_test, y_test), num_classes = load_subject_arrays(
        args.root, args.mode, args.subject, args.image_size
    )
    input_shape = (args.image_size, args.image_size, 1)

    # Build or load model
    if args.load_model and os.path.isfile(args.load_model):
        print(f"Loading model from: {args.load_model}")
        model = keras.models.load_model(args.load_model, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
        )
        trained = True
    else:
        model = build_vit(
            input_shape=input_shape,
            num_classes=num_classes,
            patch_size=32,
            projection_dim=64,
            num_heads=8,
            transformer_layers=6,
            transformer_units=(128, 64),
            mlp_head_units=(512, 256),
        )
        trained = False

    # Choose callbacks / logging
    callbacks = []
    verbose = 0 if args.progress == "tqdm" else 2  # let tqdm handle printing

    if args.progress == "tqdm":
        callbacks.append(TqdmProgress(enable=True))

    # Train (unless we loaded a pre-trained model)
    if not trained:
        # Note: with verbose=0, Keras won’t print per-batch/epoch lines; tqdm shows the progress instead.
        history = model.fit(
            x_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.val_split,
            callbacks=callbacks,
            verbose=verbose
        )

    # Evaluate on test
    logits = model.predict(x_test, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"\n[{args.mode}/{args.subject}] Test Accuracy: {acc:.4f}")
    print(f"[{args.mode}/{args.subject}] Macro Precision: {prec:.4f}  Macro Recall: {rec:.4f}  Macro F1: {f1:.4f}")

    # Save CM and model/metrics if requested
    if args.cm:
        save_confusion_matrix_png(y_test, y_pred, args.cm)
        print(f"Saved confusion matrix to: {args.cm}")

    if args.save_model:
        ensure_dir(args.save_model)
        model.save(args.save_model)
        print(f"Saved model to: {args.save_model}")

    if args.out:
        result = {
            "model": "vit",
            "mode": args.mode,
            "subject": args.subject,
            "n_classes": int(num_classes),
            "metrics": {
                "accuracy": float(acc),
                "precision_macro": float(prec),
                "recall_macro": float(rec),
                "f1_macro": float(f1),
            },
            "confusion_matrix_path": args.cm if args.cm else "",
            "params": {
                "image_size": args.image_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "val_split": args.val_split,
            },
        }
        dump_json(result, args.out)
        print(f"Saved metrics JSON to: {args.out}")

if __name__ == "__main__":
    main()
