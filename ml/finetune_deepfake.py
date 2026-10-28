"""Fine-tune the existing deepfake MobileNetV2 model."""

import json
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets" / "deepfake"
MODEL_PATH = ROOT / "models" / "deepfake_baseline.keras"
OUTPUT_PATH = ROOT / "models" / "deepfake_finetuned.keras"
REPORT_PATH = ROOT / "reports" / "deepfake_finetuned_metrics.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5


def load_dataset(split, shuffle=False):
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        class_names=["real", "fake"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


print("Loading datasets...")

train_ds = load_dataset("train", shuffle=True)
val_ds = load_dataset("val")

print("\nLoading baseline model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

# Find MobileNetV2 inside the saved model.
backbone = next(
    (
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenet" in layer.name.lower()
    ),
    None,
)

if backbone is None:
    raise RuntimeError("MobileNetV2 backbone not found.")

# Allow fine-tuning.
backbone.trainable = True

# Freeze all except the last 30 layers.
for layer in backbone.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen.
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

print("\nStarting fine-tuning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

model.save(OUTPUT_PATH)

print(f"\nFine-tuned model saved: {OUTPUT_PATH}")

validation_results = model.evaluate(
    val_ds,
    return_dict=True,
    verbose=0,
)

report = {
    "model": "deepfake_finetuned",
    "baseline": str(MODEL_PATH),
    "validation_metrics": {
        key: float(value)
        for key, value in validation_results.items()
    },
    "history": {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    },
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2))

print("\nFine-tuned validation results:")
print(json.dumps(report["validation_metrics"], indent=2))