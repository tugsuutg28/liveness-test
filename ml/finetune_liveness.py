"""Fine-tune the baseline liveness model using constant-padded face crops."""

from pathlib import Path
import json

import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("datasets/liveness_prepared_constant")

BASELINE_MODEL = Path("models/liveness_baseline.keras")
OUTPUT_MODEL = Path("models/liveness_finetuned.keras")
REPORT_PATH = Path("reports/liveness_finetuned_metrics.json")

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 1e-4
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# Load images with explicit labels
# --------------------------------------------------

def load_dataset(split, shuffle):

    paths = []
    labels = []

    for class_name, class_label in [
        ("real", 0),
        ("fake", 1),
    ]:
        folder = DATA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(folder)

        images = sorted(folder.glob("*.jpg"))

        for path in images:
            paths.append(str(path))
            labels.append(class_label)

    if not paths:
        raise RuntimeError(f"No images found for {split}")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def decode_image(path, label):

        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)

        image = tf.image.resize(
            image,
            IMAGE_SIZE,
        )

        image = tf.cast(image, tf.float32)

        # Do not divide by 255.
        # The baseline model already contains preprocessing.

        return image, tf.cast(label, tf.float32)

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"{split}: {len(paths)} images")

    return dataset


train_ds = load_dataset("train", shuffle=True)
val_ds = load_dataset("val", shuffle=False)


# --------------------------------------------------
# Load baseline model
# --------------------------------------------------

if not BASELINE_MODEL.exists():
    raise FileNotFoundError(BASELINE_MODEL)

model = tf.keras.models.load_model(
    BASELINE_MODEL,
    compile=False,
)

print("\nBaseline model loaded.")


# --------------------------------------------------
# Freeze backbone and train classifier head
# --------------------------------------------------

for layer in model.layers:
    layer.trainable = False

classifier_found = False

for layer in reversed(model.layers):

    if isinstance(layer, tf.keras.layers.Dense):
        layer.trainable = True
        classifier_found = True
        print(f"Trainable classifier: {layer.name}")
        break

if not classifier_found:
    raise RuntimeError(
        "No Dense classification layer found."
    )


# --------------------------------------------------
# Compile model
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)


# --------------------------------------------------
# Train
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

metrics = model.evaluate(
    val_ds,
    return_dict=True,
    verbose=1,
)


# --------------------------------------------------
# Save model and report
# --------------------------------------------------

OUTPUT_MODEL.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

model.save(OUTPUT_MODEL)

report = {
    "model": str(OUTPUT_MODEL),
    "baseline": str(BASELINE_MODEL),
    "dataset": str(DATA_DIR),
    "class_mapping": {
        "real": 0,
        "fake": 1,
    },
    "epochs_completed": len(history.history["loss"]),
    "validation_metrics": {
        key: float(value)
        for key, value in metrics.items()
    },
    "history": {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    },
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 45)
print("LIVENESS FINE-TUNING COMPLETE")
print("=" * 45)

for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

print(f"\nSaved model: {OUTPUT_MODEL}")
print(f"Saved report: {REPORT_PATH}")