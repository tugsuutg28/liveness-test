"""Train MobileNetV2 for face liveness, deepfake, or CIFAKE classification.

Expected:
datasets/<kind>/
    train/real/
    train/fake/
    val/real/
    val/fake/

Optional:
    test/real/
    test/fake/

Class labels:
    real = 0
    fake = 1

Model output:
    Probability of fake/spoof.
"""

import argparse
import json
from pathlib import Path


def run(kind, epochs, batch, weights):
    import tensorflow as tf

    root = Path(__file__).resolve().parents[1]
    base_dir = root / "datasets" / kind

    # Training and validation are required.
    for split in ("train", "val"):
        for label in ("real", "fake"):
            folder = base_dir / split / label

            if not folder.exists() or not any(folder.glob("*")):
                raise SystemExit(f"Missing images: {folder}")

    # Test is optional, but must contain both classes if provided.
    test_counts = {
        label: len(list((base_dir / "test" / label).glob("*")))
        for label in ("real", "fake")
    }

    if all(test_counts.values()):
        has_test = True
    elif any(test_counts.values()):
        raise SystemExit(
            "Incomplete test dataset: both real and fake images are required."
        )
    else:
        has_test = False

    def ds(split, shuffle):
        dataset = tf.keras.utils.image_dataset_from_directory(
            base_dir / split,
            class_names=["real", "fake"],
            image_size=(160, 160),
            batch_size=batch,
            shuffle=shuffle,
            seed=42,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    train = ds("train", True)
    val = ds("val", False)
    test = ds("test", False) if has_test else None

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        pooling="avg",
        weights=weights,
    )

    backbone.trainable = False

    inp = tf.keras.Input((160, 160, 3))

    x = tf.keras.layers.RandomFlip("horizontal")(inp)
    x = tf.keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1,
    )(x)

    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    print(f"\nTraining {kind} classifier...\n")

    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        callbacks=callbacks,
    )

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"{kind}.keras"
    model.save(model_path)

    print(f"\nModel saved: {model_path}")

    # Validation metrics are not independent test metrics.
    validation_results = model.evaluate(
        val,
        return_dict=True,
        verbose=0,
    )

    test_results = None

    if test is not None:
        test_results = model.evaluate(
            test,
            return_dict=True,
            verbose=0,
        )

    report = {
        "kind": kind,
        "classes": ["real", "fake"],
        "score_meaning": "P(fake)",
        "validation_metrics": {
            k: float(v)
            for k, v in validation_results.items()
        },
        "test_metrics": (
            {
                k: float(v)
                for k, v in test_results.items()
            }
            if test_results is not None
            else None
        ),
        "history": {
            k: [float(v) for v in values]
            for k, values in history.history.items()
        },
        "weights": weights,
        "warning": (
            "Subject-independent train/validation separation "
            "has not been verified. Validation metrics are "
            "not independent test results or deployment guarantees."
        ),
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / f"{kind}_metrics.json"

    report_path.write_text(
        json.dumps(report, indent=2)
    )

    print("\nValidation results:")
    print(json.dumps(report["validation_metrics"], indent=2))

    if test_results is None:
        print("\nNo independent test dataset available.")
        print("Test metrics were not calculated.")
    else:
        print("\nTest results:")
        print(json.dumps(report["test_metrics"], indent=2))

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kind",
        choices=["liveness", "deepfake", "cifake"],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
    )

    args = parser.parse_args()

    run(
        args.kind,
        args.epochs,
        args.batch,
        None if args.weights == "none" else args.weights,
    )