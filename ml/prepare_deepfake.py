import zipfile
import random
import shutil
from pathlib import Path

ZIP_PATH = Path(
    "datasets/deepfake_source/deepfake-and-real-images.zip"
)

OUTPUT_DIR = Path("datasets/deepfake")

SPLITS = {
    "Train": ("train", 5000),
    "Validation": ("val", 1000),
    "Test": ("test", 1000),
}

CLASSES = {
    "Real": "real",
    "Fake": "fake",
}

random.seed(42)

with zipfile.ZipFile(ZIP_PATH) as z:
    all_files = z.namelist()

    for source_split, (target_split, count) in SPLITS.items():
        for source_class, target_class in CLASSES.items():

            prefix = f"Dataset/{source_split}/{source_class}/"

            images = [
                name for name in all_files
                if name.startswith(prefix)
                and name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            if len(images) < count:
                raise ValueError(
                    f"Not enough images in {prefix}"
                )

            selected = random.sample(images, count)

            output_folder = (
                OUTPUT_DIR / target_split / target_class
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            print(
                f"Extracting {count} images "
                f"to {output_folder}"
            )

            for image_path in selected:
                filename = Path(image_path).name

                destination = output_folder / filename

                with z.open(image_path) as source:
                    with open(destination, "wb") as target:
                        shutil.copyfileobj(source, target)

print("\nDeepfake dataset preparation complete!")   