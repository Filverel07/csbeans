import os
import random
import shutil
import yaml
from pathlib import Path

# --- CONFIG ---
SOURCE_DIR = Path("/home/jap/projects/yolov12/crop/224_v3_2_Balanced")
EXPORT_DIR = Path("datasets/224_v3_2__balanced_split")
EXPORT_YAML = EXPORT_DIR / "dataset.yaml"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
IMG_EXT = ".jpg"

def split_dataset():
    """Split images from each class folder into train/val/test folders."""
    for class_folder in SOURCE_DIR.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        images = list(class_folder.glob(f"*{IMG_EXT}"))
        random.shuffle(images)

        total = len(images)
        n_train = int(SPLIT_RATIOS["train"] * total)
        n_val = int(SPLIT_RATIOS["val"] * total)
        n_test = total - n_train - n_val  # remainder

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, split_imgs in splits.items():
            split_class_dir = EXPORT_DIR / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_imgs:
                shutil.copy(img_path, split_class_dir / img_path.name)

    print(f"✅ Dataset split complete. Output saved to: {EXPORT_DIR}")

def generate_yaml():
    """Create dataset.yaml with split paths and class names."""
    class_names = sorted([
        d.name for d in (EXPORT_DIR / "train").iterdir()
        if d.is_dir()
    ])

    data = {
        "train": str(EXPORT_DIR / "train"),
        "val": str(EXPORT_DIR / "val"),
        "test": str(EXPORT_DIR / "test"),
        "names": class_names
    }

    with open(EXPORT_YAML, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"📄 YAML file saved to: {EXPORT_YAML}")

if __name__ == "__main__":
    split_dataset()
    generate_yaml()