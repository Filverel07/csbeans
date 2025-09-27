from PIL import Image
from pathlib import Path

# Source and destination folders
src_dir = Path("datasets/defect_Unlabeled")
dst_dir = Path("datasets/defect_Unlabeled_aug")
dst_dir.mkdir(parents=True, exist_ok=True)

# Rotation angles and prefixes
angles = [45, 90, 135, 180, 225, 270]

# Loop through all image files
for img_path in src_dir.glob("*.*"):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {img_path}: {e}")
        continue

    for angle in angles:
        rotated = img.rotate(angle, expand=True)  # expand avoids cropping corners
        prefix = f"rot{angle}_"
        rotated.save(dst_dir / f"{prefix}{img_path.name}")

print("✅ Augmentation complete.")