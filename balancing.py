from pathlib import Path
from collections import defaultdict
import shutil
import random

# 📂 Source and destination paths
src_root = Path("crop/224_v3_2")
dst_root = Path("crop/224_v3_2_Balanced")
dst_root.mkdir(parents=True, exist_ok=True)

# 🖼️ Valid image extensions
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# 🧮 Collect image paths per class
class_images = defaultdict(list)
for class_folder in src_root.iterdir():
    if not class_folder.is_dir():
        continue
    for img_path in class_folder.glob("*"):
        if img_path.suffix.lower() in valid_exts:
            class_images[class_folder.name].append(img_path)

# 📊 Print original class counts
print("\n📊 Original Class Distribution:")
for cls, imgs in sorted(class_images.items()):
    print(f"  {cls:<25} → {len(imgs):>5} images")

# 🧹 Balance classes by downsampling
min_count = min(len(imgs) for imgs in class_images.values())
print(f"\n⚖️ Balancing all classes to {min_count} images each...")

for cls, imgs in class_images.items():
    sampled_imgs = random.sample(imgs, min_count)
    dst_cls_folder = dst_root / cls
    dst_cls_folder.mkdir(parents=True, exist_ok=True)

    for img_path in sampled_imgs:
        shutil.copy(img_path, dst_cls_folder / img_path.name)

# ✅ Final check
print("\n📦 Final Balanced Class Distribution:")
for cls_folder in dst_root.iterdir():
    if cls_folder.is_dir():
        count = sum(1 for img in cls_folder.glob("*") if img.suffix.lower() in valid_exts)
        print(f"  {cls_folder.name:<25} → {count:>5} images")

print("\n✅ Dataset balanced and saved to:", dst_root)