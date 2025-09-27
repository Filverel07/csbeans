import os
import shutil
import random
from collections import defaultdict

# ✅ Config
SOURCE_DIR = 'datasets/defect_Unlabeled_aug/Images'
DEST_ROOT = 'datasets/defect_split_cls'
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
SEED = 42

# 🧹 Normalize class names for folder compatibility
def normalize_class_name(name):
    return name.replace(' ', '').replace('-', '').replace('_', '')

# 🧠 Extract class name from filename
def extract_class_name(filename):
    parts = filename.split('_')
    if len(parts) >= 2:
        return normalize_class_name(parts[1])
    return 'unknown'

# 📦 Group images by class
def group_images_by_class(source_dir):
    class_to_images = defaultdict(list)
    for fname in os.listdir(source_dir):
        if fname.lower().endswith('.jpg'):
            class_name = extract_class_name(fname)
            class_to_images[class_name].append(fname)
    return class_to_images

# ✂️ Split images into train/val/test
def split_images(images, ratios):
    random.seed(SEED)
    random.shuffle(images)
    n = len(images)
    train_end = int(ratios['train'] * n)
    val_end = train_end + int(ratios['val'] * n)
    return {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

# 📁 Organize into YOLOv12s-cls format
def organize_for_yolov12_cls(class_to_images, source_dir, dest_root, ratios):
    summary = defaultdict(lambda: defaultdict(int))  # summary[class][split] = count
    for class_name, images in class_to_images.items():
        splits = split_images(images, ratios)
        for split_name, split_imgs in splits.items():
            split_class_dir = os.path.join(dest_root, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img_name in split_imgs:
                src = os.path.join(source_dir, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy2(src, dst)
                summary[class_name][split_name] += 1

    # 📊 Logging summary
    print("\n📊 Dataset Split Summary:")
    for class_name in sorted(summary.keys()):
        print(f"  🏷️ {class_name}: ", end='')
        for split in ['train', 'val', 'test']:
            count = summary[class_name].get(split, 0)
            print(f"{split}={count} ", end='')
        print()

# 🚀 Run
if __name__ == '__main__':
    class_to_images = group_images_by_class(SOURCE_DIR)
    organize_for_yolov12_cls(class_to_images, SOURCE_DIR, DEST_ROOT, SPLIT_RATIOS)
    print("\n✅ Dataset prepared for YOLOv12s-cls training.")