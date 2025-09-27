from pathlib import Path

# Root directory to scan
src_root = Path("crops_mergedDataset_Augmentedv2_1")

# Supported image extensions
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


# Counters
subfolder_counts = {}
total_images = 0

# Scan recursively
for img_path in src_root.rglob("*.*"):
    if img_path.suffix.lower() not in valid_exts:
        continue

    relative_folder = img_path.relative_to(src_root).parent
    folder_key = str(relative_folder) or "[root]"
    subfolder_counts[folder_key] = subfolder_counts.get(folder_key, 0) + 1
    total_images += 1

# 📋 Summary
print("\n📁 Image Count per Subfolder:")
for folder, count in sorted(subfolder_counts.items()):
    print(f"  - {folder}: {count} images")

print(f"\n📦 Total images across all subfolders: {total_images}")