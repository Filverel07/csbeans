import os
from PIL import Image
from collections import defaultdict

# === CONFIG ===
input_dir = "datasets/defect_Unlabeled_aug/Images"
output_dir = "stitched_Pages/512_by_512"
tile_size = (512, 512)
page_size = (3400, 5500)
bg_color = (255, 255, 255)

# === PREP ===
os.makedirs(output_dir, exist_ok=True)
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
image_files.sort()

tiles_per_row = page_size[0] // tile_size[0]
tiles_per_col = page_size[1] // tile_size[1]
tiles_per_page = tiles_per_row * tiles_per_col

# === GROUP BY CLASS ===
def extract_classname(filename):
    parts = os.path.splitext(filename)[0].split('_')
    return parts[1] if len(parts) >= 2 else "unknown"

class_groups = defaultdict(list)
for f in image_files:
    cls = extract_classname(f)
    class_groups[cls].append(f)

# === STITCHING LOOP PER CLASS ===
for cls_name, cls_images in class_groups.items():
    cls_output_dir = os.path.join(output_dir, cls_name)
    os.makedirs(cls_output_dir, exist_ok=True)

    for i in range(0, len(cls_images), tiles_per_page):
        page_images = cls_images[i:i + tiles_per_page]
        stitched = Image.new('RGB', page_size, color=bg_color)

        for idx, filename in enumerate(page_images):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).resize(tile_size)
            x = (idx % tiles_per_row) * tile_size[0]
            y = (idx // tiles_per_row) * tile_size[1]
            stitched.paste(img, (x, y))

        output_path = os.path.join(cls_output_dir, f"page_{(i // tiles_per_page) + 1}.jpg")
        stitched.save(output_path)
        print(f"✅ Saved: {output_path}")