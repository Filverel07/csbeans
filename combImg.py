import os
from PIL import Image

# === CONFIG ===
input_dir = "datasets/defect_Unlabeled_aug/Images"  # Folder with 500x500 images
output_dir = "stitched_Pages"
tile_size = (224, 224)  # Width, Height of each image
page_size = (3400, 5500)  # Long bond paper size in pixels
bg_color = (255, 255, 255)

# === PREP ===
os.makedirs(output_dir, exist_ok=True)
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
image_files.sort()

tiles_per_row = page_size[0] // tile_size[0]
tiles_per_col = page_size[1] // tile_size[1]
tiles_per_page = tiles_per_row * tiles_per_col

# === STITCHING LOOP ===
page_count = 0
for i in range(0, len(image_files), tiles_per_page):
    page_images = image_files[i:i + tiles_per_page]
    stitched = Image.new('RGB', page_size, color=bg_color)

    for idx, filename in enumerate(page_images):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).resize(tile_size)
        x = (idx % tiles_per_row) * tile_size[0]
        y = (idx // tiles_per_row) * tile_size[1]
        stitched.paste(img, (x, y))

    output_path = os.path.join(output_dir, f"page_{page_count + 1}.jpg")
    stitched.save(output_path)
    print(f"✅ Saved: {output_path}")
    page_count += 1