import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "/home/jap/projects/yolov12/runs/detect/200epochs_LT_FT/weights/best.pt"
SOURCE_DIR = "/home/jap/projects/yolov12/datasets/mergedDataset_Augmentedv2_1"
EXPORT_DIR = "/home/jap/projects/yolov12/crops2_mergedDataset_Augmentedv2_1"
TARGET_SIZE = (224, 224)
CONF_THRESHOLD = 0.25
MAX_DET = 1  # Raise detection cap

# --- LOAD MODEL ---
model = YOLO(MODEL_PATH)

# --- TRANSFORM ---
resize_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
])

def process_image(img_path: Path, class_name: str, crop_counter: dict):
    """Run detection, crop, resize, and save crops by original folder name."""
    img = Image.open(img_path).convert("RGB")
    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False, max_det=MAX_DET)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        print(f"📸 {img_path.name}: {len(boxes)} detections")

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Crop and resize
            cropped = img.crop((x1, y1, x2, y2))
            resized = resize_transform(cropped)

            # Export path
            export_path = Path(EXPORT_DIR) / class_name
            export_path.mkdir(parents=True, exist_ok=True)

            # Unique filename
            filename = f"{class_name}_{crop_counter[class_name]:04d}.jpg"
            resized.save(export_path / filename)

            # Update counter
            crop_counter[class_name] += 1

def run_detection():
    """Iterate through dataset folders and process all images."""
    crop_counter = {}
    total_crops = 0

    for class_folder in Path(SOURCE_DIR).iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        crop_counter[class_name] = 0

        for img_file in class_folder.glob("*.jpg"):
            try:
                process_image(img_file, class_name, crop_counter)
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")

        print(f"✅ {class_name}: {crop_counter[class_name]} crops")

    total_crops = sum(crop_counter.values())
    print(f"\n🎯 Total crops across all classes: {total_crops}")
    print(f"📂 Cropped patches saved to: {EXPORT_DIR}")

if __name__ == "__main__":
    run_detection()