import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Define class labels
YOLO_CLASS = [
    "Broken", "Cut", "DryCherry", "Fade", "Floater", "FullBlack", "FullSour",
    "FungusDamage", "Husk", "Immature", "Parchment", "PartialBlack",
    "PartialSour", "SevereInsectDamage", "Shell", "SlightInsectDamage", "Withered"
]

# Load model
cls_model = YOLO("runs/classify/yolov12_cls_detNew/weights/best.pt")

def classify_images(input_folder, output_csv="classification_results.csv"):
    image_paths = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
    results = []

    for img_path in tqdm(image_paths, desc="Classifying"):
        # Run inference
        pred = cls_model(img_path)[0]
        top_idx = int(pred.probs.top1)
        top_conf = float(pred.probs.top1conf)
        top_label = YOLO_CLASS[top_idx] if top_idx < len(YOLO_CLASS) else "Unknown"

        results.append({
            "filename": img_path.name,
            "predicted_class": top_label,
            "confidence": round(top_conf, 4)
        })

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Classification complete. Results saved to: {output_csv}")

# Example usage
if __name__ == "__main__":
    classify_images("datasets/defect_Unlabeled_aug/Images")