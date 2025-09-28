import os
import io
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
#THIS IS OLD CODE
# =========================
# Classes
# =========================
YOLO_CLASS = [
    "Broken", "Dry_Cherry", "Fade", "Floater", "Foreign_Items", "Full_Black", "Full_Sour",
    "Fungus_Damage", "Good", "Husk", "Immature", "Parchment", "Partial_Black",
    "Partial_Sour", "Severe_Insect_Damage", "Shell", "Slight_Insect_Damage", "Withered"
]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (128, 128, 128), (200, 100, 50), (50, 200, 100),
    (100, 50, 200), (200, 200, 100), (100, 200, 200)
]

# =========================
# Config
# =========================
EXPORT_DIR = "exports/batch3_225_Ycls"
TARGET_SIZE = (224, 224)
CONF_THRESHOLD_CLS = 0.5

cls_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(TARGET_SIZE),
    T.ToTensor(),
])

# =========================
# Model Loader
# =========================
@st.cache_resource
def load_models(device: str = 'cuda'):
    device = torch.device(device)

    # Detection and YOLOv12 classification
    det_model = YOLO("runs/detect/200epochs_LT_FT/weights/best.pt").to(device)
    yolo_cls = YOLO("runs/classify/224_v3_2_fineTuningv2/weights/best.pt").to(device)

    # EfficientNetV2-S (TorchScript)
    effnet = torch.jit.load("models/newModels/effnet_27epoch.pt", map_location=device).eval()

    # MobileNetV3-Large (TorchScript)
    mobilenet = torch.jit.load("models/mobilenetv3s_best_ts.pt", map_location=device).eval()

    return det_model, yolo_cls, effnet, mobilenet, device

# =========================
# Detection
# =========================
def detect_objects(image: np.ndarray, model: YOLO, conf_threshold: float = 0.25) -> Any:
    results = model(image, conf=conf_threshold, iou=0.4, verbose=False)
    return results

def crop_detections(image: np.ndarray, results: Any) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    crops = []
    boxes = results[0].boxes
    if boxes is None or boxes.shape[0] == 0:
        return crops

    img_h, img_w = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append((crop, (x1, y1, x2, y2)))
    return crops

# =========================
# Classification (Top-k)
# =========================
def classify_crop(image: np.ndarray, model_type: str, yolo_model=None, effnet=None, mobilenet=None,
                  device="cpu", top_k: int = 3) -> List[Tuple[str, float]]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = cls_transform(image_rgb).unsqueeze(0).to(device)

    if model_type == "YOLOv12":
        results = yolo_model(img_tensor)
        probs = results[0].probs.data.cpu().numpy().flatten()
    else:
        model = effnet if model_type == "EfficientNet" else mobilenet
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Get top-k predictions
    top_indices = probs.argsort()[-top_k:][::-1]
    top_preds = [(YOLO_CLASS[idx], float(probs[idx])) for idx in top_indices]

    return top_preds

# =========================
# Visualization
# =========================
def visualize_results(image: np.ndarray, crops: List[Tuple], predictions: List[Dict]) -> np.ndarray:
    viz_img = image.copy()
    img_h, img_w = image.shape[:2]

    # Dynamically scale font size and thickness
    base_scale = 0.0005 * img_h
    font_scale = max(base_scale, 0.5)
    font_thickness = max(int(base_scale * 2), 1)

    for (_, (x1, y1, x2, y2)), p in zip(crops, predictions):
        top1_class = p["Top1 Class"]
        top1_conf = p["Top1 Confidence"]

        color_idx = YOLO_CLASS.index(top1_class) if top1_class in YOLO_CLASS else -1
        color = COLORS[color_idx % len(COLORS)] if color_idx >= 0 else (100, 100, 100)
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 3)

        label = f"{top1_class}: {top1_conf:.2f}"
        cv2.putText(viz_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    return viz_img

# =========================
# Pipeline
# =========================
def process_image(image_bytes: bytes, det_model: YOLO, yolo_cls: YOLO, effnet, mobilenet,
                  device: torch.device, conf_threshold: float, filename: str, classifier_choice: str):
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = detect_objects(image_bgr, det_model, conf_threshold)
    if len(results[0].boxes) == 0:
        return Image.fromarray(image), [], pd.DataFrame()

    crops = crop_detections(image_bgr, results)
    predictions = []
    crop_pils = []

    base_name = os.path.splitext(filename)[0]

    for i, (crop, _) in enumerate(crops):
        top_preds = classify_crop(crop, classifier_choice, yolo_cls, effnet, mobilenet, device, top_k=3)

        # Save all top-k predictions in dataframe
        pred_entry = {"Crop ID": i + 1}
        for rank, (cls_name, conf) in enumerate(top_preds, 1):
            pred_entry[f"Top{rank} Class"] = cls_name
            pred_entry[f"Top{rank} Confidence"] = conf
        predictions.append(pred_entry)

        # Save crop under Top-1 class folder
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cls_transform(crop_rgb)
        resized_pil = T.ToPILImage()(resized)
        export_path = Path(EXPORT_DIR) / top_preds[0][0]
        export_path.mkdir(parents=True, exist_ok=True)
        save_name = f"{base_name}_crop{i+1}.jpg"
        resized_pil.save(export_path / save_name)
        crop_pils.append(resized_pil)

    viz_bgr = visualize_results(image_bgr, crops, predictions)
    viz_rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
    viz_pil = Image.fromarray(viz_rgb)

    df = pd.DataFrame(predictions)
    return viz_pil, crop_pils, df

# =========================
# Streamlit App
# =========================
def main():
    st.title("Coffee Bean Defect Detection ☕ (Top-3 Classification)")
    device_choice = st.sidebar.selectbox("Device", ["cpu", "cuda"])
    classifier_choice = st.sidebar.selectbox("Classification Model", ["YOLOv12", "EfficientNet", "MobileNet"])

    with st.spinner("Loading models..."):
        det_model, yolo_cls, effnet, mobilenet, device = load_models(device_choice)

    st.sidebar.header("Upload Images")
    uploaded_files = st.sidebar.file_uploader(
        "Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload at least one image.")
        return

    conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

    all_dfs = []
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        viz_img, crop_imgs, df = process_image(
            uploaded_file.read(), det_model, yolo_cls, effnet, mobilenet,
            device, conf_threshold, uploaded_file.name, classifier_choice
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(viz_img, caption="Detections + Top-1 Classification", use_container_width=True)
        with col2:
            if crop_imgs:
                for i, crop_img in enumerate(crop_imgs[:3]):
                    st.image(crop_img, caption=f"Crop {i+1}", use_container_width=True)
                if len(crop_imgs) > 3:
                    st.info(f"... and {len(crop_imgs)-3} more")

        if not df.empty:
            st.dataframe(df)
            all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        st.subheader("Summary of Predictions (Top-3 per Crop)")
        st.dataframe(combined_df)
        csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "yolo_predictions.csv", "text/csv")


if __name__ == "__main__":
    main()