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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# =========================
# Configuration
# =========================
YOLO_CLASSES = [
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

DEFAULT_EXPORT_DIR = "exports/yolocls_224_batch4_test"
TARGET_SIZE = (224, 224)
CONF_THRESHOLD_CLS = 0.5

cls_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(TARGET_SIZE),
    T.ToTensor(),
])

# basin mag add ta ug normalization later for some models if trained with imagenet weights
# T.Normalize(mean=[0.485, 0.456, 0.406],

# =========================
# Model Loader
# =========================
@st.cache_resource
def load_models(device: str = 'cuda') -> Tuple[YOLO, YOLO, torch.nn.Module, torch.nn.Module, torch.device]:
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    det_model = YOLO("models/newModels/objDet/yolo_objDet.pt").to(device)
    yolo_cls = YOLO("models/newModels/Classify/yoloCLS_S.pt").to(device)
    effnet = torch.jit.load("models/newModels/Classify/effnet_27epoch.pt", map_location=device).eval()
    mobilenet = torch.jit.load("models/newModels/Classify/mobilenetv3s_best_ts.pt", map_location=device).eval()
    return det_model, yolo_cls, effnet, mobilenet, device

# =========================
# Detection and Cropping
# =========================
def detect_objects(image: np.ndarray, model: YOLO, conf_threshold: float = 0.25) -> Any:
    return model(image, conf=conf_threshold, iou=0.8, verbose=False)

def crop_detections(image: np.ndarray, results: Any, margin: float = 0) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    crops = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return crops

    img_h, img_w = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
        bw, bh = x2 - x1, y2 - y1
        dx, dy = (int(bw * margin), int(bh * margin)) if margin < 1 else (int(margin), int(margin))
        x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
        x2, y2 = min(img_w, x2 + dx), min(img_h, y2 + dy)
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append((crop, (x1, y1, x2, y2)))
    return crops

# =========================
# Classification
# =========================
def classify_batch(crops: List[np.ndarray], model_type: str, yolo_model=None, effnet=None, 
                  mobilenet=None, device="cpu", top_k: int = 3) -> List[List[Tuple[str, float]]]:
    if not crops:
        return []
    
    batch_tensors = [cls_transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).to(device) for crop in crops]
    batch = torch.stack(batch_tensors)
    
    try:
        if model_type == "YOLOv12":
            results = yolo_model(batch)
            probs_list = []
            # Handle both single result and list of results
            results_iter = results if isinstance(results, (list, tuple)) else [results]
            for result in results_iter:
                if hasattr(result, 'probs') and result.probs is not None:
                    prob = result.probs.data.cpu().numpy()
                    probs_list.append(prob if prob.ndim > 1 else prob[np.newaxis, :])
                else:
                    # Fallback: return zero probabilities if result is invalid
                    prob = np.zeros(len(YOLO_CLASSES), dtype=np.float32)
                    probs_list.append(prob[np.newaxis, :])
            probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((len(crops), len(YOLO_CLASSES)))
        else:
            model = effnet if model_type == "EfficientNet" else mobilenet
            with torch.no_grad():
                outputs = model(batch)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        # Return default probabilities for all crops to maintain length
        probs = np.zeros((len(crops), len(YOLO_CLASSES)), dtype=np.float32)
    
    # Ensure probs length matches number of crops
    if probs.shape[0] != len(crops):
        st.warning(f"Mismatch in prediction count: expected {len(crops)}, got {probs.shape[0]}. Padding with zeros.")
        probs = np.zeros((len(crops), len(YOLO_CLASSES)), dtype=np.float32)
    
    batch_predictions = []
    for prob in probs:
        top_indices = prob.argsort()[-top_k:][::-1]
        batch_predictions.append([(YOLO_CLASSES[idx], float(prob[idx])) for idx in top_indices])
    
    return batch_predictions

# =========================
# Visualization
# =========================
def visualize_results(image: np.ndarray, crops: List[Tuple], predictions: List[Dict]) -> np.ndarray:
    viz_img = image.copy()
    img_h, _ = image.shape[:2]
    base_scale = max(0.0005 * img_h, 0.5)
    font_thickness = max(int(base_scale * 2), 1)

    for (crop, (x1, y1, x2, y2)), pred in zip(crops, predictions):
        top1_class = pred["Top1 Class"]
        top1_conf = pred["Top1 Confidence"]
        crop_id = pred["Crop ID"]
        color_idx = YOLO_CLASSES.index(top1_class) if top1_class in YOLO_CLASSES else -1
        color = COLORS[color_idx % len(COLORS)] if color_idx >= 0 else (100, 100, 100)
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 3)
        label = f"ID{crop_id} {top1_class}: {top1_conf:.2f}"
        cv2.putText(viz_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, base_scale, color, font_thickness)
    return viz_img

# =========================
# Pipeline
# =========================
def process_image(image_bytes: bytes, det_model: YOLO, yolo_cls: YOLO, effnet, mobilenet,
                  device: torch.device, conf_threshold: float, filename: str, 
                  classifier_choice: str, export_dir: str, margin: float):
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = detect_objects(image_bgr, det_model, conf_threshold)
    if not results[0].boxes:
        st.warning(f"No detections for {filename}")
        return Image.fromarray(image), [], pd.DataFrame()
    
    crops = crop_detections(image_bgr, results, margin)
    if not crops:
        st.warning(f"No valid crops extracted for {filename}")
        return Image.fromarray(image), [], pd.DataFrame()
    
    crop_images = [crop for crop, _ in crops]
    batch_predictions = classify_batch(crop_images, classifier_choice, yolo_cls, effnet, mobilenet, device)
    
    # Validate lengths
    if len(batch_predictions) != len(crops):
        st.error(f"Prediction mismatch: {len(batch_predictions)} predictions for {len(crops)} crops in {filename}")
        return Image.fromarray(image), [], pd.DataFrame()
    
    predictions, crop_pils = [], []
    base_name = os.path.splitext(filename)[0]
    
    for (i, (crop, coords)), top_preds in zip(enumerate(crops), batch_predictions):
        pred_entry = {"Crop ID": i + 1}
        for rank, (cls_name, conf) in enumerate(top_preds, 1):
            pred_entry[f"Top{rank} Class"] = cls_name
            pred_entry[f"Top{rank} Confidence"] = conf
        predictions.append(pred_entry)
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cls_transform(crop_rgb)
        resized_pil = T.ToPILImage()(resized)
        export_path = Path(export_dir) / top_preds[0][0]
        export_path.mkdir(parents=True, exist_ok=True)
        resized_pil.save(export_path / f"{base_name}_crop{i+1}.jpg")
        crop_pils.append(resized_pil)
    
    viz_bgr = visualize_results(image_bgr, crops, predictions)
    viz_pil = Image.fromarray(cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB))
    
    annotated_path = Path(export_dir) / "annotated_inputs"
    annotated_path.mkdir(parents=True, exist_ok=True)
    viz_pil.save(annotated_path / f"{base_name}_annotated.jpg")
    
    df = pd.DataFrame(predictions)
    csv_path = Path(export_dir) / "csv_exports"
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f"{base_name}_predictions.csv", index=False)
    
    return viz_pil, crop_pils, df

# =========================
# LiveCam Processor
# =========================
class LiveCamProcessor(VideoTransformerBase):
    def __init__(self, det_model, yolo_cls, effnet, mobilenet, device, conf_threshold, classifier_choice, margin):
        self.det_model = det_model
        self.yolo_cls = yolo_cls
        self.effnet = effnet
        self.mobilenet = mobilenet
        self.device = device
        self.conf_threshold = conf_threshold
        self.classifier_choice = classifier_choice
        self.margin = margin
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Use recv instead of transform. Accepts an av.VideoFrame and returns an av.VideoFrame.
        Preserves input frame timing (pts / time_base).
        """
        self.frame_count += 1
        image = frame.to_ndarray(format="bgr24")
        if self.frame_count % 8 != 0:  # adjust processing frequency
            return frame

        results = detect_objects(image, self.det_model, self.conf_threshold)
        crops = crop_detections(image, results, self.margin)
        crop_images = [crop for crop, _ in crops]

        batch_predictions = classify_batch(
            crop_images, self.classifier_choice,
            self.yolo_cls, self.effnet, self.mobilenet, self.device
        )

        # Ensure predictions match crops
        if len(batch_predictions) != len(crops):
            return frame  # Skip visualization if mismatch occurs

        predictions = [{"Crop ID": i + 1, "Top1 Class": preds[0][0], "Top1 Confidence": preds[0][1]}
                       for i, preds in enumerate(batch_predictions)]

        processed = visualize_results(image, crops, predictions)  # BGR ndarray

        out_frame = av.VideoFrame.from_ndarray(processed, format="bgr24")
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base
        return out_frame

# =========================
# Streamlit App
# =========================
def main():
    st.title("Coffee Bean Defect Detection ☕ (Top-3 Classification)")
    
    st.sidebar.header("Configuration")
    device_choice = st.sidebar.selectbox("Device", ["cpu", "cuda"])
    classifier_choice = st.sidebar.selectbox("Classification Model", ["YOLOv12", "EfficientNet", "MobileNet"])
    export_dir = st.sidebar.text_input("Export Directory", DEFAULT_EXPORT_DIR)
    margin = st.sidebar.slider("Crop Margin (pixels)", 0, 50, 0, 5)
    conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

    with st.spinner("Loading models..."):
        det_model, yolo_cls, effnet, mobilenet, device = load_models(device_choice)

    st.header("Live Camera Detection 🎥")
    if st.checkbox("Enable Live Camera"):
        with st.spinner("Initializing live camera..."):
            webrtc_streamer(
                key="livecam",
                video_processor_factory=lambda: LiveCamProcessor(
                    det_model, yolo_cls, effnet, mobilenet, device, conf_threshold, classifier_choice, margin
                )
            )

    st.sidebar.header("Upload Images")
    uploaded_files = st.sidebar.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("Please upload at least one image.")
        return

    all_dfs = []
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        viz_img, crop_imgs, df = process_image(
            uploaded_file.read(), det_model, yolo_cls, effnet, mobilenet,
            device, conf_threshold, uploaded_file.name, classifier_choice, export_dir, margin
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(viz_img, caption="Detections + Top-1 Classification", use_container_width=True)
        with col2:
            if crop_imgs:
                for i, crop_img in enumerate(crop_imgs[:3]):
                    st.image(crop_img, caption=f"Crop {i+1}", use_container_width=True)
                if len(crop_imgs) > 3:
                    st.info(f"... and {len(crop_imgs)-3} more crops")

        if not df.empty:
            st.dataframe(df)
            all_dfs.append(df)
            st.success(f"CSV saved to {export_dir}/csv_exports/{os.path.splitext(uploaded_file.name)[0]}_predictions.csv")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        st.subheader("Summary of Predictions (Top-3 per Crop)")
        st.dataframe(combined_df)
        csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Combined CSV", csv, "yolo_predictions.csv", "text/csv")

if __name__ == "__main__":
    main()