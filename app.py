# app.py
# Updated Coffee Bean Defect Detection Pipeline (PyTorch with TorchScript Models)
# Modified to load EfficientNetV2-S and MobileNetV3-S as TorchScript models (.pt files) using torch.jit.load.
# Assumes your .pt files are TorchScript archives (e.g., from torch.jit.script(model) or torch.jit.trace).
# No need to create base models; directly load the scripted ones.
# Ensure the scripted models have the 13-class output head.
# Input sizes: EffNetV2-S expects 384x384, MobileNetV3-S 224x224 (transforms unchanged).
#
# Setup Instructions:
# 1. Install: pip install ultralytics streamlit opencv-python pillow numpy pandas torch torchvision
# 2. Update paths: Set to your TorchScript .pt files (e.g., 'models/effv2s_best_ts.pt').
# 3. If models were scripted after fine-tuning with num_classes=13, they should output 13 probs.
# 4. Run: streamlit run app.py
# 5. For GPU: Select 'cuda' in sidebar (requires CUDA).

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from ultralytics import YOLO
import io
from typing import List, Tuple, Dict, Any

# Define the 13 defect classes
CLASS_NAMES = ["Broken", "Dry_Cherries", "Floater", "Foreign_items", "Full_Black",
               "Full_Sour", "Fungus_damage", "Husk", "Immature", "Parchment",
               "Severe_Insect_Damage", "Shell", "Withered"]

NUM_CLASSES = len(CLASS_NAMES)

# Color map for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (128, 128, 128)
]

# Transforms for PyTorch models
transform_eff = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mob = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models(device: str = 'cuda'):
    """
    Load models: YOLO for det/cls, TorchScript for EffNetV2-S and MobileNetV3-S.
    Update paths to your TorchScript .pt files.
    """
    device = torch.device(device)

    # YOLO Detection
    det_model = YOLO('runs/detect/200epochs_LT_FT/weights/best.pt')  # Replace with 'path/to/your_fine_tuned_det.pt'
    det_model.to(device)

    # YOLO Classification
    cls_model_yolo = YOLO('models/yolov12s-cls_best.pt')  # Replace with 'path/to/your_fine_tuned_yolo_cls.pt'
    cls_model_yolo.to(device)

    # EfficientNetV2-S TorchScript
    eff_model = torch.jit.load('models/effv2s_best_ts.pt', map_location=device)
    eff_model.eval()

    # MobileNetV3-Small TorchScript
    mob_model = torch.jit.load('models/mobilenetv3s_best_ts.pt', map_location=device)  # Update path if different
    mob_model.eval()

    return det_model, cls_model_yolo, eff_model, mob_model, device

def detect_objects(image: np.ndarray, model: YOLO, conf_threshold: float = 0.25) -> Any:
    """
    Detect using YOLO, filter small objects (simplified).
    """
    results = model(image, conf=conf_threshold, iou=0.4, verbose=False)
    return results

def crop_detections(image: np.ndarray, results: Any, padding_pct: float = 0.1) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Crop with padding.
    """
    crops = []
    boxes = results[0].boxes
    if boxes is None:
        return crops
    
    img_h, img_w = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        pad_w, pad_h = (x2 - x1) * padding_pct, (y2 - y1) * padding_pct
        x1 = max(0, int(x1 - pad_w / 2))
        y1 = max(0, int(y1 - pad_h / 2))
        x2 = min(img_w, int(x2 + pad_w / 2))
        y2 = min(img_h, int(y2 + pad_h / 2))
        
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        crop_pil = pil_img.crop((x1, y1, x2, y2))
        crop_np = np.array(crop_pil)
        
        crops.append((crop_np, (x1, y1, x2, y2)))
    
    return crops

def classify_yolo(crop: np.ndarray, model: YOLO, device: torch.device) -> np.ndarray:
    """
    YOLO-cls inference.
    """
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = model(crop_rgb, verbose=False, device=device)
    probs = results[0].probs.data.cpu().numpy()
    probs = probs / 100.0  # Normalize to 0-1
    return probs

def classify_pytorch(crop: np.ndarray, model: torch.nn.Module, transform: transforms.Compose, device: torch.device) -> np.ndarray:
    """
    PyTorch/TorchScript model inference.
    """
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, torch.Tensor):
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        else:
            # Handle if model returns logits differently; adjust if needed
            probs = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
    
    return probs


    # ENSEMBLE LOGIC

def aggregate_predictions(probs_list: List[np.ndarray], weights: List[float] = None, threshold: float = 0.25) -> Tuple[str, float]:
    """
    Weighted average probs, argmax if > threshold.
    """
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    
    avg_probs = np.average(probs_list, axis=0, weights=weights)
    max_prob = np.max(avg_probs)
    if max_prob > threshold:
        pred_class_idx = np.argmax(avg_probs)
        pred_class = CLASS_NAMES[pred_class_idx]
        return pred_class, max_prob
    else:
        return "Unknown", max_prob

def visualize_results(image: np.ndarray, crops: List[Tuple], predictions: List[Dict]) -> np.ndarray:
    """
    Draw boxes and labels.
    """
    viz_img = image.copy()
    for i, ((_, box), p) in enumerate(zip(crops, predictions)):
        x1, y1, x2, y2 = box
        pred_class = p['Predicted Class']
        conf = p['Confidence']
        color_idx = CLASS_NAMES.index(pred_class) if pred_class != "Unknown" else 12
        color = COLORS[color_idx]
        
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 15)
        label = f"{pred_class}: {conf:.2f}"
        cv2.putText(viz_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
        
    
    return viz_img

def process_image(image_bytes: bytes, det_model: YOLO, cls_model_yolo: YOLO, eff_model: torch.nn.Module, 
                  mob_model: torch.nn.Module, device: torch.device, conf_threshold: float) -> Tuple[Image.Image, List[Image.Image], pd.DataFrame]:
    """
    Full pipeline.
    """
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results = detect_objects(image_bgr, det_model, conf_threshold)
    
    if len(results[0].boxes) == 0:
        return Image.fromarray(image), [], pd.DataFrame()
    
    crops = crop_detections(image_bgr, results)
    
    if not crops:
        return Image.fromarray(image), [], pd.DataFrame()
    
    predictions = []
    crop_pils = []
    for i, (crop, _) in enumerate(crops):
        # Classify
        probs_yolo = classify_yolo(crop, cls_model_yolo, device)
        probs_eff = classify_pytorch(crop, eff_model, transform_eff, device)
        probs_mob = classify_pytorch(crop, mob_model, transform_mob, device)
        
        pred_class, conf = aggregate_predictions([probs_yolo, probs_eff, probs_mob], threshold=conf_threshold)
        
        # Get class names and confidence scores for each model
        yolo_class = CLASS_NAMES[np.argmax(probs_yolo)]
        yolo_conf = np.max(probs_yolo)
        eff_class = CLASS_NAMES[np.argmax(probs_eff)]
        eff_conf = np.max(probs_eff)
        mob_class = CLASS_NAMES[np.argmax(probs_mob)]
        mob_conf = np.max(probs_mob)
        
        predictions.append({
            'Crop ID': i+1,
            'Predicted Class': pred_class,
            'Confidence': conf,
            'Yolov12n-cls': f"{yolo_class} ({yolo_conf*100:.2f})",
            'EffNetV2-S': f"{eff_class} ({eff_conf*100:.2f})",
            'MobileNetV3-S': f"{mob_class} ({mob_conf*100:.2f})",
        })
        
        # Crop display with text (crop is already RGB from crop_detections)
        crop_pil = Image.fromarray(crop)  # Directly use RGB crop
        draw = ImageDraw.Draw(crop_pil)
        draw.text((10, 10), f"{pred_class}: {conf:.2f}", fill="white")
        crop_pils.append(crop_pil)
    
    # Visualize
    viz_bgr = visualize_results(image_bgr, crops, predictions)
    viz_rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
    viz_pil = Image.fromarray(viz_rgb)
    
    df = pd.DataFrame(predictions)
    
    return viz_pil, crop_pils, df

# Streamlit App
def main():
    st.title("Updated Coffee Bean Defect Detection Pipeline (TorchScript)")
    st.markdown("Loading TorchScript models for EfficientNetV2-S and MobileNetV3-S.")
    
    device_choice = st.sidebar.selectbox("Device", ['cpu', 'cuda'])
    with st.spinner("Loading models..."):
        det_model, cls_model_yolo, eff_model, mob_model, device = load_models(device_choice)
    
    st.sidebar.header("Upload Images")
    uploaded_files = st.sidebar.file_uploader("Choose images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("Please upload at least one image.")
        return
    
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
    
    all_dfs = []
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        
        viz_img, crop_imgs, df = process_image(uploaded_file.read(), det_model, cls_model_yolo, eff_model, mob_model, device, conf_threshold)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(viz_img, caption="Original with Detections", use_container_width=True)
        with col2:
            if crop_imgs:
                for i, crop_img in enumerate(crop_imgs[:3]):
                    st.image(crop_img, caption=f"Crop {i+1}", use_container_width=True)
                if len(crop_imgs) > 3:
                    st.info(f"... and {len(crop_imgs)-3} more")
        
        if not df.empty:
            st.dataframe(df)
            all_dfs.append(df)
    
    if len(all_dfs) > 1:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        st.subheader("Overall Summary")
        st.dataframe(combined_df)
        if 'Predicted Class' in combined_df.columns:
            counts = combined_df[combined_df['Predicted Class'] != 'Unknown']['Predicted Class'].value_counts()
            st.bar_chart(counts)
        csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "defect_predictions.csv", "text/csv")
    elif all_dfs:
        csv = all_dfs[0].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "defect_predictions.csv", "text/csv")

if __name__ == "__main__":
    main()

# Notes:
# - Fixed TorchScript loading with torch.jit.load(). No more TypeError.
# - Assumed MobileNet path as 'models/mobilenetv3s_best_ts.pt'; update if different.
# - In classify_pytorch, added handling for output tensor (common for ScriptModule).
# - If your ScriptModule returns differently (e.g., list/tuple), adjust the probs extraction.
# - FlashAttention warning: Benign; scaled_dot_product_attention is used as fallback (no impact).
# - For fine-tuning your own: After training nn.Module, script with torch.jit.script(model) or trace, save as .pt.
# - Test: Ensure models output shape (1, 13) for batch_size=1.