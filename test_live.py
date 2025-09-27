# app.py
import os
import cv2
import av
import torch
import numpy as np
import streamlit as st
from PIL import Image
from collections import Counter

# UI/streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Try to import Ultralytics YOLO; but apps trained with sunsmarterjie/yolov12 can also be used
try:
    from ultralytics import YOLO as ULTRA_YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

# Try to import yolov12 repo if present (the Dockerfile below clones it to /app/yolov12)
_YOLOV12_AVAILABLE = False
try:
    import sys
    if "yolov12" not in sys.path and os.path.isdir("yolov12"):
        sys.path.insert(0, os.path.abspath("yolov12"))
    import yolov12  # noqa: F401 - presence check
    _YOLOV12_AVAILABLE = True
except Exception:
    _YOLOV12_AVAILABLE = False

import torchvision.transforms as T
import supervision as sv  # <<-- Supervision for nicer annotation

# -------------------------
# Configuration & constants
# -------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", "runs/detect/200epochs_LT_FT/weights/best.pt")
YOLOCLS_TS = os.path.join(MODEL_DIR, "runs/classify/yolov12_cls_scratch/weights/best.pt")  # optional TorchScript classifier for yolocls
EFFICIENT_TS = os.path.join(MODEL_DIR, "effv2s_best_ts.pt")
MOBILENET_TS = os.path.join(MODEL_DIR, "mobilenetv3s_best_ts.pt")

CLASS_NAMES = ["Broken", "Dry_Cherries", "Floater", "Foreign_items", "Full_Black",
               "Full_Sour", "Fungus_damage", "Husk", "Immature", "Parchment",
               "Severe_Insect_Damage", "Shell", "Withered"]

# preprocess transform for torchscript classifiers
DEFAULT_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# thresholds and parameters
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
EXPAND_MARGIN = float(os.environ.get("EXPAND_MARGIN", 0.05))
MIN_AREA = int(os.environ.get("MIN_AREA", 1000))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Coffee Bean Defect Detection", layout="wide")
st.title("☕ Coffee Bean Defect Detection — Live + Image")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Detector Confidence Threshold", 0.05, 1.0, float(DEFAULT_CONF), 0.01)
iou_threshold = st.sidebar.slider("Detector IOU/NMS Threshold", 0.01, 1.0, float(DEFAULT_IOU), 0.01)
selected_models = st.sidebar.multiselect("Select Classifiers",
                                         ["YOLOv12-cls", "EfficientNetV2", "MobileNetV3"],
                                         default=["YOLOv12-cls"])

st.sidebar.markdown("---")
st.sidebar.write(f"Device: **{DEVICE}**")
st.sidebar.write("Ensemble voting: **Uploaded Images only**")

# -------------------------
# Supervision annotators (better boxes & labels)
# -------------------------
# Use the built-in default color palette directly
color_palette = sv.ColorPalette.DEFAULT

bounding_box_annotator = sv.BoxAnnotator(
    color=color_palette,
    thickness=2
)



def get_label_annotator(image_width: int, image_height: int):
    """
    Dynamically adjust text_scale based on input image size.
    Larger images get bigger labels for better readability.
    """
    base_scale = 2  # Default scale
    reference_width = 1920  # Baseline width (you can adjust)
    reference_height = 1080   # Baseline height

    # Scale based on average dimension ratio relative to baseline
    scale_factor = ((image_width + image_height) / 2) / ((reference_width + reference_height) / 2)
    text_scale = max(0.3, min(1.2, base_scale * scale_factor))  # Clamp between 0.3 and 1.2

    return sv.LabelAnnotator(
        text_color=sv.Color.WHITE,
        text_padding=6,
        text_position=sv.Position.TOP_LEFT,
        text_scale=text_scale
    )
    

# -------------------------
# Model loading utilities
# -------------------------
@st.cache_resource
def load_detector(path: str):
    """
    Load detection model.
    Priority:
      1. If ultralytics installed -> use it
      2. Else, return yolov12 path indicator (user can adapt)
    """
    if _HAS_ULTRALYTICS:
        try:
            m = ULTRA_YOLO(path)
            try:
                m.to(DEVICE)
            except Exception:
                pass
            return ("ultralytics", m)
        except Exception:
            pass

    if _YOLOV12_AVAILABLE:
        return ("yolov12", path)

    raise RuntimeError("No compatible YOLO detection loader available. Install ultralytics or clone the yolov12 repo.")

@st.cache_resource
def load_torchscript_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        m = torch.jit.load(path, map_location=DEVICE)
        m.eval()
        return m.to(DEVICE)
    except Exception:
        m = torch.jit.load(path, map_location="cpu")
        m.eval()
        return m

# Load models
_detector_source, DETECTOR = load_detector(DETECTOR_PATH)
yolocls_ts = load_torchscript_model(YOLOCLS_TS)
effnet_ts = load_torchscript_model(EFFICIENT_TS)
mobilenet_ts = load_torchscript_model(MOBILENET_TS)

# Build active classifier map (name -> (model_object, infer_fn))
def _extract_logits(script_out):
    if isinstance(script_out, torch.Tensor):
        return script_out
    if isinstance(script_out, (list, tuple)) and len(script_out) > 0:
        return script_out[0]
    if isinstance(script_out, dict):
        for k in ("logits", "out", "pred", "scores"):
            if k in script_out:
                return script_out[k]
    raise ValueError("Unsupported TorchScript output format")

def infer_torchscript_batch(model, crops, transform=DEFAULT_TRANSFORM, device=DEVICE, micro_bs=64):
    if model is None or len(crops) == 0:
        return []
    outs = []
    with torch.no_grad():
        for i in range(0, len(crops), micro_bs):
            chunk = crops[i:i+micro_bs]
            rgb_chunk = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in chunk]
            batch = torch.stack([transform(c) for c in rgb_chunk]).to(device, non_blocking=True)
            script_out = model(batch)
            logits = _extract_logits(script_out)
            probs = torch.softmax(logits, dim=1)
            top1_ids = probs.argmax(dim=1)
            for idx, cls_id in enumerate(top1_ids):
                outs.append((CLASS_NAMES[int(cls_id)], float(probs[idx, cls_id])))
    return outs

def infer_ultralytics_cls_batch(yolo_model, crops):
    if yolo_model is None or len(crops) == 0:
        return []
    results = yolo_model.predict(crops, verbose=False)
    out = []
    for res in results:
        probs = getattr(res, "probs", None)
        if probs is not None:
            class_id = int(probs.top1)
            out.append((res.names[class_id], float(probs.top1conf)))
        else:
            out.append(("Unknown", 0.0))
    return out

MODEL_MAP = {
    "YOLOv12-cls": (yolocls_ts if yolocls_ts is not None else None, infer_torchscript_batch if yolocls_ts is not None else (infer_ultralytics_cls_batch if _HAS_ULTRALYTICS else None)),
    "EfficientNetV2": (effnet_ts, infer_torchscript_batch),
    "MobileNetV3": (mobilenet_ts, infer_torchscript_batch),
}

# -------------------------
# Detection -> produce sv.Detections
# -------------------------
def detect_images(detector_source, detector_obj, image, conf=conf_threshold, iou=iou_threshold):
    """
    Returns sv.Detections object (xyxy, confidence, class_id) for Ultralytics detector.
    Fallback returns empty Detections.
    """
    if detector_source == "ultralytics":
        res = detector_obj.predict(image, conf=conf, iou=iou, verbose=False)
        if len(res) == 0 or getattr(res[0], "boxes", None) is None:
            return sv.Detections.empty()
        boxes = res[0].boxes.xyxy.cpu().numpy()
        scores = res[0].boxes.conf.cpu().numpy()
        # class ids if available, else all zeros
        try:
            class_ids = res[0].boxes.cls.cpu().numpy().astype(int)
        except Exception:
            class_ids = np.zeros(len(boxes), dtype=int)
        return sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
    else:
        # Yolov12 repo fallback - best-effort (user may need to adapt to their repo's API)
        try:
            # If detector_obj is a path, user should adapt this section
            # Attempt to call if object-like
            res = detector_obj(image)
            if len(res) == 0 or getattr(res[0], "boxes", None) is None:
                return sv.Detections.empty()
            boxes = res[0].boxes.xyxy.cpu().numpy()
            scores = res[0].boxes.conf.cpu().numpy()
            try:
                class_ids = res[0].boxes.cls.cpu().numpy().astype(int)
            except Exception:
                class_ids = np.zeros(len(boxes), dtype=int)
            return sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
        except Exception:
            return sv.Detections.empty()

# -------------------------
# Helper functions
# -------------------------
def expand_box(box, img_shape, margin=EXPAND_MARGIN):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dw, dh = int(w * margin), int(h * margin)
    nx1, ny1 = max(int(x1 - dw), 0), max(int(y1 - dh), 0)
    nx2, ny2 = min(int(x2 + dw), img_shape[1]), min(int(y2 + dh), img_shape[0])
    return [nx1, ny1, nx2, ny2]

def ensemble_vote(predictions):
    labels = [p[0] for p in predictions]
    counts = Counter(labels)
    for label, cnt in counts.items():
        if cnt >= 2:
            matching_confs = [p[1] for p in predictions if p[0] == label]
            return label, sum(matching_confs) / len(matching_confs)
    best_pred = max(predictions, key=lambda x: x[1])
    return best_pred

# -------------------------
# Uploaded image handling (ensemble) with supervision annotation
# -------------------------
st.header("Or Upload an Image for Ensemble Processing")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
upload_col1, upload_col2 = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)[:, :, ::-1]  # RGB -> BGR for cv2

    detections = detect_images(_detector_source, DETECTOR, img_np, conf=conf_threshold, iou=iou_threshold)

    if len(detections) == 0:
        st.warning("No detections found.")
        st.image(image, use_container_width=True)
    else:
        # filter small boxes by area
        keep_mask = [(x2 - x1) * (y2 - y1) >= MIN_AREA for x1, y1, x2, y2 in detections.xyxy]
        detections = detections[keep_mask]

        if len(detections) == 0:
            st.warning("No detections after filtering small boxes.")
            st.image(image, use_container_width=True)
        else:
            # expand & crop
            crops = []
            expanded_boxes = []
            for box in detections.xyxy:
                ex = expand_box(box.astype(int), img_np.shape, margin=EXPAND_MARGIN)
                crop = img_np[ex[1]:ex[3], ex[0]:ex[2]]
                if crop.size > 0:
                    crops.append(crop)
                    expanded_boxes.append(ex)

            # Run all selected classifiers (all models) and ensemble per-crop
            all_outputs = {}
            for name in selected_models:
                model_obj, inf_fn = MODEL_MAP.get(name, (None, None))
                if model_obj is None or inf_fn is None:
                    st.warning(f"Model {name} not available or not loaded.")
                    continue
                preds = inf_fn(model_obj, crops)
                all_outputs[name] = preds

            ensemble_results = []
            for i in range(len(crops)):
                preds = []
                for name in all_outputs:
                    try:
                        preds.append(all_outputs[name][i])
                    except Exception:
                        preds.append(("Unknown", 0.0))
                final_label, final_conf = ensemble_vote(preds)
                ensemble_results.append((final_label, final_conf))

            # Prepare detections for annotation using expanded boxes
            boxes_np = np.array(expanded_boxes)
            confidences = np.array([r[1] for r in ensemble_results])
            class_ids = np.zeros(len(boxes_np), dtype=int)

            dets_for_annot = sv.Detections(
                xyxy=boxes_np,
                confidence=confidences,
                class_id=class_ids
            )

            # Prepare readable labels for each box
            labels = [f"{lbl} ({conf*100:.1f}%)" for lbl, conf in ensemble_results]

            # Annotate using supervision on the final ensemble-based detections
            annotated = bounding_box_annotator.annotate(
                scene=img_np.copy(),
                detections=dets_for_annot  # ✅ FIXED
            )
            label_annotator = get_label_annotator(img_np.shape[1], img_np.shape[0])
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=dets_for_annot,  # ✅ FIXED
                labels=labels
            )

            upload_col1.image(
                annotated[:, :, ::-1],
                use_container_width=True,
                caption="Annotated (Ensemble)"
            )

            # Show the first crop as example
            if len(crops) > 0:
                upload_col2.image(
                    crops[0][:, :, ::-1],
                    use_container_width=True,
                    caption="Example crop (first)"
                    )

# -------------------------
# Video (real-time) processing using supervision annotation
# -------------------------
st.header("Live Webcam Detection (WebRTC)")

class CoffeeBeanProcessor(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to BGR numpy array
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        detections = detect_images(_detector_source, DETECTOR, img, conf=conf_threshold, iou=iou_threshold)

        # Filter small boxes
        if len(detections):
            keep_mask = [(x2 - x1) * (y2 - y1) >= MIN_AREA for x1, y1, x2, y2 in detections.xyxy]
            detections = detections[keep_mask]

        # No detections → return original frame
        if len(detections) == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Extract crops for classification
        crops = []
        for x1, y1, x2, y2 in detections.xyxy:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        # Run classifiers (if any selected)
        all_outputs = {}
        if len(crops) > 0:
            for name in selected_models:
                model_obj, inf_fn = MODEL_MAP.get(name, (None, None))
                if model_obj is None or inf_fn is None:
                    continue
                preds = inf_fn(model_obj, crops)
                all_outputs[name] = preds

        # Build labels per detection
        labels = []
        for i in range(len(crops)):
            lines = []
            for model_name, preds in all_outputs.items():
                try:
                    lbl, conf = preds[i]
                    lines.append(f"{model_name}: {lbl} ({conf*100:.1f}%)")
                except Exception:
                    lines.append(f"{model_name}: Unknown")
            labels.append("\n".join(lines))

        # Annotate detections
        annotated = bounding_box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = get_label_annotator(img.shape[1], img.shape[0])
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="coffee-bean-stream",
    video_processor_factory=CoffeeBeanProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1920},
            "height": {"ideal": 1080}
        },
        "audio": False
    },
    async_processing=True
)
