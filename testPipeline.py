#!/usr/bin/env python3
"""
YOLOv12 + custom tiling (no SAHI) + TorchScript ensemble pipeline.

Key features:
 - Manual overlapping tiles to support high-res images on limited VRAM (4GB)
 - Merges detections across tiles and applies global NMS
 - Loads TorchScript classifiers once (CPU) and classifies crops in micro-batches
 - Produces annotated full image and annotated crops

Adjust CONFIG at top as needed.
"""

import os
import math
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.ops import nms
from ultralytics import YOLO
import supervision as sv
from collections import Counter
from typing import List, Tuple

# -----------------------
# CONFIG
# -----------------------
IMAGE_PATH = "stitched_Pages/page_1.jpg"
CROP_DIR = "crop/page_1"

YOLO_WEIGHTS = "runs/detect/200epochs_LT_FT/weights/best.pt"
CLS_PATHS = [
    ("YOLOv12-cls", "models/best.torchscript"),
    ("EfficientNetV2", "models/effv2s_best_ts.pt"),
    ("MobileNetV3", "models/mobilenetv3s_best_ts.pt")
]

CLASS_NAMES = ["Broken", "Dry_Cherries", "Floater", "Foreign_items", "Full_Black",
               "Full_Sour", "Fungus_damage", "Husk", "Immature", "Parchment",
               "Severe_Insect_Damage", "Shell", "Withered"]

# detection thresholds
CONF_THR = 0.5       # per-detection confidence threshold
TILE_NMS_IOU = 0.3  # internal NMS per tile (we still will run global NMS later)
GLOBAL_NMS_IOU = 0.45
MIN_AREA = 1000
EXPAND_MARGIN = 0.07
MAX_CROPS = 200  # max crops to classify/save

# tiling (tune for VRAM). Use square tiles for simplicity.
TILE_SIZE = 1920   # tile height & width in px (int)
TILE_OVERLAP = 640    # overlap in px between tiles (int). Must be < TILE_SIZE

# devices
DEVICE_YOLO = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_CLS = "cuda"  if torch.cuda.is_available() else "cpu"

# classification
MICRO_BS = 4

# debug
DEBUG_SLICES = False
SLICE_DEBUG_DIR = "slice_debug"

# -----------------------
# Transforms
# -----------------------
default_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def expand_box(box: List[int], img_shape: Tuple[int,int,int], margin: float = 0.07) -> List[int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dw, dh = int(w * margin), int(h * margin)
    nx1 = max(int(x1 - dw), 0)
    ny1 = max(int(y1 - dh), 0)
    nx2 = min(int(x2 + dw), img_shape[1])
    ny2 = min(int(y2 + dh), img_shape[0])
    return [nx1, ny1, nx2, ny2]

def load_yolo_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLO weights not found: {path}")
    model = YOLO(path, task="detect")
    try:
        model.to(DEVICE_YOLO)
    except Exception as e:
        print("[WARN] Could not move YOLO to requested device:", e)
    return model

def load_all_torchscript_models(paths):
    models = {}
    for name, p in paths:
        if not os.path.exists(p):
            print(f"⚠️ Classifier file not found: {p} (skipping {name})")
            continue
        try:
            m = torch.jit.load(p, map_location="cpu")
            m.eval()
            models[name] = m
            print(f"[INFO] Loaded classifier: {name}")
        except Exception as e:
            print(f"[WARN] Failed to load TorchScript {name}: {e}")
    return models

def _extract_logits(script_out):
    if isinstance(script_out, torch.Tensor):
        return script_out
    if isinstance(script_out, (list, tuple)) and len(script_out) > 0:
        for el in script_out:
            if isinstance(el, torch.Tensor):
                return el
        return script_out[0]
    if isinstance(script_out, dict):
        for k in ("logits", "out", "pred", "scores"):
            if k in script_out:
                return script_out[k]
        for v in script_out.values():
            if isinstance(v, torch.Tensor):
                return v
    raise ValueError("Unsupported TorchScript output format")

def ensemble_vote(predictions):
    if not predictions:
        return ("Unknown", 0.0)
    labels = [p[0] for p in predictions]
    counts = Counter(labels)
    for label, cnt in counts.items():
        if cnt >= 2:
            confs = [p[1] for p in predictions if p[0] == label]
            return label, sum(confs) / len(confs)
    return max(predictions, key=lambda x: x[1])

# -----------------------
# TILING
# -----------------------
def generate_tiles(img_w: int, img_h: int, tile_size: int, overlap: int):
    """
    Generate (x1, y1, x2, y2) tiles that cover the full image with overlap.
    """
    assert tile_size > 0 and overlap >= 0 and overlap < tile_size
    step = tile_size - overlap
    xs = list(range(0, max(1, img_w - overlap), step))
    ys = list(range(0, max(1, img_h - overlap), step))
    tiles = []
    for y in ys:
        for x in xs:
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
    # remove duplicates (if any)
    unique_tiles = []
    seen = set()
    for t in tiles:
        if t not in seen:
            unique_tiles.append(t)
            seen.add(t)
    return unique_tiles

# -----------------------
# DETECTION ON TILES
# -----------------------
def detect_on_tiles(detector: YOLO, image: 'np.ndarray', tile_size: int, overlap: int,
                    conf_th: float, tile_nms_iou: float, debug_slices=False):
    """
    Run detector on each tile and return list of detections in global coords:
      detections: list of dict {xyxy: [x1,y1,x2,y2], score: float, class_id: int}
    """
    import numpy as np

    h, w = image.shape[:2]
    tiles = generate_tiles(w, h, tile_size, overlap)
    all_boxes = []
    all_scores = []
    all_cls = []

    for idx, (x1, y1, x2, y2) in enumerate(tiles):
        tile = image[y1:y2, x1:x2]
        # optionally save debug tile
        if debug_slices:
            ensure_dir(SLICE_DEBUG_DIR)
            cv2.imwrite(os.path.join(SLICE_DEBUG_DIR, f"tile_{idx:04d}.jpg"), tile)

        # run inference on the tile. choose imgsz smaller than or equal to tile_size
        try:
            # keep imgsz to something reasonable to limit VRAM usage
            imgsz = min(tile_size, 1024)  # resize inference to 1024 for speed/memory safety
            res = detector(tile, conf=conf_th, iou=tile_nms_iou, imgsz=imgsz, max_det=500, verbose=False)
        except Exception as e:
            print(f"[WARN] detector failed on tile {idx}: {e}")
            continue

        if not res or len(res) == 0:
            continue
        r = res[0]
        # boxes: r.boxes.xyxy, scores: r.boxes.conf, classes: r.boxes.cls
        try:
            boxes_np = r.boxes.xyxy.cpu().numpy()  # shape (n,4)
            scores_np = r.boxes.conf.cpu().numpy()
            cls_np = r.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback: try attributes differently
            try:
                boxes_np = r.boxes.xyxy.numpy()
                scores_np = r.boxes.conf.numpy()
                cls_np = r.boxes.cls.numpy().astype(int)
            except Exception:
                continue

        # translate box coordinates to global image coords
        for b, s, c in zip(boxes_np, scores_np, cls_np):
            gx1 = int(b[0]) + x1
            gy1 = int(b[1]) + y1
            gx2 = int(b[2]) + x1
            gy2 = int(b[3]) + y1
            # clamp
            gx1 = max(0, min(gx1, w-1))
            gy1 = max(0, min(gy1, h-1))
            gx2 = max(0, min(gx2, w-1))
            gy2 = max(0, min(gy2, h-1))
            all_boxes.append([gx1, gy1, gx2, gy2])
            all_scores.append(float(s))
            all_cls.append(int(c))

    # Convert to tensors for easy NMS
    if not all_boxes:
        return []

    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    cls_t = torch.tensor(all_cls, dtype=torch.int64)

    # perform class-agnostic global NMS (if you want class-aware, do per-class NMS)
    keep_idx = nms(boxes_t, scores_t, GLOBAL_NMS_IOU).cpu().numpy().tolist()

    merged = []
    for i in keep_idx:
        merged.append({
            "xyxy": [int(x) for x in boxes_t[i].tolist()],
            "score": float(scores_t[i].item()),
            "class_id": int(cls_t[i].item())
        })
    return merged

# -----------------------
# CLASSIFICATION
# -----------------------
def classify_crops(models_dict, crops, transform=default_transform, micro_bs=4):
    """
    models_dict: {name: torchscript_model}
    crops: list of numpy BGR images
    returns: dict {model_name: [(label, conf), ...] }
    """
    outputs = {}
    if len(crops) == 0:
        for name in models_dict.keys():
            outputs[name] = []
        return outputs

    for name, model in models_dict.items():
        preds = []
        with torch.no_grad():
            for i in range(0, len(crops), micro_bs):
                chunk = crops[i:i+micro_bs]
                rgb_chunk = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in chunk]
                batch = torch.stack([transform(img) for img in rgb_chunk]).to(DEVICE_CLS)
                # run model (TorchScript)
                try:
                    out = model(batch)
                except Exception as e:
                    # attempt passing single images when batch input style mismatches
                    for img in batch:
                        try:
                            o = model(img.unsqueeze(0))
                            logits = _extract_logits(o)
                            probs = torch.softmax(logits, dim=1)
                            cls_id = int(probs.argmax(dim=1)[0].item())
                            preds.append((CLASS_NAMES[cls_id], float(probs[0, cls_id].item())))
                        except Exception:
                            preds.append(("Unknown", 0.0))
                    continue

                logits = _extract_logits(out)
                probs = torch.softmax(logits, dim=1)
                top1_ids = probs.argmax(dim=1)
                for idx in range(probs.shape[0]):
                    cls_id = int(top1_ids[idx].item())
                    preds.append((CLASS_NAMES[cls_id], float(probs[idx, cls_id].item())))
        # pad if missing
        if len(preds) < len(crops):
            preds.extend([("Unknown", 0.0)] * (len(crops) - len(preds)))
        outputs[name] = preds
    return outputs


def export_bboxes(detections, output_csv="detections.csv"):
    records = []
    for det in detections:
        x_min, y_min, x_max, y_max = det["bbox"]
        records.append({
            "image": det["image"],
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 4),
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max)
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Exported {len(df)} bounding boxes to {output_csv}")

# -----------------------
# MAIN PIPELINE
# -----------------------
def run_pipeline():
    ensure_dir(CROP_DIR)
    if DEBUG_SLICES:
        ensure_dir(SLICE_DEBUG_DIR)

    orig_img = cv2.imread(IMAGE_PATH)
    if orig_img is None:
        raise SystemExit(f"Image not found: {IMAGE_PATH}")

    print(f"[INFO] DEVICE_YOLO={DEVICE_YOLO}  DEVICE_CLS={DEVICE_CLS}")
    # load detector
    detector = load_yolo_model(YOLO_WEIGHTS)

    print("[INFO] Running tiled detection (custom slicer)...")
    merged_dets = detect_on_tiles(detector, orig_img, TILE_SIZE, TILE_OVERLAP,
                                  CONF_THR, TILE_NMS_IOU, debug_slices=DEBUG_SLICES)

    # filter by min area
    filtered = []
    for d in merged_dets:
        x1, y1, x2, y2 = d["xyxy"]
        area = (x2 - x1) * (y2 - y1)
        if area > MIN_AREA:
            filtered.append(d)
    detections = filtered[:MAX_CROPS]  # cap

    if len(detections) == 0:
        print("⚠️ No detections found after tiling/filtering.")
        cv2.imwrite(os.path.join(CROP_DIR, "full_image_no_detections.jpg"), orig_img)
        return

    print(f"[INFO] {len(detections)} detections -> creating crops...")
    crops = []
    boxes_for_annot = []
    for d in detections:
        xy = d["xyxy"]
        expanded = expand_box(xy, orig_img.shape, margin=EXPAND_MARGIN)
        crop = orig_img[expanded[1]:expanded[3], expanded[0]:expanded[2]].copy()
        crops.append(crop)
        boxes_for_annot.append(torch.tensor(xy))

    # load classifiers (once)
    models = load_all_torchscript_models(CLS_PATHS)
    if not models:
        print("[WARN] No classifiers loaded; producing Unknowns.")
        all_outputs = { "placeholder": [("Unknown", 0.0)] * len(crops) }
    else:
        all_outputs = classify_crops(models, crops, transform=default_transform, micro_bs=MICRO_BS)

    # ensemble voting per crop
    model_keys = list(all_outputs.keys())
    ensemble_results = []
    for i in range(len(crops)):
        preds = [all_outputs[k][i] for k in model_keys]
        ensemble_results.append(ensemble_vote(preds))

    # save annotated crops
    print("[INFO] Saving annotated crops...")
    for i, crop in enumerate(crops):
        label, conf = ensemble_results[i]
        out = crop.copy()
        cv2.putText(out, f"{label} ({conf*100:.2f}%)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(CROP_DIR, f"bean_{i}_annotated.jpg"), out)
        
        
    # annotate full image
    print("[INFO] Annotating full image...")

    # Build detections properly with class_id included
    detections_t = sv.Detections(
        xyxy=torch.stack(boxes_for_annot).numpy(),
        class_id=np.array([d["class_id"] for d in detections])  # <-- FIX
    )

    labels = [f"{lbl} ({conf*100:.1f}%)" for lbl, conf in ensemble_results]

    # Force BoxAnnotator to use index-based colors instead of class_id lookup
    box_annotator = sv.BoxAnnotator(
        thickness=3,
        color_lookup=sv.ColorLookup.INDEX  # <-- FIX
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_color=sv.Color.WHITE,
        smart_position=True
    )

    annotated = box_annotator.annotate(scene=orig_img.copy(), detections=detections_t)
    annotated = label_annotator.annotate(scene=annotated, detections=detections_t, labels=labels)

    full_path = os.path.join(CROP_DIR, "full_image_annotated.jpg")
    cv2.imwrite(full_path, annotated)
    print(f"\n✅ Full annotated image saved to: {full_path}")


# -----------------------
# ENTRY
# -----------------------
if __name__ == "__main__":
    run_pipeline()
    

