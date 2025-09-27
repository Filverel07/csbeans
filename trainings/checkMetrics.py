import os
from ultralytics import YOLO
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# === CONFIG ===
model_path = "models/yolo_cls.pt"
test_dir = "/home/jap/projects/yolov12/datasets/defect_split_cls/val"
save_dir = "/home/jap/projects/yolov12/runs/classify/val1/valSet"
imgsz = 224
batch_size = 32

# === LOAD MODEL ===
model = YOLO(model_path)
class_names = sorted(os.listdir(test_dir))  # Assumes folder names are class labels
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# === COLLECT LABELS ===
y_true = []
y_pred = []

for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            results = model(img)
            pred_idx = results[0].probs.top1
            y_true.append(class_to_idx[class_name])
            y_pred.append(pred_idx)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# === COMPUTE METRICS ===
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df = pd.DataFrame(report).transpose()

# === SAVE TEXT REPORT ===
os.makedirs(save_dir, exist_ok=True)
report_path = os.path.join(save_dir, "manual_metrics.txt")
with open(report_path, "w") as f:
    f.write(classification_report(y_true, y_pred, target_names=class_names))

# === PLOT HEATMAP ===
metrics_to_plot = df[['precision', 'recall', 'f1-score']].dropna()

plt.figure(figsize=(12, 8))
sns.heatmap(metrics_to_plot, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)
plt.title("Classification Metrics per Class")
plt.ylabel("Class")
plt.xlabel("Metric")
plt.tight_layout()

# === SAVE FIGURE ===
fig_path = os.path.join(save_dir, "classification_metrics_heatmap.png")
plt.savefig(fig_path)
plt.show()