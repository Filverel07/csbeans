import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the classification model
model = YOLO("runs/classify/224_v3_2_fineTuningv2/weights/best.pt")

# Define class names
class_names = {
    i: name for i, name in enumerate([
        "Broken", "Dry_Cherry", "Fade", "Floater", "Foreign_Items", "Full_Black", "Full_Sour",
        "Fungus_Damage", "Good", "Husk", "Immature", "Parchment", "Partial_Black",
        "Partial_Sour", "Severe_Insect_Damage", "Shell", "Slight_Insect_Damage", "Withered"
    ])
}

# Run validation
metrics = model.val()
print(metrics)

# Display top-k accuracy
print("\nAdditional Metrics:")
print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")

# Confusion matrix analysis
if hasattr(metrics, 'confusion_matrix'):
    print("\nConfusion Matrix is available in the saved results.")
    cm = metrics.confusion_matrix.matrix

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)

    print("\nMetrics per class:")
    class_labels = [class_names.get(i, f"Class {i}") for i in range(cm.shape[0])]

    for i, class_name in enumerate(class_labels):
        print(f"Class '{class_name}':")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1 Score: {f1[i]:.4f}")

    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    print("\nOverall (Macro Average):")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1 Score: {macro_f1:.4f}")

    # Plot metrics
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].bar(class_labels, recall, color='skyblue')
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Recall per Class")
    axes[0].tick_params(axis='x', rotation=90)

    axes[1].bar(class_labels, f1, color='salmon')
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score per Class")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()
    # Optional: Save plot
    # plt.savefig("classification_metrics.png")

# Results dictionary
if hasattr(metrics, 'results_dict'):
    print("\nResults Dictionary:")
    print(metrics.results_dict)