import os
import cv2
import albumentations as A

# Original dataset folder
DATASET_DIR = 'datasets/bean_defect'

# Output folder for augmented dataset
OUTPUT_DIR = 'datasets/bean_defect_augmented'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Brightness and contrast options with readable labels
brightness_levels = [
    (-0.3, "bright_dark"),
    (0.0, "bright_normal"),
    (0.3, "bright_bright")
]

contrast_levels = [
    (-0.3, "contrast_low"),
    (0.0, "contrast_normal"),
    (0.3, "contrast_high")
]

def augment_and_save(input_folder, output_folder):
    # Make sure output subfolder exists
    os.makedirs(output_folder, exist_ok=True)

    imgs = [f for f in os.listdir(input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in imgs:
        img_path = os.path.join(input_folder, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipped {img_path}")
            continue

        base, ext = os.path.splitext(fname)

        # Save resized original
        resized_img = A.Resize(640, 640)(image=image)['image']
        cv2.imwrite(os.path.join(output_folder, f"{base}_resized{ext}"), resized_img)

        for b_val, b_label in brightness_levels:
            for c_val, c_label in contrast_levels:
                # Skip original (normal brightness + normal contrast)
                if b_val == 0.0 and c_val == 0.0:
                    continue

                aug = A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=(b_val, b_val),
                        contrast_limit=(c_val, c_val),
                        p=1.0
                    ),
                    A.Resize(640, 640)
                ])(image=image)

                aug_img = aug['image']
                save_name = f"{base}_{b_label}_{c_label}{ext}"
                cv2.imwrite(os.path.join(output_folder, save_name), aug_img)

        print(f"Done augmenting {fname}")

if __name__ == "__main__":
    for defect_dir in os.listdir(DATASET_DIR):
        input_path = os.path.join(DATASET_DIR, defect_dir)
        if os.path.isdir(input_path):
            output_path = os.path.join(OUTPUT_DIR, defect_dir)
            print(f"Processing: {defect_dir}")
            augment_and_save(input_path, output_path)
