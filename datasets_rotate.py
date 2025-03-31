import os
import cv2
import numpy as np

source_dir = "merged_dataset"
target_dir = "merged_rotated"
img_size = 64

# Label → filename prefix map
label_map = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

# Rotate image around its center
def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

# Traverse all class folders
for label in os.listdir(source_dir):
    src_folder = os.path.join(source_dir, label)
    dst_folder = os.path.join(target_dir, label)
    os.makedirs(dst_folder, exist_ok=True)

    label_prefix = label_map[label]  # e.g., "3" → "three"

    for fname in os.listdir(src_folder):
        if not fname.lower().endswith(".jpg"):
            continue

        fpath = os.path.join(src_folder, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f" Failed to read: {fname}")
            continue

        base_name = fname.split(".")[0]  # e.g., zero_0

        # Rotate left: +5, +10, ..., +25 degrees
        for i in range(1, 6):
            angle = i * 5
            rotated = rotate(img, angle)
            save_name = f"{base_name}_L_{i}.jpg"
            save_path = os.path.join(dst_folder, save_name)
            cv2.imwrite(save_path, rotated)

        # Rotate right: -5, -10, ..., -25 degrees
        for i in range(1, 6):
            angle = -i * 5
            rotated = rotate(img, angle)
            save_name = f"{base_name}_R_{i}.jpg"
            save_path = os.path.join(dst_folder, save_name)
            cv2.imwrite(save_path, rotated)

        print(f"✔️ Generated 10 augmentations for {fname}")

        
        
import os

dataset_dir = "merged_rotated"  # Or whichever dataset you're using

total_images = 0
for label in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, label)
    if not os.path.isdir(folder_path):
        continue
    num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
    print(f"{label}: {num_images} images")
    total_images += num_images

print(f"\n Total number of images: {total_images}")
