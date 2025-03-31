import os
import shutil
import cv2
import re

original_dataset = "SL_Numbers"
orhan_root = "Orhan_Datasets"
merged_dataset = "merged_dataset"
img_size = 64

# Label → prefix mapping for filenames
label_map = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}


if os.path.exists(merged_dataset): # Copy the original dataset into the merge target
    shutil.rmtree(merged_dataset)
shutil.copytree(original_dataset, merged_dataset)
print("✅ Original dataset copied to → merged_dataset/")


for raw_label in os.listdir(orhan_root): # Append Orhan's custom images into each appropriate class folder
    if raw_label not in label_map:
        print(f"⏭️ Unknown label: {raw_label}")
        continue

    label_name = label_map[raw_label]  # e.g., zero, one, ...
    source_folder = os.path.join(orhan_root, raw_label)
    target_folder = os.path.join(merged_dataset, raw_label)  # e.g., 0, 1, ...

    if not os.path.exists(target_folder):
        print(f"⚠️ Target folder does not exist: {target_folder}")
        continue

    # Find the highest index among existing files
    max_index = 0
    for fname in os.listdir(target_folder):
        match = re.match(f"{label_name}_(\d+)", fname)
        if match:
            max_index = max(max_index, int(match.group(1)))

    # Process and add each image
    for img_name in os.listdir(source_folder):
        img_path = os.path.join(source_folder, img_name)
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))

        new_index = max_index + 1
        new_filename = f"{label_name}_{new_index}.jpg"
        save_path = os.path.join(target_folder, new_filename)
        cv2.imwrite(save_path, resized)
        max_index = new_index

        print(f"✔️ {img_name} → {target_folder}/{new_filename}")
