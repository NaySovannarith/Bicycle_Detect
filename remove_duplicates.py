import cv2
import os
from tqdm import tqdm

IMG_DIR = "dataset/images/train"
LBL_DIR = "dataset/labels/train"

hashes = {}
removed = 0

for img_name in tqdm(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith(".jpg"):
        continue

    path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(path)
    if img is None:
        continue

    h = cv2.img_hash.averageHash(img).flatten()

    found_duplicate = False
    for existing in hashes.values():
        dist = cv2.norm(h, existing, cv2.NORM_HAMMING)
        if dist < 5:  # similarity threshold
            found_duplicate = True
            break

    if found_duplicate:
        os.remove(path)
        label = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(LBL_DIR, label)
        if os.path.exists(label_path):
            os.remove(label_path)
        removed += 1
    else:
        hashes[img_name] = h

print("✅ Duplicate images removed:", removed)

