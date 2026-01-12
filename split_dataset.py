import os, random, shutil

img_dir = "dataset/images/train"
lbl_dir = "dataset/labels/train"

val_img_dir = "dataset/images/val"
val_lbl_dir = "dataset/labels/val"

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
random.shuffle(images)

split = int(0.2 * len(images))  # 20% for validation
val_images = images[:split]

for img in val_images:
    shutil.move(os.path.join(img_dir, img), os.path.join(val_img_dir, img))
    label = img.replace(".jpg", ".txt")
    shutil.move(os.path.join(lbl_dir, label), os.path.join(val_lbl_dir, label))

print("✅ Dataset split completed")

