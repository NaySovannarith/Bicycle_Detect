import os, random, shutil

train_img = "dataset/images/train"
train_lbl = "dataset/labels/train"
test_img = "dataset/images/test"
test_lbl = "dataset/labels/test"

images = [f for f in os.listdir(train_img) if f.endswith(".jpg")]
random.shuffle(images)

test_size = int(0.1 * len(images))  # 10% for test
test_images = images[:test_size]

for img in test_images:
    shutil.move(f"{train_img}/{img}", f"{test_img}/{img}")
    lbl = img.replace(".jpg", ".txt")
    shutil.move(f"{train_lbl}/{lbl}", f"{test_lbl}/{lbl}")

print("✅ Test set created")

