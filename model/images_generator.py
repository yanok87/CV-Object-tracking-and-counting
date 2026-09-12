import cv2
import os
from typing import Any

# noinspection PyPep8Naming
import albumentations as A

# Paths relative to this script (works regardless of CWD / PyCharm working dir)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------- INPUTS --------------
image_path = os.path.join(BASE_DIR, "Drones_Datasets/train/images/frame_0.jpg")
label_path = os.path.join(BASE_DIR, "Drones_Datasets/train/labels/frame_0.txt")  # YOLO format
output_dir_images = os.path.join(BASE_DIR, "Drones_Datasets/train/images")
output_dir_labels = os.path.join(BASE_DIR, "Drones_Datasets/train/labels")
num_images = 600
# ---------------------------------

# Load original image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# Load YOLO annotation
with open(label_path, "r") as f:
    yolo_data = f.readlines()

bboxes = []
classes = []

for line in yolo_data:
    cls, x, y, bw, bh = map(float, line.split())
    classes.append(int(cls))
    bboxes.append([x, y, bw, bh])

# Augmentation pipeline
def build_transform() -> Any:
    # noinspection PyTypeChecker
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.MotionBlur(p=0.3),
            A.HueSaturationValue(p=0.5),
            A.RandomGamma(p=0.4),
            A.Perspective(p=0.4),
            A.RandomScale(scale_limit=0.2, p=0.5),
            # ShiftScaleRotate is deprecated → use Affine
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.8, 1.2),
                rotate=(-15, 15),
                p=0.7,
            ),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
    )


transform = build_transform()

# Generate images
for i in range(num_images):
    print(i)
    augmented = transform(image=image, bboxes=bboxes, category_ids=classes)
    aug_img = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    aug_classes = augmented["category_ids"]

    img_name = f"frame_{i + 1}.jpg"
    txt_name = f"frame_{i + 1}.txt"

    cv2.imwrite(os.path.join(output_dir_images, img_name), aug_img)

    # Save updated YOLO label (use aug_classes — boxes can be dropped by transforms)
    with open(os.path.join(output_dir_labels, txt_name), "w") as f:
        for cls, box in zip(aug_classes, aug_bboxes):
            # int(cls): albumentations returns category_ids as floats (e.g. 0.0)
            f.write(f"{int(cls)} {' '.join(map(str, box))}\n")

print("600 images generated successfully!")
