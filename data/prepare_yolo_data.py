"""
Prepare YOLO Dataset for Model 1 (The Finder / Eye Detector).
Extracts eye annotations (Class 1) from the Goat Image Dataset,
remaps them to Class 0 ('eye'), and creates an 80/20 train/validation split.
"""
import shutil
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    GOAT_DETECTION_RAW,
    YOLO_IMAGES_TRAIN,
    YOLO_IMAGES_VAL,
    YOLO_LABELS_TRAIN,
    YOLO_LABELS_VAL,
    YOLO_CONFIG_FILE,
    YOLO_PROCESSED_DIR
)


def prepare_yolo_dataset(val_ratio: float = 0.2, seed: int = 42, max_samples: int = None):
    """
    Parses raw goat images and annotations, extracts eye labels,
    and organizes into YOLO train/val folders.
    """
    random.seed(seed)

    # Clean & create target directories
    for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL, YOLO_LABELS_TRAIN, YOLO_LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Scanning raw images in: {GOAT_DETECTION_RAW}")
    all_jpgs = sorted(list(GOAT_DETECTION_RAW.glob("*.jpg")))
    if not all_jpgs:
        raise FileNotFoundError(f"No .jpg files found in {GOAT_DETECTION_RAW}")

    if max_samples:
        all_jpgs = all_jpgs[:max_samples]

    samples_with_eyes = []
    total_eyes_found = 0

    for img_path in all_jpgs:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        eye_lines = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # In Goat Image Dataset: 0=face, 1=eye, 2=mouth, 3=ear
                if parts[0] == "1":
                    # Remap class 1 (eye) -> class 0
                    remapped_line = f"0 {' '.join(parts[1:])}\n"
                    eye_lines.append(remapped_line)

        if eye_lines:
            samples_with_eyes.append((img_path, eye_lines))
            total_eyes_found += len(eye_lines)

    print(f"Total images scanned: {len(all_jpgs)}")
    print(f"Images containing eyes: {len(samples_with_eyes)}")
    print(f"Total eyes annotated: {total_eyes_found}")

    # Shuffle and split
    random.shuffle(samples_with_eyes)
    val_count = int(len(samples_with_eyes) * val_ratio)
    val_samples = samples_with_eyes[:val_count]
    train_samples = samples_with_eyes[val_count:]

    def copy_and_write(samples, img_dest, label_dest, split_name):
        eyes_in_split = 0
        for img_path, eye_lines in samples:
            # Copy image
            target_img = img_dest / img_path.name
            shutil.copy2(img_path, target_img)

            # Write single-class eye label
            target_txt = label_dest / img_path.with_suffix(".txt").name
            with open(target_txt, "w", encoding="utf-8") as f:
                f.writelines(eye_lines)

            eyes_in_split += len(eye_lines)

        print(f"[{split_name}] Saved {len(samples)} images ({eyes_in_split} eyes) to {img_dest.parent.name}")

    copy_and_write(train_samples, YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN, "TRAIN")
    copy_and_write(val_samples, YOLO_IMAGES_VAL, YOLO_LABELS_VAL, "VAL")

    yaml_content = """# Auto-generated YOLO dataset config
path: ../processed_data/yolo
train: images/train
val: images/val

names:
  0: eye
"""
    with open(YOLO_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[DONE] YOLO dataset prepared successfully.")
    print(f"Config saved to: {YOLO_CONFIG_FILE}")


if __name__ == "__main__":
    prepare_yolo_dataset()
