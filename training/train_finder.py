"""
Training Script for Model 1: The Finder (YOLOv8 Nano Eye Detector)
Trains/fine-tunes YOLOv8n on the processed goat eye dataset.
"""
import argparse
from pathlib import Path
import sys
import shutil

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    FINDER_BASE_MODEL,
    FINDER_IMG_SIZE,
    FINDER_BATCH_SIZE,
    FINDER_EPOCHS,
    YOLO_CONFIG_FILE,
    RUNS_DIR
)


def train_finder(
    epochs: int = FINDER_EPOCHS,
    img_size: int = FINDER_IMG_SIZE,
    batch_size: int = FINDER_BATCH_SIZE,
    device: str = "",
    resume: bool = False
):
    """
    Fine-tunes YOLOv8n on the eye detection dataset.
    """
    from ultralytics import YOLO

    if not YOLO_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found at {YOLO_CONFIG_FILE}. "
            f"Please run 'python data/prepare_yolo_data.py' first."
        )

    print("=" * 60)
    print("Starting Model 1 (The Finder) Training")
    print("=" * 60)
    print(f"Base model: {FINDER_BASE_MODEL}")
    print(f"Config: {YOLO_CONFIG_FILE}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")

    model = YOLO(FINDER_BASE_MODEL)

    results = model.train(
        data=str(YOLO_CONFIG_FILE),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device if device else None,
        project=str(RUNS_DIR / "finder"),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        save=True,
        plots=True,
        verbose=True
    )

    best_weights = RUNS_DIR / "finder" / "train" / "weights" / "best.pt"
    if best_weights.exists():
        dest = RUNS_DIR / "finder" / "best.pt"
        shutil.copy2(best_weights, dest)
        print(f"\n[DONE] Best weights saved to: {dest}")

    # Run validation
    metrics = model.val()
    print("\nValidation Results:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Nano Eye Finder")
    parser.add_argument("--epochs", type=int, default=FINDER_EPOCHS, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=FINDER_IMG_SIZE, help="Image size")
    parser.add_argument("--batch", type=int, default=FINDER_BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: 'cpu', '0', etc.")
    args = parser.parse_args()

    train_finder(epochs=args.epochs, img_size=args.imgsz, batch_size=args.batch, device=args.device)
