"""
Evaluation Script for Model 1: The Finder (YOLOv8 Nano Eye Detector)
Computes mAP50, mAP50-95, Precision, Recall, and Mobile Inference Latency.
"""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import FINDER_BASE_MODEL, YOLO_CONFIG_FILE, RUNS_DIR


def evaluate_finder(weights_path: str = None, imgsz: int = 640):
    from ultralytics import YOLO

    if weights_path and Path(weights_path).exists():
        src_weights = Path(weights_path)
    elif (RUNS_DIR / "finder" / "best.pt").exists():
        src_weights = RUNS_DIR / "finder" / "best.pt"
    else:
        src_weights = FINDER_BASE_MODEL

    print("=" * 60)
    print("Evaluating Model 1 (The Finder)")
    print(f"Model: {src_weights}")
    print("=" * 60)

    model = YOLO(str(src_weights))
    metrics = model.val(data=str(YOLO_CONFIG_FILE), imgsz=imgsz, verbose=True)

    print("\n" + "=" * 60)
    print("YOLO Eye Detection Evaluation Summary:")
    print("=" * 60)
    print(f"• mAP@0.50:       {metrics.box.map50 * 100:.2f}%")
    print(f"• mAP@0.50:0.95:  {metrics.box.map * 100:.2f}%")
    print(f"• Mean Precision: {metrics.box.mp * 100:.2f}%")
    print(f"• Mean Recall:    {metrics.box.mr * 100:.2f}%")
    print(f"• Inference Time: {metrics.speed['inference']:.2f} ms / image")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Eye Finder")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    evaluate_finder(weights_path=args.weights, imgsz=args.imgsz)
