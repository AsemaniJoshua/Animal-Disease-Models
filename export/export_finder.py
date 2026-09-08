"""
Export Model 1: The Finder (YOLOv8 Nano) to ONNX and TFLite for Mobile Deployment
"""
import argparse
from pathlib import Path
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import shutil
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    FINDER_BASE_MODEL,
    FINDER_IMG_SIZE,
    RUNS_DIR,
    EXPORTED_MODELS_DIR
)


def export_finder(weights_path: str = None, imgsz: int = FINDER_IMG_SIZE, int8: bool = False):
    """
    Exports YOLOv8 Nano eye detector to ONNX and TFLite formats.
    """
    from ultralytics import YOLO

    # Find model weights
    if weights_path and Path(weights_path).exists():
        src_weights = Path(weights_path)
    elif (RUNS_DIR / "finder" / "best.pt").exists():
        src_weights = RUNS_DIR / "finder" / "best.pt"
    else:
        print(f"Trained weights not found. Using base model: {FINDER_BASE_MODEL}")
        src_weights = FINDER_BASE_MODEL

    print("=" * 60)
    print("Exporting Model 1 (The Finder)")
    print(f"Source: {src_weights}")
    print(f"Target directory: {EXPORTED_MODELS_DIR}")
    print("=" * 60)

    model = YOLO(str(src_weights))

    # 1. Export to ONNX (Universal mobile & web format)
    print("\n[1/2] Exporting to ONNX...")
    try:
        onnx_file = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
        dest_onnx = EXPORTED_MODELS_DIR / "finder_yolo.onnx"
        shutil.copy2(onnx_file, dest_onnx)
        print(f"✓ Saved ONNX model to: {dest_onnx} ({dest_onnx.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"✗ ONNX export notice: {e}")

    # 2. Export to TFLite (Android / React Native TFLite)
    print("\n[2/2] Exporting to TFLite...")
    try:
        tflite_file = model.export(format="tflite", imgsz=imgsz, int8=int8)
        dest_tflite = EXPORTED_MODELS_DIR / "finder_yolo.tflite"
        shutil.copy2(tflite_file, dest_tflite)
        print(f"✓ Saved TFLite model to: {dest_tflite} ({dest_tflite.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Note: TFLite export via ultralytics requires TensorFlow. If running on Python 3.14 on Windows, run the provided Google Colab notebook (notebooks/train_on_colab.ipynb) for direct 1-click TFLite conversion. Details: {e}")

    # 3. Export Mobile Metadata for Mobile Developer
    metadata = {
        "model_name": "Animal Eye Finder",
        "architecture": "YOLOv8n",
        "input_shape": [1, 3, imgsz, imgsz],
        "input_mean": [0.0, 0.0, 0.0],
        "input_std": [255.0, 255.0, 255.0],
        "classes": ["eye"],
        "purpose": "Detects eye coordinates and extracts eyelid crop with 15% padding"
    }
    meta_path = EXPORTED_MODELS_DIR / "finder_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved Mobile Metadata to: {meta_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 Finder Model")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=FINDER_IMG_SIZE, help="Image size (640 or 320)")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args = parser.parse_args()

    export_finder(weights_path=args.weights, imgsz=args.imgsz, int8=args.int8)
