"""
Export Model 2: The Judge (MobileNetV3-Small) to ONNX and TFLite for Mobile Deployment
"""
import argparse
from pathlib import Path
import sys
import json
import torch
import sys
# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    JUDGE_IMG_SIZE,
    JUDGE_CLASSES_3,
    IMAGENET_MEAN,
    IMAGENET_STD,
    AUDIO_MAP,
    RUNS_DIR,
    EXPORTED_MODELS_DIR
)
from models.judge import MobileNetJudge


def export_judge(weights_path: str = None):
    """
    Exports MobileNetV3-Small to ONNX format with full mobile metadata and labels.
    """
    if weights_path and Path(weights_path).exists():
        src_weights = Path(weights_path)
    elif (RUNS_DIR / "judge" / "best_model.pth").exists():
        src_weights = RUNS_DIR / "judge" / "best_model.pth"
    else:
        src_weights = None

    print("=" * 60)
    print("Exporting Model 2 (The Judge)")
    print(f"Source: {src_weights if src_weights else 'Pretrained Base Model'}")
    print(f"Target directory: {EXPORTED_MODELS_DIR}")
    print("=" * 60)

    model = MobileNetJudge(num_classes=len(JUDGE_CLASSES_3), pretrained=(src_weights is None))

    if src_weights:
        checkpoint = torch.load(src_weights, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    # 1. Export to ONNX
    dummy_input = torch.randn(1, 3, JUDGE_IMG_SIZE, JUDGE_IMG_SIZE)
    dest_onnx = EXPORTED_MODELS_DIR / "judge_mobilenet.onnx"

    print("\n[1/2] Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(dest_onnx),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None  # Static shape is ideal for mobile NPU/DSP accelerators
    )
    print(f"✓ Saved ONNX model to: {dest_onnx} ({dest_onnx.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Export Mobile Metadata & Label Mapping
    labels_file = EXPORTED_MODELS_DIR / "labels.txt"
    with open(labels_file, "w", encoding="utf-8") as f:
        for cls_name in JUDGE_CLASSES_3:
            f.write(f"{cls_name}\n")
    print(f"✓ Saved labels to: {labels_file}")

    metadata = {
        "model_name": "Goat Anemia Color Judge",
        "architecture": "MobileNetV3-Small",
        "input_shape": [1, 3, JUDGE_IMG_SIZE, JUDGE_IMG_SIZE],
        "input_mean": IMAGENET_MEAN,
        "input_std": IMAGENET_STD,
        "classes": JUDGE_CLASSES_3,
        "class_indices": {cls_name: i for i, cls_name in enumerate(JUDGE_CLASSES_3)},
        "audio_triggers": AUDIO_MAP
    }
    meta_file = EXPORTED_MODELS_DIR / "judge_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved Mobile Metadata to: {meta_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MobileNetV3 Judge Model")
    parser.add_argument("--weights", type=str, default=None, help="Path to best_model.pth")
    args = parser.parse_args()

    export_judge(weights_path=args.weights)
