"""
End-to-End Simulation of the Mobile App Flow:
[Phone Camera Photo] ──> [Model 1: Finder (YOLO)] ──> [Crops Eye]
                              │
                              ▼
                         [Model 2: Judge (MobileNet)]
                              │
                              ▼
[Screen Shows Traffic Light Circle] & [Plays Local Audio Note (Hausa / Dagbani / Gurune)]
"""
import argparse
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RUNS_DIR,
    EXPORTED_MODELS_DIR,
    AUDIO_MAP,
    SUPPORTED_LANGUAGES,
    GOAT_DETECTION_RAW
)
from models.finder import EyeFinder
from models.judge import AnemiaJudge


def run_pipeline(
    image_path: str = None,
    language: str = "hausa",
    output_dir: str = "runs/demo"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Select input photo
    if image_path and Path(image_path).exists():
        img_file = Path(image_path)
    else:
        # Pick a sample image from the raw dataset
        sample_imgs = list(GOAT_DETECTION_RAW.glob("*.jpg"))
        if not sample_imgs:
            raise FileNotFoundError("No sample images found to demonstrate.")
        img_file = sample_imgs[0]
        print(f"No image supplied. Using sample: {img_file.name}")

    print("=" * 65)
    print("      VETERINARY OFFLINE ANEMIA DIAGNOSIS PIPELINE")
    print("=" * 65)
    print(f"Input Goat Image:  {img_file}")
    print(f"Selected Language: {language.upper()}")
    print("-" * 65)

    original_img = Image.open(img_file).convert("RGB")

    # 2. Step 1: Model 1 (The Finder)
    print("\n[Step 1] Running Model 1: The Finder (YOLOv8 Nano)...")
    finder_weights = RUNS_DIR / "finder" / "best.pt"
    finder = EyeFinder(model_path=str(finder_weights) if finder_weights.exists() else None)

    best_result = finder.get_best_eye(original_img, conf=0.20, padding=0.15)

    if best_result is None:
        print("⚠ Notice: Finder did not detect an eye with high confidence.")
        print("   Using center face region fallback for inspection...")
        w, h = original_img.size
        eye_crop = original_img.crop((w * 0.3, h * 0.3, w * 0.7, h * 0.7))
        bbox = [w * 0.3, h * 0.3, w * 0.7, h * 0.7]
        conf = 0.50
    else:
        eye_crop, det_info = best_result
        bbox = det_info["bbox"]
        conf = det_info["confidence"]
        print(f"✓ Eye successfully located! Confidence: {conf * 100:.1f}%")
        print(f"  Coordinates: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

    # Save the eye crop
    crop_path = out_path / "detected_eye_crop.png"
    eye_crop.save(crop_path)
    print(f"✓ Extracted eyelid crop saved to: {crop_path}")

    # 3. Step 2: Model 2 (The Judge)
    print("\n[Step 2] Running Model 2: The Judge (MobileNetV3 Small)...")
    judge_weights = RUNS_DIR / "judge" / "best_model.pth"
    judge = AnemiaJudge(model_path=str(judge_weights) if judge_weights.exists() else None)

    pred_class, pred_conf, all_probs = judge.predict(eye_crop)

    print(f"✓ Eyelid color evaluated!")
    print(f"  Diagnosis:  {pred_class}")
    print(f"  Confidence: {pred_conf * 100:.1f}%")
    print("  Probability breakdown:")
    for cls_name, prob in all_probs.items():
        bar = "█" * int(prob * 20)
        print(f"    - {cls_name:<18}: {prob * 100:>5.1f}%  {bar}")

    # 4. Step 3: Trigger Mobile Action (Screen & Offline Audio)
    action_info = AUDIO_MAP.get(pred_class, AUDIO_MAP["Yellow_Borderline"])
    audio_file = action_info.get(language.lower(), f"{language}_warning.mp3")

    print("\n" + "=" * 65)
    print("                MOBILE APP ACTION TRIGGER")
    print("=" * 65)
    circle_symbol = "🟢" if "Green" in pred_class else ("🟡" if "Yellow" in pred_class else "🔴")
    print(f"SCREEN DISPLAY:  {circle_symbol}  [{pred_class.upper()}]  ({action_info['color']})")
    print(f"AUDIO TRIGGER:   🔊 Play 'assets/audio/{audio_file}'")
    print(f"SPOKEN ADVICE:   \"{action_info['advice_en']}\"")
    print("=" * 65)

    # 5. Create visual annotated summary image
    draw_img = original_img.copy()
    draw = ImageDraw.Draw(draw_img)
    color_hex = action_info["color"]
    # Draw eye bounding box
    draw.rectangle(bbox, outline=color_hex, width=4)
    annotated_path = out_path / "mobile_diagnosis_result.jpg"
    draw_img.save(annotated_path)
    print(f"\n[DONE] Annotated result saved to: {annotated_path}\n")

    return {
        "status": pred_class,
        "confidence": pred_conf,
        "audio_file": audio_file,
        "advice": action_info["advice_en"],
        "color": color_hex,
        "crop_path": str(crop_path),
        "result_path": str(annotated_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Mobile Diagnosis Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to goat image")
    parser.add_argument("--lang", type=str, default="hausa", choices=["hausa", "dagbani", "gurune"], help="Spoken language")
    parser.add_argument("--out", type=str, default="runs/demo", help="Output directory")
    args = parser.parse_args()

    run_pipeline(image_path=args.image, language=args.lang, output_dir=args.out)
