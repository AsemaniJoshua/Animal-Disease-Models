"""
Evaluation Script for Model 2: The Judge (MobileNetV3-Small Anemia Classifier)
Evaluates on the held-out test set, printing accuracy, precision, recall, and confusion matrix.
"""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    JUDGE_TEST_DIR,
    JUDGE_IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RUNS_DIR,
    JUDGE_CLASSES_3
)
from models.judge import MobileNetJudge


def evaluate_judge(weights_path: str = None, device: str = ""):
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if weights_path and Path(weights_path).exists():
        src_weights = Path(weights_path)
    elif (RUNS_DIR / "judge" / "best_model.pth").exists():
        src_weights = RUNS_DIR / "judge" / "best_model.pth"
    else:
        src_weights = None

    print("=" * 60)
    print("Evaluating Model 2 (The Judge: MobileNetV3)")
    print(f"Model: {src_weights if src_weights else 'Pretrained base model'}")
    print(f"Test set: {JUDGE_TEST_DIR}")
    print("=" * 60)

    eval_transform = transforms.Compose([
        transforms.Resize((JUDGE_IMG_SIZE, JUDGE_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_dataset = datasets.ImageFolder(str(JUDGE_TEST_DIR), transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    classes = test_dataset.classes

    model = MobileNetJudge(num_classes=len(classes), pretrained=(src_weights is None)).to(dev)

    if src_weights:
        checkpoint = torch.load(src_weights, map_location=dev)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(dev)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics
    total = len(all_targets)
    correct = (all_preds == all_targets).sum()
    acc = correct / total

    print(f"\nOverall Test Accuracy: {acc * 100:.2f}% ({correct}/{total})\n")

    # Confusion Matrix
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    print("Confusion Matrix:")
    header = f"{'True \\ Pred':<20}" + "".join([f"{cls[:12]:>14}" for cls in classes])
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(classes):
        row_str = f"{cls:<20}" + "".join([f"{cm[i, j]:>14}" for j in range(n_classes)])
        print(row_str)

    # Per-Class Precision, Recall, F1
    print("\nPer-Class Performance:")
    print(f"{'Class':<20}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
    print("-" * 56)
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{cls:<20}{prec * 100:>11.1f}%{rec * 100:>11.1f}%{f1 * 100:>11.1f}%")
    print("-" * 56 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MobileNet Judge")
    parser.add_argument("--weights", type=str, default=None, help="Path to best_model.pth")
    args = parser.parse_args()

    evaluate_judge(weights_path=args.weights)
