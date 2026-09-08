"""
Training Script for Model 2: The Judge (MobileNetV3-Small Anemia Classifier)
Trains on the Ghanaian CP-AnemiC conjunctiva dataset with data augmentations
and class weighting to handle imbalance.
"""
import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    JUDGE_TRAIN_DIR,
    JUDGE_VAL_DIR,
    JUDGE_TEST_DIR,
    JUDGE_IMG_SIZE,
    JUDGE_BATCH_SIZE,
    JUDGE_EPOCHS,
    JUDGE_LR,
    JUDGE_WEIGHT_DECAY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RUNS_DIR
)
from models.judge import MobileNetJudge


def get_data_loaders(batch_size: int = JUDGE_BATCH_SIZE):
    """
    Builds data loaders with domain-specific augmentations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((JUDGE_IMG_SIZE, JUDGE_IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((JUDGE_IMG_SIZE, JUDGE_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if not JUDGE_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Train directory not found at {JUDGE_TRAIN_DIR}. "
            f"Please run 'python data/prepare_judge_data.py' first."
        )

    train_dataset = datasets.ImageFolder(str(JUDGE_TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(JUDGE_VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(JUDGE_TEST_DIR), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Compute inverse class weights to handle imbalance
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets, minlength=len(train_dataset.classes))
    total_samples = len(targets)
    class_weights = total_samples / (len(train_dataset.classes) * class_counts + 1e-6)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Class counts in train: {dict(zip(train_dataset.classes, class_counts))}")

    return train_loader, val_loader, test_loader, class_weights_tensor, train_dataset.classes


def train_judge(
    epochs: int = JUDGE_EPOCHS,
    batch_size: int = JUDGE_BATCH_SIZE,
    lr: float = JUDGE_LR,
    device: str = ""
):
    """
    Trains MobileNetV3-Small on the conjunctiva dataset.
    """
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {dev}")

    train_loader, val_loader, test_loader, class_weights, classes = get_data_loaders(batch_size=batch_size)
    class_weights = class_weights.to(dev)

    model = MobileNetJudge(num_classes=len(classes), pretrained=True).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=JUDGE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = RUNS_DIR / "judge"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pth"

    best_val_loss = float("inf")
    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print("Training Model 2 (The Judge: MobileNetV3)")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(dev), labels.to(dev)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / len(val_loader.dataset)

        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "classes": classes
            }, best_model_path)

        star = "★ Best" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.1f}% {star}")

    print("\n" + "=" * 60)
    print(f"[DONE] Training complete! Best validation model saved to: {best_model_path}")
    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc * 100:.1f}%")

    # Final Test Set Evaluation
    checkpoint = torch.load(best_model_path, map_location=dev)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    test_acc = test_correct / len(test_loader.dataset)
    print(f"Final Test Accuracy: {test_acc * 100:.1f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Anemia Judge")
    parser.add_argument("--epochs", type=int, default=JUDGE_EPOCHS, help="Epochs")
    parser.add_argument("--batch", type=int, default=JUDGE_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=JUDGE_LR, help="Learning rate")
    parser.add_argument("--device", type=str, default="", help="Device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    train_judge(epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device)
