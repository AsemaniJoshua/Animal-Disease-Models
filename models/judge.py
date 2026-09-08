"""
Model 2: The Judge (MobileNetV3-Small Anemia Color Classifier)
Evaluates the mucosal color of the cropped inner eyelid to classify
anemia severity into Green (Healthy), Yellow (Borderline), and Red (Severe).
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import JUDGE_CLASSES_3, IMAGENET_MEAN, IMAGENET_STD, JUDGE_IMG_SIZE


class MobileNetJudge(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super(MobileNetJudge, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Replace final classification layer
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, num_classes)
        )
        self.classes = JUDGE_CLASSES_3 if num_classes == 3 else [f"Class_{i}" for i in range(num_classes)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AnemiaJudge:
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 3, device: Optional[str] = None):
        """
        Wrapper around MobileNetJudge handling preprocessing and inference.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MobileNetJudge(num_classes=num_classes, pretrained=(model_path is None))
        self.classes = JUDGE_CLASSES_3

        if model_path and Path(model_path).exists():
            print(f"Loading Judge weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Inference preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((JUDGE_IMG_SIZE, JUDGE_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(self, crop: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Takes an eye crop (PIL Image), predicts anemia status,
        and returns (predicted_class, confidence, probabilities_dict).
        """
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        tensor = self.preprocess(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        class_probs = {cls_name: float(p) for cls_name, p in zip(self.classes, probs)}
        best_idx = int(probs.argmax())
        best_class = self.classes[best_idx]
        best_confidence = float(probs[best_idx])

        return best_class, best_confidence, class_probs
