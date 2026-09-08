"""
Central Configuration for Animal Disease Models (Finder & Judge)
"""
from pathlib import Path

# Project Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RUNS_DIR = PROJECT_ROOT / "runs"
EXPORTED_MODELS_DIR = PROJECT_ROOT / "exported_models"

# Ensure output directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Raw Dataset Paths
GOAT_DETECTION_RAW = RAW_DATA_DIR / "Goat Image Dataset" / "upload"
CP_ANEMIC_RAW = RAW_DATA_DIR / "FAMACHA datasets" / "CP-AnemiC dataset"
CP_ANEMIC_SHEET = CP_ANEMIC_RAW / "Anemia_Data_Collection_Sheet.xlsx"
CP_ANEMIC_ANEMIC_DIR = CP_ANEMIC_RAW / "Anemic"
CP_ANEMIC_NON_ANEMIC_DIR = CP_ANEMIC_RAW / "Non-anemic"

# Processed Dataset Paths
YOLO_PROCESSED_DIR = PROCESSED_DATA_DIR / "yolo"
YOLO_IMAGES_TRAIN = YOLO_PROCESSED_DIR / "images" / "train"
YOLO_IMAGES_VAL = YOLO_PROCESSED_DIR / "images" / "val"
YOLO_LABELS_TRAIN = YOLO_PROCESSED_DIR / "labels" / "train"
YOLO_LABELS_VAL = YOLO_PROCESSED_DIR / "labels" / "val"
YOLO_CONFIG_FILE = PROJECT_ROOT / "configs" / "dataset_yolo.yaml"

JUDGE_PROCESSED_DIR = PROCESSED_DATA_DIR / "judge"
JUDGE_TRAIN_DIR = JUDGE_PROCESSED_DIR / "train"
JUDGE_VAL_DIR = JUDGE_PROCESSED_DIR / "val"
JUDGE_TEST_DIR = JUDGE_PROCESSED_DIR / "test"
JUDGE_MANIFEST_CSV = JUDGE_PROCESSED_DIR / "manifest.csv"

# ==========================================
# Model 1: The "Finder" (YOLOv8 Nano Eye Detector)
# ==========================================
FINDER_BASE_MODEL = "yolov8n.pt"
FINDER_IMG_SIZE = 640  # 640 for standard, 320 for ultra-fast mobile inference
FINDER_BATCH_SIZE = 16
FINDER_EPOCHS = 30
FINDER_CLASSES = ["eye"]  # Class 0: eye

# ==========================================
# Model 2: The "Judge" (MobileNetV3 Small Anemia Color Classifier)
# ==========================================
JUDGE_IMG_SIZE = 224
JUDGE_BATCH_SIZE = 32
JUDGE_EPOCHS = 25
JUDGE_LR = 1e-3
JUDGE_WEIGHT_DECAY = 1e-4

# Traffic light 3-class system
JUDGE_CLASSES_3 = ["Green_Healthy", "Yellow_Borderline", "Red_Severe"]

# Granular 4-class system
JUDGE_CLASSES_4 = ["Non-Anemic", "Mild", "Moderate", "Severe"]

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Severity to Traffic Light Mapping
SEVERITY_TO_TRAFFIC_LIGHT = {
    "Non-Anemic": "Green_Healthy",
    "Mild": "Yellow_Borderline",
    "Moderate": "Yellow_Borderline",
    "Severe": "Red_Severe"
}

# ==========================================
# Mobile App Audio Guidance Mapping
# ==========================================
SUPPORTED_LANGUAGES = ["hausa", "dagbani", "gurune"]

AUDIO_MAP = {
    "Green_Healthy": {
        "hausa": "hausa_healthy.mp3",
        "dagbani": "dagbani_healthy.mp3",
        "gurune": "gurune_healthy.mp3",
        "advice_en": "Your animal has healthy blood. No medicine is needed today.",
        "color": "#28a745"
    },
    "Yellow_Borderline": {
        "hausa": "hausa_warning.mp3",
        "dagbani": "dagbani_warning.mp3",
        "gurune": "gurune_warning.mp3",
        "advice_en": "Your animal is getting weak. Check its food, isolate it, and monitor for worsening pale eyes.",
        "color": "#ffc107"
    },
    "Red_Severe": {
        "hausa": "hausa_danger.mp3",
        "dagbani": "dagbani_danger.mp3",
        "gurune": "gurune_danger.mp3",
        "advice_en": "Your animal is very pale and weak inside. It has blood worms. Administer dewormer medicine today immediately.",
        "color": "#dc3545"
    }
}
