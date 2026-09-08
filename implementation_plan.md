# Mobile AI Models for Sheep & Goat Anemia Detection

Build, train, evaluate, and export two lightweight, offline AI models for mobile deployment (React Native / Android):
1. **Model 1: The "Finder" (YOLOv8 Nano)** — Detects and locates the goat/sheep eye/eyelid from live camera input.
2. **Model 2: The "Judge" (MobileNetV3 Small)** — Classifies the conjunctiva color/mucosa into FAMACHA anemia severity grades (Green: Non-anemic, Yellow: Mild/Moderate, Red: Severe) to trigger offline local audio guidance (Hausa, Dagbani, Gurune).

---

## Dataset Analysis & Verification Findings

Our inspection of the repository confirmed that both required datasets are already present and verified:
1. **Goat Image Dataset (`Goat Image Dataset/upload`)**:
   - **1,680 images** with YOLO bounding box annotations (`.jpg` + `.txt` pairs).
   - Classes verified from `classes.txt`: `0: face`, `1: eye` (3,326 annotated eyes), `2: mouth`, `3: ear`.
   - Used to train **Model 1 (The Finder)** to accurately locate and crop the eye.
2. **FAMACHA Dataset (`FAMACHA datasets/CP-AnemiC dataset`)**:
   - **710 images** of conjunctiva / inner eyelid crops collected directly in **Ghana** (Techiman, Sunyani, Atwima Nwabiagya, Ejusu).
   - Metadata verified from `Anemia_Data_Collection_Sheet.xlsx`:
     - **Non-Anemic**: 286 samples (Green / Healthy)
     - **Mild**: 144 samples (Yellow / Getting weak)
     - **Moderate**: 232 samples (Yellow / Getting weak)
     - **Severe**: 48 samples (Red / Danger / Emergency)
   - Continuous Hemoglobin level (`HB_LEVEL`) also available for each sample.

---

## User Review Required

> [!IMPORTANT]
> **Training Environment Strategy:**
> - Local system has CPU only and Python 3.14 (where PyTorch and Ultralytics install cleanly, but TensorFlow requires Python <=3.12 or Google Colab).
> - We provide a dual strategy:
>   1. **Local Python Pipeline**: Modular PyTorch/Ultralytics scripts for data preparation, local training/fine-tuning, ONNX export, and an end-to-end simulation script.
>   2. **1-Click Google Colab Notebook (`notebooks/train_on_colab.ipynb`)**: Takes full advantage of free Cloud GPUs (T4) to train both models in ~10-15 minutes and directly exports `finder_yolo.tflite` (int8 quantized, ~3-5MB) and `judge_mobilenet.tflite` (~2-3MB).
>   3. **Direct Export of Base Models**: We will generate immediate mobile-ready base models (`.tflite`) so you can test them in your mobile app repo right away.

> [!TIP]
> **Class Mapping in Model 2 (The Judge):**
> We map the Ghanaian CP-AnemiC dataset to the app's traffic-light UI:
> - `0: Green` (Healthy / Non-Anemic)
> - `1: Yellow` (Borderline / Mild & Moderate Anemia)
> - `2: Red` (Severe Anemia / Blood worm danger)
> The model will output both the 3-class traffic light probability and the 4-class granular severity (`Non-Anemic`, `Mild`, `Moderate`, `Severe`).

---

## Proposed Architecture & File Structure

```
Animal-Disease-Models/
├── configs/
│   ├── dataset_yolo.yaml         # YOLOv8 dataset configuration (train/val paths, eye class)
│   └── config.py                 # Paths, hyperparams, color thresholds, audio mappings
├── data/
│   ├── prepare_yolo_data.py      # Extracts eye annotations, splits 80/20 train/val
│   └── prepare_judge_data.py     # Structures CP-AnemiC dataset into Green/Yellow/Red & splits
├── models/
│   ├── finder.py                 # YOLOv8n detector wrapper with eye-crop extraction
│   └── judge.py                  # MobileNetV3 classifier architecture
├── training/
│   ├── train_finder.py           # Training & fine-tuning script for Model 1 (YOLOv8n)
│   └── train_judge.py            # Training script for Model 2 (MobileNetV3)
├── export/
│   ├── export_finder.py          # Exports YOLOv8n to .tflite and .onnx
│   └── export_judge.py           # Exports MobileNetV3 to .tflite and .onnx
├── evaluation/
│   ├── evaluate_finder.py        # Evaluates mAP50, Precision, Recall on validation set
│   └── evaluate_judge.py         # Evaluates Accuracy, Confusion Matrix, Classification Report
├── pipeline/
│   └── end_to_end_demo.py        # Full phone simulation: Camera photo -> Eye crop -> Color judge -> Audio advice
├── notebooks/
│   └── train_on_colab.ipynb      # Complete 1-click Google Colab notebook for GPU training
├── exported_models/              # Destination folder for generated .tflite & .onnx files
│   ├── finder_yolo.tflite
│   ├── finder_yolo.onnx
│   ├── judge_mobilenet.tflite
│   └── judge_mobilenet.onnx
├── tests/
│   └── test_pipeline.py          # Unit tests verifying dataset preparation and model forward passes
├── requirements.txt              # Production and development dependencies
└── README.md                     # Complete documentation, mobile integration guide & export instructions
```

---

## Proposed Changes

### Configuration & Utilities
#### [NEW] [configs/config.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/configs/config.py)
- Configuration defining paths to raw datasets, processed data, model artifacts, image sizes (YOLO 640x640, MobileNet 224x224), batch sizes, and Ghanaian language audio file mappings (`dagbani_*.mp3`, `hausa_*.mp3`, `gurune_*.mp3`).

#### [NEW] [configs/dataset_yolo.yaml](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/configs/dataset_yolo.yaml)
- Ultralytics YOLO configuration pointing to processed train and validation image directories for `eye` detection.

---

### Data Preparation
#### [NEW] [data/prepare_yolo_data.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/data/prepare_yolo_data.py)
- Reads `Goat Image Dataset/upload`.
- Extracts class 1 (`eye`), filters bounding boxes, and reformats them to single-class (`0: eye`) for optimal lightweight mobile detection.
- Generates 80/20 train/val splits in YOLO format.

#### [NEW] [data/prepare_judge_data.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/data/prepare_judge_data.py)
- Parses `Anemia_Data_Collection_Sheet.xlsx` and image files in `Anemic` and `Non-anemic`.
- Generates organized train/val/test splits mapped to 3 classes (`Green_Healthy`, `Yellow_Borderline`, `Red_Severe`) and 4 classes (`Non-Anemic`, `Mild`, `Moderate`, `Severe`).

---

### Training Scripts
#### [NEW] [training/train_finder.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/training/train_finder.py)
- Loads `yolov8n.pt` base model.
- Trains on processed goat eye dataset.
- Supports configurable epochs, image size (640 or 320 for ultra-fast mobile inference), device (`cpu` or `cuda`).
- Saves best model weights to `runs/detect/finder/weights/best.pt`.

#### [NEW] [training/train_judge.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/training/train_judge.py)
- Implements transfer learning with `MobileNetV3-Small`.
- Applies data augmentations (color jitter, slight rotation, horizontal flip) crucial for eyelid mucosa color assessment under variable outdoor sunlight conditions.
- Uses weighted Cross-Entropy Loss to handle the class imbalance (Severe class has 48 samples).
- Saves best weights.

---

### Model Export
#### [NEW] [export/export_finder.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/export/export_finder.py)
- Exports YOLOv8n to `.onnx` and `.tflite` (int8/fp16 quantization options).
- Copies exported model into `exported_models/finder_yolo.tflite` and `exported_models/finder_yolo.onnx`.

#### [NEW] [export/export_judge.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/export/export_judge.py)
- Exports MobileNetV3-Small to `.onnx` and `.tflite` (with default TFLite optimization / int8).
- Saves metadata labels: `labels.txt` with `[Green, Yellow, Red]`.
- Copies model into `exported_models/judge_mobilenet.tflite`.

---

### Pipeline Simulation & Verification
#### [NEW] [pipeline/end_to_end_demo.py](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/pipeline/end_to_end_demo.py)
- Simulates the entire phone workflow on any goat image:
  1. Input: Goat face photo.
  2. Model 1 runs -> locates eye bounding box -> crops eye with safety margin.
  3. Model 2 runs -> evaluates eye crop -> outputs severity score (Green / Yellow / Red).
  4. Audio trigger selector: displays the exact voice note filename to play (`hausa_danger.mp3`, `dagbani_healthy.mp3`, etc.) based on user language.

#### [NEW] [notebooks/train_on_colab.ipynb](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/notebooks/train_on_colab.ipynb)
- Interactive, self-contained Google Colab notebook with instructions to train both models on a free GPU and download the `.tflite` files directly.

#### [MODIFY] [README.md](file:///c:/Users/JOSHUA%20ASEMANI/Music/models/Animal-Disease-Models/README.md)
- Complete, easy-to-read documentation detailing:
  - Repository structure.
  - How to run data preparation.
  - How to train locally or in Google Colab.
  - How to export `.tflite` and `.onnx` files.
  - How to copy the models to the mobile app repository (`assets/models/`).

---

## Verification Plan

### Automated Tests
1. **Sanity Test (`tests/test_pipeline.py`)**:
   - Verify `prepare_yolo_data.py` outputs valid annotations and images.
   - Verify `prepare_judge_data.py` successfully reads and parses the Excel sheet and matches images.
   - Test model loading, forward pass with dummy input `(1, 3, 224, 224)` for MobileNet and `(1, 3, 640, 640)` for YOLOv8n.
   - Verify ONNX and TFLite model output dimensions.

2. **End-to-End Simulation**:
   - Run `pipeline/end_to_end_demo.py` on a sample image from the dataset and verify output:
     - Eye cropped correctly.
     - Severity score and label outputted.
     - Audio file name triggered correctly.

### Manual Verification
- Verify generated `.tflite` and `.onnx` model files in `exported_models/` and verify their file sizes (~3-5MB for YOLO nano, ~2-3MB for MobileNet).
