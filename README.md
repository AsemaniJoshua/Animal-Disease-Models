# 🐐 Animal Disease Models: Offline Mobile Diagnosis for Sheep & Goats

> **Lightweight, offline AI models designed for remote farmers in Northern Ghana (Dagbani, Hausa, Gurune).**
> Automatically detects sheep/goat eyes from mobile camera photos, evaluates inner eyelid (conjunctiva) color to diagnose blood-worm anemia (FAMACHA), and triggers offline local voice notes.

---

## 📱 System Overview

```
[Phone Camera Live Photo]
          │
          ▼
 [Model 1: The Finder]  ──► Locates and extracts eye/eyelid crop (YOLOv8 Nano, ~3-5MB)
          │
          ▼ (Padded Eyelid Crop)
 [Model 2: The Judge]   ──► Assesses mucosal color & anemia severity (MobileNetV3 Small, ~2-3MB)
          │
          ▼ (Traffic Light Classification)
 [Screen + Local Speaker]
     ├── 🟢 Green  (Healthy / Optimal)   ──► Plays '{lang}_healthy.mp3'
     ├── 🟡 Yellow (Borderline / Alert)   ──► Plays '{lang}_warning.mp3'
     └── 🔴 Red    (Severe Anemia Danger)──► Plays '{lang}_danger.mp3'
```

---

## 📁 Repository Structure

```text
Animal-Disease-Models/
├── configs/
│   ├── config.py                 # Central configurations, audio mappings & hyperparameters
│   └── dataset_yolo.yaml         # YOLOv8 dataset configuration
├── data/
│   ├── prepare_yolo_data.py      # Extracts eye annotations & splits YOLO dataset (80/20)
│   └── prepare_judge_data.py     # Parses Ghanaian CP-AnemiC data into Green/Yellow/Red
├── models/
│   ├── finder.py                 # YOLOv8n detector wrapper with eye cropping
│   └── judge.py                  # MobileNetV3-Small classifier architecture
├── training/
│   ├── train_finder.py           # Fine-tunes YOLOv8n on eye dataset
│   └── train_judge.py            # Trains MobileNetV3 on conjunctiva dataset
├── export/
│   ├── export_finder.py          # Exports YOLOv8n to .onnx and .tflite
│   └── export_judge.py           # Exports MobileNetV3 to .onnx and .tflite
├── evaluation/
│   ├── evaluate_finder.py        # Evaluates YOLO mAP, precision, and recall
│   └── evaluate_judge.py         # Evaluates MobileNet accuracy & confusion matrix
├── pipeline/
│   └── end_to_end_demo.py        # End-to-end mobile simulation (Camera -> Crop -> Judge -> Audio)
├── notebooks/
│   └── train_on_colab.ipynb      # 1-Click Google Colab GPU training notebook
├── exported_models/              # Output directory for mobile-ready .tflite / .onnx models
├── tests/
│   └── test_pipeline.py          # Unit tests verifying pipeline components
├── requirements.txt              # Core dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets
Run the automated preparation scripts to clean, structure, and split the raw datasets:
```bash
# Prepare eye detection dataset for Model 1
python data/prepare_yolo_data.py

# Prepare conjunctiva anemia dataset for Model 2
python data/prepare_judge_data.py
```

### 3. Run Pipeline Simulation Demo
You can simulate the entire mobile app workflow on any sample image right away:
```bash
# Test with Hausa audio trigger
python pipeline/end_to_end_demo.py --lang hausa

# Test with Dagbani audio trigger
python pipeline/end_to_end_demo.py --lang dagbani

# Test with Gurune audio trigger
python pipeline/end_to_end_demo.py --lang gurune
```
The output displays the located eye bounding box, saves `detected_eye_crop.png`, displays the traffic light score, and outputs the audio trigger.

---

## 🏋️ Model Training

### Option A: 1-Click Cloud Training (Free Google Colab GPU) — Recommended
Open [`notebooks/train_on_colab.ipynb`](notebooks/train_on_colab.ipynb) in [Google Colab](https://colab.research.google.com/):
1. Set runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
2. Run all cells to train both models in ~10-15 minutes.
3. Automatically exports and downloads `finder_yolo.tflite` and `judge_mobilenet.tflite`.

### Option B: Local Training
```bash
# Train Model 1 (Finder / YOLOv8 Nano)
python training/train_finder.py --epochs 30 --batch 16

# Train Model 2 (Judge / MobileNetV3)
python training/train_judge.py --epochs 25 --batch 32
```

---

## 📦 Exporting Models for Mobile App

Once trained, export the models into lightweight, mobile-ready `.tflite` and `.onnx` formats:

```bash
# Export Model 1 to ONNX / TFLite
python export/export_finder.py

# Export Model 2 to ONNX / TFLite
python export/export_judge.py
```

The exported models and metadata are saved in `exported_models/`:
- `finder_yolo.tflite` (~3-5 MB)
- `judge_mobilenet.tflite` (~2-3 MB)
- `labels.txt`
- `judge_metadata.json` & `finder_metadata.json`

---

## 📲 Mobile App Integration Guide

Copy the exported files into your mobile app repository (`assets/models/` and `assets/audio/`):

```text
my-mobile-app/
├── assets/
│   ├── models/
│   │   ├── finder_yolo.tflite
│   │   ├── judge_mobilenet.tflite
│   │   └── labels.txt
│   └── audio/
│       ├── hausa_healthy.mp3
│       ├── hausa_warning.mp3
│       ├── hausa_danger.mp3
│       ├── dagbani_healthy.mp3
│       ├── dagbani_warning.mp3
│       ├── dagbani_danger.mp3
│       ├── gurune_healthy.mp3
│       ├── gurune_warning.mp3
│       └── gurune_danger.mp3
```

### React Native / Flutter Inference Logic:
1. Capture photo from camera frame.
2. Run `finder_yolo.tflite` to get eye bounding box `[x1, y1, x2, y2]`.
3. Crop the eye region with 15% margin and resize to `224x224`.
4. Run `judge_mobilenet.tflite` on the crop to get scores `[p_green, p_yellow, p_red]`.
5. Display traffic-light color and play the corresponding offline MP3 in the farmer's selected language.

---

## 🧪 Running Tests
```bash
python -m unittest tests/test_pipeline.py
```
