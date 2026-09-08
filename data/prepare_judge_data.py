"""
Prepare Dataset for Model 2 (The Judge / MobileNetV3 Anemia Classifier).
Parses Ghanaian CP-AnemiC metadata and conjunctiva crop images, maps them into
Traffic Light categories (Green/Yellow/Red) and 4-class severities, and performs
a stratified 70/15/15 train/val/test split.
"""
import shutil
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    CP_ANEMIC_SHEET,
    CP_ANEMIC_ANEMIC_DIR,
    CP_ANEMIC_NON_ANEMIC_DIR,
    JUDGE_PROCESSED_DIR,
    JUDGE_TRAIN_DIR,
    JUDGE_VAL_DIR,
    JUDGE_TEST_DIR,
    JUDGE_MANIFEST_CSV,
    SEVERITY_TO_TRAFFIC_LIGHT,
    JUDGE_CLASSES_3
)


def prepare_judge_dataset(seed: int = 42):
    """
    Organizes CP-AnemiC images into train/val/test folders by class
    and outputs a comprehensive manifest.csv.
    """
    np.random.seed(seed)

    print(f"Reading metadata from: {CP_ANEMIC_SHEET}")
    df = pd.read_excel(CP_ANEMIC_SHEET)

    # Clean & ensure directories exist
    for split_dir in [JUDGE_TRAIN_DIR, JUDGE_VAL_DIR, JUDGE_TEST_DIR]:
        for cls_name in JUDGE_CLASSES_3:
            (split_dir / cls_name).mkdir(parents=True, exist_ok=True)

    records = []
    missing_images = 0

    for _, row in df.iterrows():
        img_id = str(row["IMAGE_ID"]).strip()
        severity = str(row["Severity"]).strip()
        hb_level = float(row["HB_LEVEL"])

        traffic_light = SEVERITY_TO_TRAFFIC_LIGHT.get(severity, "Yellow_Borderline")

        # Locate image file
        filename = f"{img_id}.png"
        src_path = None
        if (CP_ANEMIC_ANEMIC_DIR / filename).exists():
            src_path = CP_ANEMIC_ANEMIC_DIR / filename
        elif (CP_ANEMIC_NON_ANEMIC_DIR / filename).exists():
            src_path = CP_ANEMIC_NON_ANEMIC_DIR / filename

        if src_path is None or not src_path.exists():
            missing_images += 1
            continue

        records.append({
            "image_id": img_id,
            "filename": filename,
            "src_path": str(src_path),
            "severity_4class": severity,
            "traffic_light_class": traffic_light,
            "hb_level": hb_level,
            "gender": row.get("GENDER", "Unknown"),
            "district": row.get("MUNICIPALITY/DISTRICT", "Unknown"),
            "region": row.get("REGION", "Unknown")
        })

    dataset_df = pd.DataFrame(records)
    print(f"Matched {len(dataset_df)} images (Missing: {missing_images})")

    # Stratified Train / Val / Test Split (70% / 15% / 15%)
    splits = []
    for cls_name in JUDGE_CLASSES_3:
        cls_sub = dataset_df[dataset_df["traffic_light_class"] == cls_name].copy()
        n = len(cls_sub)
        indices = np.random.permutation(n)

        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        cls_sub["split"] = ""
        split_col_idx = cls_sub.columns.get_loc("split")
        cls_sub.iloc[train_idx, split_col_idx] = "train"
        cls_sub.iloc[val_idx, split_col_idx] = "val"
        cls_sub.iloc[test_idx, split_col_idx] = "test"

        splits.append(cls_sub)

    final_df = pd.concat(splits, ignore_index=True)

    # Copy files into split directories
    dest_paths = []
    for _, row in final_df.iterrows():
        split = row["split"]
        cls_name = row["traffic_light_class"]
        src = Path(row["src_path"])
        dest = JUDGE_PROCESSED_DIR / split / cls_name / row["filename"]
        shutil.copy2(src, dest)
        dest_paths.append(str(dest.relative_to(JUDGE_PROCESSED_DIR.parent)))

    final_df["processed_path"] = dest_paths
    final_df.to_csv(JUDGE_MANIFEST_CSV, index=False)

    print("\n" + "=" * 55)
    print("Judge Dataset Split Summary:")
    print("=" * 55)
    summary = final_df.groupby(["split", "traffic_light_class"]).size().unstack(fill_value=0)
    print(summary)
    print("=" * 55)
    print(f"Manifest saved to: {JUDGE_MANIFEST_CSV}")
    print("[DONE] Judge dataset prepared successfully.\n")


if __name__ == "__main__":
    prepare_judge_dataset()
