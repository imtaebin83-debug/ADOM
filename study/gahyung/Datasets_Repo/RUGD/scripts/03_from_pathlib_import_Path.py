from pathlib import Path

ROOT = Path(r"C:\Users\gahyu\RUGD")

IMAGE_DIR = (
    ROOT / "raw" / "RUGD"
    / "3.after join creek" / "image"
)

MASK_DIR = (
    ROOT / "raw" / "RUGD"
    / "3.after join creek" / "indexLabel"
)

PROCESSED_DIR = ROOT / "processed"

ALL_IMAGE_DIR = PROCESSED_DIR / "images" / "all"
ALL_MASK_DIR = PROCESSED_DIR / "annotations" / "all"

SPLIT_DIR = PROCESSED_DIR / "splits"
METADATA_DIR = PROCESSED_DIR / "metadata"
QC_DIR = PROCESSED_DIR / "qc"