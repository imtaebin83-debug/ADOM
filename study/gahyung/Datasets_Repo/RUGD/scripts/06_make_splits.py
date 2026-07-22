from pathlib import Path
import shutil

ROOT = Path(r"C:\Users\gahyu\RUGD")

SOURCE_IMAGE_DIR = ROOT / "processed" / "images" / "all"
SOURCE_MASK_DIR = ROOT / "processed" / "annotations" / "all"
SPLIT_DIR = ROOT / "processed" / "splits"

TRAIN_SEQUENCES = {
    "park-2",
    "trail",
    "trail-3",
    "trail-4",
    "trail-6",
    "trail-9",
    "trail-10",
    "trail-11",
    "trail-12",
    "trail-14",
    "trail-15",
    "village",
}

VAL_SEQUENCES = {
    "park-8",
    "trail-5",
}

TEST_SEQUENCES = {
    "creek",
    "park-1",
    "trail-7",
    "trail-13",
}

sequence_to_split = {}

for sequence in TRAIN_SEQUENCES:
    sequence_to_split[sequence] = "train"

for sequence in VAL_SEQUENCES:
    sequence_to_split[sequence] = "val"

for sequence in TEST_SEQUENCES:
    sequence_to_split[sequence] = "test"


def extract_sequence(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


for split in ("train", "val", "test"):
    (ROOT / "processed" / "images" / split).mkdir(
        parents=True,
        exist_ok=True,
    )

    (ROOT / "processed" / "annotations" / split).mkdir(
        parents=True,
        exist_ok=True,
    )

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

split_stems = {
    "train": [],
    "val": [],
    "test": [],
}

image_paths = sorted(SOURCE_IMAGE_DIR.glob("*.png"))

for image_path in image_paths:
    mask_path = SOURCE_MASK_DIR / image_path.name
    sequence = extract_sequence(image_path.stem)

    if sequence not in sequence_to_split:
        raise RuntimeError(
            f"분류되지 않은 sequence: "
            f"{sequence}, file={image_path.name}"
        )

    split = sequence_to_split[sequence]

    output_image = (
        ROOT / "processed" / "images"
        / split / image_path.name
    )

    output_mask = (
        ROOT / "processed" / "annotations"
        / split / mask_path.name
    )

    shutil.copy2(image_path, output_image)
    shutil.copy2(mask_path, output_mask)

    split_stems[split].append(image_path.stem)

for split, stems in split_stems.items():
    split_file = SPLIT_DIR / f"{split}.txt"

    split_file.write_text(
        "\n".join(stems),
        encoding="utf-8",
    )

    print(split, len(stems))