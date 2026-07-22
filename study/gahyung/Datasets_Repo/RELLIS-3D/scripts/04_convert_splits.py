from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "splits_original"
)

OUTPUT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rellis_cost4_standard"
    / "splits"
)

MASK_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rellis_cost4_standard"
    / "masks"
)


def determine_split_name(filename: str) -> str | None:
    name = filename.lower()

    if "train" in name:
        return "train"

    if "val" in name or "valid" in name:
        return "val"

    if "test" in name:
        return "test"

    return None


def convert_line(line: str) -> str | None:
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    # 한 줄에 image와 label 경로가 같이 있다면 첫 번째 경로 사용
    first_token = line.split()[0].replace("\\", "/")

    sequence_match = re.search(
        r"(?<!\d)(\d{5})(?!\d)",
        first_token,
    )

    if sequence_match is None:
        raise ValueError(
            f"sequence 번호를 찾을 수 없습니다: {line}"
        )

    sequence = sequence_match.group(1)
    stem = Path(first_token).stem

    return f"{sequence}_{stem}"


def main() -> None:
    if not SOURCE_SPLIT_ROOT.exists():
        raise FileNotFoundError(
            f"split 폴더가 없습니다: {SOURCE_SPLIT_ROOT}"
        )

    OUTPUT_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    split_files = [
        path
        for path in SOURCE_SPLIT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".lst"}
    ]

    converted: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    missing_masks: list[str] = []

    for split_file in split_files:
        split_name = determine_split_name(split_file.name)

        if split_name is None:
            continue

        with split_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            for line in file:
                sample_id = convert_line(line)

                if sample_id is None:
                    continue

                mask_path = MASK_ROOT / f"{sample_id}.png"

                if not mask_path.exists():
                    missing_masks.append(
                        f"{split_name},{sample_id}"
                    )
                    continue

                converted[split_name].add(sample_id)

    for split_name, sample_ids in converted.items():
        output_path = (
            OUTPUT_SPLIT_ROOT / f"{split_name}.txt"
        )

        with output_path.open("w", encoding="utf-8") as file:
            for sample_id in sorted(sample_ids):
                file.write(f"{sample_id}\n")

        print(
            f"{split_name}: {len(sample_ids)}개 "
            f"→ {output_path}"
        )

    if missing_masks:
        missing_path = (
            OUTPUT_SPLIT_ROOT / "missing_split_masks.csv"
        )

        missing_path.write_text(
            "split,sample_id\n"
            + "\n".join(missing_masks),
            encoding="utf-8",
        )

        print(f"split 누락 보고서: {missing_path}")


if __name__ == "__main__":
    main()