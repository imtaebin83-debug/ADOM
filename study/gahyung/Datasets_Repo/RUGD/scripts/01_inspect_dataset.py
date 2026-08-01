from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUGD_REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_INPUT_ROOT = RUGD_REPOSITORY_ROOT / "raw"


def parse_arguments() -> argparse.Namespace:
    environment_input_root = os.environ.get(
        "RUGD_INPUT_ROOT"
    )

    default_input_root = (
        Path(environment_input_root)
        if environment_input_root
        else DEFAULT_INPUT_ROOT
    )

    parser = argparse.ArgumentParser(
        description=(
            "Inspect files under a RUGD input root "
            "and report extension counts."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root,
        help=(
            "RUGD raw or extracted dataset directory. "
            "The RUGD_INPUT_ROOT environment variable "
            "is used when this argument is omitted."
        ),
    )

    return parser.parse_args()


def collect_extension_counts(
    input_root: Path,
) -> tuple[int, Counter[str]]:
    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root is not a directory: {input_root}"
        )

    file_paths = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file()
    )

    if not file_paths:
        raise RuntimeError(
            f"No files were found under input root: "
            f"{input_root}"
        )

    extension_counts: Counter[str] = Counter()

    for file_path in file_paths:
        extension = file_path.suffix.lower()

        if not extension:
            extension = "<no_extension>"

        extension_counts[extension] += 1

    return len(file_paths), extension_counts


def print_summary(
    input_root: Path,
    total_file_count: int,
    extension_counts: Counter[str],
) -> None:
    print("RUGD dataset inspection")
    print(f"검색 경로: {input_root}")
    print(f"폴더 존재 여부: {input_root.exists()}")
    print(f"전체 파일 수: {total_file_count}")
    print("확장자:")

    for extension in sorted(extension_counts):
        print(
            f"  {extension}: "
            f"{extension_counts[extension]}"
        )

    print(
        "[PASS] RUGD dataset extension "
        "inspection completed."
    )


def main() -> int:
    arguments = parse_arguments()

    input_root = (
        arguments.input_root
        .expanduser()
        .resolve()
    )

    try:
        total_file_count, extension_counts = (
            collect_extension_counts(input_root)
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
    ) as error:
        print(
            f"[ERROR] {error}",
            file=sys.stderr,
        )

        return 1

    print_summary(
        input_root=input_root,
        total_file_count=total_file_count,
        extension_counts=extension_counts,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())