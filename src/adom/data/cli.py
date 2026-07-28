from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import DatasetError
from .packaging import create_deterministic_tar
from .pipeline import inspect_dataset, prepare_dataset
from .validation import validate_package


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adom.data",
        description="Prepare and validate canonical ADOM Cost4 datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--dataset", required=True, choices=["rellis3d"])
    inspect_parser.add_argument("--input-root", required=True, type=_path)
    inspect_parser.add_argument("--mapping", required=True, type=_path)
    inspect_parser.add_argument("--report", type=_path)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset", required=True, choices=["rellis3d"])
    prepare_parser.add_argument("--input-root", required=True, type=_path)
    prepare_parser.add_argument("--output-root", required=True, type=_path)
    prepare_parser.add_argument("--mapping", required=True, type=_path)
    prepare_parser.add_argument("--split-root", required=True, type=_path)
    prepare_parser.add_argument("--version", default="v2.0")
    prepare_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset-root", required=True, type=_path)
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; canonical validation is always strict.",
    )
    validate_parser.add_argument("--skip-checksums", action="store_true")

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--dataset-root", required=True, type=_path)
    package_parser.add_argument("--archive", required=True, type=_path)
    return parser


def _emit(report: dict, report_path: Path | None = None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            report = inspect_dataset(args.dataset, args.input_root, args.mapping)
            _emit(report.to_dict(), args.report)
            report.require_success()
        elif args.command == "prepare":
            report = prepare_dataset(
                dataset=args.dataset,
                input_root=args.input_root,
                output_root=args.output_root,
                mapping_path=args.mapping,
                split_root=args.split_root,
                version=args.version,
                overwrite=args.overwrite,
            )
            _emit(report.to_dict())
            report.require_success()
        elif args.command == "validate":
            report = validate_package(
                args.dataset_root,
                verify_checksums=not args.skip_checksums,
            )
            _emit(report.to_dict())
            report.require_success()
        elif args.command == "package":
            archive, checksum = create_deterministic_tar(
                args.dataset_root,
                args.archive,
            )
            _emit({"archive": str(archive), "checksum": str(checksum)})
        else:
            raise DatasetError(f"Unknown command: {args.command}")
    except (DatasetError, FileNotFoundError, NotADirectoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
