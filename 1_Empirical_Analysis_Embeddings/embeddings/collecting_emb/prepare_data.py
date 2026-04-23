'''This script should be used to prepare and set up the extra data dataset.'''
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_SOURCE = Path("G_gender_count_3.csv")
DEFAULT_TARGET = Path("extra_data.csv")

PROMPT_PREFIXES = (
    ("Photo portrait of That person ", "A photo of one real person that "),
    ("Photo portrait of That ", "A photo of one real person that "),
)


def transform_prompt(prompt: str) -> str:
    for old_prefix, new_prefix in PROMPT_PREFIXES:
        if prompt.startswith(old_prefix):
            return f"{new_prefix}{prompt[len(old_prefix):]}"
    raise ValueError(f"Unsupported prompt format: {prompt!r}")


def transform_file(source: Path, target: Path) -> None:
    with source.open("r", newline="", encoding="utf-8") as src_file:
        reader = csv.DictReader(src_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {source}")

        rows = []
        for row in reader:
            row["Prompt"] = transform_prompt(row["Prompt"])
            rows.append(row)

    with target.open("w", newline="", encoding="utf-8") as dst_file:
        writer = csv.DictWriter(dst_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform G_gender_count_3.csv into extra_data.csv format."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Input CSV file (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Output CSV file (default: {DEFAULT_TARGET})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform_file(args.source, args.target)


if __name__ == "__main__":
    main()
