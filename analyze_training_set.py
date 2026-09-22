"""Rank illnesses in the RFMiD training set by number of labeled images."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import Counter
from pathlib import Path


LABEL_NAMES = {
    "DR": "Diabetic retinopathy",
    "MH": "Media haze",
    "ODC": "Optic disc cupping",
    "TSLN": "Tessellation",
    "DN": "Drusen",
    "MYA": "Myopia",
    "ARMD": "Age-related macular degeneration",
    "BRVO": "Branch retinal vein occlusion",
    "ODP": "Optic disc pallor",
    "ODE": "Optic disc edema",
}

NON_ILLNESS_COLUMNS = {"ID", "Disease_Risk"}


def count_illnesses(zip_path: Path) -> tuple[int, int, Counter[str]]:
    """Return image, healthy-image, and per-illness counts."""
    with zipfile.ZipFile(zip_path) as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError(f"Expected one CSV label file, found {len(csv_files)}")

        with archive.open(csv_files[0]) as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_file)
            if not reader.fieldnames:
                raise ValueError("The label CSV has no header")

            illness_columns = [
                column for column in reader.fieldnames if column not in NON_ILLNESS_COLUMNS
            ]
            counts: Counter[str] = Counter()
            record_count = 0
            healthy_count = 0

            for row_number, row in enumerate(reader, start=2):
                record_count += 1
                positive_labels = 0
                for illness in illness_columns:
                    value = row[illness].strip()
                    if value not in {"0", "1"}:
                        raise ValueError(
                            f"Invalid value {value!r} for {illness} on CSV row {row_number}"
                        )
                    counts[illness] += int(value)
                    positive_labels += int(value)
                healthy_count += positive_labels == 0

    if record_count == 0:
        raise ValueError("The label CSV contains no records")

    return record_count, healthy_count, counts


def display_name(label: str) -> str:
    name = LABEL_NAMES.get(label)
    return f"{name} ({label})" if name else label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the most prevalent illnesses in the RFMiD training-set ZIP."
    )
    parser.add_argument(
        "zip_path",
        nargs="?",
        type=Path,
        default=Path("Training_Set.zip"),
        help="path to Training_Set.zip (default: ./Training_Set.zip)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="number of illnesses to show (default: all)",
    )
    args = parser.parse_args()
    if args.top is not None and args.top < 1:
        parser.error("--top must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    try:
        record_count, healthy_count, counts = count_illnesses(args.zip_path)
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    ranking = counts.most_common(args.top)
    most_common_label, most_common_count = counts.most_common(1)[0]
    diseased_count = record_count - healthy_count

    print(f"Analyzed {record_count:,} labeled images from {args.zip_path}")
    print(f"Illness labels found: {len(counts)}\n")
    print(
        f"Without any disease: {healthy_count:,} "
        f"({healthy_count / record_count * 100:.2f}%)"
    )
    print(
        f"With one or more diseases: {diseased_count:,} "
        f"({diseased_count / record_count * 100:.2f}%)\n"
    )
    print(f"{'Rank':>4}  {'Category':<38} {'Cases':>6} {'Prevalence':>11}")
    print("-" * 65)
    for rank, (label, count) in enumerate(ranking, start=1):
        prevalence = count / record_count * 100
        print(f"{rank:>4}  {display_name(label):<38} {count:>6,} {prevalence:>10.2f}%")

    prevalence = most_common_count / record_count * 100
    print(
        f"\nMost prevalent illness: {display_name(most_common_label)} "
        f"with {most_common_count:,} cases ({prevalence:.2f}% of images)."
    )
    print("Labels are not mutually exclusive; one image may have multiple illnesses.")


if __name__ == "__main__":
    main()