from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


NON_ILLNESS_COLUMNS = {"ID", "Disease_Risk"}


def count_illnesses(zip_path: Path) -> tuple[int, int, Counter[str]]:
    with zipfile.ZipFile(zip_path) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected one CSV label file, found {len(csv_files)}"
            )

        with archive.open(csv_files[0]) as raw_file:
            text_file = io.TextIOWrapper(
                raw_file,
                encoding="utf-8-sig",
                newline="",
            )

            reader = csv.DictReader(text_file)

            if not reader.fieldnames:
                raise ValueError("The label CSV has no header")

            illness_columns = [
                column
                for column in reader.fieldnames
                if column not in NON_ILLNESS_COLUMNS
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
                            f"Invalid value {value!r} "
                            f"for {illness} on CSV row {row_number}"
                        )

                    label_value = int(value)

                    counts[illness] += label_value
                    positive_labels += label_value

                if positive_labels == 0:
                    healthy_count += 1

    if record_count == 0:
        raise ValueError("The label CSV contains no records")

    return record_count, healthy_count, counts


def plot_distribution(
    counts: Counter[str],
    record_count: int,
    top: int | None = None,
) -> None:
    ranking = counts.most_common(top)

    labels = [
        label
        for label, _ in ranking
    ]

    values = [
        count
        for _, count in ranking
    ]

    plt.figure(figsize=(10, 7))

    bars = plt.barh(labels, values)

    plt.xlabel("Кількість зображень")
    plt.ylabel("Патологічний клас")
    plt.title(
        "Розподіл патологічних класів "
        "у навчальній вибірці"
    )

    plt.gca().invert_yaxis()

    max_value = max(values)

    plt.xlim(0, max_value * 1.22)

    for bar, value in zip(bars, values):
        prevalence = value / record_count * 100

        plt.text(
            value + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({prevalence:.1f}%)",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(
        "class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the distribution of pathological "
            "classes in the training dataset."
        )
    )

    parser.add_argument(
        "zip_path",
        nargs="?",
        type=Path,
        default=Path("Training_Set.zip"),
        help="Path to Training_Set.zip",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Number of pathological classes to show",
    )

    args = parser.parse_args()

    if args.top is not None and args.top < 1:
        parser.error("--top must be at least 1")

    return args


def main() -> None:
    args = parse_args()

    try:
        record_count, healthy_count, counts = count_illnesses(
            args.zip_path
        )

    except (
        FileNotFoundError,
        PermissionError,
        zipfile.BadZipFile,
        ValueError,
    ) as error:
        raise SystemExit(
            f"Error: {error}"
        ) from error

    ranking = counts.most_common(args.top)

    diseased_count = record_count - healthy_count

    print(
        f"Analyzed {record_count:,} labeled images "
        f"from {args.zip_path}"
    )

    print(
        f"Illness labels found: {len(counts)}\n"
    )

    print(
        f"Without any disease: "
        f"{healthy_count:,} "
        f"({healthy_count / record_count * 100:.2f}%)"
    )

    print(
        f"With one or more diseases: "
        f"{diseased_count:,} "
        f"({diseased_count / record_count * 100:.2f}%)\n"
    )

    print(
        f"{'Rank':>4}  "
        f"{'Class':<12} "
        f"{'Cases':>6} "
        f"{'Prevalence':>11}"
    )

    print("-" * 40)

    for rank, (label, count) in enumerate(
        ranking,
        start=1,
    ):
        prevalence = (
            count / record_count * 100
        )

        print(
            f"{rank:>4}  "
            f"{label:<12} "
            f"{count:>6,} "
            f"{prevalence:>10.2f}%"
        )

    most_common_label, most_common_count = (
        counts.most_common(1)[0]
    )

    prevalence = (
        most_common_count
        / record_count
        * 100
    )

    print(
        f"\nMost prevalent illness: "
        f"{most_common_label} "
        f"with {most_common_count:,} cases "
        f"({prevalence:.2f}% of images)."
    )

    print(
        "Labels are not mutually exclusive; "
        "one image may have multiple illnesses."
    )

    plot_distribution(
        counts,
        record_count,
        args.top,
    )


if __name__ == "__main__":
    main()