"""Train a binary retinal classifier for five target diseases versus healthy eyes."""

from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet


IMAGE_SIZE = 224
TARGET_DISEASE_CODES = ("DR", "MH", "ODC", "TSLN", "DN")
CLASS_NAMES = ("Healthy", "Target pathology")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    image_name: str
    label: int


@dataclass(frozen=True)
class DatasetStatistics:
    included: int
    healthy: int
    pathological: int
    excluded_other_disease_only: int
    target_positive_counts: dict[str, int]


class ZipImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, zip_path: Path, samples: list[Sample], transform: Any) -> None:
        self.zip_path = zip_path
        self.samples = samples
        self.transform = transform
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.zip_path)

        with self._archive.open(sample.image_name) as raw_file:
            with Image.open(io.BytesIO(raw_file.read())) as source_image:
                image = source_image.convert("RGB")
        return self.transform(image), sample.label

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __del__(self) -> None:
        self.close()


def load_samples(zip_path: Path) -> tuple[list[Sample], DatasetStatistics]:
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one label CSV, found {len(csv_names)}")

        images_by_id = {
            Path(name).stem: name
            for name in archive.namelist()
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        }
        with archive.open(csv_names[0]) as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_file)
            required_columns = {"ID", "Disease_Risk", *TARGET_DISEASE_CODES}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    "Label CSV must contain ID, Disease_Risk, and all target disease columns"
                )
            pathology_columns = [
                column
                for column in reader.fieldnames
                if column not in {"ID", "Disease_Risk"}
            ]

            samples: list[Sample] = []
            healthy_count = 0
            pathological_count = 0
            excluded_count = 0
            target_positive_counts = {code: 0 for code in TARGET_DISEASE_CODES}
            for row_number, row in enumerate(reader, start=2):
                image_id = row["ID"].strip()
                risk = row["Disease_Risk"].strip()
                if risk not in {"0", "1"}:
                    raise ValueError(
                        f"Invalid Disease_Risk {risk!r} on CSV row {row_number}"
                    )
                if image_id not in images_by_id:
                    raise ValueError(f"No image found for ID {image_id!r}")

                pathology_values = {
                    column: row[column].strip() for column in pathology_columns
                }
                invalid_columns = [
                    column
                    for column, value in pathology_values.items()
                    if value not in {"0", "1"}
                ]
                if invalid_columns:
                    raise ValueError(
                        f"Invalid pathology label on CSV row {row_number}: "
                        f"{invalid_columns[0]}"
                    )

                target_positive = any(
                    pathology_values[code] == "1" for code in TARGET_DISEASE_CODES
                )
                any_diagnosed_pathology = any(
                    value == "1" for value in pathology_values.values()
                )
                for code in TARGET_DISEASE_CODES:
                    target_positive_counts[code] += pathology_values[code] == "1"

                if target_positive:
                    samples.append(Sample(images_by_id[image_id], 1))
                    pathological_count += 1
                elif not any_diagnosed_pathology and risk == "0":
                    samples.append(Sample(images_by_id[image_id], 0))
                    healthy_count += 1
                else:
                    # Other-disease-only eyes are neither target-positive nor healthy controls.
                    excluded_count += 1

    if not samples:
        raise ValueError("No labeled images found")
    statistics = DatasetStatistics(
        included=len(samples),
        healthy=healthy_count,
        pathological=pathological_count,
        excluded_other_disease_only=excluded_count,
        target_positive_counts=target_positive_counts,
    )
    return samples, statistics


def print_dataset_statistics(
    split_name: str, statistics: DatasetStatistics
) -> None:
    print(f"\n{split_name} dataset")
    print(f"  Included images: {statistics.included:,}")
    print(f"  Healthy images: {statistics.healthy:,}")
    print(f"  Pathological images: {statistics.pathological:,}")
    print(
        "  Excluded other-disease-only images: "
        f"{statistics.excluded_other_disease_only:,}"
    )
    print("  Target-positive examples:")
    for code in TARGET_DISEASE_CODES:
        print(f"    {code}: {statistics.target_positive_counts[code]:,}")


def stratified_split(
    samples: list[Sample], validation_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    random_generator = random.Random(seed)
    grouped = {label: [] for label in range(len(CLASS_NAMES))}
    for sample in samples:
        grouped[sample.label].append(sample)

    training: list[Sample] = []
    validation: list[Sample] = []
    for label, group in grouped.items():
        if len(group) < 2:
            raise ValueError(f"Class {CLASS_NAMES[label]} needs at least two samples")
        random_generator.shuffle(group)
        validation_size = max(1, round(len(group) * validation_fraction))
        validation_size = min(validation_size, len(group) - 1)
        validation.extend(group[:validation_size])
        training.extend(group[validation_size:])

    random_generator.shuffle(training)
    random_generator.shuffle(validation)
    return training, validation


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    training_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training_transform, validation_transform


def create_model(pretrained: bool) -> ResNet:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def set_trainable_layers(model: ResNet, fine_tune: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def set_training_mode(model: ResNet, fine_tune: bool) -> None:
    # Keep frozen BatchNorm statistics fixed while training selected layers.
    model.eval()
    model.fc.train()
    if fine_tune:
        model.layer4.train()


def calculate_metrics(
    targets: list[int], probabilities: list[float], threshold: float
) -> dict[str, float]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    true_positive = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    true_negative = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    false_positive = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    false_negative = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))

    total = len(targets)
    accuracy = (true_positive + true_negative) / max(1, total)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    f1_score = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score,
        "tp": float(true_positive),
        "tn": float(true_negative),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def find_optimal_threshold(targets: list[int], probabilities: list[float]) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(5, 96):
        threshold = step / 100
        f1 = calculate_metrics(targets, probabilities, threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def collect_outputs(
    model: ResNet,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[list[int], list[float]]:
    targets: list[int] = []
    probabilities: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            batch_probabilities = torch.sigmoid(model(images).squeeze(1))
            targets.extend(labels.tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())
    return targets, probabilities


def print_metrics(dataset_name: str, metrics: dict[str, float], threshold: float) -> None:
    print(f"\n{dataset_name} metrics (threshold={threshold:.2f})")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1-score:    {metrics['f1']:.4f}")
    print(
        f"  Confusion matrix: TP={int(metrics['tp'])}, TN={int(metrics['tn'])}, "
        f"FP={int(metrics['fp'])}, FN={int(metrics['fn'])}"
    )
    print(
        "  Matrix [[TN, FP], [FN, TP]]: "
        f"[[{int(metrics['tn'])}, {int(metrics['fp'])}], "
        f"[{int(metrics['fn'])}, {int(metrics['tp'])}]]"
    )


def train_stage(
    model: ResNet,
    training_loader: DataLoader[tuple[torch.Tensor, int]],
    evaluation_loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    device: torch.device,
    stage_name: str,
    fine_tune: bool,
) -> tuple[dict[str, torch.Tensor], float, dict[str, float]]:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, patience // 3), min_lr=1e-7
    )
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] = {}
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    print(f"\n{stage_name}")
    for epoch in range(1, epochs + 1):
        set_training_mode(model, fine_tune)
        running_loss = 0.0
        for images, labels in training_loader:
            images = images.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).squeeze(1), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(images)

        evaluation_targets, evaluation_probabilities = collect_outputs(
            model, evaluation_loader, device
        )
        threshold = find_optimal_threshold(evaluation_targets, evaluation_probabilities)
        metrics = calculate_metrics(
            evaluation_targets, evaluation_probabilities, threshold
        )
        scheduler.step(metrics["f1"])

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_threshold = threshold
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        learning_rates = ", ".join(
            f"{group['lr']:.2e}" for group in optimizer.param_groups
        )
        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"loss={running_loss / len(training_loader.dataset):.4f}, "
            f"eval_f1={metrics['f1']:.4f}, threshold={threshold:.2f}, "
            f"lr=[{learning_rates}]"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    return best_state, best_threshold, best_metrics


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA-capable device is available")
    return torch.device(requested_device)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    training_samples, training_statistics = load_samples(args.training_zip)
    evaluation_samples, evaluation_statistics = load_samples(args.evaluation_zip)
    # Test labels are read here only for the required descriptive statistics.
    # Test images and predictions remain untouched until final model selection.
    test_samples, test_statistics = load_samples(args.test_zip)
    for split_name, samples in (
        ("training", training_samples),
        ("evaluation", evaluation_samples),
        ("test", test_samples),
    ):
        if set(sample.label for sample in samples) != {0, 1}:
            raise ValueError(f"{split_name} split must contain both classes")

    print_dataset_statistics("Training", training_statistics)
    print_dataset_statistics("Evaluation", evaluation_statistics)
    print_dataset_statistics("Test", test_statistics)

    training_transform, evaluation_transform = make_transforms()
    training_dataset = ZipImageDataset(
        args.training_zip, training_samples, training_transform
    )
    evaluation_dataset = ZipImageDataset(
        args.evaluation_zip, evaluation_samples, evaluation_transform
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    training_loader = DataLoader(training_dataset, shuffle=True, **loader_options)
    evaluation_loader = DataLoader(evaluation_dataset, shuffle=False, **loader_options)

    model = create_model(pretrained=not args.no_pretrained).to(device)
    class_counts = Counter(sample.label for sample in training_samples)
    pos_weight = torch.tensor(
        [class_counts[0] / class_counts[1]],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Device: {device}")
    print(f"BCE positive-class weight: {pos_weight.item():.4f}")

    # Stage 1 trains only the randomly initialized binary classification head.
    set_trainable_layers(model, fine_tune=False)
    head_optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    stage1_state, stage1_threshold, stage1_metrics = train_stage(
        model,
        training_loader,
        evaluation_loader,
        criterion,
        head_optimizer,
        args.head_epochs,
        args.patience,
        device,
        "Stage 1: frozen backbone",
        fine_tune=False,
    )
    model.load_state_dict(stage1_state)

    # Stage 2 adapts layer4 using a lower LR while continuing to train the head.
    set_trainable_layers(model, fine_tune=True)
    fine_tune_optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": args.backbone_learning_rate},
            {"params": model.fc.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    stage2_state, stage2_threshold, stage2_metrics = train_stage(
        model,
        training_loader,
        evaluation_loader,
        criterion,
        fine_tune_optimizer,
        args.fine_tune_epochs,
        args.patience,
        device,
        "Stage 2: fine-tune layer4 and head",
        fine_tune=True,
    )

    if stage2_metrics["f1"] > stage1_metrics["f1"]:
        best_state, optimal_threshold, evaluation_metrics = (
            stage2_state,
            stage2_threshold,
            stage2_metrics,
        )
    else:
        best_state, optimal_threshold, evaluation_metrics = (
            stage1_state,
            stage1_threshold,
            stage1_metrics,
        )
    model.load_state_dict(best_state)

    # Test data is loaded only after model and threshold selection is complete.
    test_dataset = ZipImageDataset(args.test_zip, test_samples, evaluation_transform)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    test_targets, test_probabilities = collect_outputs(model, test_loader, device)
    test_metrics = calculate_metrics(test_targets, test_probabilities, optimal_threshold)

    checkpoint = {
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "architecture": "resnet18",
        "output_features": 1,
        "class_names": CLASS_NAMES,
        "positive_class": CLASS_NAMES[1],
        "target_disease_codes": TARGET_DISEASE_CODES,
        "threshold": optimal_threshold,
        "image_size": IMAGE_SIZE,
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "evaluation_metrics": evaluation_metrics,
        "test_metrics": test_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    training_dataset.close()
    evaluation_dataset.close()
    test_dataset.close()
    print(f"Saved best model to {args.output}")
    print_metrics("Evaluation set", evaluation_metrics, optimal_threshold)
    print_metrics("Test set", test_metrics, optimal_threshold)


def predict(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)

    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    _, validation_transform = make_transforms()

    with Image.open(args.image) as source_image:
        image = validation_transform(source_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(image).squeeze()).item()

    threshold = float(checkpoint["threshold"])
    predicted_index = int(probability >= threshold)
    print(f"Prediction: {CLASS_NAMES[predicted_index]}")
    print(f"Pathological probability: {probability * 100:.2f}%")
    print(f"Decision threshold: {threshold:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and validate a classifier")
    train_parser.add_argument(
        "training_zip", type=Path, nargs="?", default=Path("Training_Set.zip")
    )
    train_parser.add_argument(
        "evaluation_zip", type=Path, nargs="?", default=Path("Evaluation_Set.zip")
    )
    train_parser.add_argument(
        "test_zip", type=Path, nargs="?", default=Path("Test_Set.zip")
    )
    train_parser.add_argument("--output", type=Path, default=Path("binary_retina_model.pt"))
    train_parser.add_argument("--head-epochs", type=int, default=8)
    train_parser.add_argument("--fine-tune-epochs", type=int, default=25)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument(
        "--head-learning-rate", "--learning-rate", dest="head_learning_rate",
        type=float, default=1e-3,
    )
    train_parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=6)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.set_defaults(handler=train)

    predict_parser = subparsers.add_parser("predict", help="classify one retinal image")
    predict_parser.add_argument("model", type=Path)
    predict_parser.add_argument("image", type=Path)
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_parser.set_defaults(handler=predict)

    args = parser.parse_args()
    if args.command == "train":
        if (
            args.head_epochs < 1
            or args.fine_tune_epochs < 1
            or args.batch_size < 1
            or args.head_learning_rate <= 0
            or args.backbone_learning_rate <= 0
            or args.weight_decay < 0
            or args.patience < 1
            or args.workers < 0
        ):
            parser.error("epochs, batch size, learning rates, and patience must be positive")
        if args.backbone_learning_rate >= args.head_learning_rate:
            parser.error("backbone learning rate must be smaller than head learning rate")
    return args


def main() -> None:
    args = parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()