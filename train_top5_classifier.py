"""Train a multi-label classifier for the five most common RFMiD diseases."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet

from train_binary_classifier import IMAGE_SIZE, make_transforms, resolve_device


LABEL_CODES = ("DR", "MH", "ODC", "TSLN", "DN")
CLASS_NAMES = (
    "Diabetic retinopathy",
    "Media haze",
    "Optic disc cupping",
    "Tessellation",
    "Drusen",
)


@dataclass(frozen=True)
class MultiLabelSample:
    image_name: str
    labels: tuple[float, ...]


class ZipMultiLabelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, zip_path: Path, samples: list[MultiLabelSample], transform: Any) -> None:
        self.zip_path = zip_path
        self.samples = samples
        self.transform = transform
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.zip_path)
        with self._archive.open(sample.image_name) as raw_file:
            with Image.open(io.BytesIO(raw_file.read())) as source_image:
                image = source_image.convert("RGB")
        return self.transform(image), torch.tensor(sample.labels, dtype=torch.float32)

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


def load_samples(zip_path: Path) -> list[MultiLabelSample]:
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
            reader = csv.DictReader(io.TextIOWrapper(raw_file, encoding="utf-8-sig"))
            if not reader.fieldnames or not set(LABEL_CODES).issubset(reader.fieldnames):
                raise ValueError("Label CSV does not contain all top-five disease columns")
            samples: list[MultiLabelSample] = []
            for row_number, row in enumerate(reader, start=2):
                image_id = row["ID"].strip()
                values = tuple(row[code].strip() for code in LABEL_CODES)
                if any(value not in {"0", "1"} for value in values):
                    raise ValueError(f"Invalid disease label on CSV row {row_number}")
                if image_id not in images_by_id:
                    raise ValueError(f"No image found for ID {image_id!r}")
                samples.append(
                    MultiLabelSample(images_by_id[image_id], tuple(float(value) for value in values))
                )
    if not samples:
        raise ValueError("No labeled images found")
    return samples


def create_model(backbone_checkpoint: Path, device: torch.device) -> ResNet:
    checkpoint = torch.load(backbone_checkpoint, map_location=device, weights_only=True)
    state = {
        name: value
        for name, value in checkpoint["model_state"].items()
        if not name.startswith("fc.")
    }
    model = resnet18(weights=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError("Backbone checkpoint is incompatible with ResNet-18")
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_CODES))
    return model.to(device)


class WeightedFocalLoss(nn.Module):
    """Class-weighted BCE focal loss for imbalanced multi-label targets."""

    def __init__(self, pos_weight: torch.Tensor, gamma: float) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        binary_cross_entropy = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        probability_of_truth = probabilities * targets + (1 - probabilities) * (1 - targets)
        focal_weight = (1 - probability_of_truth).pow(self.gamma)
        return (focal_weight * binary_cross_entropy).mean()


def set_trainable_layers(model: ResNet, fine_tune: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def set_training_mode(model: ResNet, fine_tune: bool) -> None:
    # Frozen BatchNorm statistics must not drift during either training stage.
    model.eval()
    model.fc.train()
    if fine_tune:
        model.layer4.train()


def collect_predictions(
    model: ResNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            probability_batches.append(torch.sigmoid(logits).cpu())
            target_batches.append(targets)
    return torch.cat(probability_batches), torch.cat(target_batches)


def tune_thresholds(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    thresholds = torch.empty(len(LABEL_CODES))
    candidates = torch.arange(0.10, 0.91, 0.05)
    for label_index in range(len(LABEL_CODES)):
        best_threshold = 0.5
        best_f1 = -1.0
        truth = targets[:, label_index].bool()
        for threshold in candidates:
            prediction = probabilities[:, label_index] >= threshold
            true_positive = (prediction & truth).sum().item()
            false_positive = (prediction & ~truth).sum().item()
            false_negative = (~prediction & truth).sum().item()
            f1 = 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[label_index] = best_threshold
    return thresholds


def calculate_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, Any]:
    predictions = probabilities >= thresholds
    truth = targets.bool()
    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for index, code in enumerate(LABEL_CODES):
        prediction = predictions[:, index]
        actual = truth[:, index]
        true_positive = (prediction & actual).sum().item()
        false_positive = (prediction & ~actual).sum().item()
        false_negative = (~prediction & actual).sum().item()
        true_negative = (~prediction & ~actual).sum().item()
        accuracy = (true_positive + true_negative) / max(
            1, true_positive + true_negative + false_positive + false_negative
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_scores.append(f1)
        per_class[code] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": true_negative / max(1, true_negative + false_positive),
            "f1": f1,
        }
    return {"macro_f1": sum(f1_scores) / len(f1_scores), "per_class": per_class}


def print_metrics(dataset_name: str, metrics: dict[str, Any]) -> None:
    print(f"\n{dataset_name} metrics")
    print(
        f"{'Class':<6} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'Specificity':>12} {'F1-score':>10}"
    )
    print("-" * 64)
    for code in LABEL_CODES:
        class_metrics = metrics["per_class"][code]
        print(
            f"{code:<6} {class_metrics['accuracy']:>10.4f} "
            f"{class_metrics['precision']:>10.4f} {class_metrics['recall']:>10.4f} "
            f"{class_metrics['specificity']:>12.4f} {class_metrics['f1']:>10.4f}"
        )
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


def train_stage(
    model: ResNet,
    training_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    evaluation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    device: torch.device,
    stage_name: str,
    fine_tune: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, patience // 3), min_lr=1e-7
    )
    best_score = -1.0
    best_state: dict[str, torch.Tensor] = {}
    best_thresholds = torch.full((len(LABEL_CODES),), 0.5)
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0

    print(f"\n{stage_name}")
    for epoch in range(1, epochs + 1):
        set_training_mode(model, fine_tune)
        running_loss = 0.0
        for images, targets in training_loader:
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(images)

        probabilities, evaluation_targets = collect_predictions(
            model, evaluation_loader, device
        )
        thresholds = tune_thresholds(probabilities, evaluation_targets)
        metrics = calculate_metrics(probabilities, evaluation_targets, thresholds)
        scheduler.step(metrics["macro_f1"])
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_thresholds = thresholds.clone()
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
            f"eval_macro_f1={metrics['macro_f1']:.4f}, lr=[{learning_rates}]"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    return best_state, best_thresholds, best_metrics


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    training_transform, evaluation_transform = make_transforms()

    samples_by_split: dict[str, list[MultiLabelSample]] = {}
    for split, path in (("training", args.training_zip), ("evaluation", args.evaluation_zip)):
        samples_by_split[split] = load_samples(path)
        samples = samples_by_split[split]
        for label_index, code in enumerate(LABEL_CODES):
            positive_count = sum(sample.labels[label_index] for sample in samples)
            if not 0 < positive_count < len(samples):
                raise ValueError(
                    f"{split} split must contain positive and negative examples for {code}"
                )
        print(f"{split.title()} images: {len(samples):,}")

    training_dataset = ZipMultiLabelDataset(
        args.training_zip, samples_by_split["training"], training_transform
    )
    evaluation_dataset = ZipMultiLabelDataset(
        args.evaluation_zip, samples_by_split["evaluation"], evaluation_transform
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    training_loader = DataLoader(training_dataset, shuffle=True, **loader_options)
    evaluation_loader = DataLoader(evaluation_dataset, shuffle=False, **loader_options)

    model = create_model(args.backbone, device)
    training_targets = torch.tensor(
        [sample.labels for sample in samples_by_split["training"]], dtype=torch.float32
    )
    positives = training_targets.sum(dim=0)
    pos_weight = ((len(training_targets) - positives) / positives.clamp_min(1)).to(device)
    criterion = WeightedFocalLoss(pos_weight, args.focal_gamma)
    print(f"Device: {device}")
    print(
        "Positive-class weights: "
        + ", ".join(
            f"{code}={weight:.2f}" for code, weight in zip(LABEL_CODES, pos_weight.tolist())
        )
    )

    # Stage 1 learns a stable task-specific head without modifying pretrained features.
    set_trainable_layers(model, fine_tune=False)
    head_optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=args.head_learning_rate, weight_decay=args.weight_decay
    )
    stage1_state, stage1_thresholds, stage1_metrics = train_stage(
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

    # Stage 2 adapts only layer4, with a lower LR than the classification head.
    set_trainable_layers(model, fine_tune=True)
    fine_tune_optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": args.backbone_learning_rate},
            {"params": model.fc.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    stage2_state, stage2_thresholds, stage2_metrics = train_stage(
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

    if stage2_metrics["macro_f1"] > stage1_metrics["macro_f1"]:
        best_state, thresholds, evaluation_metrics = (
            stage2_state,
            stage2_thresholds,
            stage2_metrics,
        )
    else:
        best_state, thresholds, evaluation_metrics = (
            stage1_state,
            stage1_thresholds,
            stage1_metrics,
        )
    model.load_state_dict(best_state)

    # The test set is opened only after model and thresholds are fixed on evaluation data.
    test_samples = load_samples(args.test_zip)
    for label_index, code in enumerate(LABEL_CODES):
        positive_count = sum(sample.labels[label_index] for sample in test_samples)
        if not 0 < positive_count < len(test_samples):
            raise ValueError(f"test split must contain positive and negative examples for {code}")
    test_dataset = ZipMultiLabelDataset(args.test_zip, test_samples, evaluation_transform)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    test_probabilities, test_targets = collect_predictions(model, test_loader, device)
    test_metrics = calculate_metrics(test_probabilities, test_targets, thresholds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {name: value.cpu() for name, value in model.state_dict().items()},
            "label_codes": LABEL_CODES,
            "class_names": CLASS_NAMES,
            "thresholds": thresholds.tolist(),
            "image_size": IMAGE_SIZE,
            "evaluation_metrics": evaluation_metrics,
            "test_metrics": test_metrics,
        },
        args.output,
    )
    training_dataset.close()
    evaluation_dataset.close()
    test_dataset.close()
    print(f"Saved model to {args.output}")
    print(
        "Selected thresholds: "
        + ", ".join(
            f"{code}={threshold:.2f}" for code, threshold in zip(LABEL_CODES, thresholds.tolist())
        )
    )
    print_metrics("Evaluation set", evaluation_metrics)
    print_metrics("Test set", test_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-zip", type=Path, default=Path("Training_Set.zip"))
    parser.add_argument("--evaluation-zip", type=Path, default=Path("Evaluation_Set.zip"))
    parser.add_argument("--test-zip", type=Path, default=Path("Test_Set.zip"))
    parser.add_argument("--backbone", type=Path, default=Path("binary_retina_model_three_sets.pt"))
    parser.add_argument("--output", type=Path, default=Path("top5_retina_model.pt"))
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (
        args.head_epochs < 1
        or args.fine_tune_epochs < 1
        or args.batch_size < 1
        or args.head_learning_rate <= 0
        or args.backbone_learning_rate <= 0
        or args.weight_decay < 0
        or args.focal_gamma < 0
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
        train(args)
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()