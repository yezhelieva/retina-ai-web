"""Model loading and inference for retinal image screening."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image, ImageOps, ImageStat
from torch import nn
from torchvision.models import resnet18

from train_binary_classifier import make_transforms


MODEL_PATH = Path(__file__).with_name("top5_retina_model.pt")
DISPLAY_CLASS_NAMES = {
    "DR": "Діабетична ретинопатія",
    "MH": "Помутніння оптичних середовищ ока",
    "ODC": "Екскавація диска зорового нерва",
    "TSLN": "Телеангіектазія сітківки",
    "DN": "Друзи диска зорового нерва",
}


@dataclass(frozen=True)
class ScreeningResult:
    path: Path
    image: Image.Image
    label_codes: tuple[str, ...]
    class_names: tuple[str, ...]
    probabilities: tuple[float, ...]
    thresholds: tuple[float, ...]
    quality_notes: tuple[str, ...]

    @property
    def detected_count(self) -> int:
        return sum(
            probability >= threshold
            for probability, threshold in zip(self.probabilities, self.thresholds)
        )

    @property
    def attention_count(self) -> int:
        return sum(
            probability_level(probability, threshold) == "attention"
            for probability, threshold in zip(self.probabilities, self.thresholds)
        )

    @property
    def highest_level(self) -> str:
        if self.detected_count:
            return "high"
        if self.attention_count:
            return "attention"
        return "low"


def probability_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= max(0.0, threshold - 0.15):
        return "attention"
    return "low"


def load_model(
    device: torch.device | None = None,
) -> tuple[nn.Module, tuple[str, ...], tuple[str, ...], tuple[float, ...], torch.device]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Файл моделі не знайдено: {MODEL_PATH.name}")

    selected_device = device or torch.device("cpu")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=selected_device, weights_only=True)
    except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError("Файл моделі пошкоджений або має несумісний формат") from error
    label_codes = tuple(checkpoint.get("label_codes", ()))
    class_names = tuple(checkpoint.get("class_names", ()))
    thresholds = tuple(float(value) for value in checkpoint.get("thresholds", ()))
    if label_codes != tuple(DISPLAY_CLASS_NAMES) or len(class_names) != 5:
        raise ValueError("Вибрана модель не містить п'ять очікуваних класів захворювань")
    if len(thresholds) != 5 or any(
        not math.isfinite(threshold) or not 0 <= threshold <= 1
        for threshold in thresholds
    ):
        raise ValueError("Модель містить некоректні пороги класифікації")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(label_codes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(selected_device)
    model.eval()
    return model, label_codes, class_names, thresholds, selected_device


def assess_image_quality(image: Image.Image) -> tuple[str, ...]:
    notes: list[str] = []
    if min(image.size) < 224:
        notes.append(
            "Низька роздільна здатність може знизити надійність. "
            "Потрібно щонайменше 224 пікселі з кожного боку."
        )

    brightness = ImageStat.Stat(image.convert("L")).mean[0]
    if brightness < 25:
        notes.append("Знімок надто темний. Перевірте освітлення та видимість сітківки.")
    elif brightness > 230:
        notes.append(
            "Знімок надмірно освітлений. Перевірте освітлення та видимість сітківки."
        )
    return tuple(notes)


def screen_image(
    source: Path | BinaryIO,
    model: nn.Module,
    label_codes: tuple[str, ...],
    class_names: tuple[str, ...],
    thresholds: tuple[float, ...],
    device: torch.device,
    filename: str | None = None,
) -> ScreeningResult:
    with Image.open(source) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")

    _, evaluation_transform = make_transforms()
    tensor = evaluation_transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].cpu().tolist()

    result_path = source if isinstance(source, Path) else Path(filename or "image")
    return ScreeningResult(
        path=result_path,
        image=image,
        label_codes=label_codes,
        class_names=class_names,
        probabilities=tuple(probabilities),
        thresholds=thresholds,
        quality_notes=assess_image_quality(image),
    )


def screening_summary(result: ScreeningResult) -> str:
    if result.highest_level == "high":
        return (
            "Модель виявила високу ймовірність ознак патологічних змін.\n"
            "Рекомендується подальша оцінка результату офтальмологом."
        )
    if result.highest_level == "attention":
        return (
            "Модель виявила ознаки, які потребують додаткової уваги.\n"
            "Рекомендується професійна оцінка результату офтальмологом."
        )
    return (
        "Ознак патологічних змін з високою ймовірністю не виявлено.\n"
        "Рекомендується професійне офтальмологічне обстеження."
    )