"""Dedicated fundus-versus-non-fundus validation stage."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


FUNDUS_VALIDATOR_PATH = Path(__file__).with_name("fundus_validator_model.pt")
VALIDATION_CLASS_NAMES = ("non_fundus", "fundus")
VALIDATION_MESSAGES = {
    "non_fundus": (
        "Завантажене зображення не розпізнано як зображення очного дна. "
        "Аналіз не виконано."
    ),
    "uncertain": (
        "Не вдалося достовірно визначити тип зображення. "
        "Завантажте інше зображення очного дна."
    ),
}


@dataclass(frozen=True)
class FundusValidationResult:
    status: Literal["fundus", "non_fundus", "uncertain"]
    confidence: float | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "fundus"

    @property
    def message(self) -> str:
        return VALIDATION_MESSAGES.get(self.status, "")


@dataclass(frozen=True)
class FundusValidator:
    model: nn.Module
    transform: transforms.Compose
    device: torch.device
    fundus_threshold: float
    non_fundus_threshold: float


def _fallback_fundus_score(image: Image.Image) -> tuple[float, int]:
    """Estimate retinal appearance when dedicated validation weights are absent."""
    sample = ImageOps.exif_transpose(image).convert("RGB").resize((192, 192))
    pixels = np.asarray(sample, dtype=np.float32)
    red, green, blue = np.moveaxis(pixels, -1, 0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )

    y_coordinates, x_coordinates = np.ogrid[-1:1:192j, -1:1:192j]
    radius = np.sqrt(x_coordinates**2 + y_coordinates**2)
    center_mask = radius <= 0.72
    outer_mask = radius >= 0.86

    center_red = red[center_mask]
    center_green = green[center_mask]
    center_blue = blue[center_mask]
    center_luminance = luminance[center_mask]
    center_saturation = saturation[center_mask]
    outer_luminance = luminance[outer_mask]

    green_image = Image.fromarray(green.astype(np.uint8), mode="L")
    smooth_green = np.asarray(
        green_image.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32
    )
    dark_line_response = smooth_green - green
    vessel_mask = (dark_line_response > 7) & center_mask
    vessel_fraction = float(vessel_mask[center_mask].mean())
    vessel_quadrants = sum(
        float(vessel_mask[quadrant_mask].mean()) >= 0.012
        for quadrant_mask in (
            center_mask & (x_coordinates < 0) & (y_coordinates < 0),
            center_mask & (x_coordinates >= 0) & (y_coordinates < 0),
            center_mask & (x_coordinates < 0) & (y_coordinates >= 0),
            center_mask & (x_coordinates >= 0) & (y_coordinates >= 0),
        )
    )

    mean_red = float(center_red.mean())
    mean_green = float(center_green.mean())
    mean_blue = float(center_blue.mean())
    warm_pixel_fraction = float(
        ((center_red > center_green * 1.02) & (center_red > center_blue * 1.08)).mean()
    )
    strongly_warm_fraction = float(
        ((center_red > center_green * 1.15) & (center_red > center_blue * 1.30)).mean()
    )
    outer_dark_fraction = float((outer_luminance < 45).mean())
    center_brightness = float(center_luminance.mean())
    outer_brightness = float(outer_luminance.mean())

    evidence = (
        1.5 if warm_pixel_fraction >= 0.45 else 0.0,
        1.0 if strongly_warm_fraction >= 0.55 else 0.0,
        1.0 if mean_red >= mean_green * 1.25 else 0.0,
        1.0 if mean_red >= mean_blue * 1.80 else 0.0,
        1.0 if float(center_saturation.mean()) >= 0.22 else 0.0,
        1.0 if float(center_luminance.std()) >= 13 else 0.0,
        1.0 if 0.005 <= vessel_fraction <= 0.15 else 0.0,
        1.0 if vessel_fraction <= 0.15 and vessel_quadrants >= 3 else 0.0,
        2.0
        if outer_dark_fraction >= 0.12 or center_brightness >= outer_brightness * 1.12
        else 0.0,
    )
    return sum(evidence), sum(value > 0 for value in evidence)


def load_fundus_validator(
    device: torch.device,
    model_path: Path = FUNDUS_VALIDATOR_PATH,
) -> FundusValidator | None:
    """Load a model trained specifically for fundus-versus-non-fundus validation."""
    if not model_path.is_file():
        return None

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError("Файл моделі перевірки зображень пошкоджений або несумісний") from error

    if checkpoint.get("task") != "fundus_validation":
        raise ValueError("Модель перевірки не призначена для розпізнавання зображень очного дна")
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("Модель перевірки використовує непідтримувану архітектуру")
    if tuple(checkpoint.get("class_names", ())) != VALIDATION_CLASS_NAMES:
        raise ValueError("Модель перевірки містить несумісні класи")
    if checkpoint.get("positive_class") != "fundus":
        raise ValueError("Модель перевірки має некоректно визначений позитивний клас")

    fundus_threshold = float(checkpoint.get("fundus_threshold", 0.90))
    non_fundus_threshold = float(checkpoint.get("non_fundus_threshold", 0.10))
    if not (
        math.isfinite(fundus_threshold)
        and math.isfinite(non_fundus_threshold)
        and 0 <= non_fundus_threshold < fundus_threshold <= 1
    ):
        raise ValueError("Модель перевірки містить некоректні пороги")

    image_size = int(checkpoint.get("image_size", 224))
    normalization_mean = tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406)))
    normalization_std = tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225)))
    if image_size <= 0 or len(normalization_mean) != 3 or len(normalization_std) != 3:
        raise ValueError("Модель перевірки містить некоректні параметри обробки")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    validation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ]
    )
    return FundusValidator(
        model=model,
        transform=validation_transform,
        device=device,
        fundus_threshold=fundus_threshold,
        non_fundus_threshold=non_fundus_threshold,
    )


def validate_fundus_image(
    image: Image.Image,
    validator: FundusValidator | None,
) -> FundusValidationResult:
    """Classify image type, preferring dedicated model output when available."""
    if validator is None:
        score, matched_characteristics = _fallback_fundus_score(image)
        if score >= 6.0 and matched_characteristics >= 4:
            return FundusValidationResult("fundus", min(score / 10.5, 0.85))
        if score <= 2.5:
            return FundusValidationResult("non_fundus", min((10.5 - score) / 10.5, 0.85))
        return FundusValidationResult("uncertain", 0.5)

    prepared_image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = validator.transform(prepared_image).unsqueeze(0).to(validator.device)
    with torch.inference_mode():
        probability = float(torch.sigmoid(validator.model(tensor))[0, 0].cpu())

    if probability >= validator.fundus_threshold:
        return FundusValidationResult("fundus", probability)
    if probability <= validator.non_fundus_threshold:
        return FundusValidationResult("non_fundus", 1 - probability)
    return FundusValidationResult("uncertain", max(probability, 1 - probability))
