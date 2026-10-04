"""Streamlit web interface for retinal image screening."""

from __future__ import annotations

import html
import base64
import hashlib
import io
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageOps, ImageStat, UnidentifiedImageError
from torch import nn
from torchvision.models import resnet18

from fundus_validator import (
    FundusValidationResult,
    FundusValidator,
    load_fundus_validator,
    validate_fundus_image,
)
from train_binary_classifier import make_transforms, resolve_device


st.set_page_config(
    page_title="Система автоматизованого скринінгу зображень очного дна",
    page_icon="👁️",
    layout="wide",
)

MODEL_PATH = Path(__file__).with_name("top5_retina_model.pt")
MAX_UPLOADS = 10
DISPLAY_CLASS_NAMES = {
    "DR": "Діабетична ретинопатія",
    "MH": "Помутніння оптичних середовищ ока",
    "ODC": "Екскавація диска зорового нерва",
    "TSLN": "Телеангіектазія сітківки",
    "DN": "Друзи диска зорового нерва",
}

COLORS = {
    "ink": "#172321",
    "muted": "#64716d",
    "paper": "#f4f6f3",
    "panel": "#ffffff",
    "line": "#d8dfdb",
    "teal": "#176b63",
    "teal_soft": "#e3f1ed",
    "green": "#2f7d4a",
    "green_soft": "#e6f3ea",
    "red": "#a33d34",
    "red_soft": "#fae9e6",
    "amber": "#a76316",
    "amber_soft": "#fff1dc",
}

LEVEL_STYLES = {
    "high": ("Висока ймовірність", COLORS["red"], COLORS["red_soft"]),
    "attention": ("Потребує уваги", COLORS["amber"], COLORS["amber_soft"]),
    "low": ("Низька ймовірність", COLORS["green"], COLORS["green_soft"]),
}

@dataclass(frozen=True)
class ScreeningResult:
    filename: str
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


@dataclass(frozen=True)
class UploadedImage:
    image_id: str
    filename: str
    content: bytes
    size: int


@dataclass(frozen=True)
class ImageAnalysisError:
    message: str


def probability_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= max(0.0, threshold - 0.15):
        return "attention"
    return "low"


def load_model() -> tuple[
    nn.Module,
    tuple[str, ...],
    tuple[str, ...],
    tuple[float, ...],
    torch.device,
]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Файл моделі не знайдено: {MODEL_PATH.name}")

    device = resolve_device("auto")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
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
    model.to(device)
    model.eval()
    return model, label_codes, class_names, thresholds, device


@st.cache_resource
def get_model() -> tuple[
    nn.Module,
    tuple[str, ...],
    tuple[str, ...],
    tuple[float, ...],
    torch.device,
]:
    return load_model()


@st.cache_resource
def get_fundus_validator(device: torch.device) -> FundusValidator | None:
    return load_fundus_validator(device)


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
    image: Image.Image,
    filename: str,
    model: nn.Module,
    label_codes: tuple[str, ...],
    class_names: tuple[str, ...],
    thresholds: tuple[float, ...],
    device: torch.device,
) -> ScreeningResult:
    prepared_image = ImageOps.exif_transpose(image).convert("RGB")
    _, evaluation_transform = make_transforms()
    tensor = evaluation_transform(prepared_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].cpu().tolist()

    return ScreeningResult(
        filename=filename,
        image=prepared_image,
        label_codes=label_codes,
        class_names=class_names,
        probabilities=tuple(probabilities),
        thresholds=thresholds,
        quality_notes=assess_image_quality(prepared_image),
    )


def analyze_image(
    image: Image.Image,
    filename: str,
    model: nn.Module,
    label_codes: tuple[str, ...],
    class_names: tuple[str, ...],
    thresholds: tuple[float, ...],
    device: torch.device,
    validator: FundusValidator | None,
) -> ScreeningResult | FundusValidationResult:
    validation = validate_fundus_image(image, validator)
    if not validation.accepted:
        return validation
    return screen_image(
        image,
        filename,
        model,
        label_codes,
        class_names,
        thresholds,
        device,
    )


def initialize_session_state() -> None:
    if "uploaded_images" not in st.session_state:
        st.session_state["uploaded_images"] = {}
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = {}
    if "selected_image" not in st.session_state:
        st.session_state["selected_image"] = None
    if "uploader_generation" not in st.session_state:
        st.session_state["uploader_generation"] = 0


def collect_new_uploads(uploaded_files: list[object]) -> tuple[list[str], int, int]:
    stored_images: dict[str, UploadedImage] = st.session_state["uploaded_images"]
    batch_signatures: set[str] = set()
    new_image_ids: list[str] = []
    duplicate_count = 0
    overflow_count = 0
    for uploaded_file in uploaded_files:
        filename = str(getattr(uploaded_file, "name", ""))
        content = uploaded_file.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        image_id = f"{filename}:{digest}"
        if image_id in batch_signatures:
            duplicate_count += 1
            continue
        batch_signatures.add(image_id)
        if image_id in stored_images:
            continue
        if len(stored_images) >= MAX_UPLOADS:
            overflow_count += 1
            continue
        stored_images[image_id] = UploadedImage(
            image_id=image_id,
            filename=filename,
            content=content,
            size=len(content),
        )
        new_image_ids.append(image_id)
    return new_image_ids, duplicate_count, overflow_count


def select_image(image_id: str) -> None:
    st.session_state["selected_image"] = image_id


def delete_image(image_id: str) -> None:
    st.session_state["uploaded_images"].pop(image_id, None)
    st.session_state["analysis_results"].pop(image_id, None)
    if st.session_state.get("selected_image") == image_id:
        st.session_state["selected_image"] = None
    st.session_state["uploader_generation"] += 1


def choose_available_image() -> None:
    stored_images: dict[str, UploadedImage] = st.session_state["uploaded_images"]
    selected_image = st.session_state.get("selected_image")
    if selected_image in stored_images:
        return
    results = st.session_state["analysis_results"]
    valid_image_ids = [
        image_id
        for image_id in stored_images
        if is_screening_result(results.get(image_id))
    ]
    st.session_state["selected_image"] = (
        valid_image_ids[0] if valid_image_ids else next(iter(stored_images), None)
    )


def format_file_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} МБ"
    return f"{size / 1024:.1f} КБ"


def is_screening_result(analysis: object) -> TypeGuard[ScreeningResult]:
    return all(
        hasattr(analysis, attribute)
        for attribute in ("filename", "probabilities", "thresholds", "highest_level")
    )


def is_analysis_error(analysis: object) -> TypeGuard[ImageAnalysisError]:
    return type(analysis).__name__ == "ImageAnalysisError" and hasattr(analysis, "message")


def analysis_card_status(analysis: object) -> tuple[str, str, str]:
    if is_screening_result(analysis):
        return LEVEL_STYLES[analysis.highest_level]
    if isinstance(analysis, FundusValidationResult):
        if analysis.status == "non_fundus":
            return "Не розпізнано", COLORS["red"], COLORS["red_soft"]
        return "Потребує перевірки", COLORS["amber"], COLORS["amber_soft"]
    if is_analysis_error(analysis):
        return "Помилка аналізу", COLORS["red"], COLORS["red_soft"]
    return "Очікує аналізу", COLORS["muted"], "#edf1ef"


def make_circular_preview(image: Image.Image, size: int = 640) -> Image.Image:
    preview = ImageOps.fit(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    ).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    preview.putalpha(mask)
    return preview


def screening_summary(result: ScreeningResult) -> str:
    if result.highest_level == "high":
        return (
            "Модель виявила високу ймовірність ознак патологічних змін.\n\n"
            "Рекомендується подальша оцінка результату офтальмологом."
        )
    if result.highest_level == "attention":
        return (
            "Модель виявила ознаки, які потребують додаткової уваги.\n\n"
            "Рекомендується професійна оцінка результату офтальмологом."
        )
    return (
        "Ознак патологічних змін з високою ймовірністю не виявлено.\n\n"
        "Рекомендується професійне офтальмологічне обстеження."
    )


def render_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --ink: {COLORS['ink']};
            --muted: {COLORS['muted']};
            --paper: {COLORS['paper']};
            --panel: {COLORS['panel']};
            --line: {COLORS['line']};
            --teal: {COLORS['teal']};
            --teal-soft: {COLORS['teal_soft']};
        }}
        .stApp {{ background: var(--paper); color: var(--ink); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stMainBlockContainer"] {{
            box-sizing: border-box;
            max-width: 1450px;
            padding: 0 24px 1rem;
            width: 100%;
        }}
        .app-header {{
            background: #123f4a;
            box-sizing: border-box;
            box-shadow: 0 0 0 100vmax #123f4a;
            clip-path: inset(0 -100vmax);
            color: white;
            margin: 0 0 24px;
            padding-block: clamp(24px, 2.2vw, 30px);
            padding-inline: 0;
            width: 100%;
        }}
        .app-header h1 {{ font: 700 clamp(21px, 2.1vw, 28px)/1.25 "Segoe UI", sans-serif; margin: 0; }}
        h1, h2, h3, p, label, div {{ font-family: "Segoe UI", sans-serif; }}
        [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) {{
            align-items: flex-start;
            gap: 24px;
        }}
        [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) > [data-testid="stColumn"] {{
            min-width: 0;
        }}
        [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) {{
            align-items: flex-start;
            gap: 24px;
        }}
        [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) > [data-testid="stColumn"] {{
            min-width: 0;
        }}
        .dashboard-sidebar-marker,
        .result-panels-marker,
        .major-panel-marker {{ display: none; }}
        [data-testid="stElementContainer"]:has(.dashboard-sidebar-marker),
        [data-testid="stElementContainer"]:has(.result-panels-marker),
        [data-testid="stElementContainer"]:has(.major-panel-marker) {{ display: none; }}
        .section-label {{
            color: var(--muted);
            font-size: 14px;
            font-weight: 700;
            margin: 0 0 16px;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .major-panel-marker) {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            box-sizing: border-box;
            padding: 22px;
        }}
        [data-testid="stFileUploader"] {{
            min-height: 44px;
            position: relative;
            width: 100%;
        }}
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {{
            align-items: center;
            background: var(--teal);
            box-sizing: border-box;
            color: white;
            display: flex;
            font-size: 14px;
            font-weight: 700;
            border-radius: 7px;
            height: 44px;
            justify-content: center;
            margin: 0;
            padding: 0 14px;
            width: 100%;
        }}
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {{
            color: inherit;
            font-size: inherit;
            font-weight: inherit;
            margin: 0;
        }}
        [data-testid="stFileUploader"]:hover [data-testid="stWidgetLabel"] {{ background: #125c55; }}
        [data-testid="stFileUploader"]:focus-within [data-testid="stWidgetLabel"] {{
            box-shadow: 0 0 0 2px var(--paper), 0 0 0 4px var(--teal);
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: transparent;
            border: 0;
            inset: 0;
            min-height: 44px;
            opacity: 0;
            padding: 0;
            position: absolute;
            width: 100%;
            z-index: 2;
        }}
        [data-testid="stFileUploaderDropzone"] button {{ height: 44px; width: 100%; }}
        [data-testid="stFileUploaderDropzoneInstructions"] {{ display: none; }}
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileChip"] {{ display: none; }}
        [data-testid="stButton"] button {{
            border-radius: 0;
            min-height: 42px;
            max-width: 100%;
        }}
        [data-testid="stButton"] button[kind="primary"] {{
            background: var(--teal-soft);
            border-color: var(--teal);
            color: var(--teal);
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-sizing: border-box;
            margin-bottom: 12px;
            overflow: hidden;
            height: auto;
            padding: 0;
            position: relative;
            transition: background-color 120ms ease, border-color 120ms ease;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)):hover {{
            background: #f8faf9;
            border-color: #aebdb8;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)):has([class*="st-key-select_"] button[kind="primary"]) {{
            background: var(--teal-soft);
            border-color: var(--teal);
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) {{
            gap: 0;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-select_"] {{
            inset: 0;
            margin: 0;
            position: absolute;
            z-index: 1;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-select_"] [data-testid="stButton"] {{
            inset: 0;
            position: absolute;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-select_"] button {{
            background: transparent !important;
            border: 0 !important;
            inset: 0;
            color: transparent !important;
            height: 100%;
            min-height: 100%;
            padding: 0;
            position: absolute;
            width: 100%;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-select_"] button:focus-visible {{
            box-shadow: inset 0 0 0 2px var(--teal);
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-select_"] [data-testid="stIconMaterial"] {{
            display: none;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-delete_"] {{
            position: absolute;
            right: 10px;
            top: 10px;
            z-index: 3;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-delete_"] button {{
            background: transparent;
            border: 0;
            color: var(--muted);
            height: 28px;
            min-height: 28px;
            padding: 0;
            width: 28px;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-delete_"] button:hover {{
            background: {COLORS['red_soft']};
            color: {COLORS['red']};
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .image-card-marker):not(:has(.dashboard-sidebar-marker)) [class*="st-key-delete_"] button p {{
            clip: rect(0 0 0 0);
            clip-path: inset(50%);
            height: 1px;
            overflow: hidden;
            position: absolute;
            white-space: nowrap;
            width: 1px;
        }}
        .image-card-marker {{ display: none; }}
        [data-testid="stElementContainer"]:has(.image-card-content) {{
            height: auto;
            min-height: fit-content;
        }}
        [data-testid="stMarkdownContainer"]:has(.image-card-content),
        [data-testid="stMarkdownContainer"]:has(.overall-result) {{ margin-bottom: 0; }}
        .image-card-content {{
            box-sizing: border-box;
            min-width: 0;
            padding: 14px;
            pointer-events: none;
            position: relative;
            width: 100%;
            z-index: 2;
        }}
        .image-file-meta {{
            align-items: center;
            box-sizing: border-box;
            display: flex;
            gap: 9px;
            min-width: 0;
            padding-right: 34px;
            width: 100%;
        }}
        .image-file-icon {{
            color: var(--teal);
            flex: 0 0 auto;
            font-family: "Material Symbols Rounded";
            font-size: 18px;
            font-weight: normal;
            line-height: 1.2;
        }}
        .image-file-copy {{ display: flex; flex: 1 1 auto; flex-direction: column; gap: 4px; min-width: 0; }}
        .image-file-name {{
            color: var(--ink);
            font-size: 14px;
            font-weight: 600;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .image-file-size {{
            color: var(--muted);
            font-size: 12px;
            margin: 0;
        }}
        [data-testid="stImage"] {{ display: flex; justify-content: center; width: 100%; }}
        [data-testid="stImage"] img {{
            aspect-ratio: 1;
            height: auto;
            max-height: 600px;
            max-width: min(100%, 600px) !important;
            object-fit: contain;
            width: 100% !important;
        }}
        .pathology-list {{ display: grid; gap: 12px; }}
        .pathology-card {{
            background: #fafbfa;
            border: 1px solid #e2e7e4;
            border-radius: 8px;
            box-sizing: border-box;
            padding: 15px;
        }}
        .pathology-header {{
            align-items: flex-start;
            display: flex;
            gap: 16px;
            justify-content: space-between;
            min-width: 0;
        }}
        .pathology-title {{
            color: var(--ink);
            font-size: 13px;
            font-weight: 600;
            line-height: 1.35;
            min-width: 0;
            overflow-wrap: normal;
            word-break: normal;
        }}
        .pathology-meta {{ margin-top: 12px; }}
        .probability-track {{
            background: #edf1ef;
            border-radius: 999px;
            height: 6px;
            margin-top: 8px;
            overflow: hidden;
            width: 100%;
        }}
        .probability-fill {{ border-radius: inherit; height: 100%; min-width: 2px; }}
        .empty-state {{
            background: var(--panel);
            border: 1px solid var(--line);
            padding: 32px 18px;
        }}
        .empty-state h2 {{ color: var(--ink); font-size: 24px; margin: 0 0 5px; }}
        .empty-state p {{ color: var(--muted); font-size: 14px; margin: 0; }}
        .empty-state strong {{ color: var(--teal); display: block; font-size: 13px; margin-top: 18px; }}
        .status {{
            align-items: center;
            border-radius: 999px;
            display: inline-flex;
            flex-shrink: 0;
            font-size: 12px;
            font-weight: 600;
            justify-content: center;
            line-height: 1;
            max-width: 100%;
            min-height: 28px;
            overflow-wrap: normal;
            padding: 6px 10px;
            white-space: nowrap;
            word-break: normal;
        }}
        .image-status {{
            align-items: center;
            border-radius: 5px;
            box-sizing: border-box;
            display: inline-flex;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.2;
            margin-top: 4px;
            max-width: 100%;
            overflow-wrap: normal;
            padding: 5px 9px;
            text-align: left;
            width: fit-content;
            word-break: normal;
            white-space: nowrap;
        }}
        .probability {{ font-size: 13px; font-weight: 600; }}
        .threshold {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
        .overall-result {{
            border: 1px solid;
            border-radius: 10px;
            box-sizing: border-box;
            margin-top: 0;
            padding: 20px;
        }}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .result-panels-marker) {{ gap: 18px; }}
        .screening-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
            grid-template-rows: auto auto;
            column-gap: 24px;
            row-gap: 18px;
            align-items: start;
        }}
        .screening-image {{
            grid-column: 1;
            grid-row: 1;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            box-sizing: border-box;
            padding: 22px;
            min-width: 0;
        }}
        .screening-image img {{
            display: block;
            aspect-ratio: 1;
            object-fit: contain;
            width: 100%;
            max-width: 600px;
            max-height: 600px;
            margin-inline: auto;
        }}
        .screening-results-background {{
            grid-column: 2;
            grid-row: 1 / 3;
            align-self: stretch;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
        }}
        .screening-results-top {{
            grid-column: 2;
            grid-row: 1;
            padding: 22px 23px 0;
            min-width: 0;
        }}
        .screening-grid > .overall-result {{ grid-column: 1; grid-row: 2; min-width: 0; }}
        .screening-results-last {{
            grid-column: 2;
            grid-row: 2;
            padding: 0 23px 22px;
            min-width: 0;
        }}
        @media (max-width: 768px) {{
            .screening-grid {{ grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto auto auto; }}
            .screening-grid > .overall-result {{ grid-column: 1; grid-row: 2; }}
            .screening-results-background {{ grid-column: 1; grid-row: 3 / 5; }}
            .screening-results-top {{ grid-column: 1; grid-row: 3; }}
            .screening-results-last {{ grid-column: 1; grid-row: 4; }}
        }}
        @media (max-width: 480px) {{
            .screening-image {{ padding: 16px; }}
            .screening-results-top {{ padding: 16px 17px 0; }}
            .screening-results-last {{ padding: 0 17px 16px; }}
        }}
        .overall-high {{ background: {COLORS['red_soft']}; border-color: #e9bbb5; color: {COLORS['red']}; }}
        .overall-attention {{ background: {COLORS['amber_soft']}; border-color: #e9ca98; color: {COLORS['amber']}; }}
        .overall-low {{ background: {COLORS['green_soft']}; border-color: #b8d8c2; color: {COLORS['green']}; }}
        .overall-eyebrow {{ font-size: 12px; font-weight: 700; margin-bottom: 12px; }}
        .overall-heading {{ font-size: 17px; font-weight: 700; line-height: 1.35; }}
        .overall-copy {{ color: var(--ink); font-size: 13px; line-height: 1.55; margin-top: 12px; white-space: pre-line; }}
        .footer {{
            background: #e7ecea;
            box-sizing: border-box;
            box-shadow: 0 0 0 100vmax #e7ecea;
            clip-path: inset(0 -100vmax);
            color: #43534f;
            font-size: 11px;
            margin: 24px 0 -16px;
            padding-block: 9px;
            padding-inline: 0;
            text-align: center;
            width: 100%;
        }}
        .stApp:has(.home-page) {{ background: #eff3f5; color: #2d3e50; }}
        .stApp:has(.home-page) [data-testid="stMainBlockContainer"] {{
            max-width: none;
            min-height: 100vh;
            padding: 0 30px 16px;
        }}
        .stApp:has(.home-page) [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
            min-height: calc(100vh - 40px);
            gap: 0;
        }}
        .home-page.app-header {{
            align-items: center;
            background: #164650;
            box-shadow: 0 0 0 100vmax #164650;
            display: flex;
            justify-content: center;
            min-height: 63px;
            margin: 0;
            padding: 14px 48px;
            position: relative;
        }}
        .home-page.app-header h1 {{
            color: #f5f8fa;
            font-size: 21px;
            font-weight: 700;
            line-height: 1.4;
            letter-spacing: 0;
            padding: 0;
            text-align: center;
        }}
        .header-eye {{
            position: absolute;
            left: 2px;
            font-family: "Material Symbols Rounded";
            font-size: 28px;
            font-weight: normal;
            line-height: 1;
        }}
        .stApp:has(.home-page) [data-testid="stToolbar"] {{ color: #f5f8fa; }}
        .stApp:has(.home-page) [data-testid="stAppDeployButton"] {{ display: none; }}
        .stApp:has(.home-page) [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) {{
            display: block;
            margin: 0 auto;
            max-width: 704px;
            padding-top: 80px;
            width: 100%;
        }}
        .stApp:has(.home-page) [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) > [data-testid="stColumn"] {{
            width: 100%;
        }}
        .stApp:has(.home-page) [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) > [data-testid="stColumn"]:last-child {{ display: none; }}
        .stApp:has(.home-page) [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .major-panel-marker) {{
            background: transparent;
            border: 0;
            border-radius: 0;
            gap: 0;
            padding: 0;
        }}
        .stApp:has(.home-page) .home-intro {{
            color: #405467;
            font-size: 16px;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            text-align: center;
        }}
        .stApp:has(.home-page) [data-testid="stElementContainer"]:has([data-testid="stFileUploader"]) {{ margin-top: 36px; }}
        .stApp:has(.home-page) [data-testid="stElementContainer"]:has(.section-label) {{ display: none; }}
        .stApp:has(.home-page) [data-testid="stFileUploader"] {{
            background: #f7f9fa;
            border: 1px dashed #cbd6df;
            border-radius: 5px;
            box-sizing: border-box;
            height: 216px;
        }}
        .stApp:has(.home-page) [data-testid="stFileUploader"]::before {{
            color: #405467;
            content: "cloud_upload";
            font-family: "Material Symbols Rounded";
            font-size: 48px;
            line-height: 1;
            position: absolute;
            top: 24px;
            left: calc(50% - 24px);
        }}
        .stApp:has(.home-page) [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {{
            background: #1c7377;
            border-radius: 5px;
            font-size: 16px;
            font-weight: 600;
            height: 44px;
            left: calc(50% - 96px);
            position: absolute;
            top: 88px;
            width: 192px;
        }}
        .stApp:has(.home-page) [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {{ font-size: 16px; }}
        .stApp:has(.home-page) [data-testid="stFileUploader"]::after {{
            color: #718093;
            content: "JPG, PNG · до 200 МБ на файл\\a Перетягніть файли сюди або натисніть кнопку для вибору";
            font-size: 12px;
            line-height: 26px;
            left: 12px;
            right: 12px;
            position: absolute;
            top: 142px;
            text-align: center;
            white-space: pre-line;
        }}
        .stApp:has(.home-page) [data-testid="stFileUploaderDropzone"],
        .stApp:has(.home-page) [data-testid="stFileUploaderDropzone"] button {{ height: 100%; }}
        .stApp:has(.home-page) [data-testid="stElementContainer"]:has(.footer) {{ margin-top: auto; }}
        .stApp:has(.home-page) .footer {{
            background: transparent;
            box-shadow: none;
            clip-path: none;
            color: #718093;
            font-family: "Segoe UI", sans-serif;
            font-size: 10px;
            line-height: 16px;
            margin: 32px 0 0;
            padding: 0;
        }}
        @media (max-width: 768px) {{
            .stApp:has(.home-page) [data-testid="stMainBlockContainer"] {{ padding-inline: 18px; }}
            .home-page.app-header {{ padding: 14px 32px; }}
            .home-page.app-header h1 {{ font-size: 18px; }}
            .stApp:has(.home-page) [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) {{ padding-top: 40px; }}
            .stApp:has(.home-page) .home-intro {{ font-size: 14px; }}
            .stApp:has(.home-page) [data-testid="stFileUploader"] {{ height: 240px; }}
        }}
        @media (max-width: 1180px) {{
            [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) {{
                display: grid;
                grid-template-columns: minmax(0, 1fr);
            }}
            [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) > [data-testid="stColumn"] {{
                width: 100%;
            }}
        }}
        @media (max-width: 768px) {{
            [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) {{
                display: grid;
                grid-template-columns: minmax(0, 1fr);
            }}
            [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) > [data-testid="stColumn"] {{
                width: 100%;
            }}
            [data-testid="stMainBlockContainer"] {{ padding-inline: 18px; }}
            .empty-state {{ padding: 24px 16px; }}
            .empty-state h2 {{ font-size: 21px; }}
        }}
        @media (max-width: 680px) {{
            .pathology-header {{ align-items: flex-start; flex-direction: column; gap: 8px; }}
        }}
        @media (max-width: 480px) {{
            [data-testid="stMainBlockContainer"] {{ padding-inline: 14px; }}
            .app-header {{
                margin-bottom: 14px;
                padding-block: 14px;
                padding-inline: 0;
            }}
            .app-header h1 {{ font-size: 20px; }}
            [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .major-panel-marker) {{ padding: 16px; }}
            .pathology-card {{ padding: 14px; }}
            .overall-result {{ padding: 16px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: ScreeningResult) -> None:
    image_buffer = io.BytesIO()
    make_circular_preview(result.image).save(image_buffer, format="PNG")
    image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    overall_heading = {
        "high": "Висока ймовірність патологічних змін",
        "attention": "Виявлені ознаки потребують уваги",
        "low": "Низька ймовірність патологічних змін",
    }[result.highest_level]
    overall_copy = html.escape(screening_summary(result))
    pathology_cards: list[str] = []
    for code, probability, threshold in zip(
        result.label_codes, result.probabilities, result.thresholds
    ):
        level = probability_level(probability, threshold)
        status_text, color, background = LEVEL_STYLES[level]
        safe_name = html.escape(DISPLAY_CLASS_NAMES.get(code, code))
        bounded_probability = min(1.0, max(0.0, probability))
        pathology_cards.append(
                    '<article class="pathology-card">'
                    '<div class="pathology-header">'
                    f'<div class="pathology-title">{safe_name}</div>'
                    f'<span class="status" style="color:{color};background:{background}">{status_text}</span>'
                    '</div>'
                    '<div class="pathology-meta">'
                    f'<div class="probability" style="color:{color}">Ймовірність: {probability:.1%}</div>'
                    f'<div class="threshold">Порогове значення: {threshold:.1%}</div>'
                    '<div class="probability-track" role="progressbar" '
                    f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{bounded_probability * 100:.1f}">'
                    f'<div class="probability-fill" style="width:{bounded_probability * 100:.4f}%;background:{color}"></div>'
                    '</div></div></article>'
        )
    st.markdown(
        '<div class="screening-grid">'
        '<section class="screening-image">'
        '<p class="section-label">ЗОБРАЖЕННЯ ОЧНОГО ДНА</p>'
        f'<img src="data:image/png;base64,{image_data}" alt="{html.escape(result.filename, quote=True)}">'
        '</section>'
        '<div class="screening-results-background" aria-hidden="true"></div>'
        '<section class="screening-results-top">'
        '<p class="section-label">РЕЗУЛЬТАТИ МОДЕЛІ</p>'
        f'<div class="pathology-list">{"".join(pathology_cards[:-1])}</div></section>'
        f'<section class="overall-result overall-{result.highest_level}">'
        '<div class="overall-eyebrow">ЗАГАЛЬНИЙ РЕЗУЛЬТАТ СКРИНІНГУ</div>'
        f'<div class="overall-heading">{overall_heading}</div>'
        f'<div class="overall-copy">{overall_copy}</div></section>'
        f'<div class="screening-results-last">{pathology_cards[-1]}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

 
def render_selected_analysis(analysis: object | None) -> None:
    if is_screening_result(analysis):
        render_result(analysis)
    elif isinstance(analysis, FundusValidationResult):
        if analysis.status == "non_fundus":
            st.error(analysis.message)
        else:
            st.warning(analysis.message)
    elif is_analysis_error(analysis):
        st.error(analysis.message)


def run_app() -> None:
    initialize_session_state()
    render_styles()
    home_page = not st.session_state["uploaded_images"]
    header_class = "app-header home-page" if home_page else "app-header"
    header_icon = '<span class="header-eye" aria-hidden="true">visibility</span>' if home_page else ""
    st.markdown(
        f'<header class="{header_class}">{header_icon}'
        '<h1>Система автоматизованого скринінгу зображень очного дна</h1>'
        '</header>',
        unsafe_allow_html=True,
    )

    try:
        model, label_codes, class_names, thresholds, device = get_model()
        validator = get_fundus_validator(device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        st.error(f"Не вдалося завантажити модель скринінгу.\n\n{error}")
        st.stop()

    sidebar_column, result_column = st.columns([23, 77], gap="medium")
    with sidebar_column:
        st.markdown('<span class="dashboard-sidebar-marker"></span>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<span class="major-panel-marker"></span>', unsafe_allow_html=True)
            if home_page:
                st.markdown(
                    '<p class="home-intro">Завантажте зображення очного дна для автоматизованого скринінгу<br>'
                    'та виявлення ознак можливих патологічних змін.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown('<p class="section-label">ЗОБРАЖЕННЯ ДЛЯ АНАЛІЗУ</p>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Додати файли" if home_page else "+ Додати зображення",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                label_visibility="visible",
                key=f"image_uploader_{st.session_state['uploader_generation']}",
            )
            _new_image_ids, duplicate_count, overflow_count = collect_new_uploads(
                uploaded_files
            )
            if home_page and _new_image_ids:
                st.rerun()
            if duplicate_count:
                st.warning("Повторно завантажені файли не додано до аналізу.")
            if overflow_count:
                st.warning(f"Буде проаналізовано лише перші {MAX_UPLOADS} знімків.")

            pending_image_ids = [
                image_id
                for image_id in st.session_state["uploaded_images"]
                if image_id not in st.session_state["analysis_results"]
            ]
            if pending_image_ids:
                with st.spinner("Виконується аналіз нових зображень..."):
                    for image_id in pending_image_ids:
                        uploaded_image = st.session_state["uploaded_images"][image_id]
                        try:
                            with Image.open(io.BytesIO(uploaded_image.content)) as source_image:
                                source_image.load()
                                analysis = analyze_image(
                                    source_image,
                                    uploaded_image.filename,
                                    model,
                                    label_codes,
                                    class_names,
                                    thresholds,
                                    device,
                                    validator,
                                )
                            st.session_state["analysis_results"][image_id] = analysis
                        except UnidentifiedImageError:
                            st.session_state["analysis_results"][image_id] = ImageAnalysisError(
                                "Файл пошкоджений або не є підтримуваним зображенням."
                            )
                        except (OSError, ValueError, RuntimeError) as error:
                            st.session_state["analysis_results"][image_id] = ImageAnalysisError(
                                f"Не вдалося проаналізувати знімок. {error}"
                            )

            choose_available_image()
            stored_images: dict[str, UploadedImage] = st.session_state["uploaded_images"]
            for image_id, uploaded_image in stored_images.items():
                selected = image_id == st.session_state.get("selected_image")
                analysis = st.session_state["analysis_results"].get(image_id)
                with st.container(border=True):
                    st.markdown('<span class="image-card-marker"></span>', unsafe_allow_html=True)
                    st.button(
                        f"Вибрати зображення {uploaded_image.filename}",
                        key=f"select_{image_id}",
                        type="primary" if selected else "secondary",
                        width="stretch",
                        help=uploaded_image.filename,
                        on_click=select_image,
                        args=(image_id,),
                    )
                    st.button(
                        "Видалити зображення",
                        key=f"delete_{image_id}",
                        icon=":material/delete:",
                        help=f"Видалити зображення: {uploaded_image.filename}",
                        on_click=delete_image,
                        args=(image_id,),
                    )
                    status_text, status_color, status_background = analysis_card_status(analysis)
                    safe_filename = html.escape(uploaded_image.filename)
                    st.markdown(
                        f'<div class="image-card-content" title="{safe_filename}">'
                        '<div class="image-file-meta">'
                        '<span class="image-file-icon" aria-hidden="true">image</span>'
                        '<div class="image-file-copy">'
                        f'<div class="image-file-name">{safe_filename}</div>'
                        f'<div class="image-file-size">{format_file_size(uploaded_image.size)}</div>'
                        f'<span class="image-status" style="color:{status_color};background:{status_background}">'
                        f'{html.escape(status_text)}</span></div></div></div>',
                        unsafe_allow_html=True,
                    )

    with result_column:
        stored_images = st.session_state["uploaded_images"]
        selected_image = st.session_state.get("selected_image")
        if not stored_images:
            st.markdown(
                """
                <section class="empty-state">
                    <h2>Готово до аналізу зображень очного дна</h2>
                    <p>Система призначена для автоматизованого скринінгу зображень очного дна та виявлення ознак можливих патологічних змін.</p>
                    <strong>Завантажте до 10 зображень у форматі JPG або PNG.</strong>
                </section>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_selected_analysis(
                st.session_state["analysis_results"].get(selected_image)
            )

    st.markdown(
        """
        <footer class="footer">Система призначена для автоматизованого скринінгу та дослідницького аналізу зображень очного дна.<br>Результат моделі не є медичним діагнозом та не замінює консультацію лікаря-офтальмолога.</footer>
        """,
        unsafe_allow_html=True,
    )


run_app()
