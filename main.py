"""Streamlit web interface for retinal image screening."""

from __future__ import annotations

import html
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
MAX_UPLOADS = 12
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
            max-width: 1680px;
            padding: 0 clamp(14px, 2.2vw, 32px) 1rem;
            width: 100%;
        }}
        .app-header {{
            background: #123f4a;
            box-sizing: border-box;
            box-shadow: 0 0 0 100vmax #123f4a;
            clip-path: inset(0 -100vmax);
            color: white;
            margin: 0 0 20px calc(50% - 50vw);
            padding-block: clamp(16px, 2vw, 20px);
            padding-inline: max(clamp(18px, 2.5vw, 36px), calc((100vw - 1680px) / 2 + 32px));
            width: 100vw;
        }}
        .app-header h1 {{ font: 700 clamp(21px, 2.1vw, 28px)/1.25 "Segoe UI", sans-serif; margin: 0; }}
        .app-header p {{ color: #cde4e4; font: 400 14px/1.5 "Segoe UI", sans-serif; margin: 5px 0 0; }}
        h1, h2, h3, p, label, div {{ font-family: "Segoe UI", sans-serif; }}
        [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) {{
            align-items: flex-start;
            gap: clamp(12px, 1.5vw, 20px);
        }}
        [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) > [data-testid="stColumn"] {{
            min-width: 0;
        }}
        [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) {{
            align-items: flex-start;
            gap: clamp(12px, 1.5vw, 20px);
        }}
        [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) > [data-testid="stColumn"] {{
            min-width: 0;
        }}
        .dashboard-sidebar-marker,
        .result-panels-marker {{ display: none; }}
        .section-label {{
            color: var(--muted);
            font-size: 11px;
            font-weight: 700;
            margin: 0 0 10px;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel);
            border-color: var(--line);
            border-radius: 0;
        }}
        [data-testid="stFileUploaderDropzone"] {{ background: #f8faf9; border-color: var(--line); border-radius: 0; }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--teal);
            color: white;
            border: 0;
            border-radius: 0;
        }}
        [data-testid="stFileUploaderDropzone"] button p,
        [data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size: 0; }}
        [data-testid="stFileUploaderDropzone"] button p::after {{
            content: "Додати файли";
            font-size: 14px;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] span::after {{
            content: "До 200 МБ на файл · JPG, PNG";
            font-size: 12px;
        }}
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
            border-color: var(--line);
            overflow: hidden;
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
        .image-card-content {{
            min-width: 0;
            padding: 2px 34px 2px 0;
            pointer-events: none;
            position: relative;
            z-index: 2;
        }}
        .image-file-meta {{
            align-items: flex-start;
            display: flex;
            gap: 9px;
            min-width: 0;
        }}
        .image-file-icon {{
            color: var(--teal);
            flex: 0 0 auto;
            font-family: "Material Symbols Rounded";
            font-size: 21px;
            font-weight: normal;
            line-height: 1.2;
        }}
        .image-file-copy {{ min-width: 0; }}
        .image-file-name {{
            color: var(--ink);
            font-size: 13px;
            font-weight: 700;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .image-file-size {{
            color: var(--muted);
            font-size: 12px;
            margin-top: 2px;
        }}
        [data-testid="stImage"] {{ display: flex; justify-content: center; width: 100%; }}
        [data-testid="stImage"] img {{
            aspect-ratio: 1;
            height: auto;
            max-width: min(100%, 640px);
            object-fit: contain;
            width: 100%;
        }}
        [data-testid="stHorizontalBlock"]:has(.pathology-row-marker):not(:has(.result-panels-marker)) {{
            align-items: center;
            gap: 8px;
        }}
        [data-testid="stHorizontalBlock"]:has(.pathology-row-marker):not(:has(.result-panels-marker)) > [data-testid="stColumn"] {{
            min-width: 0;
        }}
        .pathology-row-marker {{ display: none; }}
        .probability-track {{
            background: #edf1ef;
            height: 7px;
            margin: 4px 0 12px;
            overflow: hidden;
            width: 100%;
        }}
        .probability-fill {{ height: 100%; min-width: 2px; }}
        .empty-state {{
            background: var(--panel);
            border: 1px solid var(--line);
            padding: 32px 18px;
        }}
        .empty-state h2 {{ color: var(--ink); font-size: 24px; margin: 0 0 5px; }}
        .empty-state p {{ color: var(--muted); font-size: 14px; margin: 0; }}
        .empty-state strong {{ color: var(--teal); display: block; font-size: 13px; margin-top: 18px; }}
        .status {{ display: inline-block; font-size: 12px; font-weight: 700; max-width: 100%; padding: 5px 8px; white-space: normal; }}
        .image-status {{
            display: inline-block;
            font-size: 12px;
            font-weight: 700;
            margin-top: 8px;
            max-width: 100%;
            padding: 5px 8px;
            white-space: normal;
        }}
        .result-name {{ color: var(--ink); font-size: 18px; font-weight: 700; margin: 3px 0 12px; overflow-wrap: anywhere; }}
        .disease-name {{ color: var(--ink); font-size: 13px; font-weight: 700; margin-top: 4px; overflow-wrap: anywhere; }}
        .probability {{ font-size: 13px; font-weight: 700; margin-top: 3px; }}
        .threshold {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
        .interpretation {{ border-top: 1px solid var(--line); margin-top: 12px; padding-top: 14px; }}
        .interpretation-text {{ color: var(--ink); font-size: 13px; margin-top: 8px; white-space: pre-line; }}
        .footer {{
            background: #e7ecea;
            box-sizing: border-box;
            box-shadow: 0 0 0 100vmax #e7ecea;
            clip-path: inset(0 -100vmax);
            color: #43534f;
            font-size: 11px;
            margin: 24px 0 -16px calc(50% - 50vw);
            padding-block: 9px;
            padding-inline: max(clamp(14px, 2.2vw, 32px), calc((100vw - 1680px) / 2 + 32px));
            text-align: center;
            width: 100vw;
        }}
        @media (max-width: 1220px) {{
            [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) {{
                display: grid;
                grid-template-columns: minmax(0, 1fr);
            }}
            [data-testid="stHorizontalBlock"]:has(.dashboard-sidebar-marker) > [data-testid="stColumn"] {{
                width: 100%;
            }}
        }}
        @media (max-width: 700px) {{
            [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) {{
                display: grid;
                grid-template-columns: minmax(0, 1fr);
            }}
            [data-testid="stHorizontalBlock"]:has(.result-panels-marker):not(:has(.dashboard-sidebar-marker)) > [data-testid="stColumn"] {{
                width: 100%;
            }}
            [data-testid="stHorizontalBlock"]:has(.pathology-row-marker):not(:has(.result-panels-marker)) {{
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
            }}
            .empty-state {{ padding: 24px 16px; }}
            .empty-state h2 {{ font-size: 21px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status(level: str) -> None:
    text, color, background = LEVEL_STYLES[level]
    st.markdown(
        f'<span class="status" style="color:{color};background:{background}">{text}</span>',
        unsafe_allow_html=True,
    )


def render_probability_bar(probability: float, color: str) -> None:
    bounded_probability = min(1.0, max(0.0, probability))
    st.markdown(
        '<div class="probability-track" role="progressbar" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{bounded_probability * 100:.1f}">'
        f'<div class="probability-fill" style="width:{bounded_probability * 100:.4f}%;background:{color}"></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_result(result: ScreeningResult) -> None:
    safe_filename = html.escape(result.filename)
    image_column, details_column = st.columns(2, gap="medium")
    with image_column:
        st.markdown('<span class="result-panels-marker"></span>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-label">ЗОБРАЖЕННЯ ОЧНОГО ДНА</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{safe_filename}</div>', unsafe_allow_html=True)
            st.image(make_circular_preview(result.image), width="stretch")
            for note in result.quality_notes:
                st.warning(note)

    with details_column:
        with st.container(border=True):
            st.markdown('<p class="section-label">РЕЗУЛЬТАТИ МОДЕЛІ</p>', unsafe_allow_html=True)
            for code, probability, threshold in zip(
                result.label_codes, result.probabilities, result.thresholds
            ):
                level = probability_level(probability, threshold)
                _, color, _ = LEVEL_STYLES[level]
                name_column, status_column = st.columns([3, 2], vertical_alignment="center")
                with name_column:
                    st.markdown('<span class="pathology-row-marker"></span>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="disease-name">{DISPLAY_CLASS_NAMES.get(code, code)}</div>',
                        unsafe_allow_html=True,
                    )
                with status_column:
                    render_status(level)
                st.markdown(
                    f'<div class="probability" style="color:{color}">Ймовірність: {probability:.1%}</div>'
                    f'<div class="threshold">Порогове значення: {threshold:.1%}</div>',
                    unsafe_allow_html=True,
                )
                render_probability_bar(probability, color)

            st.markdown(
                '<div class="interpretation"><p class="section-label">ІНТЕРПРЕТАЦІЯ СКРИНІНГУ</p></div>',
                unsafe_allow_html=True,
            )
            render_status(result.highest_level)
            st.markdown(
                f'<div class="interpretation-text">{screening_summary(result)}</div>',
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
    st.markdown(
        """
        <header class="app-header">
            <h1>Система автоматизованого скринінгу зображень очного дна</h1>
            <p>Інтелектуальний аналіз зображень для виявлення ознак можливих патологічних змін</p>
        </header>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, label_codes, class_names, thresholds, device = get_model()
        validator = get_fundus_validator(device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        st.error(f"Не вдалося завантажити модель скринінгу.\n\n{error}")
        st.stop()

    sidebar_column, result_column = st.columns([1, 3], gap="medium")
    with sidebar_column:
        st.markdown('<span class="dashboard-sidebar-marker"></span>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-label">ЗОБРАЖЕННЯ ДЛЯ АНАЛІЗУ</p>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Виберіть зображення очного дна для аналізу",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key=f"image_uploader_{st.session_state['uploader_generation']}",
            )
            _new_image_ids, duplicate_count, overflow_count = collect_new_uploads(
                uploaded_files
            )
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
                        '</div></div>'
                        f'<span class="image-status" style="color:{status_color};background:{status_background}">'
                        f'{html.escape(status_text)}</span></div>',
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
                    <strong>Завантажте до 12 зображень у форматі JPG або PNG.</strong>
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
        <footer class="footer">Система призначена для автоматизованого скринінгу та дослідницького аналізу зображень очного дна. Результат моделі не є медичним діагнозом та не замінює консультацію лікаря-офтальмолога.</footer>
        """,
        unsafe_allow_html=True,
    )


run_app()
