"""Native desktop interface for retinal image screening."""

from __future__ import annotations

import math
import pickle
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageOps, ImageStat, ImageTk, UnidentifiedImageError
from torch import nn
from torchvision.models import resnet18

from train_binary_classifier import make_transforms, resolve_device


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


def load_model() -> tuple[nn.Module, tuple[str, ...], tuple[str, ...], tuple[float, ...], torch.device]:
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


def assess_image_quality(image: Image.Image) -> tuple[str, ...]:
    notes: list[str] = []
    if min(image.size) < 224:
        notes.append("Низька роздільна здатність може знизити надійність. Потрібно щонайменше 224 пікселі з кожного боку.")

    brightness = ImageStat.Stat(image.convert("L")).mean[0]
    if brightness < 25:
        notes.append("Знімок надто темний. Перевірте освітлення та видимість сітківки.")
    elif brightness > 230:
        notes.append("Знімок надмірно освітлений. Перевірте освітлення та видимість сітківки.")
    return tuple(notes)


def screen_image(
    path: Path,
    model: nn.Module,
    label_codes: tuple[str, ...],
    class_names: tuple[str, ...],
    thresholds: tuple[float, ...],
    device: torch.device,
) -> ScreeningResult:
    with Image.open(path) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")

    _, evaluation_transform = make_transforms()
    tensor = evaluation_transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].cpu().tolist()

    return ScreeningResult(
        path=path,
        image=image,
        label_codes=label_codes,
        class_names=class_names,
        probabilities=tuple(probabilities),
        thresholds=thresholds,
        quality_notes=assess_image_quality(image),
    )


class RetinaReviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Система автоматизованого скринінгу зображень очного дна")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 760)
        self.root.configure(bg=COLORS["paper"])

        (
            self.model,
            self.label_codes,
            self.class_names,
            self.thresholds,
            self.device,
        ) = load_model()
        self.results: list[ScreeningResult] = []
        self.photo: ImageTk.PhotoImage | None = None
        self.work_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.processed_count = 0
        self.study_widgets: list[tk.Button] = []
        self.selected_index: int | None = None

        self._configure_styles()
        self._build_layout()
        self._show_empty_state()

    # GUI: visual system and reusable presentation helpers.
    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["paper"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure(
            "Primary.TButton",
            background=COLORS["teal"],
            foreground="white",
            borderwidth=0,
            padding=(18, 12),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#105b54"), ("disabled", "#93aaa5")])
        style.configure(
            "Horizontal.TProgressbar",
            background=COLORS["teal"],
            troughcolor=COLORS["line"],
            borderwidth=0,
        )

    def _card(self, parent: tk.Misc, **grid_options: object) -> tk.Frame:
        border = tk.Frame(parent, bg=COLORS["line"], padx=1, pady=1)
        border.grid(**grid_options)
        card = tk.Frame(border, bg=COLORS["panel"], padx=18, pady=16)
        card.pack(fill="both", expand=True)
        return card

    @staticmethod
    def _level_presentation(level: str) -> tuple[str, str, str]:
        styles = {
            "high": ("Висока ймовірність", COLORS["red"], COLORS["red_soft"]),
            "attention": ("Потребує уваги", COLORS["amber"], COLORS["amber_soft"]),
            "low": ("Низька ймовірність", COLORS["green"], COLORS["green_soft"]),
        }
        return styles[level]

    # GUI: header, three-column workspace, and medical disclaimer.
    def _build_layout(self) -> None:
        shell = tk.Frame(self.root, bg=COLORS["paper"])
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg="#123f4a", padx=34, pady=20)
        header.pack(fill="x")
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Система автоматизованого скринінгу зображень очного дна",
            bg="#123f4a",
            fg="white",
            font=("Segoe UI", 20, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Інтелектуальний аналіз зображень для виявлення ознак можливих патологічних змін",
            bg="#123f4a",
            fg="#cde4e4",
            font=("Segoe UI", 10),
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))

        body = tk.Frame(shell, bg=COLORS["paper"], padx=28, pady=20)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        sidebar = tk.Frame(body, bg=COLORS["panel"], width=300, padx=18, pady=18, highlightbackground=COLORS["line"], highlightthickness=1)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        sidebar.grid_propagate(False)
        tk.Label(
            sidebar,
            text="ЗОБРАЖЕННЯ ДЛЯ АНАЛІЗУ",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        self.upload_button = ttk.Button(
            sidebar,
            text="Завантажити зображення",
            style="Primary.TButton",
            command=self._select_images,
        )
        self.upload_button.pack(fill="x", pady=(10, 14))
        self.progress = ttk.Progressbar(sidebar, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 12))
        self.progress.pack_forget()
        tk.Label(
            sidebar,
            text="СПИСОК ЗАВАНТАЖЕНИХ ЗОБРАЖЕНЬ",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        list_holder = tk.Frame(sidebar, bg=COLORS["panel"])
        list_holder.pack(fill="both", expand=True, pady=(8, 0))
        self.study_canvas = tk.Canvas(list_holder, bg=COLORS["panel"], highlightthickness=0, width=260)
        list_scrollbar = ttk.Scrollbar(list_holder, orient="vertical", command=self.study_canvas.yview)
        self.study_list = tk.Frame(self.study_canvas, bg=COLORS["panel"])
        self.study_list.bind("<Configure>", lambda _event: self.study_canvas.configure(scrollregion=self.study_canvas.bbox("all")))
        self.study_canvas.create_window((0, 0), window=self.study_list, anchor="nw", width=260)
        self.study_canvas.configure(yscrollcommand=list_scrollbar.set)
        self.study_canvas.pack(side="left", fill="both", expand=True)
        list_scrollbar.pack(side="right", fill="y")

        self.result_canvas = tk.Canvas(body, bg=COLORS["paper"], highlightthickness=0)
        self.result_canvas.grid(row=0, column=1, sticky="nsew")
        result_scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.result_canvas.yview)
        result_scrollbar.grid(row=0, column=2, sticky="ns")
        self.result_area = tk.Frame(self.result_canvas, bg=COLORS["paper"])
        self.result_window = self.result_canvas.create_window((0, 0), window=self.result_area, anchor="nw")
        self.result_area.bind("<Configure>", lambda _event: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all")))
        self.result_canvas.bind("<Configure>", lambda event: self.result_canvas.itemconfigure(self.result_window, width=event.width))
        self.result_canvas.configure(yscrollcommand=result_scrollbar.set)

        footer = tk.Frame(shell, bg="#e7ecea", padx=28, pady=9)
        footer.pack(fill="x")
        tk.Label(
            footer,
            text=("Система призначена для автоматизованого скринінгу та дослідницького аналізу зображень очного дна. "
                  "Результат моделі не є медичним діагнозом та не замінює консультацію лікаря-офтальмолога."),
            bg="#e7ecea",
            fg="#43534f",
            font=("Segoe UI", 8),
            justify="center",
            wraplength=1180,
        ).pack(fill="x")

    def _clear_result_area(self) -> None:
        for widget in self.result_area.winfo_children():
            widget.destroy()

    # GUI: concise onboarding state.
    def _show_empty_state(self) -> None:
        self._clear_result_area()
        self.result_area.grid_columnconfigure(0, weight=1)
        empty = self._card(self.result_area, row=0, column=0, sticky="nsew")
        tk.Label(
            empty,
            text="Готово до аналізу зображень очного дна",
            bg=COLORS["panel"],
            fg=COLORS["ink"],
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", pady=(16, 5))
        tk.Label(
            empty,
            text=("Система призначена для автоматизованого скринінгу зображень очного дна та виявлення "
                  "ознак можливих патологічних змін."),
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
            wraplength=760,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            empty,
            text="Завантажте до 12 зображень у форматі JPG або PNG.",
            bg=COLORS["panel"],
            fg=COLORS["teal"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(18, 18))

    def _select_images(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Виберіть зображення очного дна для аналізу",
            filetypes=(("Зображення очного дна", "*.jpg *.jpeg *.png"), ("Усі файли", "*.*")),
        )
        if not selected:
            return
        paths = [Path(path) for path in selected[:MAX_UPLOADS]]
        if len(selected) > MAX_UPLOADS:
            messagebox.showinfo("Обмеження", f"Буде проаналізовано лише перші {MAX_UPLOADS} знімків.")

        self.results.clear()
        self.processed_count = 0
        self.selected_index = None
        self.study_widgets.clear()
        for widget in self.study_list.winfo_children():
            widget.destroy()
        self._show_empty_state()
        self.upload_button.configure(state="disabled")
        self.progress.configure(maximum=len(paths), value=0)
        self.progress.pack(fill="x", pady=(0, 12))
        threading.Thread(target=self._screen_paths, args=(paths,), daemon=True).start()
        self.root.after(75, self._poll_work_queue)

    def _screen_paths(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                result = screen_image(
                    path,
                    self.model,
                    self.label_codes,
                    self.class_names,
                    self.thresholds,
                    self.device,
                )
                self.work_queue.put(("result", result))
            except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as error:
                self.work_queue.put(("error", (path.name, str(error))))
        self.work_queue.put(("done", len(paths)))

    def _poll_work_queue(self) -> None:
        finished = False
        while True:
            try:
                action, payload = self.work_queue.get_nowait()
            except queue.Empty:
                break
            if action == "result":
                if isinstance(payload, ScreeningResult):
                    result = payload
                    self.processed_count += 1
                    self.results.append(result)
                    self._add_study_item(result, len(self.results) - 1)
                    self.progress.configure(value=self.processed_count)
            elif action == "error":
                self.processed_count += 1
                self.progress.configure(value=self.processed_count)
                name, reason = payload if isinstance(payload, tuple) else ("Знімок", "Невідома помилка")
                messagebox.showerror("Не вдалося проаналізувати знімок", f"{name}\n\n{reason}")
            elif action == "done":
                finished = True

        if finished:
            self.upload_button.configure(state="normal")
            self.progress.pack_forget()
            if self.results:
                self._select_result(0)
            return
        self.root.after(75, self._poll_work_queue)

    def _add_study_item(self, result: ScreeningResult, index: int) -> None:
        level_text, level_color, _ = self._level_presentation(result.highest_level)
        item = tk.Button(
            self.study_list,
            text=f"{result.path.name}\nПроаналізовано · {level_text}",
            command=lambda selected=index: self._select_result(selected),
            bg="#f8faf9",
            fg=COLORS["ink"],
            activebackground=COLORS["teal_soft"],
            activeforeground=COLORS["ink"],
            relief="flat",
            bd=0,
            highlightbackground=COLORS["line"],
            highlightthickness=1,
            anchor="w",
            justify="left",
            padx=12,
            pady=10,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        item.pack(fill="x", pady=(0, 7))
        item.configure(fg=level_color)
        self.study_widgets.append(item)

    def _select_result(self, index: int) -> None:
        self.selected_index = index
        for item_index, item in enumerate(self.study_widgets):
            result = self.results[item_index]
            _, level_color, _ = self._level_presentation(result.highest_level)
            item.configure(
                bg=COLORS["teal_soft"] if item_index == index else "#f8faf9",
                highlightbackground=COLORS["teal"] if item_index == index else COLORS["line"],
                fg=level_color,
            )
        self._show_result(self.results[index])

    # GUI: selected fundus image, model probabilities, and screening interpretation.
    def _show_result(self, result: ScreeningResult) -> None:
        self._clear_result_area()
        self.result_canvas.yview_moveto(0)
        self.result_area.grid_columnconfigure(0, weight=5)
        self.result_area.grid_columnconfigure(1, weight=5)

        image_card = self._card(self.result_area, row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        tk.Label(image_card, text="ЗОБРАЖЕННЯ ОЧНОГО ДНА", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(image_card, text=result.path.name, bg=COLORS["panel"], fg=COLORS["ink"], font=("Segoe UI", 13, "bold"), wraplength=460, justify="left").pack(anchor="w", pady=(5, 12))
        image_panel = tk.Frame(image_card, bg="#101513", height=410)
        image_panel.pack(fill="both", expand=True)
        image_panel.pack_propagate(False)
        preview = result.image.copy()
        preview.thumbnail((500, 390), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview)
        tk.Label(image_panel, image=self.photo, bg="#101513").place(relx=0.5, rely=0.5, anchor="center")

        details = self._card(self.result_area, row=0, column=1, sticky="nsew", pady=(0, 12))
        tk.Label(details, text="РЕЗУЛЬТАТИ МОДЕЛІ", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 14))

        for code, probability, threshold in zip(
            result.label_codes, result.probabilities, result.thresholds
        ):
            level = probability_level(probability, threshold)
            level_text, level_color, level_background = self._level_presentation(level)
            row = tk.Frame(details, bg=COLORS["panel"])
            row.pack(fill="x", pady=(0, 5))
            display_label = DISPLAY_CLASS_NAMES.get(code, code)
            tk.Label(row, text=display_label, bg=COLORS["panel"], fg=COLORS["ink"], font=("Segoe UI", 9, "bold"), wraplength=280, justify="left").pack(side="left")
            tk.Label(
                row,
                text=level_text,
                bg=level_background,
                fg=level_color,
                padx=7,
                pady=3,
                font=("Segoe UI", 8, "bold"),
            ).pack(side="right")
            tk.Label(details, text=f"Ймовірність: {probability:.1%}", bg=COLORS["panel"], fg=level_color, font=("Segoe UI", 9, "bold")).pack(anchor="e")
            track = tk.Canvas(details, height=7, bg=COLORS["line"], highlightthickness=0)
            track.pack(fill="x", pady=(4, 12))
            track.update_idletasks()
            track.create_rectangle(
                0,
                0,
                max(2, track.winfo_width() * probability),
                7,
                fill=level_color,
                outline="",
            )

        tk.Frame(details, bg=COLORS["line"], height=1).pack(fill="x", pady=(5, 14))
        tk.Label(details, text="ІНТЕРПРЕТАЦІЯ СКРИНІНГУ", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        summary_text = self._screening_summary(result)
        level_text, level_color, level_background = self._level_presentation(result.highest_level)
        tk.Label(details, text=level_text, bg=level_background, fg=level_color, padx=10, pady=6, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(9, 8))
        tk.Label(details, text=summary_text, bg=COLORS["panel"], fg=COLORS["ink"], font=("Segoe UI", 9), wraplength=430, justify="left", anchor="w").pack(fill="x")

    @staticmethod
    def _screening_summary(result: ScreeningResult) -> str:
        if result.highest_level == "high":
            return ("Модель виявила високу ймовірність ознак патологічних змін.\n"
                "Рекомендується подальша оцінка результату офтальмологом.")
        if result.highest_level == "attention":
            return ("Модель виявила ознаки, які потребують додаткової уваги.\n"
                "Рекомендується професійна оцінка результату офтальмологом.")
        return ("Ознак патологічних змін з високою ймовірністю не виявлено.\n"
            "Рекомендується професійне офтальмологічне обстеження.")

def main() -> None:
    root = tk.Tk()
    try:
        RetinaReviewApp(root)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        root.withdraw()
        messagebox.showerror("Огляд сітківки", f"Не вдалося завантажити модель скринінгу.\n\n{error}")
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()