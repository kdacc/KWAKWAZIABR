"""
PDF-звіт по одному кадру SPHEREx (reportlab).

У звіті: знімок кадру з виділеною червоним областю, параметри спостереження
(ціль, RA/Dec, детектор, поле зору, кількість кадрів), номер і дата кадру та нотатки користувача.

Дані параметрів беруться з результату конвеєра (`Stack.describe()`), а не від клієнта,
тому їх у формі змінити не можна. Від клієнта приходять лише номер кадру, нотатки,
область і PNG-знімок.
"""
from __future__ import annotations

import base64
import binascii
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Стандартні шрифти PDF (Helvetica) не мають кирилиці, тому беремо DejaVu з теки fonts/.
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_REGULAR, FONT_BOLD = "ReportSans", "ReportSans-Bold"
FONT_FILES = {FONT_REGULAR: "DejaVuSans.ttf", FONT_BOLD: "DejaVuSans-Bold.ttf"}

MAX_AREA_IMAGE_B64_CHARS = 12_000_000
MAX_AREA_IMAGE_SIDE_PX = 4096
MAX_NOTES_CHARS = 2000

PAGE_MARGIN = 18 * mm
LABEL_COLUMN_WIDTH = 52 * mm
MAX_IMAGE_SIDE = 115 * mm
GREY = colors.HexColor("#666666")
LIGHT_GREY = colors.HexColor("#d0d4dc")
PANEL_GREY = colors.HexColor("#f1f3f7")


@dataclass(frozen=True)
class AreaImage:
    png: bytes
    width: int
    height: int


@dataclass(frozen=True)
class ReportData:
    target: str
    ra_deg: float
    dec_deg: float
    epoch: Optional[float]
    detector: int
    band_um: tuple[float, float]
    fov_deg: float
    frame_count: int
    frame_number: int                      # з 1
    frame_date: str
    wavelength_um: Optional[float]
    grid_px: Optional[tuple[int, int]]
    region: Optional[dict]                 # частки кадру: x, y, w, h в [0, 1]
    notes: str


# ---------------------------------------------------------------------------
# Форматування координат
# ---------------------------------------------------------------------------
def format_ra_hms(ra_deg: float) -> str:
    hundredths = round(ra_deg / 15.0 * 3600 * 100) % (24 * 3600 * 100)
    hours, rest = divmod(hundredths, 3600 * 100)
    minutes, rest = divmod(rest, 60 * 100)
    return f"{hours:02d}h {minutes:02d}m {rest / 100:05.2f}s"


def format_dec_dms(dec_deg: float) -> str:
    tenths = round(abs(dec_deg) * 3600 * 10)
    degrees, rest = divmod(tenths, 3600 * 10)
    minutes, rest = divmod(rest, 60 * 10)
    sign = "−" if dec_deg < 0 else "+"
    return f"{sign}{degrees:02d}° {minutes:02d}′ {rest / 10:04.1f}″"


# ---------------------------------------------------------------------------
# Вхідні дані
# ---------------------------------------------------------------------------
def decode_area_image(encoded: str) -> AreaImage:
    """PNG у base64 (або data URL) -> перевірене зображення. ValueError, якщо воно не підходить."""
    payload = encoded.split(",", 1)[1] if encoded.startswith("data:") else encoded
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Зображення області пошкоджено (base64).") from None
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            image_format, (width, height) = image.format, image.size
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ValueError("Зображення області не є коректним PNG.") from None
    if image_format != "PNG":
        raise ValueError("Зображення області має бути у форматі PNG.")
    if max(width, height) > MAX_AREA_IMAGE_SIDE_PX:
        raise ValueError(f"Зображення області завелике (понад {MAX_AREA_IMAGE_SIDE_PX} пікселів).")
    return AreaImage(raw, width, height)


def _region_in_pixels(region: dict, grid_px: tuple[int, int]) -> tuple[int, int, int, int]:
    grid_w, grid_h = grid_px
    return (round(region["x"] * grid_w), round(region["y"] * grid_h),
            max(1, round(region["w"] * grid_w)), max(1, round(region["h"] * grid_h)))


def collect_report_data(description: dict, frame_number: int, notes: str,
                        region: Optional[dict]) -> ReportData:
    """Збирає дані звіту з `Stack.describe()`; ValueError при неправильному номері кадру."""
    frames = description["frames"]
    if not 1 <= frame_number <= len(frames):
        raise ValueError(f"Номер кадру має бути від 1 до {len(frames)}.")
    meta, frame = description["meta"], frames[frame_number - 1]
    grid = meta.get("grid_px")
    return ReportData(
        target=str(meta.get("target", "—")),
        ra_deg=float(meta["ra"]), dec_deg=float(meta["dec"]),
        epoch=meta.get("epoch"),
        detector=int(meta["detector"]), band_um=tuple(meta["band_um"]),
        fov_deg=float(meta["fov_deg"]),
        frame_count=len(frames), frame_number=frame_number,
        frame_date=frame.get("utc") or f"MJD {frame['time']:.3f}",
        wavelength_um=frame.get("wavelength_um"),
        grid_px=(int(grid[0]), int(grid[1])) if grid else None,
        region=region, notes=(notes or "").strip()[:MAX_NOTES_CHARS],
    )


# ---------------------------------------------------------------------------
# Побудова PDF
# ---------------------------------------------------------------------------
def _register_fonts() -> None:
    if FONT_REGULAR in pdfmetrics.getRegisteredFontNames():
        return
    for font_name, file_name in FONT_FILES.items():
        path = FONT_DIR / file_name
        if not path.is_file():
            raise RuntimeError(f"Немає шрифта {path}: він потрібен для кирилиці у PDF.")
        pdfmetrics.registerFont(TTFont(font_name, str(path)))
    pdfmetrics.registerFontFamily(FONT_REGULAR, normal=FONT_REGULAR, bold=FONT_BOLD,
                                  italic=FONT_REGULAR, boldItalic=FONT_BOLD)


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName=FONT_REGULAR, fontSize=9.5, leading=13)
    return {
        "cell": base,
        "label": ParagraphStyle("label", parent=base, fontName=FONT_BOLD),
        "title": ParagraphStyle("title", parent=base, fontName=FONT_BOLD, fontSize=17, leading=21),
        "muted": ParagraphStyle("muted", parent=base, fontSize=8.5, textColor=GREY),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8.5, textColor=GREY, alignment=1),
    }


def _text(value: str) -> str:
    return escape(value).replace("\n", "<br/>")


def _parameter_rows(data: ReportData) -> list[tuple[str, str]]:
    rows = [
        ("Ціль", data.target),
        ("RA", f"{data.ra_deg:.6f}° ({format_ra_hms(data.ra_deg)})"),
        ("Dec", f"{data.dec_deg:+.6f}° ({format_dec_dms(data.dec_deg)})"),
    ]
    if data.epoch is not None:
        rows.append(("Епоха координат", f"J{data.epoch}"))
    rows += [
        ("Детектор", f"D{data.detector} ({data.band_um[0]}–{data.band_um[1]} мкм)"),
        ("Поле зору (FOV)", f"{data.fov_deg}°"),
        ("Кількість кадрів", str(data.frame_count)),
        ("Номер кадру", f"{data.frame_number} з {data.frame_count}"),
        ("Дата кадру (UTC)", data.frame_date),
    ]
    if data.wavelength_um is not None:
        rows.append(("Довжина хвилі кадру", f"{data.wavelength_um:.4f} мкм"))
    if data.region and data.grid_px:
        x, y, w, h = _region_in_pixels(data.region, data.grid_px)
        rows.append(("Виділена область",
                     f"x={x}, y={y}, ширина={w}, висота={h} пікс. (кадр {data.grid_px[0]}×{data.grid_px[1]}; "
                     "початок координат у верхньому лівому куті)"))
    rows.append(("Нотатки", data.notes or "—"))
    return rows


def _image_block(area: Optional[AreaImage], styles: dict[str, ParagraphStyle], frame_number: int) -> list:
    if area is None:
        return [Paragraph("Знімок із виділеною областю не додано.", styles["muted"])]
    scale = min(MAX_IMAGE_SIDE / area.width, MAX_IMAGE_SIDE / area.height)
    picture = PdfImage(io.BytesIO(area.png), width=area.width * scale, height=area.height * scale)
    framed = Table([[picture]], style=TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    caption = Paragraph(f"Кадр {frame_number}; виділена область обведена червоним контуром.", styles["caption"])
    return [framed, Spacer(1, 3 * mm), caption]


def build_report_pdf(data: ReportData, area: Optional[AreaImage]) -> bytes:
    _register_fonts()
    styles = _styles()
    page_width, _ = A4
    content_width = page_width - 2 * PAGE_MARGIN

    table = Table(
        [[Paragraph(_text(label), styles["label"]), Paragraph(_text(value), styles["cell"])]
         for label, value in _parameter_rows(data)],
        colWidths=[LABEL_COLUMN_WIDTH, content_width - LABEL_COLUMN_WIDTH],
        style=TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
            ("BACKGROUND", (0, 0), (0, -1), PANEL_GREY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]),
    )
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    story = [
        Paragraph("SPHEREx Time-Lapse — звіт", styles["title"]),
        Paragraph(f"Створено: {generated}", styles["muted"]),
        Spacer(1, 6 * mm),
        *_image_block(area, styles, data.frame_number),
        Spacer(1, 6 * mm),
        table,
    ]
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
                      topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN,
                      title=f"SPHEREx — звіт, кадр {data.frame_number}",
                      author="SPHEREx Time-Lapse").build(story)
    return buffer.getvalue()


def make_report_pdf(description: dict, frame_number: int, notes: str = "",
                    region: Optional[dict] = None, area_image: Optional[str] = None) -> bytes:
    """
    description -- результат `Stack.describe()`; frame_number -- з 1;
    region -- {x, y, w, h} у частках кадру; area_image -- PNG у base64 / data URL.
    ValueError -- некоректні дані від клієнта.
    """
    data = collect_report_data(description, frame_number, notes, region)
    area = decode_area_image(area_image) if area_image else None
    return build_report_pdf(data, area)