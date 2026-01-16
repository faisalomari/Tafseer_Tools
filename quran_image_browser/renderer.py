# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import json
import math
import secrets
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


# =========================
# Defaults (same as your script)
# =========================

BASE_URL = "https://equran.me/read-{surah}-{start}-{end}.html"

SMALL_LOW_MEEM = "\u06ED"  # ۭ

FATHA     = "\u064E"  # َ
DAMMA     = "\u064F"  # ُ
KASRA     = "\u0650"  # ِ

FATHATAN  = "\u064B"  # ً
DAMMATAN  = "\u064C"  # ٌ
KASRATAN  = "\u064D"  # ٍ

HARAKAH_TO_TANWEEN = {
    FATHA: FATHATAN,
    DAMMA: DAMMATAN,
    KASRA: KASRATAN,
}
TANWEEN_SET = {FATHATAN, DAMMATAN, KASRATAN}

AYAH_MARK_RE = re.compile(r"(﴿\s*[0-9٠-٩]+\s*﴾)$")


@dataclass
class RenderConfig:
    # main inputs
    surah_number: int = 10
    start_verse: int = 89
    num_verses: int = 1

    # canvas
    width: int = 1920
    fixed_height: int = 200
    bg_color: str = "#ffe8cb"

    # colors
    ayah_read_color: str = "#000000"
    ayah_color: str = "#058e22"

    # logo
    logo_path: str = "logos/Yunus.png"
    logo_width: int = 180
    logo_margin: int = 0

    # typography
    font_size: int = 50
    line_height_px: int = 75
    padding_ratio: float = 0.06  # PADDING_X = int(width * 0.06)

    # fonts
    uthman_hafs_font_path: str = "Uthmanic.otf"

    # output misc
    combined_txt_out: str = "text1.txt"


# -----------------------------
# Text fix
# -----------------------------
def replace_low_meem_with_tanween_from_prev_harakah(t: str) -> str:
    out = []
    for ch in t:
        if ch != SMALL_LOW_MEEM:
            out.append(ch)
            continue

        if out and out[-1] in HARAKAH_TO_TANWEEN:
            prev = out.pop()
            out.append(HARAKAH_TO_TANWEEN[prev])
        elif out and out[-1] in TANWEEN_SET:
            pass
        else:
            pass

    return "".join(out)


def normalize_text(t: str) -> str:
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"﴿\s*", "﴿", t)
    t = re.sub(r"\s*﴾", "﴾", t)
    t = replace_low_meem_with_tanween_from_prev_harakah(t)
    return t


# -----------------------------
# Font loading (same behavior)
# -----------------------------
def get_uthman_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    p = Path(font_path)
    if not p.exists():
        raise FileNotFoundError(f"Uthman Hafs font not found:\n{font_path}")
    return ImageFont.truetype(str(p), size)


def get_windows_font(size: int, preferred: List[str]) -> ImageFont.FreeTypeFont:
    font_dir = Path(r"C:\Windows\Fonts")
    for name in preferred:
        fp = font_dir / name
        if fp.exists():
            return ImageFont.truetype(str(fp), size)
    return ImageFont.load_default()


def get_arial_bold(size: int) -> ImageFont.FreeTypeFont:
    return get_windows_font(size, ["arialbd.ttf", "Arialbd.ttf"])


def get_arial_black_or_bold(size: int) -> ImageFont.FreeTypeFont:
    font_dir = Path(r"C:\Windows\Fonts")
    arial_black = font_dir / "arialblk.ttf"
    if arial_black.exists():
        return ImageFont.truetype(str(arial_black), size)
    return get_arial_bold(size)


# -----------------------------
# Fetch verses
# -----------------------------
def build_url(surah: int, start: int, num: int) -> Tuple[str, int]:
    end = start + num - 1
    return BASE_URL.format(surah=surah, start=start, end=end), end


def fetch_verses_equran(surah: int, start: int, end: int) -> List[str]:
    url = BASE_URL.format(surah=surah, start=start, end=end)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "ar,en;q=0.8",
        "Referer": "https://equran.me/",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    tags = soup.find_all("a", class_="ayahNumber")
    if not tags:
        raise RuntimeError(f"No verses found (a.ayahNumber). URL: {url}")

    out: List[Tuple[int, str]] = []
    for a in tags:
        vnum = None
        if a.has_attr("data-ayahnumber"):
            try:
                vnum = int(str(a["data-ayahnumber"]).strip())
            except Exception:
                vnum = None

        raw = a.get_text(" ", strip=True)
        cleaned = normalize_text(raw)
        if not cleaned:
            continue

        if vnum is None:
            out.append((-1, cleaned))
        else:
            if start <= vnum <= end:
                out.append((vnum, cleaned))

    out_sorted = sorted(out, key=lambda x: (x[0] == -1, x[0]))
    verses = [t for _, t in out_sorted]

    expected = end - start + 1
    if len(verses) > expected:
        verses = verses[:expected]
    if len(verses) < expected:
        raise RuntimeError(f"Expected {expected} verses, got {len(verses)}. URL: {url}")

    return verses


# -----------------------------
# Layout helpers (same logic)
# -----------------------------
def textbbox_w(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, s: str, direction=None) -> int:
    b = draw.textbbox((0, 0), s, font=font, direction=direction)
    return b[2] - b[0]


def tokenize_for_layout(text: str) -> List[Tuple[str, str]]:
    pattern = r"(﴿\s*[0-9٠-٩]+\s*﴾)"
    parts = re.split(pattern, text)
    tokens = []

    for part in parts:
        if not part.strip():
            continue
        if re.match(pattern, part):
            tokens.append(("marker", part))
        else:
            for word in part.split():
                tokens.append(("word", word))
    return tokens


def token_width(draw, uth_font, span_font, b_font, token_type: str, token: str) -> int:
    if token_type == "word":
        b = draw.textbbox((0, 0), token, font=uth_font, direction='rtl')
        return b[2] - b[0]

    m = re.match(r"^﴿\s*([0-9٠-٩]+)\s*﴾$", normalize_text(token))
    if not m:
        return textbbox_w(draw, span_font, token)

    digits = m.group(1)
    return (
        textbbox_w(draw, b_font, "﴿")
        + textbbox_w(draw, span_font, digits)
        + textbbox_w(draw, b_font, "﴾")
    )


def build_lines_for_paragraph(draw, uth_font, span_font, b_font, paragraph: str, max_w: int):
    tokens = tokenize_for_layout(paragraph)
    if not tokens:
        return [[]]

    base_space = textbbox_w(draw, uth_font, " ")
    lines = []
    current = []
    current_w = 0

    for ttype, tok in tokens:
        tw = token_width(draw, uth_font, span_font, b_font, ttype, tok)
        add_space = base_space if current else 0

        if current and (current_w + add_space + tw) > max_w:
            lines.append(current)
            current = [(ttype, tok)]
            current_w = tw
        else:
            current_w = current_w + add_space + tw if current else tw
            current.append((ttype, tok))

    if current:
        lines.append(current)
    return lines


def wrap_text_with_logo_avoidance(draw, uth_font, span_font, b_font, text: str, max_w: int, reduced_w: int, logo_h: int, line_height_px: int, logo_margin: int):
    risk_lines = int(math.ceil((logo_h + logo_margin) / line_height_px)) + 1
    risk_lines = max(1, risk_lines)

    def wrap_all(start_idx: int):
        all_lines = []
        para_last = []
        line_idx = 0

        for para in text.splitlines():
            if not para.strip():
                all_lines.append([])
                para_last.append(len(all_lines) - 1)
                line_idx += 1
                continue

            tokens = tokenize_for_layout(para)
            cur_max_w = reduced_w if line_idx >= start_idx else max_w

            base_space = textbbox_w(draw, uth_font, " ")
            current = []
            current_w = 0

            for ttype, tok in tokens:
                tw = token_width(draw, uth_font, span_font, b_font, ttype, tok)
                add_space = base_space if current else 0

                if current and (current_w + add_space + tw) > cur_max_w:
                    all_lines.append(current)
                    line_idx += 1
                    cur_max_w = reduced_w if line_idx >= start_idx else max_w
                    current = [(ttype, tok)]
                    current_w = tw
                else:
                    current_w = (current_w + add_space + tw) if current else tw
                    current.append((ttype, tok))

            if current:
                all_lines.append(current)
                line_idx += 1

            para_last.append(len(all_lines) - 1)

        return all_lines, para_last

    lines0, para_last0 = wrap_all(start_idx=10**9)
    total0 = len(lines0)
    start_idx = max(0, total0 - risk_lines)

    all_lines, para_last = lines0, para_last0
    for _ in range(5):
        new_lines, new_para_last = wrap_all(start_idx=start_idx)
        total = len(new_lines)
        new_start = max(0, total - risk_lines)

        all_lines, para_last = new_lines, new_para_last
        if new_start == start_idx:
            break
        start_idx = new_start

    return all_lines, para_last, start_idx


def justify_line_spacing(draw, uth_font, span_font, b_font, line_tokens, max_w, is_last_line):
    if len(line_tokens) <= 1:
        return []

    base_space = textbbox_w(draw, uth_font, " ")
    widths = [token_width(draw, uth_font, span_font, b_font, t, s) for t, s in line_tokens]
    current_w = sum(widths) + base_space * (len(line_tokens) - 1)

    gaps = [base_space] * (len(line_tokens) - 1)

    if is_last_line or current_w >= max_w:
        return gaps

    extra = max_w - current_w
    eligible = [i for i in range(len(gaps)) if line_tokens[i][0] == "word" and line_tokens[i + 1][0] == "word"]
    if not eligible:
        return gaps

    q, r = divmod(extra, len(eligible))
    for idx in eligible:
        gaps[idx] += q
    for j in range(r):
        gaps[eligible[j]] += 1

    return gaps


def render_marker(draw, x_right: int, y: int, span_font, b_font, marker: str, ayah_color: str) -> int:
    marker = normalize_text(marker)
    m = re.match(r"^﴿\s*([0-9٠-٩]+)\s*﴾$", marker)

    marker_base_y = y + 12

    if not m:
        w = textbbox_w(draw, span_font, marker)
        draw.text((x_right - w, marker_base_y), marker, font=span_font, fill=ayah_color)
        return w

    digits = m.group(1)

    number_y_tweak = 4
    number_draw_y = marker_base_y + number_y_tweak

    char_right = "﴿"
    char_left = "﴾"

    w_r = textbbox_w(draw, b_font, char_right)
    w_d = textbbox_w(draw, span_font, digits)
    w_l = textbbox_w(draw, b_font, char_left)

    x = x_right
    draw.text((x - w_r, marker_base_y), char_right, font=b_font, fill=ayah_color)
    x -= w_r

    draw.text((x - w_d, number_draw_y), digits, font=span_font, fill=ayah_color)
    x -= w_d

    draw.text((x - w_l, marker_base_y), char_left, font=b_font, fill=ayah_color)

    return w_r + w_d + w_l


def draw_line_rtl(draw, uth_font, span_font, b_font, x_right, y, line_tokens, gaps, ayah_read_color: str, ayah_color: str, font_size: int):
    x = x_right
    for i, (ttype, tok) in enumerate(line_tokens):
        if ttype == "word":
            b = draw.textbbox((0, 0), tok, font=uth_font, direction='rtl')
            w = b[2] - b[0]
            draw.text((x - w, y), tok, font=uth_font, fill=ayah_read_color, direction='rtl')
            x -= w
        else:
            w = render_marker(draw, x, y, span_font, b_font, tok, ayah_color)
            x -= w

        if i < len(gaps):
            x -= gaps[i]


# -----------------------------
# Main render (kept same layout decisions; only font_size is now fixed by user)
# -----------------------------
def render_text_to_image_bytes(text: str, cfg: RenderConfig) -> bytes:
    width = cfg.width
    height = cfg.fixed_height
    padding_x = int(width * cfg.padding_ratio)
    max_w = width - 2 * padding_x

    # fonts
    chosen_main_size = int(cfg.font_size)  # USER CONTROL (requested change)
    uth_font = get_uthman_font(cfg.uthman_hafs_font_path, chosen_main_size)

    test_char = "ۭ"
    bbox = uth_font.getbbox(test_char)
    if bbox is None or (bbox[2] - bbox[0]) <= 0:
        # keep behavior non-fatal
        pass

    span_font = get_arial_bold(chosen_main_size - 5)
    b_font = get_arial_black_or_bold(chosen_main_size - 3)

    # logo
    logo = None
    if cfg.logo_path and Path(cfg.logo_path).exists():
        logo = Image.open(cfg.logo_path).convert("RGBA")
        aspect = logo.height / logo.width
        logo = logo.resize((cfg.logo_width, int(cfg.logo_width * aspect)), Image.Resampling.LANCZOS)

    # dummy draw for measurements
    dummy = Image.new("RGB", (width, 10), cfg.bg_color)
    draw_dummy = ImageDraw.Draw(dummy)

    # wrap
    if logo:
        exclusion_width = cfg.logo_width + cfg.logo_margin
        reduced_w = max(200, max_w - exclusion_width)
        all_lines, para_last_line_idx, _logo_start_idx = wrap_text_with_logo_avoidance(
            draw_dummy, uth_font, span_font, b_font,
            text, max_w, reduced_w, logo.height,
            cfg.line_height_px, cfg.logo_margin
        )
    else:
        all_lines = []
        para_last_line_idx = []
        for para in text.splitlines():
            if not para.strip():
                all_lines.append([])
                para_last_line_idx.append(len(all_lines) - 1)
                continue
            wrapped = build_lines_for_paragraph(draw_dummy, uth_font, span_font, b_font, para, max_w)
            all_lines.extend(wrapped)
            para_last_line_idx.append(len(all_lines) - 1)

    non_empty_lines = [ln for ln in all_lines if ln]
    is_one_line = (len(non_empty_lines) == 1)

    # canvas (fixed height)
    img = Image.new("RGB", (width, height), cfg.bg_color)
    draw = ImageDraw.Draw(img)

    margin_top = 10
    # keep your original condition behavior (center only if main size == 50 and one line)
    if chosen_main_size == 50 and is_one_line:
        margin_top = max(0, (height - cfg.line_height_px) // 2)

    y = margin_top
    logo_top_y = height - (logo.height if logo else 0) - cfg.logo_margin

    for idx, line_tokens in enumerate(all_lines):
        if not line_tokens:
            y += cfg.line_height_px
            continue

        current_max_w = max_w
        current_x_right = width - padding_x

        if logo and (y + cfg.line_height_px) > logo_top_y:
            exclusion_width = cfg.logo_width + cfg.logo_margin
            current_max_w = max_w - exclusion_width

        is_last = idx in para_last_line_idx
        gaps = justify_line_spacing(draw, uth_font, span_font, b_font, line_tokens, current_max_w, is_last)
        draw_line_rtl(
            draw, uth_font, span_font, b_font,
            current_x_right, y,
            line_tokens, gaps,
            cfg.ayah_read_color, cfg.ayah_color,
            chosen_main_size
        )
        y += cfg.line_height_px

    if logo:
        logo_x = cfg.logo_margin
        logo_y = height - logo.height - cfg.logo_margin
        img.paste(logo, (logo_x, logo_y), logo)

    from io import BytesIO
    bio = BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def fetch_combined_text(cfg: RenderConfig) -> Tuple[str, str]:
    if not (1 <= cfg.surah_number <= 114):
        raise ValueError("SURAH_NUMBER must be in [1, 114].")
    if cfg.start_verse < 1 or cfg.num_verses < 1:
        raise ValueError("START_VERSE and NUM_VERSES must be >= 1.")

    url, end_verse = build_url(cfg.surah_number, cfg.start_verse, cfg.num_verses)
    verses = fetch_verses_equran(cfg.surah_number, cfg.start_verse, end_verse)
    combined_text = " ".join(verses)
    return combined_text, url


# -----------------------------
# Export with key
# -----------------------------
def ensure_export_dir(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    index_path = export_dir / "index.json"
    if not index_path.exists():
        index_path.write_text(json.dumps({"items": {}}, ensure_ascii=False, indent=2), encoding="utf-8")


def export_png_with_key(png_bytes: bytes, cfg: RenderConfig, export_dir: Path) -> str:
    ensure_export_dir(export_dir)

    key = secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:10]
    out_png = export_dir / f"{key}.png"

    out_png.write_bytes(png_bytes)

    index_path = export_dir / "index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data.setdefault("items", {})
    data["items"][key] = {
        "file": out_png.name,
        "cfg": asdict(cfg),
    }
    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return key


def load_export_by_key(key: str, export_dir: Path) -> Optional[Path]:
    index_path = export_dir / "index.json"
    if not index_path.exists():
        return None
    data = json.loads(index_path.read_text(encoding="utf-8"))
    item = data.get("items", {}).get(key)
    if not item:
        return None
    p = export_dir / item["file"]
    return p if p.exists() else None
