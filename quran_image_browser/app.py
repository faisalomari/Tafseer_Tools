# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import streamlit as st

from renderer import RenderConfig, fetch_combined_text, render_text_to_image_bytes

APP_TITLE = "Ayah Image"
LOGOS_DIR = Path("logos")

# -----------------------------
# Defaults (basic + advanced)
# -----------------------------
DEFAULTS = {
    # Basic
    "surah": 10,
    "start": 89,
    "num": 1,
    "font_size": 50,
    "logo_path": "",

    # Advanced
    "width": 1920,
    "height": 200,
    "bg": "#ffe8cb",
    "read_color": "#000000",
    "mark_color": "#058e22",
    "line_h": 75,
    "pad_ratio": 0.06,
    "logo_w": 180,
    "logo_m": 0,
    "uthman_font": "Uthmanic.otf",
}


def list_logo_files() -> list[str]:
    if not LOGOS_DIR.exists():
        return []

    # One glob per extension (lowercase), then dedupe
    patterns = ["*.png", "*.webp"]

    unique: dict[str, Path] = {}
    for pat in patterns:
        for p in LOGOS_DIR.glob(pat):
            # normalize to a consistent key (absolute + lower) to dedupe on Windows
            key = str(p.resolve()).lower()
            unique[key] = p

    # return stable, sorted (by filename)
    out = [str(p).replace("\\", "/") for p in sorted(unique.values(), key=lambda x: x.name.lower())]
    return out


def init_state():
    # Pick a reasonable default logo if available
    logos = list_logo_files()
    default_logo = DEFAULTS["logo_path"]
    if not default_logo and logos:
        default_logo = logos[0]

    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "logo_path" not in st.session_state or not st.session_state["logo_path"]:
        st.session_state["logo_path"] = default_logo


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_fetch_text(surah: int, start: int, num: int):
    cfg = RenderConfig(surah_number=surah, start_verse=start, num_verses=num)
    text, url = fetch_combined_text(cfg)
    return text, url


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_render_png(cfg_dict: dict) -> bytes:
    cfg = RenderConfig(**cfg_dict)
    text, _url = cached_fetch_text(cfg.surah_number, cfg.start_verse, cfg.num_verses)
    return render_text_to_image_bytes(text, cfg)


def build_cfg_from_state() -> RenderConfig:
    return RenderConfig(
        # main
        surah_number=int(st.session_state["surah"]),
        start_verse=int(st.session_state["start"]),
        num_verses=int(st.session_state["num"]),

        # canvas
        width=int(st.session_state["width"]),
        fixed_height=int(st.session_state["height"]),
        bg_color=str(st.session_state["bg"]),

        # colors
        ayah_read_color=str(st.session_state["read_color"]),
        ayah_color=str(st.session_state["mark_color"]),

        # logo
        logo_path=str(st.session_state["logo_path"]),
        logo_width=int(st.session_state["logo_w"]),
        logo_margin=int(st.session_state["logo_m"]),

        # typography
        font_size=int(st.session_state["font_size"]),
        line_height_px=int(st.session_state["line_h"]),
        padding_ratio=float(st.session_state["pad_ratio"]),

        # fonts
        uthman_hafs_font_path=str(st.session_state["uthman_font"]),
    )


def reset_basic():
    st.session_state["surah"] = DEFAULTS["surah"]
    st.session_state["start"] = DEFAULTS["start"]
    st.session_state["num"] = DEFAULTS["num"]
    st.session_state["font_size"] = DEFAULTS["font_size"]

    logos = list_logo_files()
    st.session_state["logo_path"] = logos[0] if logos else ""


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)

    # Small/simple top row
    top_l, top_r = st.columns([1, 2], vertical_alignment="center")
    with top_l:
        st.button("Reset", on_click=reset_basic, use_container_width=True)
    with top_r:
        st.caption("Change settings → preview updates automatically.")

    left, right = st.columns([1, 1], gap="large")

    # -----------------------------
    # Controls (simple labels)
    # -----------------------------
    with left:
        st.subheader("Settings")

        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Surah", 1, 114, key="surah")
            st.number_input("Start", 1, 9999, key="start")
        with c2:
            st.number_input("Count", 1, 9999, key="num")
            st.slider("Font", 20, 120, key="font_size")

        logos = list_logo_files()
        if logos:
            # keep current if valid
            cur = st.session_state["logo_path"]
            if cur not in logos:
                st.session_state["logo_path"] = logos[0]
            st.selectbox("Logo", logos, key="logo_path")
        else:
            st.warning("Put PNG logos in ./logos")

        with st.expander("Advanced"):
            a1, a2 = st.columns(2)
            with a1:
                st.number_input("Width", 300, 6000, key="width")
                st.number_input("Height", 50, 2000, key="height")
                st.number_input("Line height", 20, 400, key="line_h")
                st.slider("Padding", 0.0, 0.20, step=0.005, key="pad_ratio")
            with a2:
                st.number_input("Logo width", 10, 1000, key="logo_w")
                st.number_input("Logo margin", 0, 300, key="logo_m")
                st.text_input("Uthman font file", key="uthman_font")

            st.markdown("**Colors**")
            st.color_picker("Background", key="bg")
            st.color_picker("Text", key="read_color")
            st.color_picker("Marker", key="mark_color")

        st.divider()

        # One button: generate + download
        st.subheader("Export")
        try:
            cfg_now = build_cfg_from_state()
            cfg_dict = cfg_now.__dict__.copy()
            png_bytes_for_download = cached_render_png(cfg_dict)

            filename = f"{cfg_now.surah_number}_{cfg_now.start_verse}_{cfg_now.start_verse + cfg_now.num_verses - 1}.png"
            st.download_button(
                "Download PNG",
                data=png_bytes_for_download,
                file_name=filename,
                mime="image/png",
                use_container_width=True,
            )
        except Exception as e:
            st.error(str(e))

    # -----------------------------
    # Preview
    # -----------------------------
    with right:
        st.subheader("Preview")
        try:
            cfg_now = build_cfg_from_state()
            text, url = cached_fetch_text(cfg_now.surah_number, cfg_now.start_verse, cfg_now.num_verses)

            png_bytes = cached_render_png(cfg_now.__dict__.copy())
            st.image(png_bytes, use_container_width=True)
            st.caption(url)
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
