# -*- coding: utf-8 -*-
"""
Shared STIX plotting utilities for the Q-vvW2 paper figures.

The helpers enforce the publication figure conventions used by this project:
white backgrounds, inward ticks, four-sided axes, STIXGeneral text, consistent
panel geometry, independent colorbar axes, and PDF/SVG/PNG export checks.
They only control visual presentation and do not alter scientific data.
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import shutil
import subprocess
import warnings
from typing import Iterable, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# ============================================================
# 0. LOCKED CONSTANTS
# ============================================================

STRICT_MODE = True

LOCKED_ENGLISH_FONT = "STIXGeneral"
LOCKED_CHINESE_FONT_KEYWORDS = (
    "NotoSerifCJK-Bold",
    "Noto Serif CJK Bold",
    "Noto Serif CJK SC",
    "Noto Serif CJK JP",
)

BLACK = "#111111"
WHITE = "#FFFFFF"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D8D8D8"

TYPE_COLORS = {
    "A": "#184C86",
    "B": "#1A9A90",
    "C": "#B2182B",
}

DEFAULT_AXIS_LINEWIDTH = 1.25
DEFAULT_LINEWIDTH = 1.80
DEFAULT_MARKEREDGEWIDTH = 0.90
DEFAULT_TICK_SIZE = 9.0
DEFAULT_AXIS_LABEL_SIZE = 12.5
DEFAULT_PANEL_TITLE_SIZE = 12.1
DEFAULT_LEGEND_SIZE = 9.8


# ============================================================
# 1. EXACT FONT RESOLUTION
# ============================================================

def _all_system_fonts():
    fonts = []
    for ext in ("ttf", "otf", "ttc"):
        try:
            fonts.extend(fm.findSystemFonts(fontpaths=None, fontext=ext))
        except Exception:
            pass
    seen, out = set(), []
    for p in fonts:
        p = str(Path(p))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _font_name(path: str) -> str:
    try:
        return fm.FontProperties(fname=path).get_name()
    except Exception:
        return ""


def _find_times_new_roman() -> str:
    """必须找到真正 STIXGeneral；禁止 Tinos/Liberation/DejaVu 替代。"""
    hard_paths = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/Library/Fonts/STIXGeneral.ttf",
        "/Library/Fonts/STIXGeneral Bold.ttf",
    ]
    for p in hard_paths:
        if Path(p).exists() and _font_name(p) == LOCKED_ENGLISH_FONT:
            return str(Path(p).resolve())

    for p in _all_system_fonts():
        if _font_name(p) == LOCKED_ENGLISH_FONT:
            return p

    raise FileNotFoundError(
        "\n[LOCKED FONT ERROR]\n"
        "未找到真正的 STIXGeneral。\n"
        "本模板禁止自动回退到 Tinos、Liberation Serif、STIX 或 DejaVu。\n"
        "请先安装 STIXGeneral 后再生成正式图。"
    )


def _find_noto_serif_cjk_bold() -> str:
    """必须找到 Noto Serif CJK Bold；禁止 DejaVu 等字体替代。"""
    hard_paths = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.otf",
        "/usr/share/fonts/noto-cjk/NotoSerifCJK-Bold.ttc",
    ]
    for p in hard_paths:
        if Path(p).exists():
            return str(Path(p).resolve())

    for p in _all_system_fonts():
        name = _font_name(p)
        token = (name + " " + Path(p).stem).lower().replace(" ", "")
        if "bold" in token and any(
            k.lower().replace(" ", "") in token
            for k in LOCKED_CHINESE_FONT_KEYWORDS
        ):
            return p

    raise FileNotFoundError(
        "\n[LOCKED FONT ERROR]\n"
        "未找到 Noto Serif CJK Bold。\n"
        "本模板禁止自动使用 DejaVu 或其他字体替代。"
    )


# 延迟初始化：避免仅查看本文件时因为机器没装字体而无法 import。
# 但一旦开始正式绘图，init_locked_style() 会强制检查。
EN_FONT_PATH = None
CN_FONT_PATH = None
EN_FONT_NAME = None
CN_FONT_NAME = None
EN_BOLD = None
CN_BOLD = None
MIXED_BOLD = None
_INITIALIZED = False


def init_locked_style():
    """User-authorized STIX variant of the supplied locked plotting template."""
    global EN_FONT_PATH, CN_FONT_PATH, EN_FONT_NAME, CN_FONT_NAME
    global EN_BOLD, CN_BOLD, MIXED_BOLD, _INITIALIZED
    if _INITIALIZED:
        return
    EN_FONT_NAME = "STIXGeneral"
    EN_FONT_PATH = fm.findfont(fm.FontProperties(family=EN_FONT_NAME, weight="bold"), fallback_to_default=False)
    if _font_name(EN_FONT_PATH) != EN_FONT_NAME:
        raise RuntimeError("STIXGeneral font is unavailable")
    CN_FONT_NAME, CN_FONT_PATH = EN_FONT_NAME, EN_FONT_PATH
    EN_BOLD = fm.FontProperties(fname=EN_FONT_PATH, family=EN_FONT_NAME, weight="bold")
    CN_BOLD = EN_BOLD
    MIXED_BOLD = EN_BOLD
    mpl.rcParams.update({
        "figure.facecolor": WHITE, "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE, "savefig.transparent": False,
        "font.family": [EN_FONT_NAME], "font.serif": [EN_FONT_NAME],
        "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.unicode_minus": False, "axes.linewidth": DEFAULT_AXIS_LINEWIDTH,
        "mathtext.fontset": "custom", "mathtext.rm": EN_FONT_NAME,
        "mathtext.it": EN_FONT_NAME + ":italic",
        "mathtext.bf": EN_FONT_NAME + ":bold", "mathtext.default": "bf",
        "mathtext.fallback": None, "mathtext.cal": EN_FONT_NAME, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "font.size": 10.0,
        "axes.labelsize": 10.0, "axes.titlesize": 10.5,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "legend.fontsize": 9.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    })
    _INITIALIZED = True


def _ensure_init():
    if not _INITIALIZED:
        init_locked_style()


def _bold_text_obj(text_obj, fontsize=None):
    _ensure_init()
    original_size = text_obj.get_fontsize()
    text_obj.set_fontproperties(MIXED_BOLD)
    text_obj.set_fontsize(original_size)
    text_obj.set_fontweight("bold")
    text_obj.set_math_fontfamily("custom")
    if fontsize is not None:
        text_obj.set_fontsize(fontsize)
    return text_obj


def make_bold_text(ax_or_fig, x, y, text, *, transform=None,
                   ha="center", va="center", fontsize=10.0,
                   zorder=20, bbox=None, clip_on=False):
    _ensure_init()
    kw = dict(
        ha=ha, va=va,
        fontproperties=MIXED_BOLD,
        fontweight="bold",
        fontsize=fontsize,
        zorder=zorder,
        bbox=bbox,
        clip_on=clip_on,
    )
    if transform is not None:
        kw["transform"] = transform
    return ax_or_fig.text(x, y, text, **kw)


# ============================================================
# 3. AXES STYLE
# ============================================================

def add_main_axes(fig, rect):
    _ensure_init()
    ax = fig.add_axes(rect)
    style_axis(ax)
    return ax


def style_axis(ax, minor=True, top_right=True):
    _ensure_init()
    ax.set_facecolor(WHITE)
    ax.grid(False)
    if minor:
        ax.minorticks_on()
    else:
        ax.minorticks_off()

    ax.tick_params(
        which="both",
        direction="in",
        top=top_right,
        right=top_right,
        width=1.0,
        pad=4,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(DEFAULT_AXIS_LINEWIDTH)

    refresh_axis_fonts(ax)
    return ax


def refresh_axis_fonts(ax, tick_size=DEFAULT_TICK_SIZE):
    _ensure_init()
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        _bold_text_obj(t, tick_size)
    if ax.xaxis.label:
        _bold_text_obj(ax.xaxis.label)
    if ax.yaxis.label:
        _bold_text_obj(ax.yaxis.label)


def set_axis_labels(ax, xlabel=None, ylabel=None,
                    xlabel_size=DEFAULT_AXIS_LABEL_SIZE,
                    ylabel_size=DEFAULT_AXIS_LABEL_SIZE,
                    xlabel_pad=7, ylabel_pad=8):
    _ensure_init()
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontproperties=MIXED_BOLD,
            fontweight="bold",
            fontsize=xlabel_size,
            labelpad=xlabel_pad,
        )
    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontproperties=MIXED_BOLD,
            fontweight="bold",
            fontsize=ylabel_size,
            labelpad=ylabel_pad,
        )


# ============================================================
# 4. PANEL TITLES
# ============================================================

def add_panel_title(fig, ax, text,
                    fontsize=DEFAULT_PANEL_TITLE_SIZE,
                    y_offset=0.022):
    _ensure_init()
    pos = ax.get_position()
    return fig.text(
        pos.x0 + pos.width / 2.0,
        pos.y1 + y_offset,
        text,
        ha="center",
        va="center",
        fontproperties=MIXED_BOLD,
        fontweight="bold",
        fontsize=fontsize,
    )


# ============================================================
# 5. LEGEND
# ============================================================

def locked_legend(ax, *args, fontsize=DEFAULT_LEGEND_SIZE, **kwargs):
    _ensure_init()
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("handlelength", 2.1)
    kwargs.setdefault("handletextpad", 0.45)
    kwargs.setdefault("borderaxespad", 0.35)
    leg = ax.legend(*args, fontsize=fontsize, **kwargs)
    if leg is not None:
        for t in leg.get_texts():
            _bold_text_obj(t, fontsize)
    return leg


# ============================================================
# 6. COLORBAR — MUST USE INDEPENDENT CAX
# ============================================================

def add_independent_colorbar_axis(fig, ref_ax, width=0.018, pad=0.012):
    pos = ref_ax.get_position()
    return fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])


def style_colorbar(cb, label=None, fontsize=11.5, tick_size=10.0):
    _ensure_init()
    if label is not None:
        cb.set_label(
            label,
            fontproperties=MIXED_BOLD,
            fontweight="bold",
            fontsize=fontsize,
            labelpad=8,
        )
    cb.ax.tick_params(direction="in", width=1.0, length=5.0, pad=4)
    for t in cb.ax.get_yticklabels():
        _bold_text_obj(t, tick_size)
    cb.outline.set_linewidth(1.20)
    cb.outline.set_edgecolor(BLACK)


# ============================================================
# 7. MAP HELPERS
# ============================================================

def set_map_aspect(ax, lon_min, lon_max, lat_min, lat_max):
    midlat = 0.5 * (lat_min + lat_max)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / max(math.cos(math.radians(midlat)), 0.2), adjustable="box")


def add_north_arrow(ax, x=0.055, y=0.92, color="#142B58"):
    _ensure_init()
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(x, y - 0.075),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.25, color=color),
    )
    ax.text(
        x, y + 0.012, "N",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontproperties=EN_BOLD,
        fontweight="bold",
        fontsize=11.5,
        color=color,
        clip_on=True,
    )


def add_scale_bar(ax, lon_min, lon_max, lat_min, lat_max,
                  km=5, xfrac=0.06, yfrac=0.055):
    _ensure_init()
    lat = lat_min + yfrac * (lat_max - lat_min)
    deg = km / (111.320 * math.cos(math.radians(lat)))
    lon0 = lon_min + xfrac * (lon_max - lon_min)
    lat0 = lat_min + yfrac * (lat_max - lat_min)

    ax.plot([lon0, lon0 + deg], [lat0, lat0], color="white", lw=4.0, zorder=30)
    ax.plot([lon0, lon0 + deg], [lat0, lat0], color=BLACK, lw=1.3, zorder=31)
    ax.text(
        lon0 + deg / 2.0,
        lat0 + 0.0005,
        f"{km} km",
        ha="center", va="bottom",
        fontproperties=EN_BOLD,
        fontweight="bold",
        fontsize=10.0,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=0.18),
        zorder=32,
        clip_on=True,
    )


def annotate_node(ax, text, x, y, dx=5, dy=5, fontsize=9.5):
    _ensure_init()
    return ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        fontproperties=MIXED_BOLD,
        fontweight="bold",
        fontsize=fontsize,
        color=BLACK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.20),
        zorder=20,
        clip_on=True,
    )


# ============================================================
# 8. COMMON CURVE / SCATTER HELPERS
# ============================================================

def locked_plot(ax, x, y, *, color="#184C86", label=None,
                lw=DEFAULT_LINEWIDTH, marker=None, ms=5.5,
                zorder=3, **kwargs):
    line, = ax.plot(
        x, y,
        color=color,
        label=label,
        lw=lw,
        marker=marker,
        ms=ms,
        markeredgecolor=WHITE if marker else None,
        markeredgewidth=DEFAULT_MARKEREDGEWIDTH if marker else None,
        zorder=zorder,
        **kwargs,
    )
    return line


def locked_scatter(ax, x, y, *, s=65, facecolor="#1A9A90",
                   edgecolor=WHITE, linewidth=0.9, zorder=4, **kwargs):
    return ax.scatter(
        x, y,
        s=s,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )


# ============================================================
# 9. AUDIT — PANEL SIZE
# ============================================================

def audit_equal_panel_sizes(main_axes: Sequence, atol=1e-10):
    if main_axes is None or len(main_axes) <= 1:
        return True

    p0 = main_axes[0].get_position().bounds
    w0, h0 = p0[2], p0[3]
    bad = []

    for i, ax in enumerate(main_axes[1:], start=2):
        p = ax.get_position().bounds
        if not (abs(p[2] - w0) <= atol and abs(p[3] - h0) <= atol):
            bad.append((i, p))

    if bad:
        raise RuntimeError("[LAYOUT FAIL] 主子图绘图区尺寸不一致：" + repr(bad))
    return True


# ============================================================
# 10. AUDIT — TEXT BOUNDS / BOLD
# ============================================================

def audit_text_out_of_canvas(fig, tol_px=1):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    bad = []

    for t in fig.findobj(match=mpl.text.Text):
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width == 0 or bb.height == 0:
            continue
        if (
            bb.x0 < fb.x0 - tol_px or bb.y0 < fb.y0 - tol_px or
            bb.x1 > fb.x1 + tol_px or bb.y1 > fb.y1 + tol_px
        ):
            bad.append(str(t.get_text()))

    if bad:
        raise RuntimeError("[TEXT BOUNDS FAIL] 发现文字越界：" + repr(bad))
    return True


def audit_all_visible_text_bold(fig):
    accepted = {
        "bold", "semibold", "demibold", "heavy", "extra bold", "black",
        600, 700, 800, 900,
    }
    bad = []

    for t in fig.findobj(match=mpl.text.Text):
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        weight = t.get_fontweight()
        if weight not in accepted:
            bad.append((str(t.get_text()), weight))

    if bad:
        raise RuntimeError("[FONT WEIGHT FAIL] 存在非粗体文字：" + repr(bad[:30]))
    return True


# ============================================================
# 11. AUDIT — AXES STYLE
# ============================================================

def _rgba_close(c1, c2, tol=0.08):
    a = np.asarray(mpl.colors.to_rgba(c1))
    b = np.asarray(mpl.colors.to_rgba(c2))
    return np.max(np.abs(a - b)) <= tol


def audit_axes_style(main_axes: Sequence):
    for idx, ax in enumerate(main_axes, start=1):
        if not _rgba_close(ax.get_facecolor(), WHITE, tol=0.03):
            raise RuntimeError(f"[AXES STYLE FAIL] Axes {idx} 不是纯白背景。")

        grids = list(ax.get_xgridlines()) + list(ax.get_ygridlines())
        visible_grids = [
            g for g in grids
            if g.get_visible() and g.get_alpha() != 0 and g.get_linewidth() > 0
        ]
        if visible_grids:
            raise RuntimeError(f"[GRID FAIL] Axes {idx} 检测到可见网格。")

        for name in ("left", "right", "top", "bottom"):
            sp = ax.spines[name]
            if not sp.get_visible():
                raise RuntimeError(f"[SPINE FAIL] Axes {idx} 的 {name} 边框未显示。")
            if not _rgba_close(sp.get_edgecolor(), BLACK, tol=0.15):
                raise RuntimeError(f"[SPINE FAIL] Axes {idx} 的 {name} 边框不是黑色。")
    return True


# ============================================================
# 12. AUDIT — ANNOTATION COLLISIONS
# ============================================================

def audit_annotations_no_overlap(ax, annotations: Iterable | None = None,
                                 expand=(1.02, 1.05)):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    if annotations is None:
        annotations = [
            c for c in ax.get_children()
            if isinstance(c, mpl.text.Annotation) and str(c.get_text()).strip()
        ]
    annotations = list(annotations)
    boxes = [a.get_window_extent(renderer=renderer).expanded(*expand) for a in annotations]

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(
                    "[ANNOTATION OVERLAP FAIL] "
                    f"{annotations[i].get_text()} vs {annotations[j].get_text()}"
                )
    return True


def audit_annotations_inside_axes(ax, annotations: Iterable | None = None,
                                  margin_px=1):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    if annotations is None:
        annotations = [
            c for c in ax.get_children()
            if isinstance(c, mpl.text.Annotation) and str(c.get_text()).strip()
        ]
    abb = ax.get_window_extent(renderer=renderer)

    for a in annotations:
        bb = a.get_window_extent(renderer=renderer)
        if (
            bb.x0 < abb.x0 + margin_px or bb.x1 > abb.x1 - margin_px or
            bb.y0 < abb.y0 + margin_px or bb.y1 > abb.y1 - margin_px
        ):
            raise RuntimeError(f"[ANNOTATION CLIP FAIL] {a.get_text()}")
    return True


# ============================================================
# 13. AUDIT — COLORBAR GEOMETRY
# ============================================================

def audit_colorbar_axes_not_overlapping(main_axes: Sequence,
                                        colorbar_axes: Sequence | None):
    if not colorbar_axes:
        return True

    for cax in colorbar_axes:
        cb = cax.get_position()
        for i, ax in enumerate(main_axes, start=1):
            ab = ax.get_position()
            x_overlap = min(cb.x1, ab.x1) > max(cb.x0, ab.x0)
            y_overlap = min(cb.y1, ab.y1) > max(cb.y0, ab.y0)
            if x_overlap and y_overlap:
                raise RuntimeError(
                    f"[COLORBAR OVERLAP FAIL] colorbar 与主子图 {i} 重叠。"
                )
    return True


# ============================================================
# 14. AUDIT — VECTOR FONT
# ============================================================

FORBIDDEN_FONT_TOKENS = (
    "DejaVu Sans", "DejaVu Serif", "DejaVuSans", "DejaVuSerif",
)


def audit_svg_fonts(svg_path):
    svg_path = Path(svg_path)
    text = svg_path.read_text(encoding="utf-8", errors="ignore")

    bad = [token for token in FORBIDDEN_FONT_TOKENS if token not in ("STIX", "stix") and token in text]
    if bad:
        raise RuntimeError("[SVG FONT FAIL] 检测到禁止字体：" + repr(bad))

    if "STIXGeneral" not in text:
        raise RuntimeError("[SVG FONT FAIL] SVG 中未检测到 STIXGeneral。")
    return True


def audit_pdf_fonts(pdf_path):
    exe = shutil.which("pdffonts")
    if exe is None:
        warnings.warn(
            "[PDF FONT AUDIT] 系统无 pdffonts；跳过 PDF 字体表检查。",
            RuntimeWarning,
        )
        return None

    proc = subprocess.run(
        [exe, str(pdf_path)], capture_output=True, text=True, check=False
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if "DejaVu" in output:
        raise RuntimeError("[PDF FONT FAIL] PDF 中检测到 DejaVu。")

    normalized = output.lower().replace(" ", "").replace("-", "")
    if "stixgeneral" not in normalized:
        raise RuntimeError("[PDF FONT FAIL] PDF 字体表中未检测到 STIXGeneral。")
    return output


# ============================================================
# 15. PDF REAL RENDER AUDIT
# ============================================================

def render_pdf_for_audit(pdf_path, out_png, dpi=220):
    pdf_path = Path(pdf_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    exe = shutil.which("pdftoppm")
    if exe is None:
        warnings.warn(
            "[PDF RENDER AUDIT] 系统无 pdftoppm；无法自动回渲染 PDF。",
            RuntimeWarning,
        )
        return None

    prefix = out_png.with_suffix("")
    proc = subprocess.run(
        [
            exe, "-singlefile", "-png", "-r", str(dpi),
            str(pdf_path), str(prefix),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    generated = prefix.with_suffix(".png")

    if proc.returncode != 0 or not generated.exists():
        raise RuntimeError(
            "[PDF RENDER FAIL] PDF 回渲染失败。\n"
            + (proc.stdout or "") + "\n" + (proc.stderr or "")
        )
    if generated.stat().st_size <= 1000:
        raise RuntimeError("[PDF RENDER FAIL] 回渲染 PNG 文件异常过小。")
    return generated


# ============================================================
# 16. OUTPUT FILE AUDIT
# ============================================================

def audit_output_files(*paths):
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise RuntimeError(f"[OUTPUT FAIL] 文件不存在：{p}")
        if p.stat().st_size <= 1000:
            raise RuntimeError(f"[OUTPUT FAIL] 文件异常过小：{p}")
    return True


# ============================================================
# 17. FINAL SAVE — OFFICIAL ENTRY ONLY
# ============================================================

def save_final_figure(fig, out_base, *, main_axes: Sequence,
                      colorbar_axes: Sequence | None = None,
                      dpi=300, pdf_render_dpi=220,
                      run_pdf_render=True):
    """
    正式图唯一推荐保存入口。

    输出：
        XXX.png
        XXX.pdf
        XXX.svg
        XXX_PDF_RENDER_AUDIT.png  （若 pdftoppm 可用）

    正式 ZIP 最终只保留：
        XXX_FINAL.py
        XXX_FINAL.pdf
        XXX_FINAL.svg
    """
    _ensure_init()

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    svg_path = out_base.with_suffix(".svg")

    # ---------- PRE-SAVE AUDIT ----------
    fig.canvas.draw()
    audit_equal_panel_sizes(main_axes)
    audit_axes_style(main_axes)
    audit_colorbar_axes_not_overlapping(main_axes, colorbar_axes)

    for ax in main_axes:
        refresh_axis_fonts(ax)

    if colorbar_axes:
        for cax in colorbar_axes:
            for t in list(cax.get_xticklabels()) + list(cax.get_yticklabels()):
                _bold_text_obj(t)

    fig.canvas.draw()
    audit_text_out_of_canvas(fig)
    audit_all_visible_text_bold(fig)

    # ---------- EXPORT ----------
    fig.savefig(png_path, dpi=dpi, facecolor=WHITE, transparent=False)
    fig.savefig(pdf_path, facecolor=WHITE, transparent=False)
    fig.savefig(svg_path, facecolor=WHITE, transparent=False)

    # ---------- POST-SAVE AUDIT ----------
    audit_output_files(png_path, pdf_path, svg_path)
    audit_svg_fonts(svg_path)
    audit_pdf_fonts(pdf_path)

    render_path = None
    if run_pdf_render:
        render_path = out_base.parent / (out_base.name + "_PDF_RENDER_AUDIT.png")
        render_path = render_pdf_for_audit(pdf_path, render_path, dpi=pdf_render_dpi)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "svg": str(svg_path),
        "pdf_render_audit": str(render_path) if render_path is not None else None,
        "english_font": EN_FONT_NAME,
        "chinese_font": CN_FONT_NAME,
        "status": "PASS",
    }


# ============================================================
# 18. FORBIDDEN EXAMPLES — READ ONLY
# ============================================================

def forbidden_examples():
    """
    正式图禁止以下做法：

    1. 禁止自动挤压 colorbar：
       fig.colorbar(im, ax=ax)

       必须：
       cax = add_independent_colorbar_axis(fig, ax)
       cb = fig.colorbar(im, cax=cax)

    2. 禁止使用 DejaVu / Tinos / Liberation 替代 STIXGeneral。

    3. 禁止直接 fig.savefig() 绕过 save_final_figure() 的审计。

    4. 禁止为了美化：
       - 平滑真实曲线
       - 改数据点
       - 改排序
       - 改路线
       - 改节点位置
       - 改时间
       - 改单位
       - 改类别
       - 改结论

    5. 禁止多子图主绘图区尺寸不一致。

    6. 禁止图例遮挡数据。

    7. 禁止任何文字、标签、标题或色条发生裁切还继续提交。
    """
    raise RuntimeError("本函数仅用于记录禁止事项，不应实际调用。")


# ============================================================
# 19. ENVIRONMENT CHECK
# ============================================================

def environment_check():
    """只检查字体环境，不画图。"""
    init_locked_style()
    return {
        "status": "PASS",
        "english_font": EN_FONT_NAME,
        "english_font_path": EN_FONT_PATH,
        "chinese_font": CN_FONT_NAME,
        "chinese_font_path": CN_FONT_PATH,
        "pdffonts": shutil.which("pdffonts"),
        "pdftoppm": shutil.which("pdftoppm"),
    }


if __name__ == "__main__":
    info = environment_check()
    print("=" * 70)
    print("Q-vvW2 STIX PLOT STYLE — ENVIRONMENT CHECK")
    print("=" * 70)
    for k, v in info.items():
        print(f"{k}: {v}")
    print("=" * 70)
