#!/usr/bin/env python3
"""
Generate publication-quality SVG + PDF pipeline figure for LaTeX.
Output: results/figures/pipeline_methodology.svg  (and .pdf if cairosvg available)

Usage:
    python docs/generate_pipeline_figure.py
"""

from pathlib import Path

# ── Layout constants (all in px; viewBox maps to 185mm width for A4) ─────
W, H = 700, 330
SEC_Y, SEC_H = 18, 278
BADGE_Y = SEC_Y + SEC_H + 12          # badge centre
ARROW_W = 24
GAP = 3                               # between section edge and arrow

# Section X positions and widths  (A/C/D compact, B gets extra room)
SEC_A_X, SEC_A_W = 8,   125
ARR1_X = SEC_A_X + SEC_A_W + GAP
SEC_B_X, SEC_B_W = ARR1_X + ARROW_W + GAP, 210
ARR2_X = SEC_B_X + SEC_B_W + GAP
SEC_C_X, SEC_C_W = ARR2_X + ARROW_W + GAP, 140
ARR3_X = SEC_C_X + SEC_C_W + GAP
SEC_D_X, SEC_D_W = ARR3_X + ARROW_W + GAP, 115

# Colours
BLUE      = "#1a56db"
BLUE_MID  = "#3b82f6"
BLUE_LT   = "#bfdbfe"
BLUE_FAINT= "#eff6ff"
ORANGE    = "#ea580c"
ORANGE_LT = "#ffedd5"
SLATE     = "#475569"
SLATE_LT  = "#94a3b8"
GREY_BG   = "#f7f8fc"
GREY_BD   = "#c5cde0"
WHITE     = "#ffffff"
RED_LT    = "#fca5a5"
TEXT_PRI   = "#1e293b"
TEXT_SEC   = "#4b5572"
TEXT_MUT   = "#8a93b0"
GREEN     = "#059669"
MIN_FONT_SIZE = 8  # px – legible at A4 print size


# ── Helpers ──────────────────────────────────────────────────────────────

def _estimate_text_width(txt, size, family="Helvetica, Arial, sans-serif", weight="normal"):
    """Rough SVG text-width estimate in px for auto-fitting."""
    text = str(txt).replace("&amp;", "&")
    mono = any(token in family for token in ("Menlo", "Consolas", "monospace"))
    ratio = 0.62 if mono else 0.54
    if str(weight).lower() in {"bold", "600", "700", "800", "900"}:
        ratio *= 1.05
    return max(len(text), 1) * float(size) * ratio


def _font_size(size, txt="", family="Helvetica, Arial, sans-serif", weight="normal", max_width=None, min_size=MIN_FONT_SIZE):
    """Clamp and optionally shrink font-size to fit a target width."""
    try:
        numeric = float(size)
    except (TypeError, ValueError):
        numeric = MIN_FONT_SIZE
    if max_width is not None and max_width > 0:
        est_w = _estimate_text_width(txt, numeric, family, weight)
        if est_w > max_width:
            numeric *= max_width / est_w
    return f"{max(numeric, min_size):g}"

def _rect(x, y, w, h, rx=8, fill=WHITE, stroke=GREY_BD, sw=1.2, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def _text(
    x,
    y,
    txt,
    size=8,
    fill=TEXT_PRI,
    weight="normal",
    anchor="middle",
    family="Helvetica, Arial, sans-serif",
    max_width=None,
    min_size=MIN_FONT_SIZE,
):
    safe = str(txt).replace("&", "&amp;")
    fitted = _font_size(size, txt=txt, family=family, weight=weight, max_width=max_width, min_size=min_size)
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{fitted}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{safe}</text>'

def _arrow_right(x, y, length=ARROW_W, h=14, fill=BLUE_LT):
    """Chevron arrow pointing right, centred at (x, y)."""
    tip = x + length
    body = tip - 8
    ht = h // 2
    hb = h // 2 + 2  # slight extra for arrowhead
    pts = f"{x},{y-ht//2} {body},{y-ht//2} {body},{y-hb} {tip},{y} {body},{y+hb} {body},{y+ht//2} {x},{y+ht//2}"
    return f'<polygon points="{pts}" fill="{fill}"/>'

def _mini_arrow_down(cx, y, size=7, fill=BLUE_LT):
    pts = f"{cx-size},{y} {cx},{y+size} {cx+size},{y}"
    return f'<polygon points="{pts}" fill="{fill}"/>'

def _badge(cx, cy, letter, r=13):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{BLUE}" stroke="{GREY_BG}" stroke-width="3"/>'
            f'<text x="{cx}" y="{cy+4.5}" font-family="Helvetica, Arial, sans-serif" font-size="{_font_size(12, txt=letter, weight="bold")}" '
            f'fill="{WHITE}" font-weight="bold" text-anchor="middle">{letter}</text>')

def _cell(x, y, w=14, h=8, fill=GREY_BD, rx=2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>'

def _matrix(ox, oy, pattern, cw=14, ch=8, gap=2):
    """Draw a small data matrix. pattern is list-of-lists of colour strings."""
    lines = []
    for r, row in enumerate(pattern):
        for c, colour in enumerate(row):
            lines.append(_cell(ox + c * (cw + gap), oy + r * (ch + gap), cw, ch, fill=colour))
    return "\n".join(lines)

# Colour shorthand for matrix patterns
E = WHITE    # empty / valid data
G = GREY_BD  # generic filled cell
R = RED_LT   # missing (red)
B = BLUE_MID
S = SLATE_LT
O = ORANGE

def _db_icon(cx, cy, s=10):
    """Tiny database cylinder icon."""
    return (
        f'<ellipse cx="{cx}" cy="{cy-s}" rx="{s}" ry="{s//3}" fill="none" stroke="{BLUE}" stroke-width="1.4"/>'
        f'<path d="M{cx-s},{cy-s} v{s} c0,{s//3} {2*s},{ s//3} {2*s},0 v-{s}" fill="none" stroke="{BLUE}" stroke-width="1.4"/>'
        f'<path d="M{cx-s},{cy-s//2} c0,{s//3} {2*s},{s//3} {2*s},0" fill="none" stroke="{BLUE}" stroke-width="1" opacity="0.5"/>'
    )

def _filter_icon(cx, cy, s=9):
    return f'<polygon points="{cx-s},{cy-s} {cx+s},{cy-s} {cx+2},{cy+2} {cx+2},{cy+s} {cx-2},{cy+s} {cx-2},{cy+2}" fill="none" stroke="{SLATE}" stroke-width="1.3"/>'

def _folder_icon(cx, cy, s=8, fill=GREY_BD, stroke=SLATE_LT):
    return (f'<path d="M{cx-s},{cy-s//2+2} a1.5,1.5 0 0 1 1.5,-1.5 h{s//2} l{s//4},{s//4} h{s} '
            f'a1.5,1.5 0 0 1 1.5,1.5 v{s} a1.5,1.5 0 0 1 -1.5,1.5 h-{int(1.7*s)} '
            f'a1.5,1.5 0 0 1 -1.5,-1.5 z" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')

def _brain_icon(cx, cy, s=10):
    return (
        f'<circle cx="{cx-3}" cy="{cy-3}" r="{s//2}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>'
        f'<circle cx="{cx+3}" cy="{cy-3}" r="{s//2}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>'
        f'<circle cx="{cx-3}" cy="{cy+3}" r="{s//2}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>'
        f'<circle cx="{cx+3}" cy="{cy+3}" r="{s//2}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>'
        f'<circle cx="{cx}" cy="{cy}" r="2" fill="{BLUE}"/>'
    )

def _clock_icon(cx, cy, r=9):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>'
            f'<polyline points="{cx},{cy-r+3} {cx},{cy} {cx+4},{cy+2}" fill="none" stroke="{BLUE}" stroke-width="1.3" stroke-linecap="round"/>')

def _chart_icon(cx, cy, s=9):
    return (f'<rect x="{cx-s}" y="{cy-s}" width="{2*s}" height="{2*s}" rx="2" fill="none" stroke="{SLATE}" stroke-width="1.3"/>'
            f'<rect x="{cx-6}" y="{cy-3}" width="4" height="10" rx="1" fill="{SLATE}" opacity="0.6"/>'
            f'<rect x="{cx-1}" y="{cy-6}" width="4" height="13" rx="1" fill="{SLATE}" opacity="0.6"/>'
            f'<rect x="{cx+4}" y="{cy+1}" width="4" height="6" rx="1" fill="{SLATE}" opacity="0.6"/>')

def _gear_icon(cx, cy, r=8):
    spokes = ""
    for a in range(0, 360, 45):
        import math
        rx = cx + r * math.cos(math.radians(a))
        ry = cy + r * math.sin(math.radians(a))
        spokes += f'<line x1="{cx}" y1="{cy}" x2="{rx:.1f}" y2="{ry:.1f}" stroke="{SLATE}" stroke-width="1.2"/>'
    return (f'<circle cx="{cx}" cy="{cy}" r="{r-2}" fill="none" stroke="{SLATE}" stroke-width="1.3"/>'
            f'{spokes}')

def _chip(cx, cy, label, color=BLUE, bg=BLUE_FAINT, border=BLUE_LT, w=None):
    if w is None:
        w = max(len(label) * 7.5 + 14, 55)
    return (f'<rect x="{cx-w/2}" y="{cy-9}" width="{w}" height="18" rx="9" fill="{bg}" stroke="{border}" stroke-width="1.2"/>'
            f'<text x="{cx}" y="{cy+4}" font-family="Menlo, Consolas, monospace" font-size="{_font_size(9, txt=label, family="Menlo, Consolas, monospace", weight="500", max_width=w-10)}" fill="{color}" '
            f'text-anchor="middle" font-weight="500">{label}</text>')

def _split_lines(x1, x2, y_top, y_mid, y_bot, n=3):
    """Three-way fork connector."""
    xs = [x1 + i * (x2 - x1) / (n - 1) for i in range(n)]
    mid_x = (x1 + x2) / 2
    lines = [
        f'<line x1="{mid_x}" y1="{y_top}" x2="{mid_x}" y2="{y_mid}" stroke="{BLUE_LT}" stroke-width="1.8"/>',
        f'<line x1="{xs[0]}" y1="{y_mid}" x2="{xs[-1]}" y2="{y_mid}" stroke="{BLUE_LT}" stroke-width="1.8"/>',
    ]
    for xi in xs:
        lines.append(f'<line x1="{xi}" y1="{y_mid}" x2="{xi}" y2="{y_bot}" stroke="{BLUE_LT}" stroke-width="1.8"/>')
    return "\n".join(lines)

def _metric_tag(cx, cy, label):
    safe = label.replace("&", "&amp;")
    w = len(label) * 6.5 + 10
    return (f'<rect x="{cx-w/2}" y="{cy-8}" width="{w}" height="16" rx="4" fill="{GREY_BG}" stroke="{GREY_BD}" stroke-width="0.8"/>'
            f'<text x="{cx}" y="{cy+4}" font-family="Menlo, Consolas, monospace" font-size="{_font_size(9, txt=label, family="Menlo, Consolas, monospace", max_width=w-8)}" fill="{TEXT_SEC}" text-anchor="middle">{safe}</text>')


# ── Build SVG ────────────────────────────────────────────────────────────

def build_svg():
    parts = []

    # ── Background (transparent for LaTeX) ──

    # ====================================================================
    # SECTION A: Data Preparation & Splitting
    # ====================================================================
    ax, aw = SEC_A_X, SEC_A_W
    parts.append(_rect(ax, SEC_Y, aw, SEC_H, dash=True))

    # Title
    cx_a = ax + aw / 2
    sec_a_text_w = aw - 14
    parts.append(_text(cx_a, SEC_Y + 16, "DATA PREPARATION", 11, TEXT_PRI, "bold", max_width=sec_a_text_w))
    parts.append(_text(cx_a, SEC_Y + 28, "& Splitting", 10, TEXT_MUT, max_width=sec_a_text_w))

    # DB icon + label
    db_block_y = SEC_Y + 40
    db_w = aw - 16
    parts.append(_rect(ax + 8, db_block_y, db_w, 58, 6, BLUE_FAINT, BLUE_LT, 0.8))
    parts.append(_db_icon(cx_a, db_block_y + 15))
    parts.append(_text(cx_a, db_block_y + 35, "SisRHC / INCA", 10.5, TEXT_PRI, "600", max_width=db_w - 10))
    parts.append(_text(cx_a, db_block_y + 48, "~5.4M records (raw)", 9, TEXT_SEC, max_width=db_w - 10))

    # Arrow down
    parts.append(_mini_arrow_down(cx_a, db_block_y + 62))

    # Filter block
    flt_y = db_block_y + 74
    flt_w = aw - 16
    parts.append(_rect(ax + 8, flt_y, flt_w, 90, 6, WHITE, GREY_BD, 0.8))
    parts.append(_filter_icon(cx_a, flt_y + 12))
    parts.append(_text(cx_a, flt_y + 30, "Data Cleaning", 10.5, TEXT_PRI, "600", max_width=flt_w - 10))
    parts.append(_text(cx_a, flt_y + 44, "Date filter 2013-2023", 9, TEXT_SEC, max_width=flt_w - 10))
    parts.append(_text(cx_a, flt_y + 55, "Dictionary enforcement", 9, TEXT_SEC, max_width=flt_w - 10))
    parts.append(_text(cx_a, flt_y + 66, "Non-informative -> NaN", 9, TEXT_SEC, max_width=flt_w - 10))
    parts.append(_text(cx_a, flt_y + 80, "~2.3M after filters", 10, BLUE, "bold", max_width=flt_w - 10))

    # Arrow down
    parts.append(_mini_arrow_down(cx_a, flt_y + 94))

    # 5-fold CV block
    cv_y = flt_y + 108
    cv_w = aw - 16
    parts.append(_rect(ax + 8, cv_y, cv_w, 52, 6, WHITE, GREY_BD, 0.8))
    # folder icons
    for i in range(5):
        fx = cx_a - 36 + i * 16
        fc = BLUE_LT if i == 2 else GREY_BD
        fs = BLUE if i == 2 else SLATE_LT
        parts.append(_folder_icon(fx, cv_y + 13, 7, fc, fs))
    parts.append(_text(cx_a, cv_y + 30, "Stratified 5-fold CV", 10, TEXT_PRI, "600", max_width=cv_w - 10))
    parts.append(_text(cx_a, cv_y + 43, "Encoding fit per fold", 9, TEXT_SEC, max_width=cv_w - 10))

    # Badge A
    parts.append(_badge(cx_a, BADGE_Y, "A"))

    # ====================================================================
    # Arrow A → B
    # ====================================================================
    arr1_cy = SEC_Y + SEC_H / 2
    parts.append(_arrow_right(ARR1_X, arr1_cy))

    # ====================================================================
    # SECTION B: Missing Data Handling
    # ====================================================================
    bx, bw = SEC_B_X, SEC_B_W
    parts.append(_rect(bx, SEC_Y, bw, SEC_H, dash=True))
    cx_b = bx + bw / 2

    sec_b_text_w = bw - 14
    parts.append(_text(cx_b, SEC_Y + 16, "MISSING DATA HANDLING", 11, TEXT_PRI, "bold", max_width=sec_b_text_w))
    parts.append(_text(cx_b, SEC_Y + 28, "Imputation Strategies", 10, TEXT_MUT, max_width=sec_b_text_w))

    # Natural missing matrix
    mat_y = SEC_Y + 38
    nat_pattern = [
        [G, G, R, G],
        [G, R, G, R],
        [R, G, G, G],
        [G, R, R, G],
    ]
    mat_ox = cx_b - 33
    parts.append(_matrix(mat_ox, mat_y, nat_pattern))
    parts.append(_text(cx_b, mat_y + 46, "Natural Missing Data", 10, TEXT_PRI, "600", max_width=bw - 24))
    parts.append(_text(cx_b, mat_y + 58, "Real-world clinical records", 9, TEXT_SEC, max_width=bw - 24))

    # Split lines
    split_top = mat_y + 64
    split_mid = split_top + 8
    split_bot = split_mid + 10
    col_w = (bw - 30) / 3
    col_centres = [bx + 12 + col_w / 2 + i * (col_w + 3) for i in range(3)]
    parts.append(_split_lines(col_centres[0], col_centres[2], split_top, split_mid, split_bot))

    # Strategy columns
    strat_y = split_bot + 2
    strat_h = 120
    strat_configs = [
        # (title_lines, sub-lines, matrix colour)
        (["Baseline", "NoImpute"], ["XGBoost native", "CatBoost native"], S),
        (["Classical"], ["Mean / Mode", "Median / Mode"], B),
        (["ML Methods"], ["MICE", "MICE-XGBoost", "MissForest", "kNN (k=5)"], O),
    ]

    for i, (title_lines, subs, mcol) in enumerate(strat_configs):
        sx = bx + 10 + i * (col_w + 3)
        parts.append(_rect(sx, strat_y, col_w, strat_h, 6, GREY_BG, GREY_BD, 0.8))
        ccx = sx + col_w / 2

        # mini matrix inside
        m_pattern = [
            [G, G, mcol, G],
            [G, mcol, G, mcol],
            [mcol, G, G, G],
        ]
        # mini matrix inside (4 cols × 9px + 3 gaps × 2px = 42px wide)
        mat_w = 4 * 9 + 3 * 2  # 42
        parts.append(_matrix(ccx - mat_w / 2, strat_y + 6, m_pattern, 9, 6, 2))
        for t, tline in enumerate(title_lines):
            parts.append(_text(ccx, strat_y + 40 + t * 12, tline, 9.5, TEXT_PRI, "600", max_width=col_w - 8))
        title_offset = len(title_lines) * 12
        for j, sub in enumerate(subs):
            parts.append(_text(ccx, strat_y + 42 + title_offset + j * 13, sub, 8.5, TEXT_MUT, max_width=col_w - 8))

    # Badge B
    parts.append(_badge(cx_b, BADGE_Y, "B"))

    # ====================================================================
    # Arrow B → C
    # ====================================================================
    parts.append(_arrow_right(ARR2_X, arr1_cy))

    # ====================================================================
    # SECTION C: Predictive Modeling
    # ====================================================================
    cx_c = SEC_C_X + SEC_C_W / 2
    parts.append(_rect(SEC_C_X, SEC_Y, SEC_C_W, SEC_H, dash=True))

    sec_c_text_w = SEC_C_W - 14
    parts.append(_text(cx_c, SEC_Y + 16, "PREDICTIVE MODELING", 11, TEXT_PRI, "bold", max_width=sec_c_text_w))
    parts.append(_text(cx_c, SEC_Y + 28, "Oncology Staging", 10, TEXT_MUT, max_width=sec_c_text_w))

    # Brain icon + label
    ml_y = SEC_Y + 40
    ml_w = SEC_C_W - 16
    parts.append(_rect(SEC_C_X + 8, ml_y, ml_w, 50, 6, BLUE_FAINT, BLUE_LT, 0.8))
    parts.append(_brain_icon(cx_c, ml_y + 12))
    parts.append(_text(cx_c, ml_y + 32, "ML Classification", 10.5, TEXT_PRI, "600", max_width=ml_w - 10))
    parts.append(_text(cx_c, ml_y + 44, "6-class staging", 9, TEXT_SEC, max_width=ml_w - 10))

    # Classifier chips - Gradient Boosting (stacked vertically to fit)
    chip_y1 = ml_y + 58
    parts.append(_text(cx_c, chip_y1, "Gradient Boosting", 9, TEXT_MUT, max_width=SEC_C_W - 20))
    parts.append(_chip(cx_c, chip_y1 + 15, "XGBoost", BLUE, BLUE_FAINT, BLUE_LT))
    parts.append(_chip(cx_c, chip_y1 + 35, "CatBoost", BLUE, BLUE_FAINT, BLUE_LT))

    # GPU-accelerated
    chip_y2 = chip_y1 + 56
    parts.append(_text(cx_c, chip_y2, "GPU-accelerated (RAPIDS)", 9, TEXT_MUT, max_width=SEC_C_W - 20))
    parts.append(_chip(cx_c, chip_y2 + 15, "cuML RF", ORANGE, ORANGE_LT, "#fed7aa"))
    parts.append(_chip(cx_c, chip_y2 + 35, "cuML SVM", ORANGE, ORANGE_LT, "#fed7aa"))

    # Divider
    div_y = chip_y2 + 52
    parts.append(f'<line x1="{SEC_C_X + 12}" y1="{div_y}" x2="{SEC_C_X + SEC_C_W - 12}" y2="{div_y}" stroke="{GREY_BD}" stroke-width="0.8"/>')

    # Tuning block
    tune_y = div_y + 8
    tune_w = SEC_C_W - 16
    parts.append(_rect(SEC_C_X + 8, tune_y, tune_w, 48, 6, WHITE, GREY_BD, 0.8))
    parts.append(_gear_icon(cx_c - 10, tune_y + 14, 7))
    parts.append(_gear_icon(cx_c + 10, tune_y + 12, 5))
    parts.append(_text(cx_c, tune_y + 30, "Hyperparameter Tuning", 9, TEXT_PRI, "600", max_width=tune_w - 10))
    parts.append(_text(cx_c, tune_y + 42, "RandomizedSearchCV (inner 5-fold)", 8, TEXT_SEC, max_width=tune_w - 10))

    # Badge C
    parts.append(_badge(cx_c, BADGE_Y, "C"))

    # ====================================================================
    # Arrow C → D
    # ====================================================================
    parts.append(_arrow_right(ARR3_X, arr1_cy))

    # ====================================================================
    # SECTION D: Evaluation (Processing Time + Metrics only)
    # ====================================================================
    cx_d = SEC_D_X + SEC_D_W / 2
    parts.append(_rect(SEC_D_X, SEC_Y, SEC_D_W, SEC_H, dash=True))

    sec_d_text_w = SEC_D_W - 14
    parts.append(_text(cx_d, SEC_Y + 16, "EVALUATION", 11, TEXT_PRI, "bold", max_width=sec_d_text_w))
    parts.append(_text(cx_d, SEC_Y + 28, "Performance Assessment", 10, TEXT_MUT, max_width=sec_d_text_w))

    # Processing Time block
    pt_y = SEC_Y + 40
    pt_w = SEC_D_W - 16
    parts.append(_rect(SEC_D_X + 8, pt_y, pt_w, 68, 6, BLUE_FAINT, BLUE_LT, 0.8))
    parts.append(_clock_icon(cx_d, pt_y + 14, 8))
    parts.append(_text(cx_d, pt_y + 33, "Processing Time", 10.5, TEXT_PRI, "600", max_width=pt_w - 10))
    parts.append(_text(cx_d, pt_y + 46, "Imputation", 9, TEXT_SEC, max_width=pt_w - 10))
    parts.append(_text(cx_d, pt_y + 57, "Tuning / Prediction", 9, TEXT_SEC, max_width=pt_w - 10))

    # Arrow down
    parts.append(_mini_arrow_down(cx_d, pt_y + 72))

    # Metrics block
    met_y = pt_y + 86
    met_w = SEC_D_W - 16
    parts.append(_rect(SEC_D_X + 8, met_y, met_w, 148, 6, WHITE, GREY_BD, 0.8))
    parts.append(_chart_icon(cx_d, met_y + 14, 8))
    parts.append(_text(cx_d, met_y + 34, "Metrics & Accuracy", 10.5, TEXT_PRI, "600", max_width=met_w - 10))

    # Metric tags — stack vertically to avoid overlap
    tags = ["F1-Weighted", "AUC", "Accuracy", "Conf. Matrix"]
    for j, label in enumerate(tags):
        parts.append(_metric_tag(cx_d, met_y + 50 + j * 20, label))

    # Direct & indirect evaluation labels
    parts.append(_text(cx_d, met_y + 128, "Direct & indirect evaluation", 8, TEXT_MUT, max_width=met_w - 10))
    parts.append(_text(cx_d, met_y + 140, "comparison across imputers", 8, TEXT_MUT, max_width=met_w - 10))

    # Badge D
    parts.append(_badge(cx_d, BADGE_Y, "D"))


    return parts


def generate():
    parts = build_svg()
    body = "\n  ".join(parts)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {W} {H}"
     width="185mm" height="{185 * H / W:.1f}mm"
     font-family="Helvetica, Arial, sans-serif">
  {body}
</svg>
"""

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "pipeline_methodology.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"✓ SVG saved: {svg_path}  ({svg_path.stat().st_size / 1024:.1f} KB)")

    # Try to convert to PDF for LaTeX \includegraphics
    try:
        import cairosvg
        pdf_path = out_dir / "pipeline_methodology.pdf"
        cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(pdf_path))
        print(f"✓ PDF saved: {pdf_path}  ({pdf_path.stat().st_size / 1024:.1f} KB)")
    except ImportError:
        print("ℹ  Install cairosvg (`pip install cairosvg`) to also export PDF.")
        print("   Alternatively, use Inkscape:  inkscape pipeline_methodology.svg --export-filename=pipeline_methodology.pdf")

    # Also try high-res PNG
    try:
        import cairosvg
        png_path = out_dir / "pipeline_methodology.png"
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path), scale=4)
        print(f"✓ PNG saved: {png_path}  ({png_path.stat().st_size / 1024:.1f} KB, 4× scale)")
    except ImportError:
        pass

    return svg_path


if __name__ == "__main__":
    generate()
