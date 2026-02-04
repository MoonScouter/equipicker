"""Generate the Weekly Scoring Board PDF report."""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepInFrame,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image,
)

from xml.sax.saxutils import escape

from equipicker_connect import get_report_dataframe, get_scoring_dataframe, CACHE_DIR
from report_config import DEFAULT_CONFIG_PATH, load_report_config

logger = logging.getLogger(__name__)

PAGE_SIZE = letter
LEFT_MARGIN = RIGHT_MARGIN = 36
TOP_MARGIN = 34
BOTTOM_MARGIN = 34
CONTENT_WIDTH = PAGE_SIZE[0] - LEFT_MARGIN - RIGHT_MARGIN

BRAND_COLORS: Dict[str, colors.Color] = {
    "primary": colors.HexColor("#0B2D5C"),
    "secondary": colors.HexColor("#006C8E"),
    "accent": colors.HexColor("#22B5D8"),
    "muted_text": colors.HexColor("#425466"),
    "row_alt": colors.HexColor("#F6FBFF"),
    "score_bg": colors.HexColor("#F0F7FB"),
}
HIGHLIGHT_COLOR = colors.HexColor("#0C97FF")
SECTION_BAND_COLOR = colors.HexColor("#1F4A82")
TABLE_BAND_COLOR = colors.HexColor("#E3EFFB")
SCORE_ARROW_COLOR = colors.HexColor("#22B573")
LOGO_PATH = Path(__file__).resolve().parent / "logo.jpg"
BANNER_PATH = Path(__file__).resolve().parent / "banner.png"
BANNER_HEIGHT = 140
REPORT_NAME = "Scoring Board Report"

# unicode-capable font registration
UNICODE_FONT_NAME = "EquipickerSans"
UNICODE_FONT_BOLD = "EquipickerSans-Bold"
_FONT_CANDIDATE_PAIRS = [
    (
        Path(__file__).resolve().parent / "DejaVuSans.ttf",
        Path(__file__).resolve().parent / "DejaVuSans-Bold.ttf",
    ),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
    (
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ),
    (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ),
]
_font_registered = False
for regular, bold in _FONT_CANDIDATE_PAIRS:
    try:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont(UNICODE_FONT_NAME, str(regular)))
            pdfmetrics.registerFont(TTFont(UNICODE_FONT_BOLD, str(bold)))
            pdfmetrics.registerFontFamily(
                UNICODE_FONT_NAME,
                normal=UNICODE_FONT_NAME,
                bold=UNICODE_FONT_BOLD,
                italic=UNICODE_FONT_NAME,
                boldItalic=UNICODE_FONT_BOLD,
            )
            _font_registered = True
            break
    except Exception as exc:  # pragma: no cover - best-effort font registration
        logger.warning("Failed to register font %s / %s: %s", regular, bold, exc)
if not _font_registered:
    UNICODE_FONT_NAME = "Helvetica"

TABLE_BODY_STYLE = ParagraphStyle(
    "table_body",
    fontName=UNICODE_FONT_NAME,
    fontSize=6.2,
    leading=7.2,
    textColor=BRAND_COLORS["muted_text"],
)

SECTOR_OVERVIEW_BODY_STYLE = ParagraphStyle(
    "sector_overview_body",
    parent=TABLE_BODY_STYLE,
    fontSize=8.2,
    leading=9.2,
)

SECTOR_OVERVIEW_CENTER_STYLE = ParagraphStyle(
    "sector_overview_center",
    parent=SECTOR_OVERVIEW_BODY_STYLE,
    alignment=TA_CENTER,
)

SUMMARY_BODY_STYLE = ParagraphStyle(
    "summary_body",
    parent=TABLE_BODY_STYLE,
    fontSize=7.5,
    leading=13,
    alignment=TA_JUSTIFY,
    textColor=BRAND_COLORS["muted_text"],
)

MASTHEAD_STYLE = ParagraphStyle(
    "masthead",
    parent=TABLE_BODY_STYLE,
    fontName="Times-Bold",
    fontSize=28,
    leading=32,
    alignment=TA_CENTER,
    textColor=BRAND_COLORS["primary"],
)

MASTHEAD_DATE_STYLE = ParagraphStyle(
    "masthead_date",
    parent=TABLE_BODY_STYLE,
    fontSize=10,
    italic=True,
    alignment=TA_CENTER,
    textColor=BRAND_COLORS["muted_text"],
)

SCORE_COLOR_BANDS: List = [
    (80, colors.HexColor("#0BA360")),
    (60, colors.HexColor("#6FCF97")),
    (40, colors.HexColor("#F2994A")),
    (0, colors.HexColor("#EB5757")),
]

def wrap_table_text(value) -> Paragraph:
    if pd.isna(value):
        return Paragraph("", TABLE_BODY_STYLE)
    return Paragraph(str(value), TABLE_BODY_STYLE)


def slugify(text: str) -> str:
    return "-".join(filter(None, "".join(ch.lower() if ch.isalnum() else "-" for ch in text).split("-")))


def chunk_table_groups(tables: List[Dict], chunk_size: int = 3):
    for offset in range(0, len(tables), chunk_size):
        chunk = tables[offset : offset + chunk_size]
        yield chunk, offset > 0

COLUMN_RENAMES = {
    "name": "company_name",
    "fundamental_total_score": "total_score",
    "fundamental_value": "pillar_value",
    "fundamental_growth": "pillar_growth",
    "fundamental_quality": "pillar_quality",
    "fundamental_risk": "pillar_risk",
    "fundamental_momentum": "pillar_momentum",
}

REQUIRED_COLUMNS = [
    "ticker",
    "company_name",
    "exchange",
    "sector",
    # "industry",
    "market_cap",
    # "market_cap_category",
    # "beta",
    # "style",
    "total_score",
    "pillar_value",
    "pillar_growth",
    "pillar_quality",
    "pillar_risk",
    "pillar_momentum",
]

SCORE_COLUMNS = {"total_score", "pillar_value", "pillar_growth", "pillar_quality", "pillar_risk", "pillar_momentum"}
MOMENTUM_COLUMN = "pillar_momentum"

TABLE_COLUMN_DEFS = [
    {"key": "ticker", "label": "Ticker", "formatter": lambda v: wrap_table_text(v)},
    {"key": "company_name", "label": "Company name", "formatter": lambda v: wrap_table_text(v)},
    {"key": "sector", "label": "Sector", "formatter": lambda v: wrap_table_text(v)},
    # {"key": "industry", "label": "Industry", "formatter": lambda v: wrap_table_text(v)},
    {"key": "market_cap", "label": "Market cap", "formatter": lambda v: wrap_table_text(format_market_cap(v))},
    # {"key": "market_cap_category", "label": "Market cap category", "formatter": lambda v: wrap_table_text(v)},
    # {"key": "beta", "label": "Beta", "formatter": lambda v: wrap_table_text(format_beta(v))},
    # {"key": "style", "label": "Style", "formatter": lambda v: wrap_table_text(v)},
    {"key": "total_score", "label": "Total", "formatter": lambda v: _format_score_badge(v)},
    {"key": "pillar_value", "label": "P1", "formatter": lambda v: _format_score_badge(v)},
    {"key": "pillar_growth", "label": "P2", "formatter": lambda v: _format_score_badge(v)},
    {"key": "pillar_quality", "label": "P3", "formatter": lambda v: _format_score_badge(v)},
    {"key": "pillar_risk", "label": "P4", "formatter": lambda v: _format_score_badge(v)},
    {"key": "pillar_momentum", "label": "P5", "formatter": lambda v: _format_score_badge(v)},
]

TABLE_COLUMN_WIDTHS = [60, 100, 70, 50, 30, 30, 28, 28, 28, 28, 28]
COLUMN_INDEX = {col["key"]: idx for idx, col in enumerate(TABLE_COLUMN_DEFS)}

METRIC_TABLES = [
    {"metric": "total_score", "title": "Top 5 - Total Fundamental Score"},
    {"metric": "pillar_value", "title": "Top 5 - Value (P1)"},
    {"metric": "pillar_growth", "title": "Top 5 - Growth (P2)"},
    {"metric": "pillar_quality", "title": "Top 5 - Quality (P3)"},
    {"metric": "pillar_risk", "title": "Top 5 - Risk (P4)"},
    {"metric": "pillar_momentum", "title": "Top 5 - Momentum (P5)"},
]

SECTOR_OVERVIEW_ANCHOR = "sector-overview-board"
SUMMARY_PAGE_ANCHOR = "summary-highlights"
SUMMARY_TEXT_FILE = CACHE_DIR / "text_generated.json"
DISCLAIMER_TEXT_FILE = CACHE_DIR / "disclaimer_text.json"
POSITIVE_TEXT_HEX = "#0BA360"
NEGATIVE_TEXT_HEX = "#EB5757"
NEUTRAL_TEXT_HEX = "#425466"
BREADTH_THRESHOLD = 50.0
SECTOR_PULSE_COLUMN_WIDTHS = [170, 80, 80, 80, 80, 50]
CROSS_SECTOR_COLUMN_WIDTHS = [170, 70, 60, 60, 60, 60, 60]
ROCKET_ICON = "&#128640;"  # legacy icon (no longer used but kept for potential reuse)
POSITIVE_BULLET_COLOR = colors.HexColor("#0BA360")
NEGATIVE_BULLET_COLOR = colors.HexColor("#EB5757")
NEUTRAL_BULLET_COLOR = colors.HexColor("#C4CBD6")
SECTOR_SCORE_BADGE_DIAMETER = 18
SECTOR_NOTE_TEXT = (
    "Notă: P1 - Value, P2 - Growth, P3 - Quality, P4 - Risk, P5 - Momentum; "
    'Pentru explicații legate de metodologia de calcul al scoring-ului, puteți consulta '
    '<link href="https://equipicker.com/ro/cum-interpretam-scorurile-din-equipicker/">'
    "<u>acest ghid Equupicker</u></link>."
)
SUMMARY_SECTION_DEFS = [
    ("sector_pulse_snapshot", "Sector Pulse"),
    ("fundamental_heatmap_snapshot", "Fundamental Heatmap"),
    ("how_to_read", "Ce informații puteți găsi în raport"),
]
DISCLAIMER_SECTION_DEFS = [
    ("metodologie", "Metodologie"),
    ("sector_pulse", "Sector Pulse"),
    ("cross_sector", "Cross-Sector Fundamental Scoring"),
    ("top_companies", "Topurile de companii"),
    ("disclaimer", "Disclaimer"),
]
DISCLAIMER_PAGE_BREAK_SECTIONS = {"top_companies"}
SECTOR_SCORE_COLUMNS = [
    ("avg_total_score", "fundamental_total_score", "Total"),
    ("avg_value", "fundamental_value", "P1"),
    ("avg_growth", "fundamental_growth", "P2"),
    ("avg_quality", "fundamental_quality", "P3"),
    ("avg_risk", "fundamental_risk", "P4"),
    ("avg_momentum", "fundamental_momentum", "P5"),
]


def format_market_cap(value) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        billions = float(value) / 1_000_000_000
    except (TypeError, ValueError):
        return "n/a"
    return f"${billions:.1f}B"


def format_beta(value) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def format_score_value(value) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _score_fill_color(value):
    if pd.isna(value):
        return colors.HexColor("#B0BEC5")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = 0
    for threshold, band_color in SCORE_COLOR_BANDS:
        if numeric_value >= threshold:
            return band_color
    return SCORE_COLOR_BANDS[-1][1]


class ScoreBadge(Flowable):
    def __init__(self, value, diameter: int = 22):
        super().__init__()
        self.value = value
        self.diameter = diameter

    def wrap(self, availWidth, availHeight):
        return self.diameter, self.diameter

    def draw(self):
        padding = 1
        radius = (self.diameter / 2) - padding
        fill_color = _score_fill_color(self.value)
        self.canv.setFillColor(fill_color)
        self.canv.setStrokeColor(fill_color)
        cx = cy = self.diameter / 2
        self.canv.circle(cx, cy, radius, stroke=0, fill=1)
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Helvetica-Bold", 7.2)
        text = format_score_value(self.value)
        text_width = self.canv.stringWidth(text, "Helvetica-Bold", 7.2)
        self.canv.drawString(cx - (text_width / 2), cy - 3, text)


def _format_score_badge(value, diameter: int = 22):
    return ScoreBadge(value, diameter=diameter)


class SectorPulseBullet(Flowable):
    def __init__(self, fill_color: colors.Color, diameter: int = 14):
        super().__init__()
        self.fill_color = fill_color
        self.diameter = diameter

    def wrap(self, availWidth, availHeight):
        return self.diameter, self.diameter

    def draw(self):
        radius = self.diameter / 2
        self.canv.setFillColor(self.fill_color)
        self.canv.setStrokeColor(self.fill_color)
        self.canv.circle(radius, radius, radius - 1, stroke=0, fill=1)


class MiniScoreDot(Flowable):
    def __init__(self, fill_color: colors.Color, diameter: int = 10):
        super().__init__()
        self.fill_color = fill_color
        self.diameter = diameter

    def wrap(self, availWidth, availHeight):
        return self.diameter, self.diameter

    def draw(self):
        radius = self.diameter / 2
        self.canv.setFillColor(self.fill_color)
        self.canv.setStrokeColor(self.fill_color)
        self.canv.circle(radius, radius, radius - 1, stroke=0, fill=1)


def prepare_scoring_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAMES).copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df["company_name"] = df["company_name"].fillna("")
    df["sector"] = df["sector"].fillna("Unspecified")
    df["industry"] = df["industry"].fillna("Unspecified")
    df["style"] = df["style"].fillna("-")
    return df


def shortlist_top(df: pd.DataFrame, metric: str, top_n: int = 5) -> pd.DataFrame:
    if metric not in df.columns:
        logger.warning("Metric '%s' missing from dataframe.", metric)
        return pd.DataFrame(columns=df.columns)
    filtered = df.dropna(subset=[metric])
    if filtered.empty:
        return filtered
    sort_cols = [metric]
    ascending = [False]
    if metric == MOMENTUM_COLUMN:
        sort_cols.append("market_cap")
        ascending.append(False)
    else:
        if MOMENTUM_COLUMN in filtered.columns:
            sort_cols.append(MOMENTUM_COLUMN)
            ascending.append(False)
        if "market_cap" in filtered.columns:
            sort_cols.append("market_cap")
            ascending.append(False)
    shortlisted = filtered.sort_values(sort_cols, ascending=ascending).head(top_n)
    return shortlisted.reset_index(drop=True)


def build_report_pages(df: pd.DataFrame) -> List[Dict]:
    pages: List[Dict] = []
    if df.empty:
        return pages

    pages.append({
        "title": "Entire Universe",
        "scope": "Universe - Top Scoring Stocks",
        "tables": build_tables_for_slice(df)
    })

    for sector in sorted(df["sector"].dropna().unique()):
        sector_df = df[df["sector"] == sector]
        if sector_df.empty:
            logger.warning("Skipping sector '%s' - no rows returned.", sector)
            continue
        pages.append({
            "title": sector,
            "scope": f"Sector - {sector}",
            "tables": build_tables_for_slice(sector_df)
        })
    return pages


def build_tables_for_slice(df: pd.DataFrame) -> List[Dict]:
    tables = []
    for table_meta in METRIC_TABLES:
        tables.append({
            "title": table_meta["title"],
            "metric": table_meta["metric"],
            "data": shortlist_top(df, table_meta["metric"])
        })
    return tables


def _format_percentage_value(value) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_ratio(success_count: int, total_count: int) -> Tuple[str, Optional[float]]:
    if not total_count:
        return "N/A", np.nan
    pct_value = round((success_count / total_count) * 100, 0)
    return f"{pct_value:.0f}%", pct_value


def _prepare_sector_overview_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    working = df.copy()
    working["sector"] = working.get("sector", pd.Series(dtype=str)).fillna("Unspecified")
    one_m_close = pd.to_numeric(working.get("1m_close"), errors="coerce")
    eod_price = pd.to_numeric(working.get("eod_price_used"), errors="coerce")
    ic_eod_price = pd.to_numeric(working.get("ic_eod_price_used"), errors="coerce")
    market_cap = pd.to_numeric(working.get("market_cap"), errors="coerce")

    valid_override = (
        eod_price.notna()
        & ic_eod_price.notna()
        & (ic_eod_price != 0)
    )
    market_cap_current = np.where(
        valid_override,
        (eod_price / ic_eod_price) * market_cap,
        market_cap,
    )
    market_cap_current = pd.Series(market_cap_current, index=working.index)
    market_cap_current_num = pd.to_numeric(market_cap_current, errors="coerce")

    valid_prices = (
        one_m_close.notna()
        & eod_price.notna()
        & market_cap_current_num.notna()
        & (one_m_close != 0)
        & (eod_price != 0)
    )

    working["market_cap_numeric"] = market_cap_current_num
    working["1m_market_cap"] = np.where(
        valid_prices,
        (one_m_close / eod_price) * market_cap_current_num,
        np.nan,
    )
    working["1m_price_var_pct_num"] = np.where(
        valid_prices,
        (eod_price / one_m_close - 1.0) * 100.0,
        np.nan,
    )
    working["1m_price_var_pct"] = working["1m_price_var_pct_num"].apply(_format_percentage_value)
    return working


def compute_sector_overview_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    processed = _prepare_sector_overview_dataframe(df)
    results: List[Dict] = []
    for sector, group in processed.groupby("sector", dropna=False):
        overlap_mask = group["market_cap_numeric"].notna() & group["1m_market_cap"].notna()
        sum_current_cap = group.loc[overlap_mask, "market_cap_numeric"].sum()
        sum_prior_cap = group.loc[overlap_mask, "1m_market_cap"].sum()
        if sum_prior_cap and not pd.isna(sum_prior_cap):
            sector_var_num = round((sum_current_cap / sum_prior_cap - 1.0) * 100.0, 2)
            sector_var = f"{sector_var_num:.2f}%"
        else:
            sector_var_num = np.nan
            sector_var = "N/A"

        valid_var_mask = group["1m_price_var_pct_num"].notna()
        total_var_count = int(valid_var_mask.sum())
        positive_var = int((group.loc[valid_var_mask, "1m_price_var_pct_num"] > 0).sum())
        market_breadth, market_breadth_num = _format_ratio(positive_var, total_var_count)

        rs_monthly = pd.to_numeric(group.get("rs_monthly"), errors="coerce")
        valid_rs_mask = rs_monthly.notna()
        rs_success = int(((rs_monthly > 0) & valid_rs_mask).sum())
        total_rs = int(valid_rs_mask.sum())
        rs_breadth, rs_breadth_num = _format_ratio(rs_success, total_rs)

        obvm_monthly = pd.to_numeric(group.get("obvm_monthly"), errors="coerce")
        valid_obvm_mask = obvm_monthly.notna()
        obvm_success = int(((obvm_monthly > 0) & valid_obvm_mask).sum())
        total_obvm = int(valid_obvm_mask.sum())
        obvm_breadth, obvm_breadth_num = _format_ratio(obvm_success, total_obvm)

        score_avgs = {}
        for target_col, source_col, _ in SECTOR_SCORE_COLUMNS:
            col_values = pd.to_numeric(group.get(source_col), errors="coerce")
            mean_value = col_values.mean()
            score_avgs[target_col] = round(mean_value, 2) if pd.notna(mean_value) else np.nan

        signal_flag = (
            _breadth_exceeds_threshold(market_breadth_num)
            and _breadth_exceeds_threshold(rs_breadth_num)
            and _breadth_exceeds_threshold(obvm_breadth_num)
        )

        results.append({
            "sector": sector,
            "sector_1m_var_pct": sector_var,
            "sector_1m_var_pct_num": sector_var_num,
            "market_breadth": market_breadth,
            "market_breadth_num": market_breadth_num,
            "rs_breadth": rs_breadth,
            "rs_breadth_num": rs_breadth_num,
            "obvm_breadth": obvm_breadth,
            "obvm_breadth_num": obvm_breadth_num,
            "signal": signal_flag,
            **score_avgs,
        })

    sector_df = pd.DataFrame(results)
    if not sector_df.empty:
        sector_df = sector_df.sort_values(
            by="sector_1m_var_pct_num",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    return sector_df


def _build_sector_anchor_map(pages: List[Dict]) -> Dict[str, str]:
    anchors: Dict[str, str] = {}
    for page in pages:
        if page["title"] == "Entire Universe":
            continue
        anchors[page["title"]] = slugify(page["title"])
    return anchors


def _sector_label_cell(sector: str, anchor: Optional[str]) -> Paragraph:
    label = escape(str(sector or "Unspecified"))
    if anchor:
        return Paragraph(f'<link href="#{anchor}">{label}</link>', SECTOR_OVERVIEW_BODY_STYLE)
    return Paragraph(label, SECTOR_OVERVIEW_BODY_STYLE)


def _colored_value_cell(text: str, color_hex: Optional[str], centered: bool = False) -> Paragraph:
    safe_text = escape(text or "N/A")
    if color_hex:
        safe_text = f'<font color="{color_hex}">{safe_text}</font>'
    style = SECTOR_OVERVIEW_CENTER_STYLE if centered else SECTOR_OVERVIEW_BODY_STYLE
    return Paragraph(safe_text, style)


def _variation_color(value: Optional[float]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if value > 0:
        return POSITIVE_TEXT_HEX
    if value < 0:
        return NEGATIVE_TEXT_HEX
    return None


def _breadth_color(value: Optional[float]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return POSITIVE_TEXT_HEX if value > BREADTH_THRESHOLD else NEGATIVE_TEXT_HEX


def _breadth_exceeds_threshold(value: Optional[float]) -> bool:
    return value is not None and not pd.isna(value) and value > BREADTH_THRESHOLD


def _resolve_signal(color_codes: List[Optional[str]]) -> Tuple[str, colors.Color]:
    cleaned = [code for code in color_codes if code]
    if len(cleaned) != len(color_codes):
        return "mixed", NEUTRAL_BULLET_COLOR
    all_positive = all(code == POSITIVE_TEXT_HEX for code in cleaned)
    all_negative = all(code == NEGATIVE_TEXT_HEX for code in cleaned)
    if all_positive:
        return "positive", POSITIVE_BULLET_COLOR
    if all_negative:
        return "negative", NEGATIVE_BULLET_COLOR
    return "mixed", NEUTRAL_BULLET_COLOR


def _build_signal_cell(color_codes: List[Optional[str]]) -> Flowable:
    _, fill = _resolve_signal(color_codes)
    return SectorPulseBullet(fill)


def export_sector_stats_json(sector_stats: pd.DataFrame, report_date: date) -> Optional[Path]:
    output = CACHE_DIR / f"sector_data_json_{report_date.isoformat()}.json"
    txt_output = CACHE_DIR / f"sector_data_tables_{report_date.isoformat()}.txt"
    try:
        if sector_stats is None or sector_stats.empty:
            payload: List[Dict] = []
            table_text = ""
        else:
            filtered = sector_stats[[col for col in sector_stats.columns if not str(col).endswith("_num")]]
            rename_map = {
                "avg_total_score": "total_score",
                "avg_value": "score_value",
                "avg_growth": "score_growth",
                "avg_quality": "score_quality",
                "avg_risk": "score_risk",
                "avg_momentum": "score_momentum",
            }
            export_df = filtered.rename(columns=rename_map)
            payload = json.loads(export_df.to_json(orient="records"))
            lines = ["<TABLES>"]
            lines.append("<SECTOR_PULSE>")
            lines.append("Sector | 1m_market_cap_var | market_breadth | rel_perf_breadth | rel_vol_breadth | signal")
            for idx, row in export_df.iterrows():
                original_row = sector_stats.loc[idx]
                signal_status, _ = _resolve_signal([
                    _variation_color(original_row.get("sector_1m_var_pct_num")),
                    _breadth_color(original_row.get("market_breadth_num")),
                    _breadth_color(original_row.get("rs_breadth_num")),
                    _breadth_color(original_row.get("obvm_breadth_num")),
                ])
                lines.append(
                    f"{row.get('sector','')} | {row.get('sector_1m_var_pct','')} | "
                    f"{row.get('market_breadth','')} | {row.get('rs_breadth','')} | "
                    f"{row.get('obvm_breadth','')} | {signal_status}"
                )
            lines.append("</SECTOR_PULSE>")
            lines.append("")
            lines.append("<CROSS_SECTOR_FUNDAMENTAL>")
            lines.append("Sector | total_score | score_value | score_growth | score_quality | score_risk | score_momentum")
            for _, row in export_df.iterrows():
                lines.append(
                    f"{row.get('sector','')} | {row.get('total_score','')} | {row.get('score_value','')} | "
                    f"{row.get('score_growth','')} | {row.get('score_quality','')} | "
                    f"{row.get('score_risk','')} | {row.get('score_momentum','')}"
                )
            lines.append("</CROSS_SECTOR_FUNDAMENTAL>")
            lines.append("</TABLES>")
            table_text = "\n".join(lines)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if table_text:
            txt_output.write_text(table_text, encoding="utf-8")
        return output
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to export sector stats JSON: %s", exc)
        return None


def load_or_initialize_summary_text() -> Dict[str, List[str]]:
    defaults = {key: ["TBD"] for key, _ in SUMMARY_SECTION_DEFS}
    path = SUMMARY_TEXT_FILE
    data: Dict[str, List[str]]
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = defaults.copy()
        except Exception:
            data = defaults.copy()
    else:
        data = defaults.copy()

    updated = False
    for key in defaults:
        if key not in data or not isinstance(data[key], list) or not data[key]:
            data[key] = ["TBD"]
            updated = True

    if updated or not path.exists():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def load_or_initialize_disclaimer_text() -> Dict[str, List[str]]:
    defaults = {
        "metodologie": [
            "În această secțiune îți explicăm, pe scurt și simplu, cum să înțelegi datele și scorurile pe care le-ai găsit în raportul nostru. Tot ce vezi aici este rezultatul unei metodologii proprii Equipicker, creată special pentru investitorii care vor să se orienteze rapid într-o piață dinamică, fără să piardă vremea cu zgomotul de fundal.",
        ],
        "sector_pulse": [
            "Tabelul Sector Pulse îți arată cum se mișcă fiecare sector din piață, dintr-o privire. Iată ce înseamnă fiecare coloană:",
            "• <b>1-month %</b> – reprezintă variația capitalizării sectoriale din ultima lună. Dacă procentul e pozitiv, sectorul a crescut în ultimele 30 de zile.",
            "• <b>Market Breadth</b> – procentul de companii din sector care au înregistrat creșteri în ultima lună. Un nivel de peste 50% înseamnă că majoritatea companiilor „trag sectorul în sus”.",
            "• <b>Relative Performance Breadth</b> – câte dintre companiile sectorului performează mai bine decât S&P 500 în prezent.",
            "• <b>Relative Volume Breadth</b> – câte companii înregistrează intrări de bani în prezent.",
            "Culorile de lângă fiecare sector sunt o formă rapidă de codificare vizuală:",
            "• <font color=\"#0BA360\"><b><font size=\"15\">●</font> Verde</b></font> – toți indicatorii sunt bullish: sectorul a crescut în ultima lună, peste jumătate din companii au performanță mai bună decât piața și volum pozitiv (intrări de bani).",
            "• <font color=\"#EB5757\"><b><font size=\"15\">●</font> Roșu</b></font> – toți indicatorii sunt bearish: sectorul a scăzut, majoritatea companiilor subperformează, iar volumele sunt predominant negative (ieșiri de bani).",
            "• <font color=\"#A0A8B5\"><b><font size=\"15\">●</font> Gri</b></font> – situație mixtă, fără o direcție clară.",
            "Sectoarele sunt afișate în ordine descrescătoare în funcție de variația lunară a capitalizării bursiere.",
        ],
        "cross_sector": [
            "Aici ai o imagine de ansamblu asupra scorurilor fundamentale pentru fiecare sector – atât scorul total, cât și scorurile pe piloni specifici. Cu cât scorul este mai mare, cu atât sectorul stă mai bine la acel capitol. Cadrăm automat cele mai ridicate valori pentru a le evidenția vizual.",
            "La ce se referă acești piloni? Sunt 5 piloni de analiză fundamentală, gândiți pentru a acoperi toate unghiurile importante când analizezi o companie:",
            "• <b>Value</b> – dacă acțiunea este scumpă sau ieftină în raport cu alte companii similare, prin prisma multiplilor de evaluare relativă și a consistenței politicilor de remunerare a acționarilor.",
            "• <b>Growth</b> – cât de repede scalează afacerea: vânzări, profituri, marje.",
            "• <b>Quality</b> – stabilitate și eficiență în utilizarea capitalului, sustenabilitatea profiturilor și capacitatea de conversie a acestora în cash.",
            "• <b>Risk</b> – risc asociat bilanțului: gradul de îndatorare, riscul de lichiditate și cât de robustă este structura financiară.",
            "• <b>Momentum</b> – soliditatea rezultatelor din trimestrul cel mai recent în ceea ce privește creșterea veniturilor, profitabilitatea și marjele.",
            "Dacă vrei să aprofundezi cum se calculează fiecare scor și ce semnificație are, găsești detalii complete pe pagina noastră: &#128073; <link href=\"https://equipicker.com/ro/cum-interpretam-scorurile-din-equipicker/\"><u>https://equipicker.com/ro/cum-interpretam-scorurile-din-equipicker/</u></link>",
        ],
        "top_companies": [
            "În paginile dedicate topurilor ai cele mai bine cotate companii din perspectiva scoring-urilor fundamentale calculate prin metodologia Equipicker, atât la nivelul întregii piețe (universul Equipicker = S&P 500 + Nasdaq 100 + companii cu lichiditate mai mică, dar relevante), cât și pentru fiecare sector în parte. Pentru fiecare grup, îți prezentăm:",
            "• Top 5 companii după scorul total;",
            "• Top 5 pentru fiecare pilon fundamental.",
        ],
        "disclaimer": [
            "Lista companiilor prezentate are scop informativ. Nu este o recomandare de investiție, ci un instrument de selecție și orientare. Tu ești cel care decide ce analizează mai departe și în ce investește – noi îți oferim doar un punct de plecare structurat, eficient și ușor de folosit.",
        ],
    }
    path = DISCLAIMER_TEXT_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = defaults.copy()
        except Exception:
            data = defaults.copy()
    else:
        data = defaults.copy()

    updated = False
    for key, value in defaults.items():
        if key not in data or not isinstance(data[key], list) or not data[key]:
            data[key] = value
            updated = True
    if updated or not path.exists():
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def _extract_score_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _build_sector_pulse_table(sector_stats: pd.DataFrame, anchors: Dict[str, str]) -> Table:
    headers = [
        "Sector",
        "1-month %",
        "Mk Breadth",
        "Rel. Perf. Breadth",
        "Rel Vol. Breadth",
        "",
    ]
    table_data: List[List] = [headers]
    span_empty = False

    if sector_stats.empty:
        table_data.append(
            [Paragraph("No sector data available.", SECTOR_OVERVIEW_BODY_STYLE)]
            + [Paragraph("", SECTOR_OVERVIEW_BODY_STYLE) for _ in range(5)]
        )
        span_empty = True
    else:
        for _, row in sector_stats.iterrows():
            anchor = anchors.get(row["sector"])
            variation_color = _variation_color(row["sector_1m_var_pct_num"])
            market_color = _breadth_color(row["market_breadth_num"])
            rs_color = _breadth_color(row["rs_breadth_num"])
            obvm_color = _breadth_color(row["obvm_breadth_num"])
            table_data.append([
                _sector_label_cell(row["sector"], anchor),
                _colored_value_cell(row["sector_1m_var_pct"], variation_color, centered=True),
                _colored_value_cell(row["market_breadth"], market_color, centered=True),
                _colored_value_cell(row["rs_breadth"], rs_color, centered=True),
                _colored_value_cell(row["obvm_breadth"], obvm_color, centered=True),
                _build_signal_cell([variation_color, market_color, rs_color, obvm_color]),
            ])

    table = Table(
        table_data,
        colWidths=SECTOR_PULSE_COLUMN_WIDTHS,
        repeatRows=1,
        hAlign="LEFT",
    )
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLORS["primary"]),
        ("ALIGN", (1, 0), (-2, 0), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (4, -1), "CENTER"),
        ("ALIGN", (5, 1), (5, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BOX", (0, 0), (4, -1), 0.25, colors.HexColor("#D3E1EA")),
        ("INNERGRID", (0, 0), (4, -1), 0.25, colors.HexColor("#D3E1EA")),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BACKGROUND", (5, 0), (5, 0), colors.white),
        ("BACKGROUND", (5, 1), (5, -1), colors.white),
        ("TEXTCOLOR", (5, 0), (5, 0), colors.white),
    ]
    for row_idx in range(1, len(table_data)):
        bg = BRAND_COLORS["row_alt"] if row_idx % 2 else colors.white
        style_cmds.append(("BACKGROUND", (0, row_idx), (4, row_idx), bg))
    if span_empty:
        style_cmds.append(("SPAN", (0, 1), (5, 1)))
    table.setStyle(TableStyle(style_cmds))
    return table


def _build_cross_sector_table(sector_stats: pd.DataFrame, anchors: Dict[str, str]) -> Table:
    headers = ["Sector"] + [col[2] for col in SECTOR_SCORE_COLUMNS]
    table_data: List[List] = [headers]
    span_empty = False
    if sector_stats.empty:
        table_data.append(
            [Paragraph("No sector scoring data available.", SECTOR_OVERVIEW_BODY_STYLE)]
            + [Paragraph("", SECTOR_OVERVIEW_BODY_STYLE) for _ in range(len(headers) - 1)]
        )
        span_empty = True
    else:
        for _, row in sector_stats.iterrows():
            anchor = anchors.get(row["sector"])
            row_values: List = [_sector_label_cell(row["sector"], anchor)]
            for target_col, _, _ in SECTOR_SCORE_COLUMNS:
                row_values.append(_format_score_badge(row.get(target_col), diameter=SECTOR_SCORE_BADGE_DIAMETER))
            table_data.append(row_values)

    table = Table(
        table_data,
        colWidths=CROSS_SECTOR_COLUMN_WIDTHS,
        repeatRows=1,
        hAlign="LEFT",
    )
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLORS["primary"]),
        ("ALIGN", (1, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D3E1EA")),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
    ]
    for row_idx in range(1, len(table_data)):
        bg = BRAND_COLORS["row_alt"] if row_idx % 2 else colors.white
        style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))
    if span_empty:
        style_cmds.append(("SPAN", (0, 1), (-1, 1)))
    else:
        for offset, (target_col, _, _) in enumerate(SECTOR_SCORE_COLUMNS, start=1):
            col_values = pd.to_numeric(sector_stats[target_col], errors="coerce")
            valid_values = col_values.dropna()
            if valid_values.empty:
                continue
            max_value = valid_values.max()
            max_rows = col_values[col_values == max_value].index.tolist()
            for row_idx in max_rows:
                table_row = row_idx + 1
                style_cmds.extend(
                    [
                        ("BOX", (offset, table_row), (offset, table_row), 1.5, HIGHLIGHT_COLOR),
                        ("LINEBEFORE", (offset, table_row), (offset, table_row), 1.5, HIGHLIGHT_COLOR),
                        ("LINEAFTER", (offset, table_row), (offset, table_row), 1.5, HIGHLIGHT_COLOR),
                    ]
                )
    table.setStyle(TableStyle(style_cmds))
    return table


def build_sector_overview_page(
    styles: Dict[str, ParagraphStyle],
    sector_stats: pd.DataFrame,
    anchors: Dict[str, str],
    include_note: bool = True,
    eod_date: Optional[str] = None,
) -> List:
    flowables: List = []
    flowables.extend(
        _build_scope_title(
            styles,
            "Sector Overview Board",
            anchor=SECTOR_OVERVIEW_ANCHOR,
            arrow_target=None,
        )
    )
    flowables.append(_table_title_band("Sector Pulse", styles))
    flowables.append(Spacer(1, 4))
    flowables.append(_build_sector_pulse_table(sector_stats, anchors))
    if include_note:
        note_text = (
            f"Notă: Variația 1-month și Market breadth sunt calculate pe baza evoluției capitalizării companiilor în ultimele 30 de zile față de data de referință {eod_date or 'latest close'}; "
            f"Indicatorii Relative Performance Breadth și Relative Volume Breadth sunt calculați la data de {eod_date or 'latest close'} pe baza seriilor monthly de preț și volum."
        )
        flowables.append(Spacer(1, 3))
        flowables.append(Paragraph(note_text, styles["table_note"]))
    flowables.append(Spacer(1, 10))
    flowables.append(_table_title_band("Cross-Sector Fundamental Scoring", styles))
    flowables.append(Spacer(1, 4))
    flowables.append(_build_cross_sector_table(sector_stats, anchors))
    if include_note:
        flowables.append(Spacer(1, 2))
        flowables.append(Paragraph(SECTOR_NOTE_TEXT, styles["table_note"]))
    return flowables


def build_summary_page(
    styles: Dict[str, ParagraphStyle],
    summary_lines: Dict[str, List[str]],
    report_date: date,
) -> List:
    flowables: List = []

    flowables.append(Spacer(1, 26))
    if BANNER_PATH.exists():
        banner = Image(str(BANNER_PATH), width=CONTENT_WIDTH, height=BANNER_HEIGHT)
        flowables.append(banner)
        flowables.append(Spacer(1, 20))
    else:
        flowables.append(Spacer(1, 10))
    masthead = Table(
        [[Paragraph("Equipicker Scoring Board Report", ParagraphStyle(
            "masthead_band",
            parent=styles["scope"],
            alignment=TA_CENTER,
            textColor=colors.white,
            fontSize=16,
            fontName="Helvetica-Bold",
        ))]],
        colWidths=[CONTENT_WIDTH],
    )
    masthead.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SECTION_BAND_COLOR),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    flowables.append(masthead)
    flowables.append(Paragraph(report_date.strftime("%B %d, %Y"), styles["masthead_date"]))
    flowables.append(Spacer(1, 18))

    def _card(title: str, lines: List[str], include_dots: bool = False, width: float = CONTENT_WIDTH / 2 - 6):
        header = _table_title_band(title, styles, width=width)
        body_rows = []
        entries = lines or ["TBD"]
        for entry in entries:
            if include_dots:
                score = _extract_score_from_text(entry)
                color = _score_fill_color(score) if score is not None else colors.HexColor("#B0BEC5")
                body_rows.append([MiniScoreDot(color, diameter=10), Paragraph(entry, styles["summary_body"])])
            else:
                body_rows.append([Paragraph(f"• {entry}", styles["summary_body"])])
        if include_dots:
            body_table = Table(body_rows, colWidths=[12, width - 20])
            body_style = TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        else:
            body_table = Table([[row[0]] for row in body_rows], colWidths=[width - 8])
            body_style = TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        body_table.setStyle(body_style)
        card = Table([[header], [Spacer(1, 6)], [body_table]], colWidths=[width])
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        _, card_height = card.wrap(width, 0)
        return card, card_height

    half_width = CONTENT_WIDTH / 2 - 8
    left_card, left_height = _card(
        "Sector Pulse",
        summary_lines.get("sector_pulse_snapshot", []),
        include_dots=False,
        width=half_width,
    )
    right_card, right_height = _card(
        "Fundamental Heatmap",
        summary_lines.get("fundamental_heatmap_snapshot", []),
        include_dots=False,
        width=half_width,
    )
    max_card_height = max(left_height, right_height)
    left_cell = KeepInFrame(half_width, max_card_height, [left_card], hAlign="LEFT", vAlign="TOP")
    right_cell = KeepInFrame(half_width, max_card_height, [right_card], hAlign="LEFT", vAlign="TOP")
    cards_table = Table(
        [[left_cell, right_cell]],
        colWidths=[CONTENT_WIDTH / 2 - 6, CONTENT_WIDTH / 2 - 6],
        rowHeights=[max_card_height],
    )
    cards_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("BOX", (0, 0), (0, 0), 0.75, colors.HexColor("#D3E1EA")),
                ("BOX", (1, 0), (1, 0), 0.75, colors.HexColor("#D3E1EA")),
            ]
        )
    )
    flowables.append(cards_table)
    flowables.append(Spacer(1, 3))

    rule = Table([[""]], colWidths=[CONTENT_WIDTH])
    rule.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.6, colors.HexColor("#D3E1EA")),
            ]
        )
    )
    flowables.append(rule)
    flowables.append(Spacer(1, 10))

    how_to_read_card, _ = _card(
        "Ce informații puteți găsi în raport",
        summary_lines.get("how_to_read", []),
        include_dots=False,
        width=CONTENT_WIDTH,
    )
    flowables.append(how_to_read_card)
    return flowables


def _build_styles():
    sample = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=BRAND_COLORS["primary"],
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=sample["Heading2"],
            fontName="Helvetica",
            fontSize=12,
            textColor=BRAND_COLORS["muted_text"],
            spaceAfter=2,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=sample["Normal"],
            fontSize=10,
            textColor=BRAND_COLORS["muted_text"],
            spaceAfter=4,
        ),
        "toc_entry": ParagraphStyle(
            "toc_entry",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=12,
            textColor=BRAND_COLORS["primary"],
            spaceAfter=4,
        ),
        "button": ParagraphStyle(
            "button",
            parent=sample["Heading4"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=SCORE_ARROW_COLOR,
            alignment=TA_CENTER,
            backColor=None,
            borderColor=None,
            borderWidth=0,
            borderPadding=(0, 0, 0, 0),
            spaceAfter=0,
        ),
        "scope": ParagraphStyle(
            "scope",
            parent=sample["Heading3"],
            fontSize=13,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            spaceAfter=6,
        ),
        "table_title": ParagraphStyle(
            "table_title",
            parent=sample["Heading4"],
            fontSize=10.5,
            fontName=UNICODE_FONT_BOLD,
            textColor=BRAND_COLORS["primary"],
            spaceAfter=0,
        ),
        "summary_body": SUMMARY_BODY_STYLE,
        "masthead": MASTHEAD_STYLE,
        "masthead_date": MASTHEAD_DATE_STYLE,
        "disclaimer_title": ParagraphStyle(
            "disclaimer_title",
            parent=sample["Heading3"],
            fontName=UNICODE_FONT_NAME,
            fontSize=12,
            textColor=colors.white,
        ),
        "table_note": ParagraphStyle(
            "table_note",
            parent=sample["BodyText"],
            fontName=UNICODE_FONT_NAME,
            fontSize=6,
            leading=8,
            textColor=BRAND_COLORS["muted_text"],
        ),
        "disclaimer_body": ParagraphStyle(
            "disclaimer_body",
            parent=sample["BodyText"],
            fontName=UNICODE_FONT_NAME,
            fontSize=10,
            leading=13,
            textColor=BRAND_COLORS["muted_text"],
        ),
    }
    return styles


def _table_title_band(title: str, styles: Dict[str, ParagraphStyle], width: float = CONTENT_WIDTH) -> Table:
    band = Table([[Paragraph(title, styles["table_title"])]], colWidths=[width])
    band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TABLE_BAND_COLOR),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return band


def _disclaimer_title_band(title: str, styles: Dict[str, ParagraphStyle], width: float = CONTENT_WIDTH) -> Table:
    band = Table([[Paragraph(title, styles["disclaimer_title"])]], colWidths=[width])
    band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SECTION_BAND_COLOR),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return band


def _build_table_block(table_info: Dict, styles: Dict[str, ParagraphStyle]) -> KeepTogether:
    table = _create_data_table(table_info["data"], table_info["metric"])
    title_band = _table_title_band(table_info["title"], styles)
    return KeepTogether([
        title_band,
        Spacer(1, 4),
        table,
        Spacer(1, 18),
    ])


def _create_data_table(df: pd.DataFrame, metric: str) -> Table:
    table_data = [[col["label"] for col in TABLE_COLUMN_DEFS]]

    if df.empty:
        table_data.append([column["formatter"](pd.NA) for column in TABLE_COLUMN_DEFS])
    else:
        for _, row in df.iterrows():
            row_values = []
            for column in TABLE_COLUMN_DEFS:
                raw_value = row.get(column["key"])
                formatted = column["formatter"](raw_value)
                row_values.append(formatted)
            table_data.append(row_values)

    table = Table(table_data, colWidths=TABLE_COLUMN_WIDTHS, repeatRows=1)
    style_cmds = _table_style_commands(len(table_data), metric)
    table.setStyle(TableStyle(style_cmds))
    return table



def _table_style_commands(row_count: int, metric: str) -> List:
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLORS["primary"]),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 6),
        ("TEXTCOLOR", (0, 1), (-1, -1), BRAND_COLORS["muted_text"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D3E1EA")),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 2),
    ]

    for row in range(1, row_count):
        bg = BRAND_COLORS["row_alt"] if row % 2 else colors.white
        commands.append(("BACKGROUND", (0, row), (-1, row), bg))

    highlight_idx = COLUMN_INDEX.get(metric)
    if highlight_idx is not None:
        commands.extend(
            [
                ("BACKGROUND", (highlight_idx, 0), (highlight_idx, 0), HIGHLIGHT_COLOR),
                ("TEXTCOLOR", (highlight_idx, 0), (highlight_idx, 0), colors.white),
                ("BOX", (highlight_idx, 0), (highlight_idx, -1), 1.5, HIGHLIGHT_COLOR),
                ("LINEBEFORE", (highlight_idx, 0), (highlight_idx, -1), 1.5, HIGHLIGHT_COLOR),
                ("LINEAFTER", (highlight_idx, 0), (highlight_idx, -1), 1.5, HIGHLIGHT_COLOR),
            ]
        )

    # Alignment tweaks
    for key in ["market_cap", "style"]:
        idx = COLUMN_INDEX.get(key)
        if idx is not None:
            align = "CENTER" if key != "market_cap" else "RIGHT"
            commands.append(("ALIGN", (idx, 1), (idx, -1), align))
    for key in SCORE_COLUMNS:
        idx = COLUMN_INDEX.get(key)
        if idx is not None:
            commands.append(("ALIGN", (idx, 1), (idx, -1), "CENTER"))
    return commands


def _build_page_header(
    styles: Dict[str, ParagraphStyle],
    report_date: date,
    scope_label: str,
    anchor: Optional[str] = None,
    arrow_target: Optional[str] = SECTOR_OVERVIEW_ANCHOR,
) -> List:
    header: List = []
    label_text = f'<a name="{anchor}"/>{scope_label}' if anchor else scope_label
    arrow = Paragraph(
        f'<link href="#{arrow_target}">&#9650;</link>',
        styles["button"],
    ) if arrow_target else Paragraph("", styles["button"])
    header_table = Table(
        [[Paragraph(label_text, styles["scope"]), arrow]],
        colWidths=[CONTENT_WIDTH - 50, 50],
    )
    band.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("BACKGROUND", (0, 0), (0, 0), SECTION_BAND_COLOR),
                ("BACKGROUND", (1, 0), (1, 0), SECTION_BAND_COLOR),
            ]
        )
    )
    header.append(header_table)
    header.append(Spacer(1, 18))
    return header


def _build_tables_flowables(tables: List[Dict], styles: Dict[str, ParagraphStyle]) -> List:
    flowables: List = []
    for table_info in tables:
        flowables.append(_build_table_block(table_info, styles))
        flowables.append(Spacer(1, 6))
    return flowables


def _build_scope_title(
    styles: Dict[str, ParagraphStyle],
    scope_label: str,
    anchor: Optional[str] = None,
    arrow_target: Optional[str] = SECTOR_OVERVIEW_ANCHOR,
) -> List:
    label_text = f'<a name="{anchor}"/>{scope_label}' if anchor else scope_label
    arrow = Paragraph(
        f'<link href="#{arrow_target}">&#9650;</link>',
        styles["button"],
    ) if arrow_target else Paragraph("", styles["button"])
    band = Table(
        [[Paragraph(label_text, styles["scope"]), arrow]],
        colWidths=[CONTENT_WIDTH - 50, 50],
    )
    band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (1, 0), SECTION_BAND_COLOR),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    # 10pt spacer above band, 6pt below – tweak if needed
    return [Spacer(1, 20), band, Spacer(1, 20)]


def _build_disclaimer(styles: Dict[str, ParagraphStyle], report_date: date) -> List:
    content = load_or_initialize_disclaimer_text()
    flowables: List = []
    flowables.append(Spacer(1, 20))
    for key, title in DISCLAIMER_SECTION_DEFS:
        if key in DISCLAIMER_PAGE_BREAK_SECTIONS and flowables:
            flowables.append(PageBreak())
            flowables.append(Spacer(1, 20))
        else:
            flowables.append(Spacer(1, 6))
        flowables.append(_disclaimer_title_band(title, styles))
        paragraphs = content.get(key, ["TBD"])
        for paragraph in paragraphs:
            flowables.append(Paragraph(paragraph, styles["disclaimer_body"]))
            flowables.append(Spacer(1, 4))
    return flowables


def make_header_footer(report_date: date):
    def _draw(canvas, doc):
        canvas.saveState()
        header_y = PAGE_SIZE[1] - TOP_MARGIN + 20

        if doc.page == 1:
            title_y = header_y - 8
            if LOGO_PATH.exists():
                logo_width = 120
                logo_height = 70
                canvas.drawImage(
                    str(LOGO_PATH),
                    PAGE_SIZE[0] - RIGHT_MARGIN - logo_width,
                    title_y - (logo_height - 30),
                    width=logo_width,
                    height=logo_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            else:
                canvas.setFont("Helvetica-Bold", 12)
                canvas.setFillColor(BRAND_COLORS["secondary"])
                canvas.drawRightString(
                    PAGE_SIZE[0] - RIGHT_MARGIN,
                    title_y,
                    "EQUIPICKER",
                )
        else:
            title_y = header_y - 8
            canvas.setFont("Times-BoldItalic", 12)
            canvas.setFillColor(BRAND_COLORS["primary"])
            canvas.drawString(LEFT_MARGIN, title_y, REPORT_NAME)

            date_y = title_y - 10
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(BRAND_COLORS["muted_text"])
            canvas.drawString(LEFT_MARGIN, date_y, report_date.strftime("%b %d, %Y"))

            if LOGO_PATH.exists():
                logo_width = 100
                logo_height = 60
                canvas.drawImage(
                    str(LOGO_PATH),
                    PAGE_SIZE[0] - RIGHT_MARGIN - logo_width,
                    title_y - (logo_height - 30),
                    width=logo_width,
                    height=logo_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            else:
                canvas.setFont("Helvetica-Bold", 9)
                canvas.setFillColor(BRAND_COLORS["secondary"])
                canvas.drawRightString(
                    PAGE_SIZE[0] - RIGHT_MARGIN,
                    title_y,
                    "EQUIPICKER",
                )

            line_y = date_y - 6
            canvas.setStrokeColor(colors.HexColor("#D6DFEB"))
            canvas.setLineWidth(0.5)
            canvas.line(LEFT_MARGIN, line_y, PAGE_SIZE[0] - RIGHT_MARGIN, line_y)

        # Footer
        footer_text = f"Equipicker - {REPORT_NAME}   |   Page {doc.page}"
        canvas.setFillColor(BRAND_COLORS["muted_text"])
        canvas.setFont("Helvetica", 8)
        canvas.drawString(LEFT_MARGIN, BOTTOM_MARGIN - 20, footer_text)
        canvas.restoreState()

    return _draw



def generate_weekly_scoring_board_pdf(
    output_path: Path | str,
    report_date: Optional[date] = None,
    run_sql: bool = True,
    use_cache: bool = True,
    config_path: Optional[Path | str] = None,
    eod_as_of_date: Optional[date] = None,
    cache_date: Optional[date] = None,
    use_config: bool = True,
) -> Path:
    """Create the Weekly Scoring Board PDF at the given location."""
    config = None
    if use_config and (config_path or DEFAULT_CONFIG_PATH.exists()):
        config = load_report_config(config_path)
    if report_date is None:
        report_date = config.report_date if config else date.today()
    if eod_as_of_date is None and config:
        eod_as_of_date = config.eod_as_of_date
    if cache_date is None and config:
        cache_date = config.cache_date
    logger.info("Fetching scoring dataframe (run_sql=%s, use_cache=%s)", run_sql, use_cache)
    df = get_scoring_dataframe(run_sql=run_sql, use_cache=use_cache, cache_date=cache_date)
    if df.empty:
        raise ValueError("Scoring query returned no data.")

    logger.info("Fetching sector overview dataframe (run_sql=%s, use_cache=%s)", run_sql, use_cache)
    sector_overview_df = get_report_dataframe(
        run_sql=run_sql,
        use_cache=use_cache,
        cache_date=cache_date,
        eod_as_of_date=eod_as_of_date,
    )
    df = prepare_scoring_dataframe(df)
    pages = build_report_pages(df)
    if not pages:
        raise ValueError("No report pages could be built from scoring data.")
    sector_stats = compute_sector_overview_stats(sector_overview_df)
    eod_date = None
    if sector_overview_df is not None and not sector_overview_df.empty:
        if "eod_price_date" in sector_overview_df.columns:
            eod_date = pd.to_datetime(sector_overview_df["eod_price_date"]).max()
            if pd.notna(eod_date):
                eod_date = eod_date.strftime("%Y-%m-%d")
    sector_anchor_map = _build_sector_anchor_map(pages)
    summary_lines = load_or_initialize_summary_text()
    export_sector_stats_json(sector_stats, report_date)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=PAGE_SIZE,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
    )

    styles = _build_styles()
    story: List = []
    story.extend(build_summary_page(styles, summary_lines, report_date))
    story.append(PageBreak())
    story.extend(build_sector_overview_page(styles, sector_stats, sector_anchor_map, eod_date=eod_date))
    story.append(PageBreak())
    for idx, page in enumerate(pages):
        anchor = slugify(page["title"])
        for chunk_idx, (chunk_tables, is_cont) in enumerate(chunk_table_groups(page["tables"], chunk_size=3)):
            if idx or chunk_idx:
                story.append(PageBreak())
            logger.info("Rendering page %s: %s", idx + 1, page["scope"])
            scope_label = page["scope"] if not is_cont else f"{page['scope']} (cont.)"
            story.extend(
                _build_scope_title(
                    styles,
                    scope_label,
                    anchor=anchor if chunk_idx == 0 else None,
                )
            )
            story.extend(_build_tables_flowables(chunk_tables, styles))

    story.append(PageBreak())
    story.extend(_build_disclaimer(styles, report_date))

    logger.info("Generating PDF report with %s pages", len(pages) + 1)
    header_footer = make_header_footer(report_date)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    logger.info("Weekly Scoring Board saved to %s", output_path)
    return output_path


def main():
    if DEFAULT_CONFIG_PATH.exists():
        config = load_report_config()
        report_date = config.report_date
    else:
        report_date = date.today()
    reports_dir = Path("reports")
    filename = reports_dir / f"Monthly_Scoring_Board_{report_date.isoformat()}.pdf"
    generated = generate_weekly_scoring_board_pdf(filename, report_date=report_date, run_sql=False, use_cache=True)
    print(f"Weekly scoring board written to: {generated.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    main()
