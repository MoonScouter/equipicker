"""Generate the Weekly Scoring Board PDF report."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from xml.sax.saxutils import escape

from equipicker_connect import get_dataframe, get_scoring_dataframe

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
REPORT_NAME = "Scoring Board Report"

TABLE_BODY_STYLE = ParagraphStyle(
    "table_body",
    fontName="Helvetica",
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
POSITIVE_TEXT_HEX = "#0BA360"
NEGATIVE_TEXT_HEX = "#EB5757"
NEUTRAL_TEXT_HEX = "#425466"
BREADTH_THRESHOLD = 50.0
SECTOR_PULSE_COLUMN_WIDTHS = [170, 95, 95, 90, 90, 30]
CROSS_SECTOR_COLUMN_WIDTHS = [170, 70, 60, 60, 60, 60, 60]
ROCKET_ICON = "&#128640;"
SECTOR_SCORE_COLUMNS = [
    ("avg_total_score", "fundamental_total_score", "Total"),
    ("avg_value", "fundamental_value", "P1"),
    ("avg_growth", "fundamental_growth", "P2"),
    ("avg_risk", "fundamental_risk", "P3"),
    ("avg_quality", "fundamental_quality", "P4"),
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


def _format_score_badge(value):
    return ScoreBadge(value)


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
    if MOMENTUM_COLUMN in filtered.columns and metric != MOMENTUM_COLUMN:
        sort_cols.append(MOMENTUM_COLUMN)
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
    pct_value = round((success_count / total_count) * 100, 2)
    return f"{pct_value:.2f}%", pct_value


def _prepare_sector_overview_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    working = df.copy()
    working["sector"] = working.get("sector", pd.Series(dtype=str)).fillna("Unspecified")
    one_m_close = pd.to_numeric(working.get("1m_close"), errors="coerce")
    eod_price = pd.to_numeric(working.get("eod_price_used"), errors="coerce")
    market_cap = pd.to_numeric(working.get("market_cap"), errors="coerce")

    valid_prices = (
        one_m_close.notna()
        & eod_price.notna()
        & (one_m_close != 0)
        & (eod_price != 0)
    )

    working["market_cap_numeric"] = market_cap
    working["1m_market_cap"] = np.where(
        valid_prices,
        (one_m_close / eod_price) * market_cap,
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

        rs_daily = pd.to_numeric(group.get("rs_daily"), errors="coerce")
        rs_sma20 = pd.to_numeric(group.get("rs_sma20"), errors="coerce")
        valid_rs_mask = rs_daily.notna() & rs_sma20.notna()
        rs_success = int(((rs_daily > 0) & (rs_daily > rs_sma20) & valid_rs_mask).sum())
        total_rs = int(valid_rs_mask.sum())
        rs_breadth, rs_breadth_num = _format_ratio(rs_success, total_rs)

        obvm_daily = pd.to_numeric(group.get("obvm_daily"), errors="coerce")
        obvm_sma20 = pd.to_numeric(group.get("obvm_sma20"), errors="coerce")
        valid_obvm_mask = obvm_daily.notna() & obvm_sma20.notna()
        obvm_success = int(((obvm_daily > 0) & (obvm_daily > obvm_sma20) & valid_obvm_mask).sum())
        total_obvm = int(valid_obvm_mask.sum())
        obvm_breadth, obvm_breadth_num = _format_ratio(obvm_success, total_obvm)

        score_avgs = {}
        for target_col, source_col, _ in SECTOR_SCORE_COLUMNS:
            score_avgs[target_col] = pd.to_numeric(group.get(source_col), errors="coerce").mean()

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


def _colored_value_cell(text: str, color_hex: Optional[str]) -> Paragraph:
    safe_text = escape(text or "N/A")
    if color_hex:
        safe_text = f'<font color="{color_hex}">{safe_text}</font>'
    return Paragraph(safe_text, SECTOR_OVERVIEW_BODY_STYLE)


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


def _build_signal_cell(has_signal: bool) -> Paragraph:
    if has_signal:
        return Paragraph(f'<font color="{POSITIVE_TEXT_HEX}">{ROCKET_ICON}</font>', SECTOR_OVERVIEW_BODY_STYLE)
    return Paragraph("", SECTOR_OVERVIEW_BODY_STYLE)


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
            table_data.append([
                _sector_label_cell(row["sector"], anchor),
                _colored_value_cell(row["sector_1m_var_pct"], _variation_color(row["sector_1m_var_pct_num"])),
                _colored_value_cell(row["market_breadth"], _breadth_color(row["market_breadth_num"])),
                _colored_value_cell(row["rs_breadth"], _breadth_color(row["rs_breadth_num"])),
                _colored_value_cell(row["obvm_breadth"], _breadth_color(row["obvm_breadth_num"])),
                _build_signal_cell(bool(row.get("signal"))),
            ])

    table = Table(table_data, colWidths=SECTOR_PULSE_COLUMN_WIDTHS, repeatRows=1)
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
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
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
                row_values.append(_format_score_badge(row.get(target_col)))
            table_data.append(row_values)

    table = Table(table_data, colWidths=CROSS_SECTOR_COLUMN_WIDTHS, repeatRows=1)
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
    flowables.append(Spacer(1, 18))
    flowables.append(_table_title_band("Cross-Sector Fundamental Scoring", styles))
    flowables.append(Spacer(1, 4))
    flowables.append(_build_cross_sector_table(sector_stats, anchors))
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
            fontName="Helvetica-Bold",
            textColor=BRAND_COLORS["primary"],
            spaceAfter=0,
        ),
        "disclaimer_body": ParagraphStyle(
            "disclaimer_body",
            parent=sample["BodyText"],
            fontSize=10,
            leading=14,
            textColor=BRAND_COLORS["muted_text"],
        ),
    }
    return styles


def _table_title_band(title: str, styles: Dict[str, ParagraphStyle]) -> Table:
    band = Table([[Paragraph(title, styles["table_title"])]], colWidths=[CONTENT_WIDTH])
    band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TABLE_BAND_COLOR),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
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
    paragraphs = [
        Paragraph("Disclaimer", styles["title"]),
        Paragraph(
            "This Weekly Scoring Board is generated from Equipicker's quantitative models and summarizes the highest scoring equities based on fundamental pillars.",
            styles["disclaimer_body"],
        ),
        Paragraph(
            "Scores express relative strength within our universe and should not be interpreted as personalized investment advice, a solicitation, or a guarantee of future performance.",
            styles["disclaimer_body"],
        ),
        Paragraph(
            "Investors should perform their own due diligence, review all available financial disclosures, and consult a licensed financial advisor before taking action.",
            styles["disclaimer_body"],
        ),
        Paragraph(
            "This document is provided for informational purposes only. Equipicker and its affiliates assume no responsibility for losses arising from reliance on these materials.",
            styles["disclaimer_body"],
        ),
    ]
    return paragraphs


def make_header_footer(report_date: date):
    def _draw(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            header_y = PAGE_SIZE[1] - TOP_MARGIN + 20

            # Title on the left
            title_y = header_y - 8
            canvas.setFont("Times-BoldItalic", 12)
            canvas.setFillColor(BRAND_COLORS["primary"])
            canvas.drawString(LEFT_MARGIN, title_y, REPORT_NAME)

            # Date just under the title
            date_y = title_y - 10
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(BRAND_COLORS["muted_text"])
            canvas.drawString(LEFT_MARGIN, date_y, report_date.strftime("%b %d, %Y"))

            # Right-hand element: logo or EQUIPICKER text
            if LOGO_PATH.exists():
                logo_width = 100
                logo_height = 60
                canvas.drawImage(
                    str(LOGO_PATH),
                    PAGE_SIZE[0] - RIGHT_MARGIN - logo_width,
                    title_y - (logo_height - 30),  # roughly aligned with title
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

            # Separator line just under the date
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
) -> Path:
    """Create the Weekly Scoring Board PDF at the given location."""
    report_date = report_date or date.today()
    logger.info("Fetching scoring dataframe (run_sql=%s, use_cache=%s)", run_sql, use_cache)
    df = get_scoring_dataframe(run_sql=run_sql, use_cache=use_cache)
    if df.empty:
        raise ValueError("Scoring query returned no data.")

    logger.info("Fetching sector overview dataframe (run_sql=%s, use_cache=%s)", run_sql, use_cache)
    sector_overview_df = get_dataframe(run_sql=run_sql, use_cache=use_cache)
    df = prepare_scoring_dataframe(df)
    pages = build_report_pages(df)
    if not pages:
        raise ValueError("No report pages could be built from scoring data.")
    sector_stats = compute_sector_overview_stats(sector_overview_df)
    sector_anchor_map = _build_sector_anchor_map(pages)

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
    story.extend(build_sector_overview_page(styles, sector_stats, sector_anchor_map))
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
    today = date.today()
    reports_dir = Path("reports")
    filename = reports_dir / f"Weekly_Scoring_Board_{today.isoformat()}.pdf"
    generated = generate_weekly_scoring_board_pdf(filename, report_date=today, run_sql=True, use_cache=False)
    print(f"Weekly scoring board written to: {generated.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    main()
