"""Generate the Weekly Scoring Board PDF report."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from equipicker_connect import get_scoring_dataframe

logger = logging.getLogger(__name__)

PAGE_SIZE = landscape(letter)
LEFT_MARGIN = RIGHT_MARGIN = 30
TOP_MARGIN = 36
BOTTOM_MARGIN = 42

BRAND_COLORS: Dict[str, colors.Color] = {
    "primary": colors.HexColor("#0B2D5C"),
    "secondary": colors.HexColor("#006C8E"),
    "accent": colors.HexColor("#22B5D8"),
    "muted_text": colors.HexColor("#425466"),
    "row_alt": colors.HexColor("#F6FBFF"),
    "score_bg": colors.HexColor("#F0F7FB"),
}

SCORE_COLOR_BANDS: List = [
    (80, colors.HexColor("#0BA360")),
    (60, colors.HexColor("#6FCF97")),
    (40, colors.HexColor("#F2994A")),
    (0, colors.HexColor("#EB5757")),
]

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
    "industry",
    "market_cap",
    "market_cap_category",
    "beta",
    "style",
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
    {"key": "ticker", "label": "Ticker", "formatter": lambda v: v or "-"},
    {"key": "company_name", "label": "Company name", "formatter": lambda v: v or "-"},
    {"key": "sector", "label": "Sector", "formatter": lambda v: v or "-"},
    {"key": "industry", "label": "Industry", "formatter": lambda v: v or "-"},
    {"key": "market_cap", "label": "Market cap", "formatter": lambda v: format_market_cap(v)},
    {"key": "market_cap_category", "label": "Market cap category", "formatter": lambda v: v or "-"},
    {"key": "beta", "label": "Beta", "formatter": lambda v: format_beta(v)},
    {"key": "style", "label": "Style", "formatter": lambda v: v or "-"},
    {"key": "total_score", "label": "Total score", "formatter": lambda v: format_score_value(v)},
    {"key": "pillar_value", "label": "Pillar 1", "formatter": lambda v: format_score_value(v)},
    {"key": "pillar_growth", "label": "Pillar 2", "formatter": lambda v: format_score_value(v)},
    {"key": "pillar_quality", "label": "Pillar 3", "formatter": lambda v: format_score_value(v)},
    {"key": "pillar_risk", "label": "Pillar 4", "formatter": lambda v: format_score_value(v)},
    {"key": "pillar_momentum", "label": "Pillar 5", "formatter": lambda v: format_score_value(v)},
]

TABLE_COLUMN_WIDTHS = [40, 75, 55, 75, 55, 55, 35, 40, 50, 43, 43, 43, 43, 43]
COLUMN_INDEX = {col["key"]: idx for idx, col in enumerate(TABLE_COLUMN_DEFS)}

METRIC_TABLES = [
    {"metric": "total_score", "title": "Top 5 - Total Fundamental Score"},
    {"metric": "pillar_value", "title": "Top 5 - Pillar 1 (Value)"},
    {"metric": "pillar_growth", "title": "Top 5 - Pillar 2 (Growth)"},
    {"metric": "pillar_quality", "title": "Top 5 - Pillar 3 (Quality)"},
    {"metric": "pillar_risk", "title": "Top 5 - Pillar 4 (Risk)"},
    {"metric": "pillar_momentum", "title": "Top 5 - Pillar 5 (Momentum)"},
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


def _score_border_color(value):
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
        "scope": ParagraphStyle(
            "scope",
            parent=sample["Heading3"],
            fontSize=14,
            fontName="Helvetica-Bold",
            textColor=BRAND_COLORS["secondary"],
            spaceAfter=10,
        ),
        "table_title": ParagraphStyle(
            "table_title",
            parent=sample["Heading4"],
            fontSize=12,
            fontName="Helvetica-Bold",
            textColor=BRAND_COLORS["primary"],
            spaceAfter=4,
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


def _build_table_block(table_info: Dict, styles: Dict[str, ParagraphStyle]) -> KeepTogether:
    table = _create_data_table(table_info["data"], table_info["metric"])
    return KeepTogether([
        Paragraph(table_info["title"], styles["table_title"]),
        Spacer(1, 4),
        table,
    ])


def _create_data_table(df: pd.DataFrame, metric: str) -> Table:
    table_data = [[col["label"] for col in TABLE_COLUMN_DEFS]]
    extra_styles: List = []

    if df.empty:
        table_data.append(["n/a"] * len(TABLE_COLUMN_DEFS))
    else:
        for _, row in df.iterrows():
            row_values = []
            for col_idx, column in enumerate(TABLE_COLUMN_DEFS):
                raw_value = row.get(column["key"])
                formatted = column["formatter"](raw_value)
                row_values.append(formatted)
                if column["key"] in SCORE_COLUMNS:
                    extra_styles.extend(_score_cell_styles(len(table_data), col_idx, raw_value))
            table_data.append(row_values)

    table = Table(table_data, colWidths=TABLE_COLUMN_WIDTHS, repeatRows=1)
    style_cmds = _table_style_commands(len(table_data), metric)
    table.setStyle(TableStyle(style_cmds + extra_styles))
    return table


def _score_cell_styles(row_idx: int, col_idx: int, value) -> List:
    border_color = _score_border_color(value)
    return [
        ("BOX", (col_idx, row_idx), (col_idx, row_idx), 1, border_color),
        ("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), BRAND_COLORS["score_bg"]),
    ]


def _table_style_commands(row_count: int, metric: str) -> List:
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLORS["primary"]),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), BRAND_COLORS["muted_text"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D3E1EA")),
    ]

    for row in range(1, row_count):
        bg = BRAND_COLORS["row_alt"] if row % 2 else colors.white
        commands.append(("BACKGROUND", (0, row), (-1, row), bg))

    highlight_idx = COLUMN_INDEX.get(metric)
    if highlight_idx is not None:
        commands.extend([
            ("BACKGROUND", (highlight_idx, 0), (highlight_idx, 0), BRAND_COLORS["secondary"]),
            ("LINEBEFORE", (highlight_idx, 1), (highlight_idx, -1), 1, BRAND_COLORS["secondary"]),
            ("LINEAFTER", (highlight_idx, 1), (highlight_idx, -1), 1, BRAND_COLORS["secondary"]),
        ])

    # Alignment tweaks
    for key in ["market_cap", "market_cap_category", "beta", "style"]:
        idx = COLUMN_INDEX.get(key)
        if idx is not None:
            align = "CENTER" if key != "market_cap" else "RIGHT"
            commands.append(("ALIGN", (idx, 1), (idx, -1), align))
    for key in SCORE_COLUMNS:
        idx = COLUMN_INDEX.get(key)
        if idx is not None:
            commands.append(("ALIGN", (idx, 1), (idx, -1), "CENTER"))
    return commands


def _build_page_header(styles: Dict[str, ParagraphStyle], report_date: date, scope_label: str) -> List:
    formatted_date = report_date.strftime("%B %d, %Y")
    return [
        Paragraph("Weekly Scoring Board", styles["title"]),
        Paragraph("Equipicker", styles["subtitle"]),
        Paragraph(f"Report date: {formatted_date}", styles["meta"]),
        Paragraph(scope_label, styles["scope"]),
    ]


def _build_tables_flowables(tables: List[Dict], styles: Dict[str, ParagraphStyle]) -> List:
    flowables: List = []
    for table_info in tables:
        flowables.append(_build_table_block(table_info, styles))
        flowables.append(Spacer(1, 10))
    return flowables


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


def _draw_footer(canvas, doc):
    canvas.saveState()
    footer_text = f"Equipicker - Weekly Scoring Board   |   Page {doc.page}"
    canvas.setFillColor(BRAND_COLORS["muted_text"])
    canvas.setFont("Helvetica", 8)
    canvas.drawString(LEFT_MARGIN, BOTTOM_MARGIN - 20, footer_text)
    canvas.restoreState()


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

    df = prepare_scoring_dataframe(df)
    pages = build_report_pages(df)
    if not pages:
        raise ValueError("No report pages could be built from scoring data.")

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
    for idx, page in enumerate(pages):
        if idx:
            story.append(PageBreak())
        logger.info("Rendering page %s: %s", idx + 1, page["scope"])
        story.extend(_build_page_header(styles, report_date, page["scope"]))
        story.extend(_build_tables_flowables(page["tables"], styles))

    story.append(PageBreak())
    story.extend(_build_disclaimer(styles, report_date))

    logger.info("Generating PDF report with %s pages", len(pages) + 1)
    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
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
