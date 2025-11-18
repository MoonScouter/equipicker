"""Generate the Weekly Scoring Board PDF report."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
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

from equipicker_connect import get_scoring_dataframe

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
LOGO_PATH = Path(__file__).resolve().parent / "logo.png"

TABLE_BODY_STYLE = ParagraphStyle(
    "table_body",
    fontName="Helvetica",
    fontSize=6.2,
    leading=7.2,
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


def _build_toc(styles: Dict[str, ParagraphStyle], pages: List[Dict]) -> List:
    flowables: List = [
        Paragraph('<a name="toc"/>Table of Contents', styles["title"]),
        Spacer(1, 12),
    ]
    for page in pages:
        anchor = slugify(page["title"])
        flowables.append(
            Paragraph(f'<link href="#{anchor}">{page["scope"]}</link>', styles["toc_entry"])
        )
    return flowables


def _build_table_block(table_info: Dict, styles: Dict[str, ParagraphStyle]) -> KeepTogether:
    table = _create_data_table(table_info["data"], table_info["metric"])
    title_band = Table(
        [[Paragraph(table_info["title"], styles["table_title"])]],
        colWidths=[CONTENT_WIDTH],
    )
    title_band.setStyle(
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
) -> List:
    header: List = []
    label_text = f'<a name="{anchor}"/>{scope_label}' if anchor else scope_label
    arrow = Paragraph('<link href="#toc">&#9650;</link>', styles["button"])
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
) -> List:
    label_text = f'<a name="{anchor}"/>{scope_label}' if anchor else scope_label
    arrow = Paragraph('<link href="#toc">&#9650;</link>', styles["button"])
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
            canvas.drawString(LEFT_MARGIN, title_y, "Weekly Scoring Board")

            # Date just under the title
            date_y = title_y - 10
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(BRAND_COLORS["muted_text"])
            canvas.drawString(LEFT_MARGIN, date_y, report_date.strftime("%b %d, %Y"))

            # Right-hand element: logo or EQUIPICKER text
            if LOGO_PATH.exists():
                logo_width = 60
                logo_height = 20
                canvas.drawImage(
                    str(LOGO_PATH),
                    PAGE_SIZE[0] - RIGHT_MARGIN - logo_width,
                    title_y - (logo_height - 6),  # roughly aligned with title
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
        footer_text = f"Equipicker - Weekly Scoring Board   |   Page {doc.page}"
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
    story.extend(_build_toc(styles, pages))
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
