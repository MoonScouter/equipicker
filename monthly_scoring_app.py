"""Streamlit UI for managing Monthly Scoring Board generation and helper files."""
from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
from streamlit.components.v1 import html

from equipicker_connect import (
    bucharest_today_str,
    report_cache_path,
    scoring_cache_path,
)
from report_config import DEFAULT_CONFIG_PATH, ReportConfig, load_report_config, save_report_config
from weekly_scoring_board import generate_weekly_scoring_board_pdf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPT_PATH = BASE_DIR / "summary_prompt.txt"
REPORTS_DIR = BASE_DIR / "reports"
SUMMARY_JSON_PATH = DATA_DIR / "text_generated.json"
CONFIG_PATH = DEFAULT_CONFIG_PATH


def _format_ts(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return "(not found)"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M")


def get_latest_sector_tables_file() -> Optional[Path]:
    candidates = sorted(DATA_DIR.glob("sector_data_tables_*.txt"))
    return candidates[-1] if candidates else None


def read_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_all_texts() -> Dict[str, str]:
    latest_sector = get_latest_sector_tables_file()
    return {
        "prompt": read_text(PROMPT_PATH),
        "sector_path": str(latest_sector) if latest_sector else "",
        "sector_text": read_text(latest_sector),
        "summary_json": read_text(SUMMARY_JSON_PATH),
    }


def sync_editors(force: bool = False) -> None:
    bundle = st.session_state.setdefault("file_bundle", load_all_texts())
    if force or "prompt_content" not in st.session_state:
        st.session_state["prompt_content"] = bundle.get("prompt", "")
    if force or "sector_content" not in st.session_state:
        st.session_state["sector_content"] = bundle.get("sector_text", "")
    if force or "summary_content" not in st.session_state:
        st.session_state["summary_content"] = bundle.get("summary_json", "")
    st.session_state.setdefault("logs", "")


def refresh_files() -> None:
    st.session_state["file_bundle"] = load_all_texts()
    sync_editors(force=True)


def append_log(message: str) -> None:
    current = st.session_state.get("logs", "")
    st.session_state["logs"] = (current + ("\n" if current else "") + message)
    placeholder = st.session_state.get("log_placeholder")
    if placeholder is not None:
        placeholder.code(st.session_state["logs"] or "(no logs yet)")


class StreamlitLogHandler(logging.Handler):
    """Logging handler that feeds messages into the Streamlit log panel."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI side-effects
        append_log(self.format(record))


def run_generation(
    report_date: date,
    run_sql: bool,
    eod_as_of_date: Optional[date],
    cache_date: Optional[date],
) -> None:
    output = REPORTS_DIR / f"Monthly_Scoring_Board_{report_date.isoformat()}.pdf"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    handler = StreamlitLogHandler()
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            generate_weekly_scoring_board_pdf(
                output,
                report_date=report_date,
                run_sql=run_sql,
                use_cache=True,
                eod_as_of_date=eod_as_of_date,
                cache_date=cache_date,
                use_config=False,
            )
        append_log(f"PDF generated successfully: {output}")
        st.success(f"PDF generated: {output}")
    except Exception as exc:  # pragma: no cover - show in UI
        append_log(f"Generation failed: {exc}")
        st.error(f"Generation failed: {exc}")
    finally:
        if stdout_buffer.getvalue():
            append_log(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            append_log(stderr_buffer.getvalue())
        refresh_files()
        root_logger.removeHandler(handler)


def summary_counts(summary_text: str) -> str:
    try:
        parsed = json.loads(summary_text)
    except json.JSONDecodeError:
        return "JSON invalid – cannot compute counts."
    counts = {
        key: len(value) if isinstance(value, list) else 0
        for key, value in parsed.items()
    }
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


def copy_button(text: str, key: str) -> None:
    """Render a copy-to-clipboard button using Streamlit's HTML component."""
    html(
        f"""
        <button onclick='navigator.clipboard.writeText({json.dumps(text)})'
                style='padding:6px 12px; margin-top:18px;' id='{key}'>
            Copy to clipboard
        </button>
        """,
        height=60,
    )


st.set_page_config(page_title="Monthly Scoring Board", layout="wide")
force_sync = st.session_state.pop("force_sync", False)
sync_editors(force=force_sync)

st.title("Monthly Scoring Board")
try:
    config = load_report_config(CONFIG_PATH)
except Exception as exc:  # pragma: no cover - UI feedback
    st.error(f"Invalid config: {exc}")
    config = ReportConfig(report_date=date.today())

st.subheader("Report Config")
report_date_value = st.date_input("Report date", value=config.report_date, key="report_date")
override_eod = st.checkbox("Override EOD as-of date", value=config.eod_as_of_date is not None)
eod_as_of_value = None
if override_eod:
    eod_as_of_value = st.date_input(
        "EOD as-of date (30-day window anchor)",
        value=config.eod_as_of_date or report_date_value,
        key="eod_as_of_date",
    )
override_cache = st.checkbox("Override cache date", value=config.cache_date is not None)
cache_date_value = None
if override_cache:
    cache_date_value = st.date_input(
        "Cache date",
        value=config.cache_date or report_date_value,
        key="cache_date",
    )
if st.button("Save config"):
    save_report_config(ReportConfig(report_date_value, eod_as_of_value, cache_date_value), CONFIG_PATH)
    st.success(f"Saved config: {CONFIG_PATH}")
    st.rerun()

effective_cache_date = cache_date_value.isoformat() if cache_date_value else bucharest_today_str()
st.caption(f"Cache date in use: {effective_cache_date}")
st.caption(f"Report cache: {report_cache_path(cache_date=cache_date_value)}")
st.caption(f"Scoring cache: {scoring_cache_path(cache_date=cache_date_value)}")

st.caption(report_date_value.strftime("%B %d, %Y"))
output_hint = REPORTS_DIR / f"Monthly_Scoring_Board_{report_date_value.isoformat()}.pdf"
st.caption(f"Output path: {output_hint}")

controls_col, _ = st.columns([1, 3])
with controls_col:
    run_sql_toggle = st.checkbox("Run SQL (ignore cache)", value=False)
    if st.button("Generate PDF", use_container_width=True):
        run_generation(report_date_value, run_sql_toggle, eod_as_of_value, cache_date_value)
    if st.button("Refresh Files", use_container_width=True):
        refresh_files()
        st.session_state["force_sync"] = True
        st.rerun()

logs_tab, prompt_tab, sector_tab, summary_tab = st.tabs(
    ["Logs", "Summary Prompt", "Sector Data Tables", "Summary Text (JSON)"]
)

with logs_tab:
    placeholder = st.empty()
    st.session_state["log_placeholder"] = placeholder
    placeholder.code(st.session_state.get("logs", "") or "(no logs yet)")
    if st.button("Clear logs"):
        st.session_state["logs"] = ""
        placeholder.code("(no logs yet)")

bundle = st.session_state["file_bundle"]

with prompt_tab:
    st.caption(f"File: {PROMPT_PATH} (last updated: {_format_ts(PROMPT_PATH)})")
    prompt_text = st.text_area(
        "Prompt contents",
        value=st.session_state.get("prompt_content", ""),
        height=300,
    )
    st.session_state["prompt_content"] = prompt_text
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save prompt"):
            write_text(PROMPT_PATH, prompt_text)
            refresh_files()
            st.session_state["force_sync"] = True
            st.rerun()
    with c2:
        copy_button(prompt_text, "prompt_copy")

with sector_tab:
    sector_path_str = bundle.get("sector_path")
    if not sector_path_str:
        st.warning("No sector_data_tables file found. Generate the PDF first.")
    else:
        sector_path = Path(sector_path_str)
        st.caption(f"Latest file: {sector_path.name} (last updated: {_format_ts(sector_path)})")
        sector_text = st.text_area(
            "Sector tables",
            value=st.session_state.get("sector_content", ""),
            height=300,
        )
        st.session_state["sector_content"] = sector_text
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save sector tables"):
                write_text(sector_path, sector_text)
                refresh_files()
                st.session_state["force_sync"] = True
                st.rerun()
        with c2:
            copy_button(sector_text, "sector_copy")

with summary_tab:
    st.caption(f"File: {SUMMARY_JSON_PATH} (last updated: {_format_ts(SUMMARY_JSON_PATH)})")
    summary_text = st.text_area(
        "Summary JSON",
        value=st.session_state.get("summary_content", ""),
        height=300,
    )
    st.session_state["summary_content"] = summary_text
    st.caption(f"Paragraph counts: {summary_counts(summary_text)}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save summary JSON"):
            try:
                json.loads(summary_text)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            else:
                write_text(SUMMARY_JSON_PATH, summary_text)
                refresh_files()
                st.session_state["force_sync"] = True
                st.rerun()
    with c2:
        copy_button(summary_text, "summary_copy")

st.markdown("---")
st.markdown(
    "**Shortcut tip:** target `C:/Users/razva/PycharmProjects/equipicker/equipicker/.venv/Scripts/python.exe -m streamlit run monthly_scoring_app.py` with start-in folder `C:/Users/razva/PycharmProjects/equipicker/equipicker`."
)
