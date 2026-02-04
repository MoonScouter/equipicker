# equipicker

## Environment

This repo targets a modern Python 3 interpreter (3.11+ recommended). Set up an isolated environment before running the app or scripts:

1. `python -m venv .venv`
2. Activate the virtual environment:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`

Once dependencies are installed, you can launch `streamlit run equipicker_app.py` or run other scripts in this environment.

## Weekly Scoring Board report

To render the PDF scoring report:

1. Activate the virtual environment and ensure dependencies (including `reportlab`) are installed: `pip install -r requirements.txt`.
2. Run the generator from the project root: `python weekly_scoring_board.py`.
   - The script fetches fresh scoring data by default and writes a file such as `reports/Weekly_Scoring_Board_YYYY-MM-DD.pdf`.
   - Pass a different output path or reuse cached SQL results by calling `generate_weekly_scoring_board_pdf` directly from Python.

### Report config (JSON)

The report generator reads `config/report_config.json` if it exists. You can also edit these values from the Streamlit UI.

- `report_date`: date used in the filename and first page (YYYY-MM-DD)
- `eod_as_of_date`: optional EOD date used only for the 30-day window fields (YYYY-MM-DD or null)
- `cache_date`: optional date used to load/write cache files (YYYY-MM-DD or null)

Report cache files:
- `data/report_select_YYYY-MM-DD.xlsx` (report query)
- `data/scoring_YYYY-MM-DD.xlsx` (scoring query)
