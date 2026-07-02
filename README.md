# equipicker

## Environment

This repo targets a modern Python 3 interpreter (3.11+ recommended). Set up an isolated environment before running the app or scripts:

1. `python -m venv .venv`
2. Activate the virtual environment:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`

Once dependencies are installed, you can launch:

- `streamlit run equipilot_app.py` (main cockpit app)
- `streamlit run monthly_scoring_app.py` (backward-compatible entrypoint)
- `streamlit run equipicker_app.py` (legacy screener runner)

## Weekly Scoring Board report

To render the PDF scoring report:

1. Activate the virtual environment and ensure dependencies (including `reportlab`) are installed: `pip install -r requirements.txt`.
2. Run the generator from the project root: `python weekly_scoring_board.py`.
   - The script fetches fresh scoring data by default and writes a file such as `reports/Monthly_Scoring_Board_YYYY-MM-DD.pdf`.
   - Pass a different output path or reuse cached SQL results by calling `generate_weekly_scoring_board_pdf` directly from Python.

### Report config (JSON)

The report generator reads `config/report_config.json` if it exists. You can also edit these values from the Streamlit UI.

- `report_date`: date used in the filename and first page (YYYY-MM-DD)
- `eod_as_of_date`: optional EOD date used only for the 30-day window fields (YYYY-MM-DD or null)

Report cache files:
- `data/report_select_YYYY-MM-DD.xlsx` (report query)
- `data/scoring_YYYY-MM-DD.xlsx` (scoring query)

## Scheduled report_select generation

You can run the same report-select generation flow used in Home > Report Excel Import on an automatic schedule.

### Remote database connection

`report_select` generation reads the remote MySQL connection from environment variables or from `.env` in this folder. Copy `.env.example` to `.env` and fill in the current host, database, username, and password supplied by the hosting/admin team.

You can configure either a full SQLAlchemy URL:

- `EQUIPICKER_DATABASE_URL=mysql+mysqlconnector://user:password@host:3306/database?charset=utf8mb4`

Or separate values:

- `EQUIPICKER_DB_HOST`
- `EQUIPICKER_DB_PORT`
- `EQUIPICKER_DB_NAME`
- `EQUIPICKER_DB_USER`
- `EQUIPICKER_DB_PASSWORD`
- `EQUIPICKER_DB_CHARSET`

If Streamlit or the scheduled task is already running when `.env` changes, restart it so the new connection settings are loaded.

### CLI runner

- Script: `scheduled_report_select.py`
- Default mode: previous NYSE trading day relative to Bucharest local date
- Example manual run:
  - `python scheduled_report_select.py --mode previous-us-trading-day --run-sql --time-zone Europe/Bucharest`

Output/log behavior:
- Writes or overwrites `data/report_select_YYYY-MM-DD.xlsx` for resolved anchor date
- Logs to console and by default to `logs/report_select_scheduler.log`
- Returns non-zero exit code on failure

### Windows Task Scheduler wrapper

- Batch wrapper: `run_report_select_job.bat`
- Default wrapper behavior:
  - Activates `.venv`
  - Runs `scheduled_report_select.py --mode previous-us-trading-day --run-sql --time-zone Europe/Bucharest`

### Recommended Task Scheduler setup

1. Open `Task Scheduler` and click `Create Task...` (not Basic Task).
2. In `General`:
   - Name: `Equipilot Report Select Daily`
   - Select your Windows user.
   - Optional: enable `Run whether user is logged on or not` for unattended runs.
3. In `Triggers`:
   - Click `New...`
   - Begin the task: `On a schedule`
   - Settings: `Daily`
   - Start time: `20:00`
   - Ensure `Enabled` is checked.
4. In `Actions`:
   - Click `New...`
   - Action: `Start a program`
   - Program/script: `cmd.exe`
   - Add arguments: `/c "C:\\Users\\razva\\PycharmProjects\\equipicker\\equipicker\\run_report_select_job.bat"`
   - Start in: `C:\\Users\\razva\\PycharmProjects\\equipicker\\equipicker`
5. In `Settings` (recommended):
   - Enable `Allow task to be run on demand`
   - Enable `If the task fails, restart every` 5 minutes, `Attempt to restart up to` 3 times
6. Click `OK` and provide credentials if prompted.
7. Validate immediately:
   - Right-click the task and choose `Run`
   - Confirm `Last Run Result` is `0x0`
   - Confirm log updates in `logs/report_select_scheduler.log`
   - Confirm expected file exists: `data/report_select_<anchor-date>.xlsx`

### Scheduler troubleshooting checklist

1. Verify venv/interpreter exists: `C:\\Users\\razva\\PycharmProjects\\equipicker\\equipicker\\.venv\\Scripts\\python.exe`
2. Run wrapper manually from cmd:
   - `cd C:\\Users\\razva\\PycharmProjects\\equipicker\\equipicker`
   - `run_report_select_job.bat`
3. Ensure dependencies are installed in `.venv`: `pip install -r requirements.txt`
4. Check task `History` tab for start/action/exit events.

## Trade Ideas filters

The `Trade Ideas` tab builds six setup baskets, each as **hard eligibility gates + a 0–100
Setup Score** ranking layer. The logic lives in `_filter_trade_idea_basket`,
`_compute_trade_idea_setup_scores` and `TRADE_IDEA_SCORE_WEIGHTS` in `equipilot_app.py`.

Baskets: **Full Acceleration**, **Uptrend Losing Steam** (watch/trim),
**Pullback Reclaim**, **Around MA200 daily**, **Around MA200 weekly**, **Positive Divergence**.

Every basket also enforces a **liquidity floor** — 20-session average dollar volume
(`adv_usd_20`, fed by the `volume` column added to the daily price cache) ≥ `$5M`
(`TRADE_IDEA_MIN_ADV_USD`) — and the acceleration basket caps extension at
`atr_vs_ma20 ≤ 4` ATRs (`TRADE_IDEA_EXTENSION_ATR_CAP`) for reasonable entries.

Full per-basket gates, the Setup Score formula, sub-scores (RS / Flow / Trend / Entry) and
per-basket weights are documented in **[config/trade_ideas_methodology.md](config/trade_ideas_methodology.md)**.

> The older `equipicker_filters.py` presets (`extreme_accel_up`, `accel_up_weak`,
> `extreme_accel_down`, `accel_down_weak`) are retired from this tab and superseded by the
> baskets above.

