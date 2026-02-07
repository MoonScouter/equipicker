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

You can run the same report-select generation flow used in Home tab on an automatic schedule.

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

The `Trade Ideas` tab uses four presets from `equipicker_filters.py`:
- `extreme_accel_up`
- `accel_up_weak`
- `extreme_accel_down`
- `accel_down_weak`

Legacy names `extreme_accel` and `accel_weak` are kept as compatibility wrappers to the corresponding `*_up` filters.

### `extreme_accel_up`

Bullish high-conviction acceleration:
- Scores pinned near max (`general_technical_score`, `relative_performance`, `relative_volume`, `momentum`, `intermediate_trend`, `long_term_trend`)
- Positive RS/OBVM thrust (`rs_daily > rs_sma20`, `obvm_daily > 1.5 * obvm_sma20`)
- Optional guards when fields exist:
  - `rs_monthly >= 2.0`
  - `obvm_weekly > 0` and `obvm_monthly > 0`
  - `rsi_weekly >= 70` and `rsi_daily >= 80`
  - MA stack + price extension above MA20
- Sort: strongest bullish first (`fundamental_total_score DESC`, then `general_technical_score DESC` when available)

### `accel_up_weak`

Bullish moderate acceleration:
- `general_technical_score >= 72`
- `relative_volume >= 75`
- `relative_performance >= 65`
- `momentum >= 72`
- `intermediate_trend >= 80`
- `long_term_trend >= 70`
- Optional guards when fields exist:
  - `obvm_monthly > 0`
  - `rs_monthly >= 0` (tighten mode)
  - RSI band: weekly `[60,75]`, daily `[60,70]`, and `daily <= weekly`
  - `rs_daily < rs_sma20`
  - `obvm_daily < obvm_sma20`
- Sort: `general_technical_score DESC`, then `fundamental_total_score DESC`

### `extreme_accel_down`

Bearish high-conviction acceleration (mirror of extreme up):
- `general_technical_score <= 0.1`
- `relative_performance <= 0.1` and optional `rs_monthly <= -2.0`
- `relative_volume <= 0.1` and optional `obvm_weekly < 0`, `obvm_monthly < 0`
- `momentum <= 0.1` and optional `rsi_weekly <= 30`, `rsi_daily <= 20`
- `intermediate_trend <= 0.1`
- `long_term_trend <= 0.1`
- Optional MA mirror:
  - `sma_daily_20 <= 0.98 * sma_daily_50`
  - `sma_daily_50 <= 0.98 * sma_daily_200`
  - `eod_price_used <= 0.97 * sma_daily_20`
- Downside thrust mirror:
  - `rs_daily < rs_sma20`
  - `obvm_daily < (obvm_sma20 / 1.5)`
- Sort: strongest bearish first (`general_technical_score ASC`, then `fundamental_total_score ASC`)

### `accel_down_weak`

Bearish moderate acceleration (mirror of weak up):
- `general_technical_score <= 28`
- `relative_volume <= 25` and optional `obvm_monthly < 0`
- `relative_performance <= 35` and optional `rs_monthly <= 0` (tighten mode)
- `momentum <= 28`
- `intermediate_trend <= 20`
- `long_term_trend <= 30`
- Optional RSI mirror:
  - `rsi_weekly` in `[25,40]`
  - `rsi_daily` in `[30,40]`
  - `rsi_daily >= rsi_weekly`
- Weak downside thrust mirror:
  - `rs_daily > rs_sma20`
  - `obvm_daily > obvm_sma20`
- Sort: `general_technical_score ASC`, then `fundamental_total_score ASC`

All four filters format `market_cap` for display as `x.xxB` or `x.xxM`.
