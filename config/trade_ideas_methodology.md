# Trade Ideas — Methodology

How the Trade Ideas tab turns the daily indicator universe into a ranked, tradeable
shortlist. Each basket is **hard eligibility gates** (does it qualify?) plus a **Setup
Score** ranking layer (of the names that qualify, which are the best to trade right now?).

The goal is an *edge*: surface names where trend, money flow, relative strength, RSI
regime and entry quality line up, rank them so the strongest setups are reviewed first,
and never propose something that cannot actually be traded.

Source of truth for the logic: `_filter_trade_idea_basket`, `_compute_trade_idea_setup_scores`
and `TRADE_IDEA_SCORE_WEIGHTS` in `equipilot_app.py`.

---

## Universe-wide gates (apply to every basket)

- **Liquidity floor** — 20-session **Average Dollar Volume** (`adv_usd_20 = mean(volume ×
  adjusted_close)`) must be **≥ $5,000,000** (`TRADE_IDEA_MIN_ADV_USD`). Keeps proposals
  to names that can be entered and exited without excessive slippage.
  - Degrades to a no-op if the daily price cache predates the `volume` column, so a stale
    cache cannot silently empty the tab. Regenerate the daily cache to activate the floor.
- **Market-cap bucket** — Small / Mid / Large / Mega (excludes Micro / Nano).
- **Fundamental gates** — default thresholds are `0` (disaster filter only). Quality
  filtering is opt-in via the tab controls; raise them to require real fundamental quality.

> Note: no minimum **price** filter is applied — low-priced but genuinely liquid names are
> allowed through on purpose.

---

## Baskets — hard gates

### Full Acceleration (`acceleration`) — momentum continuation, *reasonable entry*
Bullish names accelerating but not yet over-extended, so there is still room to enter.
- RSI regime > 70, weekly RSI > 55, daily RSI > 70
- **Monthly relative strength > 0** (leadership, not merely "not falling")
- Daily RS > its 20-day SMA; monthly OBVM > 0; daily OBVM > its 20-day SMA
- Trend aligned: `price ≥ SMA20`, `price > SMA50`, `price > SMA200`, `SMA20 ≥ SMA50 ≥ SMA200`
- **Controlled extension**: `atr_vs_ma20 ≤ 4` ATRs above the 20-day MA
  (`TRADE_IDEA_EXTENSION_ATR_CAP`) — this is the lever that biases toward a sane entry
  instead of chasing parabolic moves. Missing extension data is allowed through.
- RSI-regime cross not negative; no bearish divergence (none / negative-confirmed / extension-negative)

### Uptrend Losing Steam (`acceleration_weakening`) — watch / trim list
Same strong acceleration base, but **at least one** confirmation is cracking. This is **not
a buy list** — it flags strong names where the thesis is starting to fail.
- Keeps the strong base (regime, RSI, monthly RS/OBVM, aligned trend) — note this basket
  intentionally keeps the looser `price > 0.9 × SMA20` and `rs_monthly > -0.1` so it can
  catch the early breakdown.
- Fires on any of: daily OBVM ≤ its SMA, daily RS ≤ its SMA, negative RSI-regime cross, or
  bearish RSI divergence.

### Pullback Reclaim (`pullback_reclaim`) — bull-trend dip with early recovery
- Aligned uptrend (`price > SMA50 > … `, `SMA20 ≥ SMA50 ≥ SMA200`, `price > 0.7 × SMA20`)
- Cooled momentum: daily RSI in (40, 70), weekly RSI in (30, 70), RSI regime > 60
- A real pullback happened (confirmed bearish divergence, or price < 0.9 × last bear pivot)
- Early recovery (daily OBVM > its SMA **or** daily RS > its SMA); monthly RS > -0.1;
  RSI-regime cross not negative

### Around MA200 daily (`below_ma200`) / weekly (`around_ma200_weekly`) — trend repair
- Constructive repair guardrails: `price > SMA50`, `SMA20 ≥ SMA50`, daily OBVM > its SMA,
  daily RS > its SMA, daily RSI > 60, weekly RSI > 40, regime > 40, cross not negative, no
  bearish divergence
- Distance to the (daily / weekly) 200-MA between **-20% and +10%**

### Positive Divergence (`positive_divergence_bottoming`) — early bottoming
- Positive RSI divergence (daily or weekly), daily OBVM > its SMA, daily RS > its SMA,
  RSI-regime cross not negative

---

## Setup Score (ranking layer)

Every eligible name gets a **0–100 Setup Score** plus four transparent sub-scores so a high
rank is explainable. Component signals are **percentile-ranked cross-sectionally across the
liquid universe** (`Series.rank(pct=True)`), which turns unitless levels like `rs` and
`obvm` into market-relative, comparable numbers.

### The four sub-scores (each 0–100)
- **RS** — leadership: `pct_rank(rs_daily)`, RS turning up `pct_rank(rs_daily − rs_sma20)`,
  medium-term `pct_rank(rs_monthly)`.
- **Flow** — money flow: `pct_rank(obvm_daily − obvm_sma20)`, `pct_rank(obvm_weekly)`,
  `pct_rank(obvm_monthly)`.
- **Trend** — `pct_rank` of MA separation `0.5·(price/SMA200−1) + 0.5·(SMA50/SMA200−1)`
  blended with the RSI regime score.
- **Entry** — basket-shaped:
  - *Momentum baskets* (acceleration, weakening): reward **controlled extension**
    (`(CAP − atr_vs_ma20)/CAP`) and strong-but-not-parabolic RSI (`rsi_daily` mapped 50→85).
  - *Repair / pullback / divergence baskets*: reward **proximity to the 20-MA** and RSI
    reclaiming the 35→55 zone.

### Composite weights per basket (`TRADE_IDEA_SCORE_WEIGHTS`)

| Basket | RS | Flow | Trend | Entry |
|---|---|---|---|---|
| Full Acceleration | 0.30 | 0.25 | 0.25 | 0.20 |
| Pullback Reclaim | 0.25 | 0.25 | 0.25 | 0.25 |
| Around MA200 daily | 0.25 | 0.25 | 0.30 | 0.20 |
| Around MA200 weekly | 0.25 | 0.25 | 0.30 | 0.20 |
| Positive Divergence | 0.25 | 0.25 | 0.20 | 0.30 |

**Uptrend Losing Steam** is the exception: it is *not* scored by the table above. It is
ranked by a **deterioration score** — `0.6 × (count of weakening tells / 4) + 0.4 ×
percentile of how negative daily flow and RS are vs their averages`. Higher = cracking
harder = review for trim first.

Baskets are sorted by Setup Score descending (ties broken by general technical score, then
fundamental score). The score and its four sub-scores are shown as columns in the grid.

### Tuning
All knobs are constants near the top of the filter logic in `equipilot_app.py`:
`TRADE_IDEA_MIN_ADV_USD`, `TRADE_IDEA_EXTENSION_ATR_CAP`, `TRADE_IDEA_ADV_WINDOW`, and
`TRADE_IDEA_SCORE_WEIGHTS`. These are starting points — tune against live output and the
2-week forward-return backtest on the Analysis tab.
