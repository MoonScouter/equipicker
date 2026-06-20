# Market / Regime Methodology

This page documents the deterministic Market layer used in Equipilot. The engine does not use AI for scoring. AI is reserved for later commentary and explanation.

All scores are normalized to a `0-100` scale unless noted otherwise. A score above `50` is constructive, a score below `50` is defensive or deteriorating, and labels are deterministic translations of the final score bands shown below.

## Analytical map

The Market layer is a chain, not a set of isolated indicators:

`Stock RSI Regime -> Sector Participation -> Sector Rotation -> Family Rotation -> Market Regime`

`Market Regime + Sector Strength -> Sector Regime Fit`

`Market Regime + Sector Regime Fit + Stock/Fundamental Quality -> Setup Readiness`

The practical reading is:

- Stock RSI Regime describes whether individual stocks are acting like bull, neutral, or bear vehicles.
- Sector Participation aggregates those stock regimes to show whether strength is broad inside each sector.
- Sector Rotation combines participation, actual technical score, RSI participation, and recent change.
- Family Rotation aggregates sector rotation into offensive, cyclical, defensive, and rate-sensitive leadership.
- Market Regime blends market-wide RSI participation, risk appetite, and family-adjusted sector rotation.
- Sector Regime Fit asks whether a sector's strength is appropriate for the current market regime.
- Setup Readiness is the final stock-level context score after market, sector, technical, and fundamental inputs are combined.

## Anchors

The Values tab uses four user-controlled anchors:

- Evaluation date: the current snapshot.
- RSI regime start date: the first date included in the stock RSI regime window.
- 1 month ago date: the anchor for one-month returns and monthly sector change.
- 1 week ago date: the anchor for short-term sector change.

The RSI interval uses all daily and weekly RSI values between `RSI regime start date` and `Evaluation date`, inclusive. If an anchor date has no exact report file, the app resolves to the closest available report date on or before the target date.

## Stock RSI Regime

Stock RSI Regime is computed per ticker from weekly RSI evidence plus a daily confirmation block.

Minimum data rule:

```text
weekly_observation_count >= min_weekly_rsi_observations
```

If weekly history is too sparse, the score is unavailable for that ticker.

The engine computes bull evidence and bear evidence separately, then converts the spread into the final score:

```text
stock_rsi_regime_score = clip(50 + 0.5 * (bull_evidence - bear_evidence), 0, 100)
```

Bull evidence uses four blocks:

```text
bull_evidence =
  weighted_average(
    bull_persist,
    bull_expand,
    bull_resilience,
    bull_daily
  )
```

Bear evidence uses the same structure:

```text
bear_evidence =
  weighted_average(
    bear_persist,
    bear_expand,
    bear_resilience,
    bear_daily
  )
```

The default weights are:

```text
weekly_range_persistence = 45%
expansion_frequency      = 25%
pullback_resilience      = 20%
daily_confirmation       = 10%
```

If daily RSI is missing, the daily block is dropped and the remaining weights are renormalized.

### Stock RSI sub-formulas

Weekly persistence rewards stocks that avoid bearish RSI zones or fail to sustain bullish RSI zones:

```text
bull_persist = clip((pct_weekly_RSI_ge_40 - 50) / 40, 0, 1) * 100
bear_persist = clip((pct_weekly_RSI_le_60 - 50) / 40, 0, 1) * 100
```

Expansion frequency rewards repeated bullish or bearish RSI pushes:

```text
bull_expand =
  0.6 * clip(count_weekly_RSI_ge_60 / 4, 0, 1) * 100
  + 0.4 * clip(count_weekly_RSI_ge_70 / 2, 0, 1) * 100

bear_expand =
  0.6 * clip(count_weekly_RSI_le_40 / 4, 0, 1) * 100
  + 0.4 * clip(count_weekly_RSI_le_30 / 2, 0, 1) * 100
```

Resilience penalizes repeated or persistent breaks into the opposite RSI zone:

```text
bull_resilience =
  100 - (
    0.6 * clip(count_weekly_RSI_lt_40 / 4, 0, 1)
    + 0.4 * clip(max_consecutive_weekly_RSI_lt_40 / 3, 0, 1)
  ) * 100

bear_resilience =
  100 - (
    0.6 * clip(count_weekly_RSI_gt_60 / 4, 0, 1)
    + 0.4 * clip(max_consecutive_weekly_RSI_gt_60 / 3, 0, 1)
  ) * 100
```

Daily confirmation uses the most recent daily RSI observations, capped by `max_daily_rsi_lookback`:

```text
bull_daily =
  0.5 * clip((recent_daily_RSI_average - 50) / 15, 0, 1) * 100
  + 0.5 * clip((latest_daily_RSI - 55) / 15, 0, 1) * 100

bear_daily =
  0.5 * clip((50 - recent_daily_RSI_average) / 15, 0, 1) * 100
  + 0.5 * clip((45 - latest_daily_RSI) / 15, 0, 1) * 100
```

## Sector and Market RSI Participation

Sector participation is the aggregation of stock RSI regime outputs within each sector:

```text
sector_rsi_breadth_pct_60  = percent_of_sector_stocks(stock_rsi_regime_score >= 60)
sector_rsi_breadth_pct_75  = percent_of_sector_stocks(stock_rsi_regime_score >= 75)
sector_rsi_breadth_pct_lt40 = percent_of_sector_stocks(stock_rsi_regime_score < 40)

sector_rsi_participation_composite_score =
  average(stock_rsi_regime_score for scored stocks in sector)
```

The same logic is applied to the full evaluation-date universe:

```text
market_rsi_breadth_pct_60  = percent_of_all_scored_stocks(stock_rsi_regime_score >= 60)
market_rsi_breadth_pct_75  = percent_of_all_scored_stocks(stock_rsi_regime_score >= 75)
market_rsi_breadth_pct_lt40 = percent_of_all_scored_stocks(stock_rsi_regime_score < 40)

market_rsi_participation_composite_score =
  average(stock_rsi_regime_score for all scored stocks)
```

## Sector Participation Score `P`

`P` is the market-breadth participation score used by Sector Rotation. It is built from the existing sector overview statistics:

- one-month sector market-cap variation;
- market breadth;
- relative-performance breadth;
- relative-volume breadth.

The one-month market-cap variation is ranked across sectors:

```text
mc_var_score = percentile_rank(sector_1m_market_cap_variation) * 100
```

That rank is discounted when the absolute one-month return is weak:

```text
if one_month_pct >  5: multiplier = 1.0
if one_month_pct >  0: multiplier = 0.9
if one_month_pct > -5: multiplier = 0.8
else:                  multiplier = 0.7

mc_var_score_discounted = clip(mc_var_score * multiplier, 0, 100)
```

Then:

```text
P =
  average(
    mc_var_score_discounted,
    market_breadth,
    relative_performance_breadth,
    relative_volume_breadth
  )
```

This means a sector cannot get full participation credit from market-cap movement alone if the underlying breadth is poor.

## Sector Technical Score `T`

`T` is the average general technical score of all companies in the sector:

```text
T = average(general_technical_score for companies in sector)
```

`P` answers "how broad is sector participation?" while `T` answers "how technically strong is the average company?"

## Trend of Change

The trend-of-change block compares the current sector state with the selected 1-week and 1-month anchors.

```text
dT_1w = T_now - T_1w_ago
dP_1w = P_now - P_1w_ago
dT_1m = T_now - T_1m_ago
dP_1m = P_now - P_1m_ago

dT = weekly_weight * dT_1w + monthly_weight * (dT_1m / monthly_divisor)
dP = weekly_weight * dP_1w + monthly_weight * (dP_1m / monthly_divisor)

dT_score = clip(50 + score_factor * dT, 0, 100)
dP_score = clip(50 + score_factor * dP, 0, 100)

trend_of_change_score = average(dT_score, dP_score)
```

Default settings:

```text
weekly_weight   = 60%
monthly_weight  = 40%
monthly_divisor = 4
score_factor    = 8
```

The monthly change is divided by `4` so the one-month move is scaled closer to a weekly pace. Scores above `50` indicate improvement; scores below `50` indicate deterioration.

## Sector Rotation

Sector Rotation combines four linked inputs:

```text
sector_rotation_score =
  weighted_average(
    P_now,
    T_now,
    sector_rsi_participation_composite_score,
    trend_of_change_score
  )
```

Default weights:

```text
P_now                                    = 35%
T_now                                    = 35%
sector_rsi_participation_composite_score = 20%
trend_of_change_score                    = 10%
```

Interpretation:

- `P_now` checks whether sector strength is broad.
- `T_now` checks whether the sector's companies have strong current technical scores.
- `sector_rsi_participation_composite_score` checks whether RSI regime behavior confirms the technical score.
- `trend_of_change_score` checks whether the sector is improving or fading versus the selected anchors.

## Family Rotation

Each sector belongs to a configured family: offensive, cyclical, defensive, or rate-sensitive.

Family rotation is the average sector rotation score within that family:

```text
family_sector_rotation_score =
  average(sector_rotation_score for sectors in family)
```

The market-level sector rotation score is the average of the family scores:

```text
market_sector_rotation_score =
  average(family_sector_rotation_score for available families)
```

The leading family classifier is the top family by sector rotation score. If more than one family is within the configured tie band of the leader, the label becomes mixed leadership.

## Risk Appetite

Risk Appetite compares two cohorts:

- Quality / Defensive: `fundamental_quality >= 70` and `fundamental_risk >= 60`.
- Speculative: (`fundamental_quality <= 45` or `fundamental_risk <= 40`) and `general_technical_score >= 55`.

The score blends a one-month return spread and a current technical-score spread:

```text
return_1m = 100 * (evaluation_close / month_anchor_close - 1)

ret_spread_1m = average(speculative_return_1m) - average(quality_defensive_return_1m)
tech_spread   = average(speculative_general_technical_score) - average(quality_defensive_general_technical_score)

ret_component  = clip(50 + 50 * (ret_spread_1m / return_spread_divisor), 0, 100)
tech_component = clip(50 + 50 * (tech_spread / technical_spread_divisor), 0, 100)

risk_appetite_score =
  weighted_average(ret_component, tech_component)
```

Default weights and divisors:

```text
ret_component_weight  = 60%
tech_component_weight = 40%
return_spread_divisor = 8
technical_spread_divisor = 15
```

Higher values mean speculative stocks are outperforming quality/defensive stocks. Lower values mean the tape favors defensiveness.

## Market Regime

The Market Regime score blends participation, risk appetite, and family-adjusted sector rotation:

```text
market_regime_score =
  weighted_average(
    market_rsi_participation_composite_score,
    risk_appetite_score,
    market_sector_rotation_score
  )
```

Default weights:

```text
market_rsi_participation_composite_score = 40%
risk_appetite_score                      = 30%
market_sector_rotation_score             = 30%
```

The label describes tape direction:

- Risk-Off
- Late Cycle / Defensive
- Neutral / Transition
- Recovery
- Risk-On

## Market Regime Confidence and Status

Confidence does not change the Market Regime score. It measures whether the regime score is trustworthy.

The confidence inputs are:

```text
component_agreement_score =
  weighted_percent_of_market_regime_components_on_same_side_of_50_as_market_regime_score

distance_from_neutral_score =
  min(100, 2 * abs(market_regime_score - 50))

persistence_score =
  percent_of_recent_matching_cached_market_snapshots_on_same_side_of_50_as_current_score
```

Then:

```text
market_regime_confidence =
  weighted_sum(
    component_agreement_score,
    distance_from_neutral_score,
    persistence_score
  )
```

When matching prior snapshots are available, the default confidence weights are:

```text
component_agreement_score  = 45%
distance_from_neutral_score = 35%
persistence_score          = 20%
```

When matching prior snapshots are not available, persistence receives reduced weight, so confidence naturally caps lower.

Status rule:

```text
Confirmed =
  market_regime_confidence >= 70
  and component_agreement_score >= 70

Tentative =
  market_regime_confidence >= 50
  and not Confirmed

Mixed / Transitional =
  market_regime_confidence < 50
```

## Sector Regime Fit

Sector Regime Fit adjusts sector strength for the current Market Regime. It answers: "Is this sector the kind of leadership we want in this regime?"

```text
sector_regime_fit_score =
  weighted_average(
    T_now,
    P_now,
    sector_rsi_participation_composite_score,
    regime_preference_score
  )
```

Default weights:

```text
T_now                                    = 25%
P_now                                    = 25%
sector_rsi_participation_composite_score = 20%
regime_preference_score                  = 30%
```

`regime_preference_score` comes from the configured regime/family matrix, with sector-specific overrides where needed. For example, offensive and cyclical sectors are favored in Risk-On or Recovery regimes, while defensive sectors receive higher preference in Risk-Off or Late Cycle / Defensive regimes.

Visible flags:

```text
Avoid   = 0 to <50
Neutral = 50 to <70
Favored = 70 to 100
```

## Market Alignment

Market Alignment is a stock-level mapping from:

```text
market_regime_label + sector_family + stock_rsi_regime_label
```

to a configured `market_alignment_score`.

This creates a direct bridge from the top-down tape to the stock-level RSI regime. A bullish stock in an offensive family receives high alignment in a Risk-On market, but much lower alignment in Risk-Off. A defensive stock can receive high alignment in Risk-Off even if offensive leadership is being penalized.

## Setup Readiness

Setup Readiness is the final stock-level context score. It blends stock, sector, market, and fundamental context:

```text
setup_readiness_score =
  weighted_average(
    general_technical_score,
    stock_rsi_regime_score,
    fundamental_total_score,
    sector_regime_fit_score,
    market_alignment_score
  )
```

Default weights:

```text
general_technical_score  = 25%
stock_rsi_regime_score   = 20%
fundamental_total_score  = 20%
sector_regime_fit_score  = 20%
market_alignment_score   = 15%
```

The intended interpretation is:

- `general_technical_score`: is the stock technically attractive now?
- `stock_rsi_regime_score`: is the stock structurally behaving like a bull or bear vehicle?
- `fundamental_total_score`: is the business/fundamental profile supportive?
- `sector_regime_fit_score`: is the sector favored in the current market regime?
- `market_alignment_score`: does this stock's regime fit the broader tape?

This phase stops at caching and displaying the score. Trade Ideas integration and historical charting come later.

## Missing data behavior

The engine uses available weighted averages. If an input is unavailable, that component is dropped and the remaining available weights are renormalized for weighted averages. Stock RSI Regime requires enough weekly RSI history; otherwise the ticker is explicitly marked unavailable.

Risk Appetite also shows cohort counts. If either cohort is too small, the score can still be computed, but the Values tab shows a warning so the user can judge reliability.

## Current exclusions

This implementation phase intentionally excludes:

- RSI divergence;
- Trade Ideas tab reconfiguration;
- historical regime charting;
- AI-generated commentary and news synthesis.
