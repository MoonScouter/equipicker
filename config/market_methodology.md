# Market / Regime Methodology

This page documents the deterministic Market layer used in Equipilot. The engine does not use AI for scoring. AI is reserved for later commentary and explanation.

## Analytical spine

`Market Regime -> Sector Rotation / Sector Fit -> Stock RSI Regime -> Setup Readiness`

Each layer answers a different question:

- Market Regime: what type of tape are we in?
- Sector Rotation: where is capital flowing and how healthy is that leadership?
- Sector Regime Fit: which sectors are appropriate to focus on for the current backdrop?
- Stock RSI Regime: is a stock structurally behaving like a bull, bear, or neutral vehicle over the chosen interval?
- Setup Readiness: how attractive is the stock after combining stock, sector, and market context?

All scores stay on a `0-100` scale. Labels are deterministic translations of scores.

## Anchors

The Values tab uses four user-controlled anchors:

- Evaluation date
- RSI regime start date
- 1 month ago date
- 1 week ago date

The RSI interval uses all daily and weekly RSI values between `RSI regime start date` and `Evaluation date`, inclusive.

## Stock RSI Regime

The stock regime score is built from four blocks:

1. Weekly range persistence
2. Weekly expansion frequency
3. Pullback resilience / rally failure
4. Daily confirmation

The engine computes separate bull and bear evidence, then translates the difference into the final `stock_rsi_regime_score`.

If daily RSI data is missing, the daily-confirmation block is dropped and the remaining weights are renormalized. If weekly history is too sparse, the stock regime output is marked unavailable.

## Sector and Market Participation

Sector participation is computed from stock RSI regime outputs:

- `% of stocks with score >= 60`
- `% of stocks with score >= 75`
- `% of stocks with score < 40`
- average stock RSI regime score

The same metrics are then aggregated across the full evaluation-date universe for the market-wide participation view.

## Sector Rotation

Sector rotation combines four inputs:

- `P`: sector participation / leadership using the Quadrants participation formula
- `T`: sector-average actual technical score
- sector RSI participation composite
- trend-of-change score from `dT` and `dP`

`dT` and `dP` compare current sector state with the selected 1-week and 1-month anchors. Scores above `50` indicate improvement, below `50` indicate deterioration.

## Risk Appetite

Risk appetite compares two cohorts:

- Quality / Defensive
- Speculative

The score blends:

- 1-month return spread between the cohorts
- technical-score spread between the cohorts

The Values tab also shows the number of companies in both cohorts so the user can judge whether the comparison is meaningful.

## Market Regime

The top-down market score blends:

- market RSI participation composite
- risk appetite score
- market sector rotation score

The market regime label describes tape direction, while confidence and status describe how aligned and trustworthy the tape is.

Confidence uses:

- component agreement
- distance from neutral
- persistence from prior cached market snapshots with matching interval settings

If not enough history exists, persistence gets reduced weight and confidence naturally caps lower.

## Sector Regime Fit

Sector regime fit adjusts sector strength for the current backdrop. It combines:

- sector technical score
- sector participation score
- sector RSI participation composite
- regime preference score from config

Visible flags are:

- `Favored`
- `Neutral`
- `Avoid`

## Setup Readiness

Setup Readiness is the final stock-level context score. It blends:

- general technical score
- stock RSI regime score
- fundamental total score
- sector regime fit score
- market alignment score

This phase stops at caching and displaying the score. Trade Ideas integration and historical charting come later.

## Current exclusions

This implementation phase intentionally excludes:

- RSI divergence
- Trade Ideas tab reconfiguration
- historical regime charting
- AI-generated commentary and news synthesis
