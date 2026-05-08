from pathlib import Path

from prices_service import (
    DIVERGENCE_OB_LEVEL,
    _build_active_divergence_series,
    _find_price_pivots,
    compute_rsi_divergence_state,
    divergence_settings_for_frequency,
    load_prices_cache,
)


df = load_prices_cache(Path("data/prices_daily_2026.jsonl"))
df = df[df["ticker"].eq("SNDK.US")].sort_values("date").reset_index(drop=True)
settings = divergence_settings_for_frequency("daily")
pivots = _find_price_pivots(df["adjusted_high"], df["rsi_14"], kind="high", settings=settings)
active = _build_active_divergence_series(
    pivots=pivots,
    settings=settings,
    row_count=len(df),
    anchor_threshold=DIVERGENCE_OB_LEVEL,
    anchor_cmp=lambda current, threshold: current > threshold,
    anchor_break_cmp=lambda current, anchor: current > anchor,
    price_divergence_cmp=lambda current, anchor: current > anchor,
    rsi_divergence_cmp=lambda current, anchor: current < anchor,
    prefer_more_extreme_cmp=lambda current, previous: current > previous,
)
state = compute_rsi_divergence_state(df, frequency="daily")
print("PIVOTS")
for pivot in pivots:
    if "2026-02-01" <= str(df.loc[pivot.index, "date"]) <= "2026-04-30":
        print(
            pivot.index,
            df.loc[pivot.index, "date"],
            "high",
            df.loc[pivot.index, "adjusted_high"],
            "rsi",
            df.loc[pivot.index, "rsi_14"],
            "confirm_on",
            df.loc[pivot.confirmation_index, "date"],
        )
print("\nROWS")
for idx, row in df.iterrows():
    if "2026-03-16" <= str(row["date"]) <= "2026-04-21":
        active_state = active[idx]
        pair = ""
        if active_state is not None:
            pair = f"{df.loc[active_state.anchor_index, 'date']} -> {df.loc[active_state.pivot2_index, 'date']}"
        print(
            idx,
            row["date"],
            "high",
            row["adjusted_high"],
            "rsi",
            round(float(row["rsi_14"]), 4),
            "flag",
            state.loc[idx, "rsi_divergence_flag"],
            "confirmed",
            bool(state.loc[idx, "rsi_divergence_confirmed"]),
            pair,
        )
