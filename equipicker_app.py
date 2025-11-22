import io
import streamlit as st
from pathlib import Path
import pandas as pd

from equipicker_connect import get_dataframe, CACHE_DIR, cache_path
from equipicker_filters import extreme_accel, accel_normal, accel_weak, wake_up, peg_value
from equipicker_stats import compute_and_save_stats


st.set_page_config(page_title="Screener Runner", layout="wide")

st.title("Equipicker — Screener Runner")

# controls
run_sql = st.sidebar.checkbox("Run SQL (ignore cache)", value=False)

if st.button("Run all filters"):
    # load base dataframe (cached or fresh)
    df = get_dataframe(run_sql=run_sql)
    st.success(f"Base rows: {len(df)} • cache file: {cache_path('xlsx').name}")

    # apply filters in cascade (each from the same df)
    # NEW: Market statistics tab
    tabs = st.tabs(["Market statistics", "Extreme Accel", "Accel Normal", "Accel Weak", "Wake Up", "PEG Value"])

    with tabs[0]:
        market_df, sector_only_df, sector_ind_df, saved_path = compute_and_save_stats(df, CACHE_DIR)

        st.subheader("Market-wide")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Near 1Y High (count)", int(market_df.get("near_1y_high_count", pd.Series([0])).iloc[0]))
        c2.metric("Near 1Y Low (count)", int(market_df.get("near_1y_low_count", pd.Series([0])).iloc[0]))
        c3.metric("Avg RS coeff",
                  round(market_df.get("avg_rs_coeff", pd.Series([float("nan")])).iloc[0], 4) if pd.notna(
                      market_df.get("avg_rs_coeff", pd.Series([None])).iloc[0]) else "n/a")
        c4.metric("Avg OBVM coeff",
                  round(market_df.get("avg_obvm_coeff", pd.Series([float("nan")])).iloc[0], 4) if pd.notna(
                      market_df.get("avg_obvm_coeff", pd.Series([None])).iloc[0]) else "n/a")

        st.write("Market averages")
        market_cols = [c for c in [
            "fundamental_scoring", "technical_scoring", "relative_volume", "relative_performance", "momentum",
            "medium_trend", "long_trend", "rsi_daily", "rsi_weekly", "w1_variation",
            "avg_rs_coeff", "avg_obvm_coeff"
        ] if c in market_df.columns]
        st.dataframe(market_df[["date"] + market_cols], width='stretch', hide_index=True)

        st.subheader("Sector (averages)")
        if sector_only_df.empty:
            st.info("No sector column present.")
        else:
            st.dataframe(sector_only_df.sort_values(["sector"]), width='stretch', hide_index=True)
            # Download button for SECTOR-ONLY stats (your request)
            buf = io.BytesIO();
            sector_only_df.to_excel(buf, index=False)
            st.download_button(
                "Download Sector stats (xlsx)",
                data=buf.getvalue(),
                file_name="sector_stats_current.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.subheader("Sector / Industry (averages)")
        if sector_ind_df.empty:
            st.info("No sector/industry columns present.")
        else:
            st.dataframe(sector_ind_df.sort_values(["sector", "industry"]),
                         width='stretch', hide_index=True)
            # no download button here (sector/industry is already auto-saved daily)
            if saved_path:
                st.caption(f"Daily upsert saved to: {saved_path.name}")

    with tabs[1]:
        res = extreme_accel(df, CACHE_DIR)
        st.write(f"Rows: {len(res)}")
        st.dataframe(res, width='stretch', hide_index=True)

        buf = io.BytesIO()
        res.to_excel(buf, index=False)
        st.download_button(
            "Download Extreme Accel (xlsx)",
            data=buf.getvalue(),
            file_name="extreme_accel.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with tabs[2]:
        res = accel_normal(df, CACHE_DIR)
        st.write(f"Rows: {len(res)}")
        st.dataframe(res, width='stretch')
        buf = io.BytesIO();
        res.to_excel(buf, index=False)
        st.download_button("Download Accel Normal (xlsx)", data=buf.getvalue(),
                           file_name="accel_normal.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tabs[3]:
        res = accel_weak(df, CACHE_DIR)
        st.write(f"Rows: {len(res)}")
        st.dataframe(res, width='stretch')
        buf = io.BytesIO();
        res.to_excel(buf, index=False)
        st.download_button("Download Accel Weak (xlsx)", data=buf.getvalue(),
                           file_name="accel_weak.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tabs[4]:
        res = wake_up(df, CACHE_DIR)
        st.write(f"Rows: {len(res)}")
        st.dataframe(res, width='stretch')
        buf = io.BytesIO();
        res.to_excel(buf, index=False)
        st.download_button("Download Wake Up (xlsx)", data=buf.getvalue(),
                           file_name="wake_up.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tabs[5]:
        res = peg_value(df, CACHE_DIR)  # uses: peg_ratio, peg_ratio_score, fundamental_total_score, sector, market_cap
        st.caption("Result set based on PEG ratio: absolute (<1), and total fundamental scoring ≥ 60.")

        # display per market-cap bucket
        order = ["Mega","Large","Mid","Small","Micro","Nano","Unknown"]
        cols = [c for c in [
            "ticker", "sector", "industry", "market_cap",
            "fundamental_total_score", "fundamental_momentum",
            "peg_ratio", "peg_ratio_score"
        ] if c in res.columns]

        for cat in order:
            sub = res[res.get("cap_category","Unknown").eq(cat)]
            if len(sub):
                st.markdown(f"**{cat} ({len(sub)})**")
                st.dataframe(sub[cols], width="stretch", hide_index=True)

        # download
        buf = io.BytesIO(); res.to_excel(buf, index=False)
        st.download_button("Download PEG Value (xlsx)",
                           data=buf.getvalue(),
                           file_name="peg_value.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Press 'Run all filters' to execute.")
