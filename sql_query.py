

SQL_QUERY = """
WITH latest AS (
  SELECT ic.ticker, ic.eod_price_date
  FROM indicators_computed ic
  -- WHERE ic.ticker IN ('OKLO.US')
),
hist AS (
  SELECT t.ticker, t.date, t.rs, t.obvm, t.adjusted_close
  FROM (
    SELECT
      ed.ticker, ed.date, ed.rs, ed.obvm, ed.adjusted_close,
      ROW_NUMBER() OVER (PARTITION BY ed.ticker ORDER BY ed.date DESC) AS rn
    FROM eod_data ed
    JOIN latest l
      ON l.ticker = ed.ticker
     AND ed.date <= l.eod_price_date
  ) t
  WHERE t.rn <= 50
),
ranked AS (
  SELECT
    h.*,
    ROW_NUMBER() OVER (PARTITION BY h.ticker ORDER BY h.date DESC) AS rn,
    AVG(h.rs)   OVER (PARTITION BY h.ticker ORDER BY h.date
                      ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS rs_sma20,
    AVG(h.obvm) OVER (PARTITION BY h.ticker ORDER BY h.date
                      ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS obvm_sma20
  FROM hist h
),
lastrow AS (
  SELECT
    ticker,
    date,
    rs AS rs_daily,
    rs_sma20,
    obvm AS obvm_daily,
    obvm_sma20,
    adjusted_close AS last_adj_close
  FROM ranked
  WHERE rn = 1
),
fiveago AS (
  SELECT
    ticker,
    adjusted_close AS adj_close_5ago
  FROM ranked
  WHERE rn = 6
),

-- NEW: pick date and close ~30 days before latest eod date, from the 50-day hist window
monthago AS (
  SELECT
    h1.ticker,
    h1.date       AS date_30d_ago,
    h1.adjusted_close AS adj_close_30d_ago
  FROM hist h1
  JOIN latest l
    ON l.ticker = h1.ticker
  WHERE h1.date = (
    SELECT MAX(h2.date)
    FROM hist h2
    WHERE h2.ticker = h1.ticker
      AND h2.date <= DATE_SUB(l.eod_price_date, INTERVAL 30 DAY)
  )
),

monthly_hist AS (
  SELECT em.ticker, em.date, em.high, em.low,
  em.rs, em.obvm,
         ROW_NUMBER() OVER (PARTITION BY em.ticker ORDER BY em.date DESC) AS rn
  FROM eod_monthly em
  JOIN latest l
    ON l.ticker = em.ticker
   AND em.date <= l.eod_price_date
),

monthly_last AS (
  SELECT
    ticker,
    rs  AS rs_monthly,
    obvm AS obvm_monthly
  FROM monthly_hist
  WHERE rn = 1
),

monthly_agg AS (
  SELECT
    ticker,
    MAX(CASE WHEN rn <= 12 THEN high END) AS y1_high,
    MIN(CASE WHEN rn <= 12 THEN low  END) AS y1_low
  FROM monthly_hist
  GROUP BY ticker
),

pillar_scores AS (
  SELECT
    fs.ticker,
    fs.total_score AS fundamental_total_score,
    MAX(CASE WHEN fsp.code = 'VALUE'    THEN fsp.score END) AS fundamental_value,
    MAX(CASE WHEN fsp.code = 'GROWTH'   THEN fsp.score END) AS fundamental_growth,
    MAX(CASE WHEN fsp.code = 'RISK'     THEN fsp.score END) AS fundamental_risk,
    MAX(CASE WHEN fsp.code = 'QUALITY'  THEN fsp.score END) AS fundamental_quality,
    MAX(CASE WHEN fsp.code = 'MOMENTUM' THEN fsp.score END) AS fundamental_momentum
  FROM fundamental_scores fs
  LEFT JOIN fundamental_score_pillars fsp
    ON fsp.fundamental_score_id = fs.id
  GROUP BY fs.ticker, fs.total_score
),
pegdata AS (
  SELECT
    fs.ticker,
    eod.peg_ratio,
    fsc.score AS peg_ratio_score
  FROM fundamental_scores fs
  LEFT JOIN fundamental_score_pillars fsp
    ON fsp.fundamental_score_id = fs.id
  JOIN fundamental_score_criteria fsc
    ON fsc.pillar_id = fsp.id AND fsc.code = 'PEG_RATIO'
  LEFT JOIN indicators_eodhd eod
    ON eod.ticker = fs.ticker
)
SELECT
  s.ticker,
  tk.market_cap,
  tk.sector,
  tk.industry,
  ic.trend,
  s.relative_performance,
  s.relative_volume,
  s.momentum,
  s.intermediate_trend,
  s.long_term_trend,
  s.general_technical_score,
  ic.eod_price_date,
  ic.eod_price_used,
  lr.rs_daily,
  lr.rs_sma20,
  lr.obvm_daily,
  lr.obvm_sma20,
  ma.y1_high AS `1y_high`,
  ma.y1_low  AS `1y_low`,
  CASE WHEN ABS(ic.eod_price_used - ma.y1_high) / NULLIF(ma.y1_high,0) <= 0.05 THEN 'yes' ELSE 'no' END AS near_1y_high_5pct,
  CASE WHEN ABS(ic.eod_price_used - ma.y1_low)  / NULLIF(ma.y1_low,0)  <= 0.05 THEN 'yes' ELSE 'no' END  AS near_1y_low_5pct,
  CASE WHEN ABS(ic.eod_price_used - ic.sma_daily_20) / NULLIF(ic.sma_daily_20,0) <= 0.05 THEN 'yes' ELSE 'no' END AS near_ma20_5pct,
  CASE WHEN ABS(ic.eod_price_used - ic.sma_daily_50) / NULLIF(ic.sma_daily_50,0) <= 0.05 THEN 'yes' ELSE 'no' END AS near_ma50_5pct,
  CASE WHEN ABS(ic.eod_price_used - ic.sma_daily_200)/ NULLIF(ic.sma_daily_200,0) <= 0.05 THEN 'yes' ELSE 'no' END AS near_ma200_5pct,
  100.0 * (lr.last_adj_close / NULLIF(f5.adj_close_5ago, 0) - 1.0) AS `1w_variation`,

  -- NEW: 30-days-ago date and close (from hist)
  mo.date_30d_ago     AS `1m_date`,
  mo.adj_close_30d_ago AS `1m_close`,

  ic.rsi_daily, ic.rsi_weekly,
  ps.fundamental_total_score,
  ps.fundamental_value,
  ps.fundamental_growth,
  ps.fundamental_risk,
  ps.fundamental_quality,
  ps.fundamental_momentum,
  pd.peg_ratio,
  pd.peg_ratio_score,
  ml.rs_monthly,
  ml.obvm_monthly
FROM technical_scoring AS s
JOIN tickers AS tk
  ON tk.ticker = s.ticker
JOIN indicators_computed AS ic
  ON ic.ticker = s.ticker
JOIN lastrow AS lr
  ON lr.ticker = s.ticker
LEFT JOIN monthly_agg AS ma
  ON ma.ticker = s.ticker
LEFT JOIN monthly_last AS ml
  ON ml.ticker = s.ticker
LEFT JOIN pillar_scores AS ps
  ON ps.ticker = s.ticker 
LEFT JOIN fiveago f5
  ON f5.ticker = s.ticker
LEFT JOIN monthago mo
  ON mo.ticker = s.ticker
LEFT JOIN pegdata pd
  ON pd.ticker = s.ticker
WHERE tk.exclude_from_screener = 0
AND tk.market_cap >= 300000000
"""