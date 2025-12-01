

SQL_QUERY_SCORING = """
SELECT
    fs.ticker, t.name, t.exchange, t.sector, t.industry, t.market_cap, t.market_cap_category, eodhd.beta,
    fs.total_score AS fundamental_total_score, fs.style AS style,fs.most_recent_quarter AS most_recent_quarter,
    MAX(CASE WHEN fsp.code = 'VALUE'    THEN fsp.score END) AS fundamental_value,
    MAX(CASE WHEN fsp.code = 'GROWTH'   THEN fsp.score END) AS fundamental_growth,
    MAX(CASE WHEN fsp.code = 'RISK'     THEN fsp.score END) AS fundamental_risk,
    MAX(CASE WHEN fsp.code = 'QUALITY'  THEN fsp.score END) AS fundamental_quality,
    MAX(CASE WHEN fsp.code = 'MOMENTUM' THEN fsp.score END) AS fundamental_momentum
  FROM fundamental_scores fs
  LEFT JOIN fundamental_score_pillars fsp
      ON fsp.fundamental_score_id = fs.id
  LEFT JOIN tickers t 
  	  ON t.ticker = fs.ticker
  LEFT JOIN indicators_eodhd eodhd 
  	  ON eodhd.ticker = fs.ticker
   WHERE t.exclude_from_screener = 0
   AND t.market_cap >= 5000000000
  GROUP BY fs.ticker, fs.total_score;
"""