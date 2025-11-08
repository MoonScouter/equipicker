import pandas as pd
import requests
import os
from time import sleep

# ===== CONFIGURATION =====
# Path to input tickers Excel file
TICKERS_FILE = r"C:\Users\razva\Desktop\Portfolio\Trading Team\Equipicker\Analiza Fundamentala - curs\tickers.xlsx"
# Output file path
OUTPUT_FILE = r"C:\Users\razva\Desktop\Portfolio\Trading Team\Equipicker\Analiza Fundamentala - curs\analyst_ratings.xlsx"
# EODHD API token
API_TOKEN = "6787b8a805ada2.08489565"
# Use Excel file or manual ticker list for testing
USE_MANUAL_LIST = False  # Set to False to use Excel file

# ===== MANUAL TICKERS (for testing) =====
MANUAL_TICKERS = [
    {"full_ticker": "WRAP.US", "Name": "Uber", "Sector": "Technology", "Industry": "Software"},
    {"full_ticker": "UBER.US", "Name": "Uber", "Sector": "Technology", "Industry": "Software"},
    {"full_ticker": "MSFT.US", "Name": "Microsoft", "Sector": "Technology", "Industry": "Software"},
    # {"full_ticker": "AAPL.US", "Name": "Apple", "Sector": "Technology", "Industry": "Consumer Electronics"},
    # {"full_ticker": "BMY.US", "Name": "Bristol-Myers Squibb Company", "Sector": "Healthcare", "Industry": "Drug Manufacturers - General"},
    # {"full_ticker": "MMM.US", "Name": "3M Company", "Sector": "Industrials", "Industry": "Conglomerates"},
    # {"full_ticker": "F.US", "Name": "Ford Motor Company", "Sector": "Consumer Cyclical", "Industry": "Auto Manufacturers"},
    # {"full_ticker": "BBY.US", "Name": "Best Buy Co. Inc", "Sector": "Consumer Cyclical",
    #  "Industry": "Specialty Retail"},
]


# ===== FUNCTION TO FETCH FUNDAMENTALS FROM EODHD =====
def get_analyst_ratings(ticker):
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={API_TOKEN}&fmt=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        ratings = data.get("AnalystRatings", {})
        return ratings
    except Exception as e:
        print(f"[ERROR] Ticker {ticker}: {e}")
        return None


# ===== MAIN PROCESSING =====
def main():
    print("[INFO] Starting analyst ratings extraction...")

    # Load tickers
    if USE_MANUAL_LIST:
        tickers_df = pd.DataFrame(MANUAL_TICKERS)
    else:
        tickers_df = pd.read_excel(TICKERS_FILE, sheet_name='db')

    # Prepare result list
    results = []
    total_companies = len(tickers_df)
    for idx, row in tickers_df.iterrows():
        ticker = row['full_ticker']
        name = row.get('Name', '')
        sector = row.get('Sector', '')
        industry = row.get('Industry', '')
        print(f"[INFO] ({idx + 1}/{total_companies}) Processing {ticker} ({name})...")

        ratings = get_analyst_ratings(ticker)
        if ratings:
            strong_buy = ratings.get("StrongBuy")
            buy = ratings.get("Buy")
            hold = ratings.get("Hold")
            sell = ratings.get("Sell")
            strong_sell = ratings.get("StrongSell")
            rating = ratings.get("Rating", None)
            target_price = ratings.get("TargetPrice", None)
            ratings_list = [strong_buy, buy, hold, sell, strong_sell]
            # Only sum values that are not None
            total = sum(val for val in ratings_list if val is not None)

            if total >= 3:
                pct_strong_buy = (strong_buy / total * 100) if strong_buy is not None else None
                pct_buy = (buy / total * 100) if buy is not None else None
                pct_hold = (hold / total * 100) if hold is not None else None
                pct_sell = (sell / total * 100) if sell is not None else None
                pct_strong_sell = (strong_sell / total * 100) if strong_sell is not None else None
            else:
                pct_strong_buy = pct_buy = pct_hold = pct_sell = pct_strong_sell = None

        else:
            strong_buy = buy = hold = sell = strong_sell = rating = target_price = None
            pct_strong_buy = pct_buy = pct_hold = pct_sell = pct_strong_sell = None
            total = None

        results.append({
            "Ticker": ticker,
            "Name": name,
            "Sector": sector,
            "Industry": industry,
            "Rating": rating,
            "TargetPrice": target_price,
            "StrongBuy": strong_buy,
            "Buy": buy,
            "Hold": hold,
            "Sell": sell,
            "StrongSell": strong_sell,
            "TotalRatings": total,
            "%StrongBuy": pct_strong_buy,
            "%Buy": pct_buy,
            "%Hold": pct_hold,
            "%Sell": pct_sell,
            "%StrongSell": pct_strong_sell,
        })
        print(f"[INFO] Completed {ticker}")
        sleep(1)  # To avoid API rate limiting

    # Convert results to DataFrame
    res_df = pd.DataFrame(results)

    # Compute global and sector-level averages
    def avg_if(df, col, sector=None):
        data = df[df['TotalRatings'] >= 3]
        if sector:
            data = data[data['Sector'] == sector]
        return data[col].mean()

    # List of columns for percentages
    pct_cols = ["%StrongBuy", "%Buy", "%Hold", "%Sell", "%StrongSell"]
    summary = []
    # Global averages
    global_avg = {col: avg_if(res_df, col) for col in pct_cols}
    global_avg["Sector"] = "ALL"
    summary.append(global_avg)
    # Sector averages
    for sector in res_df['Sector'].dropna().unique():
        sector_avg = {col: avg_if(res_df, col, sector=sector) for col in pct_cols}
        sector_avg["Sector"] = sector
        summary.append(sector_avg)
    summary_df = pd.DataFrame(summary).set_index("Sector")

    # ===== SAVE TO EXCEL =====
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        res_df.to_excel(writer, index=False, sheet_name="AnalystRatings")
        summary_df.to_excel(writer, sheet_name="Averages")
    print(f"[INFO] Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
