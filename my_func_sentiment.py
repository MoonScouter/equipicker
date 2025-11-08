
import my_func_support as sup
import my_func as f
import requests
import os
from datetime import datetime


api_key = os.getenv('eodhd')


# Function to retrieve sentiment series for a given symbol
def sentiment_series(symbol, market='US', period='0d', news_day=None):

    # Validate that the symbol is provided
    if not symbol:
        raise ValueError("Symbol must be provided.")

    # Construct ticker
    ticker = f"{symbol}.{market}"

    # Handle specific day (news_day) or period
    if news_day:
        # Validate the 'news_day' format
        try:
            # Parse the date string (YYYY-MM-DD format)
            datetime.strptime(news_day, '%Y-%m-%d')
            # If valid, set both 'from' and 'to' to the news_day
            from_date = news_day
            to_date = news_day
        except ValueError:
            raise ValueError("Invalid 'news_day' format. It should be in 'YYYY-MM-DD' format.")
    else:
        # If 'news_day' is not provided, determine 'from' and 'to' based on 'period'
        from_date, to_date = sup.parse_period(period)

    # Prepare parameters for the API call
    params = {
        's': ticker,  # Symbol
        'from': from_date,  # Start date
        'to': to_date,      # End date
        'api_token': api_key,
    }

    # Prepare the endpoint URL for sentiment data
    endpoint = f.generate_url('sentiments')

    # Make the API call
    response = requests.get(endpoint, params=params)

    # Check the status of the API response
    if response.status_code == 200:
        data = response.json()
    else:
        raise ValueError(f"Sentiment data could not be retrieved. Response status code: {response.status_code}")

    # Process the data to extract required fields
    processed_sentiments = {}
    for ticker_code, sentiments in data.items():
        for sentiment in sentiments:
            date = sentiment.get('date')
            sentiment_count = sentiment.get('count', 0)
            normalized_score = sentiment.get('normalized', 0)
            processed_sentiments[date] = {
                'articles_count': sentiment_count,
                'normalized': normalized_score
            }

    # Return the processed sentiments (date as key, sentiment data as value)
    return processed_sentiments