import my_func_support as sup
import my_func as f
import requests
import os
import json

api_key = os.getenv('eodhd')

# Helper function to fetch price data for a given date range
def fetch_eod(symbol, market='US', interval='1d', frequency='d'):

    # Validate that the symbol is provided
    if not symbol:
        raise ValueError("Symbol must be provided.")

    ticker = f"{symbol}.{market}"

    from_date, to_date = sup.parse_period(interval)

    params = {
        'from': from_date,
        'to': to_date,
        'period': frequency,
        'order': 'd',  # Ascending order, from oldest to newest
        'api_token': api_key,
        'fmt': 'json'
    }

    # Prepare the endpoint URL for EOD data
    endpoint = f.generate_url('eod', ticker=ticker)

    # Make the API call
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
    else:
        raise ValueError(f"Price data could not be retrieved. Response status code is {response.status_code}")

    return data