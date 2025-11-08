

import my_func as f
import pandas as pd
import time
import requests
import json
import io
import os
import csv
import importlib

importlib.reload(f)


# List of symbols is represented by tickers variable from api_testing.py module

# Initialize a DataFrame to store the data
df = pd.DataFrame()
processed_symbols = set()  # Set to keep track of processed symbols

# API call counter
api_calls = 0
processed = 10

# Maximum API calls per minute
max_api_calls_per_minute = 1000

for symbol in tickers:
    try:

        # Skip if symbol has already been processed
        if symbol['Code'] in processed_symbols:
            print(f"Symbol {symbol['Code']} already processed. Skipping.")
            continue

        # Debug: Print the current symbol to see its structure
        print(f"Current symbol: {symbol['Code']} to be processed")

        if api_calls >= max_api_calls_per_minute:
            print("API call limit reached. Sleeping for 60 seconds...")
            time.sleep(60)  # Sleep for 60 seconds
            api_calls = 0  # Reset counter

        # Call the business function and append the result to the DataFrame
        result = f.business(symbol['Code'], fmt='json')

        # Create a new DataFrame for the result and concatenate it
        new_df = pd.DataFrame([result])
        df = pd.concat([df, new_df], ignore_index=True)

        # Add the symbol to the set of processed symbols
        processed_symbols.add(symbol['Code'])

        api_calls += 10
        processed += 1

        print(f"{processed}: Symbol {symbol['Code']} processed succesfully")

    except ValueError as e:
        print(f"Error processing symbol {symbol['Code']}: {e}")

# Save the DataFrame to a CSV file
df.to_csv('us_stock_data.csv', index=False)
df.to_csv('us_stock_data.txt', sep = ",", index=False)


print("Current Working Directory:", os.getcwd())

print(df)
print(f.business(symbol='PATH'))
print(tickers[:2])
print(tickers[1]['Code'])

for symbol in tickers[:1]:
    # Debug: Print the current symbol to see its structure
    print(type(tickers))
    print(type(symbol))
    print(symbol)
    print(f"Current symbol: {symbol['Code']}")




