
import os
import requests
import json
import my_func as f
import importlib

api_key = os.getenv('eodhd')
# print(api_key)

symbol = 'UBER.US'

# FUNDAMENTAL DATA API
endpoint_fundamental =f.generate_url('fundamentals', ticker=symbol)

params_fundamental = {
    'api_token': api_key,
    'fmt' : 'json'
}


response_fundamental = requests.get(endpoint_fundamental, params=params_fundamental)

if response_fundamental.status_code == 200:
    data_fundamental = response_fundamental.json()
    pretty_data_fundamental = json.dumps(data_fundamental, indent=4)
    print(f"Worked!")
else:
    print("Failed to fetch data")

# Extract top-level keys
top_level_keys_fundamental = data_fundamental.keys()
print("Top-level keys:", top_level_keys_fundamental)

# Extract all keys including nested ones
all_keys_fundamental = list(f.extract_keys(data_fundamental))
print("All keys including nested ones:", all_keys_fundamental)


##################################################################################################

# EXCHANGES API
endpoint_exchanges = f.generate_url('exchanges')

params_exchanges = {
    'api_token': api_key,
    'fmt' : 'json'
}

response_exchanges = requests.get(endpoint_exchanges, params=params_exchanges)

if response_exchanges.status_code == 200:
    data_exchanges = response_exchanges.json()
    pretty_data_exchanges = json.dumps(data_exchanges, indent=4)
    print(f"Worked!")
else:
    print("Failed to fetch data")


# EXCHANGES SYMBOL LIST API
code = 'US'
endpoint_exchange_symbol_list = f.generate_url('symbols', exchange_code=code)

params_exchange_symbol_list = {
    'api_token': api_key,
    'fmt': 'json',
    'type': 'common_stock'
}

response_exchange_symbol_list = requests.get(endpoint_exchange_symbol_list, params=params_exchange_symbol_list)

if response_exchange_symbol_list.status_code == 200:
    data_exchange_symbol_list = response_exchange_symbol_list.json()
    pretty_data_exchange_symbol_list = json.dumps(data_exchange_symbol_list, indent=4)
    print(f"Worked!")
else:
    print("Failed to fetch data")


# Get unique exchanges
unique_exchanges = f.get_unique_exchanges(data_exchange_symbol_list)
print("Unique Exchanges:", unique_exchanges)
# Get exchanges summary
exchange_summary = json.dumps(f.get_exchange_summary(data_exchange_symbol_list, unique_exchanges), indent =4)


nasdaq = f.get_exchange_stocks(data_exchange_symbol_list, 'NASDAQ')
nyse = f.get_exchange_stocks(data_exchange_symbol_list, 'NYSE')
tickers = nasdaq + nyse

importlib.reload(f)

print(f.generate_url('fundamentals', exchange_code='US'))
print(f.generate_url('fundamentals', ticker='AACIW.US'))
print(f.generate_url('exchanges', ticker='AAPL.US'))
print(f.generate_url('exchanges'))
print(f.generate_url('symbols', exchange_code='US'))
print(f.generate_url('symbols', ticker='US'))


print(f.business('AAPL'))







