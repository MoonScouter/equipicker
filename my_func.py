import csv
import io
import os
import requests
import pandas as pd
import my_func_support as sup
import my_func_ratios as ratio
import my_func_capital as cap
import my_func_profitability as profit
import my_func_cashflow as cash
import my_func_news as newsflow
import my_func_sentiment as sentiment

api_key = os.getenv('eodhd')


# Function to construct the endpoint url depending on API needed to retrieve data
def generate_url(endpoint_name, ticker=None, exchange_code=None):
    base_url = 'https://eodhd.com/api/'

    # Define the allowed endpoints and their URL patterns
    endpoints = {
        'fundamentals': 'fundamentals/{}',
        'exchanges': 'exchanges-list/',
        'symbols': 'exchange-symbol-list/{}',
        'news': 'news/',
        'sentiments': 'sentiments/',
        'eod': 'eod/{}'
        # Add other endpoints here as needed
    }

    # Validate parameters for each endpoint
    if endpoint_name == 'fundamentals':
        if ticker and not exchange_code:
            endpoint_url = endpoints[endpoint_name].format(ticker)
        else:
            raise ValueError("Fundamentals endpoint requires only the 'ticker' parameter")

    elif endpoint_name == 'exchanges':
        if not exchange_code and not ticker:
            endpoint_url = endpoints[endpoint_name]
        else:
            raise ValueError("Exchange list endpoint does not require any parameter")

    elif endpoint_name == 'symbols':
        if exchange_code and not ticker:
            endpoint_url = endpoints[endpoint_name].format(exchange_code)
        else:
            raise ValueError("Exchange symbol list endpoint requires only the 'exchange_code' parameter")

    elif endpoint_name == 'news':
        if not exchange_code and not ticker:
            endpoint_url = endpoints[endpoint_name]
        else:
            raise ValueError("News endpoint does not require any parameter")

    elif endpoint_name == 'sentiments':
        if not exchange_code and not ticker:
            endpoint_url = endpoints[endpoint_name]
        else:
            raise ValueError("Sentiment endpoint does not require any parameter")

    elif endpoint_name == 'eod':
        if ticker and not exchange_code:
            endpoint_url = endpoints[endpoint_name].format(ticker)
        else:
            raise ValueError("Eod endpoint requires only the 'ticker' parameter")

    else:
        raise ValueError(f"Invalid endpoint: {endpoint_name}")

    return base_url + endpoint_url


# Function calling to fundamentals API for business activity related information for symbol
def business(symbol, market='US', fmt='json', raw_data=None):

    if raw_data is None:
        # Generate endpoint
        endpoint = generate_url('fundamentals', ticker=f'{symbol}.{market}')

        params = {
            'api_token': api_key,
            'fmt': fmt
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
        else:
            raise ValueError(f"Data could not be retrieved. Response status code is {response.status_code}")
    else:
        data = raw_data

    # Handle null values in MarketCapitalization
    market_cap = data["Highlights"].get("MarketCapitalization")
    if market_cap is None:
        market_cap_value = "-"
    else:
        market_cap_value = round(market_cap / 1e9, 2)  # Convert to billions

    # Extract required fields
    extracted_data = {
        "Code": data["General"]["Code"],
        "Name": data["General"]["Name"],
        "Exchange": data["General"]["Exchange"],
        "FiscalYearEnd": data["General"]["FiscalYearEnd"],
        "GicSector": data["General"]["GicSector"],
        "GicIndustry": data["General"]["GicIndustry"],
        "GicSubIndustry": data["General"]["GicSubIndustry"],
        "Description": data["General"]["Description"],
        "Beta": data["Technicals"].get("Beta"),
        "MarketCapitalization": market_cap_value  # Convert to billions
        }
    # Categorize based on Market Capitalization
    market_cap = extracted_data["MarketCapitalization"]
    if market_cap_value == "-":
        cap_category = "Unknown"
    elif market_cap_value < 0.05:
        cap_category = "Nano"
    elif 0.05 <= market_cap_value < 0.3:
        cap_category = "Micro"
    elif 0.3 <= market_cap_value < 2:
        cap_category = "Small"
    elif 2 <= market_cap_value < 10:
        cap_category = "Mid"
    elif 10 <= market_cap_value < 200:
        cap_category = "Large"
    else:
        cap_category = "Mega"

    extracted_data["MarketCapCategory"] = cap_category

    # Return data in the requested format
    if fmt == 'json':
        return extracted_data
    elif fmt == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(extracted_data.keys())
        writer.writerow(extracted_data.values())
        return output.getvalue().strip()
    else:
        raise ValueError("Unsupported format. Please use 'json' or 'csv'.")


# # Function to retrieve ratios overview for list of stocks
# def ratios(symbols: list, market='US', type='multiples'):
#
#     # Classify type parameter
#     type_class = {'multiples': 1, 'overview': 2, 'dupont': 3, 'capital': 4}
#     type_instance = type_class[type]
#
#     result = {}
#
#     # Iterate through the list of symbols
#     for symbol in symbols:
#         ratios = ratio.get_ratios(symbol, market=market)
#
#         extracted_data = {}
#         for key, value in ratios.items():
#             if type_instance in value['type']:
#                 extracted_data[value['label']] = value['value']
#
#         # Store extracted data in result dictionary
#         result[symbol] = extracted_data
#
#     result = pd.DataFrame(result)
#
#     # Add average when there is more than one symbol
#     if len(symbols) > 1:
#         # Apply the function to each row, excluding 'FiscalYearEnd' and 'LatestQuarter'
#         result['Average'] = result.apply(
#             lambda row: 'NA' if row.name in ['FiscalYearEnd', 'MostRecentQuarter', 'MarketCapitalization']
#             else sup.calculate_mean(row), axis=1)
#
#     # print('It worked!')
#     return result


# Function to retrieve ratios overview for list of stocks - NEW
def ratios(symbols: list, market='US', type='multiples', convert='bln', ref_symbol=None, raw_data=None):

    # Classify type parameter
    type_class = {'multiples': 1, 'overview': 2, 'dupont': 3, 'capital': 4}
    type_instance = type_class[type]

    result = {}

    # Iterate through the list of symbols
    for symbol in symbols:
        if ref_symbol is not None and raw_data is not None and symbol == ref_symbol:
            ratios = ratio.get_ratios(symbol, market=market, convert=convert, raw_data=raw_data)

            extracted_data = {}
            for key, value in ratios.items():
                if type_instance in value['type']:
                    extracted_data[value['label']] = value['value']

            # Store extracted data in result dictionary
            result[symbol] = extracted_data
        else:
            ratios = ratio.get_ratios(symbol, market=market, convert=convert)

            extracted_data = {}
            for key, value in ratios.items():
                if type_instance in value['type']:
                    extracted_data[value['label']] = value['value']

            # Store extracted data in result dictionary
            result[symbol] = extracted_data

    result = pd.DataFrame(result)

    # Add average when there is more than one symbol
    if len(symbols) > 1:
        # Apply the function to each row, excluding 'FiscalYearEnd' and 'LatestQuarter'
        result['Average'] = result.apply(
            lambda row: 'NA' if row.name in ['FiscalYearEnd', 'MostRecentQuarter', 'MarketCapitalization', 'QuarterlyRevenueGrowthYOY','QuarterlyEarningsGrowthYOY']
            else sup.calculate_mean(sup.convert_float(value, convert=None, decimal=2) for value in row), axis=1)

    # print('It worked!')
    return result


# # Function to retrieve capital structure for a given stock
# def capital_old(symbol, market='US', type='overview'):
#
#     # Classify type parameter
#     type_class = {'overview': 1, 'indicators': 2}
#     type_instance = type_class[type]
#
#     rows = []
#
#     capital_structure = cap.capital_structure(symbol=symbol, market=market)['capital_structure_indicators']
#
#     for key, value in capital_structure.items():
#         if type_instance in value['type']:
#             row = value['value']
#             row['metric'] = value['label']
#             row['comment'] = value['comment']
#             rows.append(row)
#
#     # Create dataframe
#     result = pd.DataFrame(rows)
#
#     # Set the 'label' column as the index of the DataFrame
#     result.set_index('metric', inplace=True)
#
#     # print('It worked!')
#     return result


# Function to retrieve capital structure for a given stock - NEW
def capital(symbol, market='US', type='overview', convert='bln', raw_data=None):

    # Classify type parameter
    type_class = {'overview': 1, 'indicators': 2}
    type_instance = type_class[type]

    rows = []

    capital_structure = cap.capital_structure(symbol=symbol, market=market, convert=convert, raw_data=raw_data)

    for key, value in capital_structure.items():
        if type_instance in value['type']:
            row = value['value']
            row['metric'] = value['label']
            row['comment'] = value['comment']
            rows.append(row)

    # Create dataframe
    result = pd.DataFrame(rows)

    # Set the 'label' column as the index of the DataFrame
    result.set_index('metric', inplace=True)

    # print('It worked!')
    return result


# Function to retrieve profitability data for a given stock
def profitability(symbol, market='US', type='overview', convert='bln', earnings_time=5, raw_data=None):

    # Classify type parameter
    type_class = {'overview': 1, 'indicators': 2, 'margins': 3, 'earnings_surprise': 4, 'momentum': 5}
    type_instance = type_class[type]

    perf = profit.profitability(symbol=symbol, market=market, convert=convert, earnings_time=earnings_time, raw_data=raw_data)

    rows = []
    metrics = perf['profitability']
    for key, value in metrics.items():
        if type_instance in value['type']:
            row = value['value']
            row['metric'] = value['label']
            row['comment'] = value['comment']
            rows.append(row)

    if rows:
        # Create dataframe for profitability metrics
        result_profit = pd.DataFrame(rows)
        # Set the 'label' column as the index of the DataFrame
        result_profit.set_index('metric', inplace=True)
    else:
        result_profit = pd.DataFrame()

    rows = []
    earnings = perf['earnings']
    for key, value in earnings.items():
        if type_instance in value['type']:
            row = value['value']
            row['metric'] = value['label']
            row['comment'] = value['comment']
            rows.append(row)

    if rows:
        # Create dataframe for earnings surprise data
        result_earnings = pd.DataFrame(rows)
        # Set the 'label' column as the index of the DataFrame
        result_earnings.set_index('metric', inplace=True)
        # Transpose Dataframe
        result_earnings.transpose()
    else:
        result_earnings = pd.DataFrame()

    result = {'profit_indicators_and_margins': result_profit, 'earnings_surprise': result_earnings}

    # print('It worked!')
    return result


# Function to retrieve cash flow data for a given stock
def cashflow(symbol, market='US', type='overview', convert='bln', raw_data=None):

    # Classify type parameter
    type_class = {'overview': 1, 'indicators': 2}
    type_instance = type_class[type]

    cashflow = cash.cashflow(symbol=symbol, market=market, convert=convert, raw_data=raw_data)

    rows = []
    for key, value in cashflow.items():
        if type_instance in value['type']:
            row = value['value']
            row['metric'] = value['label']
            row['comment'] = value['comment']
            rows.append(row)

    # Create dataframe for profitability metrics
    result = pd.DataFrame(rows)
    # Set the 'label' column as the index of the DataFrame
    result.set_index('metric', inplace=True)

    # print('It worked!')
    return result


# Function to retrieve all fundamental data for a given stock
def one_shot_analysis(symbol, peers: list = [], market='US', type='overview', convert='bln', earnings_time=5):
    result = {}

    endpoint = generate_url('fundamentals', ticker=f'{symbol}.{market}')

    params = {
        'api_token': api_key,
        'fmt': 'json'
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
    else:
        raise ValueError(f"Data could not be retrieved. Response status code is {response.status_code}")

    company_description = business(symbol=symbol, raw_data=data)
    capital_structure_data = capital(symbol=symbol, market=market, type=type, convert=convert, raw_data=data)
    profitability_data = profitability(symbol=symbol, market=market, type=type, convert=convert, earnings_time=earnings_time, raw_data=data)
    cashflow_data = cashflow(symbol, market=market, type=type, convert=convert, raw_data=data)

    peers_all = [symbol] + peers

    relative_analysis_data = ratios(symbols=peers_all, market=market, type='overview', convert=convert, ref_symbol=symbol, raw_data=data)

    result['company_description'] = company_description
    result['capital_structure_data'] = capital_structure_data
    result['profitability_data'] = profitability_data
    result['cashflow_data'] = cashflow_data
    result['relative_analysis_data'] = relative_analysis_data

    return result


# Function to retrieve news flow for listed stocks
def news(symbol=None, market='US', limit=5, period='0d', news_day=None, tag=None, mode='latest'):

    result = newsflow.fetch_news(symbol=symbol, market=market, limit=limit, period=period, news_day=news_day, tag=tag, mode=mode)

    return result


# Function to retrieve news flow for listed stocks
def news_sentiment(symbol, market='US', period='0d', news_day=None):

    result = sentiment.sentiment_series(symbol=symbol, market=market, period=period, news_day=news_day)

    return result

