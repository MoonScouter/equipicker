

import my_func_support as sup
import my_func_capital as cap
import my_func_profitability as profit
import my_func as f
import requests
import os
from datetime import datetime

api_key = os.getenv('eodhd')

def get_ratios(symbol, market='US', convert = 'bln', raw_data=None):

    if raw_data is None:
        # Prepare url
        endpoint = f.generate_url('fundamentals', ticker=f'{symbol}.{market}')

        params = {
            'api_token': api_key,
            'fmt': 'json'
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
        else:
            raise ValueError(f"Data could not be retrieved. Response status code is {response.status_code}")
    else:
        data = raw_data

    # Check most recent quarter info
    quarter_info = sup.get_quarter_info(data)
    mrq = quarter_info['mrq']
    prev_year_mrq = quarter_info['prev_year_mrq']
    prev_q = quarter_info['prev_q']
    prev_year_prev_q = quarter_info['prev_year_prev_q']
    check_mrq = quarter_info['check_mrq']
    mrq_formatted = datetime.strptime(mrq, '%Y-%m-%d').strftime('%b-%y')  # Formatted as mmm-yy, ex. Jun-23

    # Create labels
    mrq_label = f"Most Recent Quarter - {mrq_formatted}"
    yoy_change = 'Y/Y Change'
    qoq_change = 'Q/Q Change'
    ttm = 'Trailing Twelve Months'
    ttm_change = 'Trailing Twelve Months Y/Y change'

    # Initialize some variables
    currency = data['General']['CurrencyCode']
    decimal = 2
    decimal_mg = 4

    perc_suffix = '%'
    pp_suffix = 'p.p'
    x_suffix = 'x'
    convert_suffix = f'{convert}'

    data_set_income_statement = data["Financials"]["Income_Statement"]
    data_set_balance_sheet = data["Financials"]["Balance_Sheet"]

    # Create dictionary with fields
    fields = {
        "FiscalYearEnd": {'label': "FiscalYearEnd", 'type': [2], 'value': None},
        "MostRecentQuarter": {'label': "MostRecentQuarter", 'type': [1, 2], 'value': None},
        "MarketCapitalization": {'label': "MarketCapitalization (bln)", 'type': [2], 'value': None},
        "TrailingPE": {'label': "TrailingPE", 'type': [1, 2], 'value': None},
        "ForwardPE": {'label': "ForwardPE", 'type': [1, 2], 'value': None},
        "PriceSalesTTM": {'label': "PriceToSales", 'type': [1, 2], 'value': None},
        "PriceBookMRQ": {'label': "PriceToBook", 'type': [1, 2], 'value': None},
        "PEGRatio": {'label': "PEGRatio", 'type': [1, 2], 'value': None},
        "EnterpriseValueRevenue": {'label': "EVToRevenue", 'type': [1, 2], 'value': None},
        "EnterpriseValueEbitda": {'label': "EVToEBITDA", 'type': [1, 2], 'value': None},
        "DividendShare": {'label': "DividendShare", 'type': [2], 'value': None},
        "DividendYield": {'label': "DividendYield %", 'type': [2], 'value': None},
        "EarningsShare": {'label': "EPS", 'type': [2], 'value': None},
        "DilutedEpsTTM": {'label': "DilutedEPS", 'type': [2], 'value': None},
        "EPSEstimateCurrentQuarter": {'label': "EPSEstimateCurrentQuarter", 'type': [2], 'value': None},
        "EPSEstimateNextQuarter": {'label': "EPSEstimateNextQuarter", 'type': [2], 'value': None},
        "EPSEstimateCurrentYear": {'label': "EPSEstimateCurrentYear", 'type': [2], 'value': None},
        "EPSEstimateNextYear": {'label': "EPSEstimateNextYear", 'type': [2], 'value': None},
        "RevenuePerShareTTM": {'label': "RevenuePerShare", 'type': [2], 'value': None},
        "ReturnOnAssetsTTM": {'label': "ROA %", 'type': [2, 3], 'value': None},
        "ReturnOnEquityTTM": {'label': "ROE %", 'type': [2, 3], 'value': None},
        "ProfitMarginTTM": {'label': "ProfitMargin %", 'type': [2, 3], 'value': None},
        "OperatingMarginTTM": {'label': "OperatingMargin %", 'type': [2], 'value': None},
        "AssetTurnover": {'label': "AssetTurnover", 'type': [2, 3], 'value': None},
        "Leverage": {'label': "Leverage", 'type': [2, 3], 'value': None},
        "QuarterlyRevenueGrowthYOY": {'label': "QuarterlyRevenueGrowthYOY", 'type': [2], 'value': None},
        "QuarterlyEarningsGrowthYOY": {'label': "QuarterlyEarningsGrowthYOY", 'type': [2], 'value': None},
        "Beta": {'label': "Beta", 'type': [2], 'value': None},
        "DebtToEquity": {'label': "Total Liabilities-To-Equity", 'type': [2,4], 'value': None},
        "LTDebtToEquity": {'label': "Non Current Liabilities-To-Equity", 'type': [2,4], 'value': None},
        "DebtToCapital": {'label': "DebtToCapital", 'type': [2,4], 'value': None},
        "LTDebtToCapital": {'label': "LTDebtToCapital", 'type': [2,4], 'value': None},
        "InterestCoverageRatio": {'label': "InterestCoverageRatio", 'type': [2,4], 'value': None}
        }

    # Get values for fields
    fields['FiscalYearEnd']['value'] = data["General"].get("FiscalYearEnd")
    fields['MostRecentQuarter']['value'] = mrq

    # Handle computed value in MarketCapitalization
    market_cap = data["Highlights"].get("MarketCapitalization")
    if market_cap is None:
        market_cap_value = "-"
    else:
        market_cap_value = sup.convert_float(market_cap, convert=convert, decimal=decimal)  # Convert to billions
    # Assign value to dictionary
    fields["MarketCapitalization"]['value'] = market_cap_value

    fields['TrailingPE']['value'] = sup.convert_float(data["Valuation"].get("TrailingPE"))
    fields['ForwardPE']['value'] = sup.convert_float(data["Valuation"].get("ForwardPE"))
    fields['PriceSalesTTM']['value'] = sup.convert_float(data["Valuation"].get("PriceSalesTTM"))
    fields['PriceBookMRQ']['value'] = sup.convert_float(data["Valuation"].get("PriceBookMRQ"))
    fields['PEGRatio']['value'] = sup.convert_float(data["Highlights"].get("PEGRatio"))
    fields['EnterpriseValueRevenue']['value'] = sup.convert_float(data["Valuation"].get("EnterpriseValueRevenue"))
    fields['EnterpriseValueEbitda']['value'] = sup.convert_float(data["Valuation"].get("EnterpriseValueEbitda"))
    fields['DividendShare']['value'] = sup.convert_float(data["Highlights"].get("DividendShare"))
    fields['DividendYield']['value'] = sup.safe_multiply(sup.convert_float(data["Highlights"].get("DividendYield"), decimal=decimal_mg),100)
    fields['EarningsShare']['value'] = sup.convert_float(data["Highlights"].get("EarningsShare"))
    fields['DilutedEpsTTM']['value'] = sup.convert_float(data["Highlights"].get("DilutedEpsTTM"))
    fields['EPSEstimateCurrentQuarter']['value'] = sup.convert_float(data["Highlights"].get("EPSEstimateCurrentQuarter"))
    fields['EPSEstimateNextQuarter']['value'] = sup.convert_float(data["Highlights"].get("EPSEstimateNextQuarter"))
    fields['EPSEstimateCurrentYear']['value'] = sup.convert_float(data["Highlights"].get("EPSEstimateCurrentYear"))
    fields['EPSEstimateNextYear']['value'] = sup.convert_float(data["Highlights"].get("EPSEstimateNextYear"))
    fields['RevenuePerShareTTM']['value'] = sup.convert_float(data["Highlights"].get("RevenuePerShareTTM"))
    fields['Beta']['value'] =  sup.convert_float(data["Technicals"].get("Beta"), decimal=decimal_mg)


    # Handle computed values for profitability and capital indicators
    # Get values from profitability and capital functions
    profit_indicators = profit.profitability(symbol=symbol, market=market, convert=convert, earnings_time=5)
    capital_indicators = cap.capital_structure(symbol=symbol, market=market, convert=convert)

    # Profitability
    fields["ReturnOnAssetsTTM"]['value'] = sup.safe_multiply(sup.safe_divide(sup.convert_float(profit_indicators['profitability']['netIncome']['value'][ttm]), sup.convert_float(capital_indicators['totalAssets']['value'][ttm]), decimal=decimal_mg), element2 = 100)
    fields["ReturnOnEquityTTM"]['value'] = sup.safe_multiply(sup.safe_divide(sup.convert_float(profit_indicators['profitability']['netIncome']['value'][ttm]), sup.convert_float(capital_indicators['totalStockholderEquity']['value'][ttm]), decimal=decimal_mg), element2 = 100)
    fields["ProfitMarginTTM"]['value'] = profit_indicators['profitability']['netMargin']['value'][ttm]
    fields["OperatingMarginTTM"]['value'] = profit_indicators['profitability']['operatingMargin']['value'][ttm]
    fields["QuarterlyRevenueGrowthYOY"]['value'] = profit_indicators['profitability']['totalRevenue']['value'][yoy_change]
    fields["QuarterlyEarningsGrowthYOY"]['value'] = profit_indicators['profitability']['epsActual']['value'][yoy_change]

    # Capital
    fields["DebtToEquity"]['value'] = capital_indicators['DebtToEquity']['value'][ttm]
    fields["LTDebtToEquity"]['value'] = capital_indicators['LTDebtToEquity']['value'][ttm]
    fields["DebtToCapital"]['value'] = capital_indicators['DebtToCapital']['value'][ttm]
    fields["LTDebtToCapital"]['value'] = capital_indicators['LTDebtToCapital']['value'][ttm]
    fields["InterestCoverageRatio"]['value'] = capital_indicators['InterestCoverageRatio']['value'][ttm]
    fields["AssetTurnover"]['value'] = capital_indicators['AssetTurnoverDupont']['value'][ttm]
    fields["Leverage"]['value'] = capital_indicators['LeverageDupont']['value'][ttm]

    return fields






