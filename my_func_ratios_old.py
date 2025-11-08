

import my_func_support as sup
import my_func_capital_old as cap
import my_func as f
import requests
import os

api_key = os.getenv('eodhd')

def get_ratios(symbol, market='US'):

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

    # Create dictionary with fields
    fields = {
        "FiscalYearEnd": {'label': "FiscalYearEnd", 'type': [2], 'value': data["General"].get("FiscalYearEnd")},
        "MostRecentQuarter": {'label': "MostRecentQuarter", 'type': [1, 2],
                              'value': data["Highlights"].get("MostRecentQuarter")},
        "MarketCapitalization": {'label': "MarketCapitalization (bln)", 'type': [2], 'value': None},
        "TrailingPE": {'label': "TrailingPE", 'type': [1, 2],
                       'value': sup.convert_float(data["Valuation"].get("TrailingPE"))},
        "ForwardPE": {'label': "ForwardPE", 'type': [1, 2],
                      'value': sup.convert_float(data["Valuation"].get("ForwardPE"))},
        "PriceSalesTTM": {'label': "PriceToSales", 'type': [1, 2],
                          'value': sup.convert_float(data["Valuation"].get("PriceSalesTTM"))},
        "PriceBookMRQ": {'label': "PriceToBook", 'type': [1, 2],
                         'value': sup.convert_float(data["Valuation"].get("PriceBookMRQ"))},
        "PEGRatio": {'label': "PEGRatio", 'type': [1, 2],
                     'value': sup.convert_float(data["Highlights"].get("PEGRatio"))},
        "EnterpriseValueRevenue": {'label': "EVToRevenue", 'type': [1, 2],
                                   'value': sup.convert_float(data["Valuation"].get("EnterpriseValueRevenue"))},
        "EnterpriseValueEbitda": {'label': "EVToEBITDA", 'type': [1, 2],
                                  'value': sup.convert_float(data["Valuation"].get("EnterpriseValueEbitda"))},
        "DividendShare": {'label': "DividendShare", 'type': [2],
                          'value': sup.convert_float(data["Highlights"].get("DividendShare"))},
        "DividendYield": {'label': "DividendYield %", 'type': [2],
                          'value': sup.safe_multiply(sup.convert_float(data["Highlights"].get("DividendYield"), decimal=4),100)},
        "EarningsShare": {'label': "EPS", 'type': [2],
                          'value': sup.convert_float(data["Highlights"].get("EarningsShare"))},
        "DilutedEpsTTM": {'label': "DilutedEPS", 'type': [2],
                          'value': sup.convert_float(data["Highlights"].get("DilutedEpsTTM"))},
        "EPSEstimateCurrentYear": {'label': "EPSEstimateCurrentYear", 'type': [2],
                                   'value': sup.convert_float(data["Highlights"].get("EPSEstimateCurrentYear"))},
        "EPSEstimateNextYear": {'label': "EPSEstimateNextYear", 'type': [2],
                                'value': sup.convert_float(data["Highlights"].get("EPSEstimateNextYear"))},
        "EPSEstimateNextQuarter": {'label': "EPSEstimateNextQuarter", 'type': [2],
                                   'value': sup.convert_float(data["Highlights"].get("EPSEstimateNextQuarter"))},
        "EPSEstimateCurrentQuarter": {'label': "EPSEstimateCurrentQuarter", 'type': [2],
                                      'value': sup.convert_float(data["Highlights"].get("EPSEstimateCurrentQuarter"))},
        "RevenuePerShareTTM": {'label': "RevenuePerShare", 'type': [2],
                               'value': sup.convert_float(data["Highlights"].get("RevenuePerShareTTM"))},
        "ReturnOnAssetsTTM": {'label': "ROA %", 'type': [2, 3], 'value': None},
        # 'value': sup.safe_multiply(sup.convert_float(data["Highlights"].get("ReturnOnAssetsTTM"), decimal=4), 100)},
        "ReturnOnEquityTTM": {'label': "ROE %", 'type': [2, 3], 'value': None},
        # 'value': sup.safe_multiply(sup.convert_float(data["Highlights"].get("ReturnOnEquityTTM"), decimal=4),  100)},
        "ProfitMarginTTM": {'label': "ProfitMargin %", 'type': [2, 3], 'value': None},
        "OperatingMarginTTM": {'label': "OperatingMargin %", 'type': [2], 'value': None},
        "AssetTurnover": {'label': "AssetTurnover", 'type': [2, 3], 'value': None},
        "Leverage": {'label': "Leverage", 'type': [2, 3], 'value': None},
        # "ProfitMargin": {'label': "ProfitMargin %", 'type': [2],
        # 'value': sup.safe_multiply(sup.convert_float(data["Highlights"].get("ProfitMargin"), decimal=4),100)},
        # "OperatingMarginTTM": {'label': "OperatingMargin %", 'type': [2],
        #  'value': sup.safe_multiply(sup.convert_float(data["Highlights"].get("OperatingMarginTTM"), decimal=4),100)},
        "QuarterlyRevenueGrowthYOY": {'label': "QuarterlyRevenueGrowthYOY %", 'type': [2],
                                      'value': sup.safe_multiply(sup.convert_float(data["Highlights"].get("QuarterlyRevenueGrowthYOY"), decimal=4), 100)},
        "QuarterlyEarningsGrowthYOY": {'label': "QuarterlyEarningsGrowthYOY %", 'type': [2],
                                       'value': sup.safe_multiply(sup.convert_float(data["Highlights"].get("QuarterlyEarningsGrowthYOY"),decimal=4), 100)},
        "Beta": {'label': "Beta", 'type': [2], 'value': sup.convert_float(data["Technicals"].get("Beta"), decimal=4)},
        "DebtToEquity": {'label': "DebtToEquity", 'type': [2,4], 'value': None},
        "LTDebtToEquity": {'label': "LTDebtToEquity", 'type': [2,4], 'value': None},
        "DebtToCapital": {'label': "DebtToCapital", 'type': [2,4], 'value': None},
        "LTDebtToCapital": {'label': "LTDebtToCapital", 'type': [2,4], 'value': None},
        "InterestCoverageRatio": {'label': "InterestCoverageRatio", 'type': [2,4], 'value': None}
        }

    # Handle computed value in MarketCapitalization
    market_cap = data["Highlights"].get("MarketCapitalization")
    if market_cap is None:
        market_cap_value = "-"
    else:
        market_cap_value = sup.convert_float(market_cap, convert='bln', decimal=2)  # Convert to billions
    # Assign value to dictionary
    fields["MarketCapitalization"]['value'] = market_cap_value

    # Handle computed value in parameters used for Dupont analysis indicators

    # Check if most recent quarter is the last quarter of FY
    mrq = sup.get_mrq(data)
    check_mrq = sup.check_mrq(data)

    # Compute parameters for TTM (trailing_twelve-months)
    if check_mrq:
        net_profit_ttm = sup.get_statement_value(data["Financials"]["Income_Statement"]["yearly"], key=mrq,
                                                 field="netIncome", decimal=2)
        operating_income_ttm = sup.get_statement_value(data["Financials"]["Income_Statement"]["yearly"], key=mrq,
                                                       field="operatingIncome", decimal=2)
        total_revenue_ttm = sup.get_statement_value(data["Financials"]["Income_Statement"]["yearly"], key=mrq,
                                                    field="totalRevenue", decimal=2)
        avg_total_assets = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,
                                                 field="totalAssets", n=5)
        avg_equity = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,
                                           field="totalStockholderEquity", n=5)

    else:
        net_profit_ttm = sup.sum_n_consecutive(data["Financials"]["Income_Statement"]["quarterly"], start_key=mrq,
                                               field="netIncome", decimal=2, n=4)
        operating_income_ttm = sup.sum_n_consecutive(data["Financials"]["Income_Statement"]["quarterly"], start_key=mrq,
                                                     field="operatingIncome", decimal=2, n=4)
        total_revenue_ttm = sup.sum_n_consecutive(data["Financials"]["Income_Statement"]["quarterly"], start_key=mrq,
                                                  field="totalRevenue", decimal=2, n=4)
        avg_total_assets = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,
                                                 field="totalAssets", n=5)
        avg_equity = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,
                                           field="totalStockholderEquity", n=5)

    # Compute Dupont indicators:
    net_profit_margin = sup.safe_multiply(sup.safe_divide(net_profit_ttm, total_revenue_ttm, 4), 100)
    asset_turnover = sup.safe_divide(total_revenue_ttm, avg_total_assets, 2)
    leverage = sup.safe_divide(avg_total_assets, avg_equity, 2)

    # Compute ROA and ROE
    roa = sup.safe_multiply(sup.safe_divide(net_profit_ttm, avg_total_assets, 4), 100)
    roe = sup.safe_multiply(sup.safe_divide(net_profit_ttm, avg_equity, 4), 100)

    # Compute Operating margin
    operating_margin = sup.safe_multiply(sup.safe_divide(operating_income_ttm, total_revenue_ttm, 4), 100)

    # Assign values to dictionary
    fields["ReturnOnAssetsTTM"]['value'] = roa
    fields["ReturnOnEquityTTM"]['value'] = roe
    fields["ProfitMarginTTM"]['value'] = net_profit_margin
    fields["OperatingMarginTTM"]['value'] = operating_margin
    fields["AssetTurnover"]['value'] = asset_turnover
    fields["Leverage"]['value'] = leverage

    capital = cap.capital_structure(symbol=symbol, market='US')['capital_ratios']

    # Assign values to capital structure indicators
    fields["DebtToEquity"]['value'] = capital["DebtToEquity"]
    fields["LTDebtToEquity"]['value'] = capital["LTDebtToEquity"]
    fields["DebtToCapital"]['value'] = capital["DebtToCapital"]
    fields["LTDebtToCapital"]['value'] = capital["LTDebtToCapital"]
    fields["InterestCoverageRatio"]['value'] = capital["InterestCoverageRatio"]

    return fields






