

import my_func_support as sup
import my_func as f
import requests
import os
from datetime import datetime


api_key = os.getenv('eodhd')


# Function defined for capital structure indicators
def capital_structure(symbol, market='US', convert='bln', raw_data=None):

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
        "MostRecentQuarter": {'label': "MostRecentQuarter", 'type': [], 'value': {mrq_label: mrq}},
        "totalAssets": {'label': f"Total Assets ({convert})", 'type': [1],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "totalLiab": {'label': f"Total Liabilities ({convert})", 'type': [1],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "totalCurrentLiabilities": {'label': f"Current Liabilities ({convert})", 'type': [1],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "nonCurrentLiabilitiesTotal": {'label': f"Non Current Liabilities ({convert})", 'type': [1],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "shortLongTermDebtTotal": {'label': f"Total Debt ({convert})", 'type': [1],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} computed as average of interest bearing debt over last four quarters"},
        "longTermDebt": {'label': f"Long Term Debt ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} computed as average of interest bearing LT debt over last four quarters"},
        "totalStockholderEquity": {'label': f"Shareholders Equity ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "DebtToEquity": {'label': f"Total Liabilities-To-Equity", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"Total liabilities / Shareholders'equity; {ttm} computed as average indicator value over last four quarters"},
        "LTDebtToEquity": {'label': f"Non Current Liabilities-To-Equity", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"Total non-current liabilities / Shareholders'equity; {ttm} computed as average indicator value over last four quarters"},
        "DebtToCapital": {'label': f"Debt-To-Capital", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"Total interest bearing debt / (Interest bearing debt + Shareholders'equity); {ttm} computed as average indicator value over last four quarters"},
        "LTDebtToCapital": {'label': f"LongTermDebt-To-LongTermCapital", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"Total interest bearing LT debt / (Interest bearing LT debt + Shareholders'equity); {ttm} computed as average indicator value over last four quarters"},
        "ProfitMarginDupont": {'label': f"Profit Margin (Dupont) (%)", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} indicator value is based on trailing twelve months of net income and revenues"},
        "AssetTurnoverDupont": {'label': f"Asset Turnover (Dupont)", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} indicator value is based on average value of Total Assets over last four quarters and trailing twelve months of revenues"},
        "LeverageDupont": {'label': f"Leverage (Dupont)", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} is based on average values of Total Assets and Shareholders' equity over last four quarters"},
        "InterestCoverageRatio": {'label': f"Interest Coverage Ratio", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': f"{ttm} is based on trailing twelve months values of operating income and interest expenses"},
            }

    # Handle computed values for indicators
    # TotalAssets
    api_field = "totalAssets"
    totalAssets = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field=api_field, decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(totalAssets['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(totalAssets['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(totalAssets['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(totalAssets['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(totalAssets['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # TotalLiabilities
    api_field = "totalLiab"
    totalLiab = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field=api_field, decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(totalLiab['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(totalLiab['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(totalLiab['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(totalLiab['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(totalLiab['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # totalCurrentLiabilities
    api_field = "totalCurrentLiabilities"
    totalCurrentLiabilities = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field=api_field, decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(totalCurrentLiabilities['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(totalCurrentLiabilities['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(totalCurrentLiabilities['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(totalCurrentLiabilities['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(totalCurrentLiabilities['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # nonCurrentLiabilitiesTotal
    api_field = "nonCurrentLiabilitiesTotal"
    nonCurrentLiabilitiesTotal = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field=api_field, decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(nonCurrentLiabilitiesTotal['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(nonCurrentLiabilitiesTotal['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(nonCurrentLiabilitiesTotal['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(nonCurrentLiabilitiesTotal['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(nonCurrentLiabilitiesTotal['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # totalDebt
    api_field = "shortLongTermDebtTotal"
    shortLongTermDebtTotal = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field=api_field, decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(shortLongTermDebtTotal['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(shortLongTermDebtTotal['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(shortLongTermDebtTotal['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(shortLongTermDebtTotal['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(shortLongTermDebtTotal['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # LTDebt
    api_field = "longTermDebt"
    longTermDebt = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field=api_field, decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(longTermDebt['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(longTermDebt['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(longTermDebt['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(longTermDebt['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(longTermDebt['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # ShareholdersEquity
    api_field = "totalStockholderEquity"
    totalStockholderEquity = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field=api_field, decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(totalStockholderEquity['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(totalStockholderEquity['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(totalStockholderEquity['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(totalStockholderEquity['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(totalStockholderEquity['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # DebtToEquity
    api_field = "DebtToEquity"
    value_mrq_debt_to_equity = sup.safe_divide(totalLiab['value_mrq'], totalStockholderEquity['value_mrq'], decimal=decimal)
    value_prev_year_mrq_debt_to_equity = sup.safe_divide(totalLiab['value_prev_year_mrq'], totalStockholderEquity['value_prev_year_mrq'], decimal=decimal)
    value_prev_q_debt_to_equity = sup.safe_divide(totalLiab['value_prev_q'], totalStockholderEquity['value_prev_q'], decimal=decimal)
    yoy_delta_change_debt_to_equity = sup.safe_delta(value_mrq_debt_to_equity,value_prev_year_mrq_debt_to_equity, decimal=decimal)
    qoq_delta_change_debt_to_equity = sup.safe_delta(value_mrq_debt_to_equity,value_prev_q_debt_to_equity, decimal=decimal)

    ttm_mrq_debt_to_equity = sup.safe_divide(totalLiab['value_ttm'], totalStockholderEquity['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_debt_to_equity = sup.safe_divide(totalLiab['value_prev_year_ttm'], totalStockholderEquity['value_prev_year_ttm'], decimal=decimal)
    yoy_delta_change_ttm_debt_to_equity = sup.safe_delta(ttm_mrq_debt_to_equity, ttm_prev_year_mrq_debt_to_equity, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_debt_to_equity, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_debt_to_equity,decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_debt_to_equity,decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_debt_to_equity, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_debt_to_equity, decimal=decimal, suffix=None)

    # LTDebtToEquity
    api_field = "LTDebtToEquity"
    value_mrq_lt_debt_to_equity = sup.safe_divide(nonCurrentLiabilitiesTotal['value_mrq'], totalStockholderEquity['value_mrq'], decimal=decimal)
    value_prev_year_mrq_lt_debt_to_equity = sup.safe_divide(nonCurrentLiabilitiesTotal['value_prev_year_mrq'], totalStockholderEquity['value_prev_year_mrq'], decimal=decimal)
    value_prev_q_lt_debt_to_equity = sup.safe_divide(nonCurrentLiabilitiesTotal['value_prev_q'], totalStockholderEquity['value_prev_q'], decimal=decimal)
    yoy_delta_change_lt_debt_to_equity = sup.safe_delta(value_mrq_lt_debt_to_equity, value_prev_year_mrq_lt_debt_to_equity, decimal=decimal)
    qoq_delta_change_lt_debt_to_equity = sup.safe_delta(value_mrq_lt_debt_to_equity, value_prev_q_lt_debt_to_equity, decimal=decimal)

    ttm_mrq_lt_debt_to_equity = sup.safe_divide(nonCurrentLiabilitiesTotal['value_ttm'], totalStockholderEquity['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_lt_debt_to_equity = sup.safe_divide(nonCurrentLiabilitiesTotal['value_prev_year_ttm'], totalStockholderEquity['value_prev_year_ttm'], decimal=decimal)
    yoy_delta_change_ttm_lt_debt_to_equity = sup.safe_delta(ttm_mrq_lt_debt_to_equity, ttm_prev_year_mrq_lt_debt_to_equity, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_lt_debt_to_equity, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_lt_debt_to_equity,decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_lt_debt_to_equity,decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_lt_debt_to_equity, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_lt_debt_to_equity, decimal=decimal, suffix=None)

    # DebtToCapital
    api_field = "DebtToCapital"

    value_mrq_total_capital = sup.safe_addition(shortLongTermDebtTotal['value_mrq'], totalStockholderEquity['value_mrq'], decimal=decimal)
    value_prev_year_mrq_total_capital = sup.safe_addition(shortLongTermDebtTotal['value_prev_year_mrq'], totalStockholderEquity['value_prev_year_mrq'], decimal=decimal)
    value_prev_q_total_capital = sup.safe_addition(shortLongTermDebtTotal['value_prev_q'], totalStockholderEquity['value_prev_q'], decimal=decimal)
    ttm_mrq_total_capital = sup.safe_addition(shortLongTermDebtTotal['value_ttm'], totalStockholderEquity['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_total_capital = sup.safe_addition(shortLongTermDebtTotal['value_prev_year_ttm'], totalStockholderEquity['value_prev_year_ttm'], decimal=decimal)

    value_mrq_debt_to_capital = sup.safe_divide(shortLongTermDebtTotal['value_mrq'], value_mrq_total_capital, decimal=decimal)
    value_prev_year_mrq_debt_to_capital = sup.safe_divide(shortLongTermDebtTotal['value_prev_year_mrq'], value_prev_year_mrq_total_capital, decimal=decimal)
    value_prev_q_debt_to_capital = sup.safe_divide(shortLongTermDebtTotal['value_prev_q'], value_prev_q_total_capital, decimal=decimal)
    yoy_delta_change_debt_to_capital = sup.safe_delta(value_mrq_debt_to_capital, value_prev_year_mrq_debt_to_capital, decimal=decimal)
    qoq_delta_change_debt_to_capital = sup.safe_delta(value_mrq_debt_to_capital, value_prev_q_debt_to_capital, decimal=decimal)

    ttm_mrq_debt_to_capital = sup.safe_divide(shortLongTermDebtTotal['value_ttm'], ttm_mrq_total_capital, decimal=decimal)
    ttm_prev_year_mrq_debt_to_capital = sup.safe_divide(shortLongTermDebtTotal['value_prev_year_ttm'], ttm_prev_year_mrq_total_capital, decimal=decimal)
    yoy_delta_change_ttm_debt_to_capital = sup.safe_delta(ttm_mrq_debt_to_capital, ttm_prev_year_mrq_debt_to_capital, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_debt_to_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_debt_to_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_debt_to_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_debt_to_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_debt_to_capital, decimal=decimal, suffix=None)

    # LTDebtToCapital
    api_field = "LTDebtToCapital"

    value_mrq_total_lt_capital = sup.safe_addition(longTermDebt['value_mrq'], totalStockholderEquity['value_mrq'], decimal=decimal)
    value_prev_year_mrq_total_lt_capital = sup.safe_addition(longTermDebt['value_prev_year_mrq'], totalStockholderEquity['value_prev_year_mrq'], decimal=decimal)
    value_prev_q_total_lt_capital = sup.safe_addition(longTermDebt['value_prev_q'], totalStockholderEquity['value_prev_q'], decimal=decimal)
    ttm_mrq_total_lt_capital = sup.safe_addition(longTermDebt['value_ttm'], totalStockholderEquity['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_total_lt_capital = sup.safe_addition(longTermDebt['value_prev_year_ttm'], totalStockholderEquity['value_prev_year_ttm'], decimal=decimal)

    value_mrq_lt_debt_to_lt_capital = sup.safe_divide(longTermDebt['value_mrq'], value_mrq_total_lt_capital, decimal=decimal)
    value_prev_year_mrq_lt_debt_to_lt_capital = sup.safe_divide(longTermDebt['value_prev_year_mrq'], value_prev_year_mrq_total_lt_capital, decimal=decimal)
    value_prev_q_lt_debt_to_lt_capital = sup.safe_divide(longTermDebt['value_prev_q'], value_prev_q_total_lt_capital, decimal=decimal)
    yoy_delta_change_lt_debt_to_lt_capital = sup.safe_delta(value_mrq_lt_debt_to_lt_capital, value_prev_year_mrq_lt_debt_to_lt_capital, decimal=decimal)
    qoq_delta_change_lt_debt_to_lt_capital = sup.safe_delta(value_mrq_lt_debt_to_lt_capital, value_prev_q_lt_debt_to_lt_capital, decimal=decimal)

    ttm_mrq_lt_debt_to_lt_capital = sup.safe_divide(longTermDebt['value_ttm'], ttm_mrq_total_lt_capital, decimal=decimal)
    ttm_prev_year_mrq_lt_debt_to_lt_capital = sup.safe_divide(longTermDebt['value_prev_year_ttm'], ttm_prev_year_mrq_total_lt_capital, decimal=decimal)
    yoy_delta_change_ttm_lt_debt_to_lt_capital = sup.safe_delta(ttm_mrq_lt_debt_to_lt_capital, ttm_prev_year_mrq_lt_debt_to_lt_capital, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_lt_debt_to_lt_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_lt_debt_to_lt_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_lt_debt_to_lt_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_lt_debt_to_lt_capital, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_lt_debt_to_lt_capital, decimal=decimal, suffix=None)

    # ProfitMarginDupont
    api_field = "ProfitMarginDupont"

    revenues = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field="totalRevenue", decimal=decimal)
    netIncome = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field="netIncome", decimal=decimal)

    value_mrq_profit_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_mrq'], revenues['value_mrq'], decimal=decimal_mg), element2 =100)
    value_prev_year_mrq_profit_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_prev_year_mrq'], revenues['value_prev_year_mrq'], decimal=decimal_mg), element2 =100)
    value_prev_q_profit_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_prev_q'], revenues['value_prev_q'], decimal=decimal_mg), element2 =100)
    yoy_delta_change_profit_margin = sup.safe_delta(value_mrq_profit_margin, value_prev_year_mrq_profit_margin, decimal=decimal)
    qoq_delta_change_profit_margin = sup.safe_delta(value_mrq_profit_margin, value_prev_q_profit_margin, decimal=decimal)

    ttm_mrq_profit_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_ttm'], revenues['value_ttm'], decimal=decimal_mg), element2 =100)
    ttm_prev_year_mrq_profit_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_prev_year_ttm'], revenues['value_prev_year_ttm'], decimal=decimal_mg), element2 =100)
    yoy_delta_change_ttm_profit_margin = sup.safe_delta(ttm_mrq_profit_margin, ttm_prev_year_mrq_profit_margin, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_profit_margin, decimal=decimal)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_profit_margin,decimal=decimal, suffix=pp_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_profit_margin,decimal=decimal, suffix=pp_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_profit_margin, decimal=decimal)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_profit_margin, decimal=decimal,suffix=pp_suffix)

    # AssetTurnoverDupont
    api_field = "AssetTurnoverDupont"

    average_last_two_q_total_assets = sup.avg_n_consecutive(data_set_balance_sheet["quarterly"], start_key=mrq,field="totalAssets", n=2)
    average_last_two_q_prev_total_assets = sup.avg_n_consecutive(data_set_balance_sheet["quarterly"], start_key=prev_q,field="totalAssets", n=2)
    average_last_two_q_prev_year_total_assets = sup.avg_n_consecutive(data_set_balance_sheet["quarterly"], start_key=prev_year_mrq,field="totalAssets", n=2)

    value_mrq_asset_turnover = sup.safe_divide(revenues['value_mrq'], average_last_two_q_total_assets, decimal=decimal)
    value_prev_year_mrq_asset_turnover = sup.safe_divide(revenues['value_prev_year_mrq'], average_last_two_q_prev_year_total_assets, decimal=decimal)
    value_prev_q_asset_turnover = sup.safe_divide(revenues['value_prev_q'], average_last_two_q_prev_total_assets, decimal=decimal)
    yoy_delta_change_asset_turnover = sup.safe_delta(value_mrq_asset_turnover, value_prev_year_mrq_asset_turnover, decimal=decimal)
    qoq_delta_change_asset_turnover = sup.safe_delta(value_mrq_asset_turnover, value_prev_q_asset_turnover, decimal=decimal)

    ttm_mrq_asset_turnover = sup.safe_divide(revenues['value_ttm'], totalAssets['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_asset_turnover = sup.safe_divide(revenues['value_prev_year_ttm'], totalAssets['value_prev_year_ttm'], decimal=decimal)
    yoy_delta_change_ttm_asset_turnover = sup.safe_delta(ttm_mrq_asset_turnover, ttm_prev_year_mrq_asset_turnover, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_asset_turnover, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_asset_turnover, decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_asset_turnover, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_asset_turnover, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_asset_turnover, decimal=decimal, suffix=None)

    # LeverageDupont
    api_field = "LeverageDupont"

    average_last_two_q_shareholders_equity = sup.avg_n_consecutive(data_set_balance_sheet["quarterly"], start_key=mrq,field="totalStockholderEquity", n=2)
    average_last_two_q_prev_shareholders_equity = sup.avg_n_consecutive(data_set_balance_sheet["quarterly"], start_key=prev_q,field="totalStockholderEquity", n=2)
    average_last_two_q_prev_year_shareholders_equity = sup.avg_n_consecutive(data_set_balance_sheet["quarterly"], start_key=prev_year_mrq,field="totalStockholderEquity", n=2)

    value_mrq_leverage = sup.safe_divide(average_last_two_q_total_assets, average_last_two_q_shareholders_equity, decimal=decimal)
    value_prev_year_mrq_leverage = sup.safe_divide(average_last_two_q_prev_year_total_assets, average_last_two_q_prev_year_shareholders_equity, decimal=decimal)
    value_prev_q_leverage = sup.safe_divide(average_last_two_q_prev_total_assets, average_last_two_q_prev_shareholders_equity, decimal=decimal)
    yoy_delta_change_leverage = sup.safe_delta(value_mrq_leverage, value_prev_year_mrq_leverage, decimal=decimal)
    qoq_delta_change_leverage = sup.safe_delta(value_mrq_leverage, value_prev_q_leverage, decimal=decimal)

    ttm_mrq_leverage = sup.safe_divide(totalAssets['value_ttm'], totalStockholderEquity['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_leverage = sup.safe_divide(totalAssets['value_prev_year_ttm'], totalStockholderEquity['value_prev_year_ttm'], decimal=decimal)
    yoy_delta_change_ttm_leverage = sup.safe_delta(ttm_mrq_leverage, ttm_prev_year_mrq_leverage, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_leverage, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_leverage, decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_leverage, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_leverage, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_leverage, decimal=decimal, suffix=None)

    # InterestCoverageRatio
    api_field = "InterestCoverageRatio"

    interest = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field="interestExpense", decimal=decimal)
    op_income = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field="operatingIncome", decimal=decimal)

    value_mrq_interest_coverage = sup.safe_divide(op_income['value_mrq'], interest['value_mrq'], decimal=decimal)
    value_prev_year_mrq_interest_coverage = sup.safe_divide(op_income['value_prev_year_mrq'], interest['value_prev_year_mrq'], decimal=decimal)
    value_prev_q_interest_coverage = sup.safe_divide(op_income['value_prev_q'], interest['value_prev_q'], decimal=decimal)
    yoy_delta_change_interest_coverage = sup.safe_delta(value_mrq_interest_coverage, value_prev_year_mrq_interest_coverage, decimal=decimal)
    qoq_delta_change_interest_coverage = sup.safe_delta(value_mrq_interest_coverage, value_prev_q_interest_coverage, decimal=decimal)

    ttm_mrq_interest_coverage = sup.safe_divide(op_income['value_ttm'], interest['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_interest_coverage = sup.safe_divide(op_income['value_prev_year_ttm'], interest['value_prev_year_ttm'], decimal=decimal)
    yoy_delta_change_ttm_interest_coverage = sup.safe_delta(ttm_mrq_interest_coverage, ttm_prev_year_mrq_interest_coverage, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_interest_coverage, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_interest_coverage,decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_interest_coverage,decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_interest_coverage, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_interest_coverage, decimal=decimal,suffix=None)

    return fields




