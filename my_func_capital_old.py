

import my_func_support as sup
import my_func as f
import requests
import os
from datetime import datetime


api_key = os.getenv('eodhd')


# Function defined for capital structure indicators
def capital_structure(symbol, market='US'):

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

    # Check if most recent quarter is the last quarter of FY; Get quarter from a year before period
    mrq = sup.get_mrq(data)
    prev_year_mrq = sup.get_previous_year_mrq(data)
    check_mrq = sup.check_mrq(data)
    mrq_formatted = datetime.strptime(mrq, '%Y-%m-%d').strftime('%b-%y')  # Formatted as mmm-yy, ex. Jun-23

    # Create labels
    mrq_label = f"Most Recent Quarter: ({mrq_formatted})"
    yoy_change = 'Y/Y Change'
    ttm = 'Trailing Twelve Months'

    # Create dictionary with fields
    fields = {
        "MostRecentQuarter": {'label': "MostRecentQuarter", 'type': [], 'value': {mrq_label: mrq}},
        "TotalAssets": {'label': "TotalAssets", 'type': [1],'value': {mrq_label: sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq,
                            field="totalAssets", convert='bln', decimal=2), yoy_change: None, ttm: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "TotalLiabilities": {'label': "TotalLiabilities", 'type': [1],'value': {mrq_label: sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq,
                            field="totalLiab", convert='bln', decimal=2),yoy_change: None, ttm: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "totalCurrentLiabilities": {'label': "totalCurrentLiabilities", 'type': [1],'value': {mrq_label: sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq,
                            field="totalCurrentLiabilities", convert='bln', decimal=2), yoy_change: None, ttm: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "nonCurrentLiabilitiesTotal": {'label': "nonCurrentLiabilitiesTotal", 'type': [1],'value': {mrq_label: sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq,
                            field="nonCurrentLiabilitiesTotal", convert='bln', decimal=2), yoy_change: None, ttm: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "totalDebt": {'label': "shortLongTermDebtTotal", 'type': [1],'value': {mrq_label: sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq,
                            field="shortLongTermDebtTotal", convert='bln', decimal=2),yoy_change: None, ttm: None}, 'comment': f"refers to all interest bearing debt; {ttm} computed as average over last four quarters"},
        "LTDebt": {'label': "LTDebt", 'type': [1], 'value': {mrq_label: sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq,
                            field="longTermDebt", convert='bln', decimal=2), yoy_change: None, ttm: None}, 'comment': f"refers to interest bearing LT debt; {ttm} computed as average over last four quarters"},
        "ShareholdersEquity": {'label': "ShareholdersEquity", 'type': [1], 'value': {mrq_label: sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq,
                            field="totalStockholderEquity", convert='bln', decimal=2), yoy_change: None,ttm: None}, 'comment': f"{ttm} computed as average over last four quarters"},
        "DebtToEquity": {'label': "DebtToEquity", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"computed as total liabilities / shareholders'equity; yoy_change computed as delta change of percentage points; {ttm} is based on average values over last four quarters"},
        "LTDebtToEquity": {'label': "LTDebtToEquity", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"computed as total non-current liabilities / shareholders'equity; yoy_change computed as delta change of percentage points; {ttm} is based on average values over last four quarters"},
        "DebtToCapital": {'label': "DebtToCapital", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"computed as total interest bearing debt / (interest bearing debt + shareholders'equity); yoy_change computed as delta change of percentage points; {ttm} is based on average values over last four quarters"},
        "LTDebtToCapital": {'label': "LTDebtToCapital", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"computed as total interest bearing LT debt / (interest bearing LT debt + shareholders'equity); yoy_change computed as delta change of percentage points; {ttm} is based on average values over last four quarters"},
        "ProfitMarginDupont": {'label': "ProfitMarginDupont", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"yoy_change computed as delta change of percentage points; {ttm} is based on trailing twelve months values"},
        "AssetTurnoverDupont": {'label': "AssetTurnoverDupont", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"yoy_change computed as delta change; {ttm} is based on average value of totalAssets over last four quarters"},
        "LeverageDupont": {'label': "LeverageDupont", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"yoy_change computed as delta change; {ttm} is based on average values over last four quarters"},
        "InterestCoverageRatio": {'label': "InterestCoverageRatio", 'type': [1,2], 'value': {mrq_label: None, yoy_change: None, ttm: None}, 'comment': f"yoy_change computed as delta change of percentage points; {ttm} is based on trailing twelve months values"},
            }

    # Handle computed values for indicators
    # TotalAssets
    value_mrq_total_assets = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="totalAssets", decimal=2)
    value_prev_year_quarter_total_assets = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="totalAssets", decimal=2)
    percentage_change_total_assets = sup.safe_multiply(sup.relative_change(value_mrq_total_assets, value_prev_year_quarter_total_assets),100)
    average_ttm_total_assets = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="totalAssets", n=5, convert='bln')
    fields["TotalAssets"]['value'][yoy_change] = sup.formatted_value(percentage_change_total_assets, 2, '%')
    fields["TotalAssets"]['value'][ttm] = average_ttm_total_assets

    # TotalLiabilities
    value_mrq_total_liabilities = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="totalLiab", decimal=2)
    value_prev_year_quarter_total_liabilities = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="totalLiab", decimal=2)
    percentage_change_total_liabilities = sup.safe_multiply(sup.relative_change(value_mrq_total_liabilities, value_prev_year_quarter_total_liabilities),100)
    average_ttm_total_liabilities = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="totalLiab", n=5, convert='bln')
    fields["TotalLiabilities"]['value'][yoy_change] = sup.formatted_value(percentage_change_total_liabilities, 2, '%')
    fields["TotalLiabilities"]['value'][ttm] = average_ttm_total_liabilities

    # totalCurrentLiabilities
    value_mrq_total_current_liabilities = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="totalCurrentLiabilities", decimal=2)
    value_prev_year_quarter_total_current_liabilities = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="totalCurrentLiabilities", decimal=2)
    percentage_change_total_current_liabilities = sup.safe_multiply(sup.relative_change(value_mrq_total_current_liabilities, value_prev_year_quarter_total_current_liabilities),100)
    average_ttm_total_current_liabilities = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="totalCurrentLiabilities", n=5, convert='bln')
    fields["totalCurrentLiabilities"]['value'][yoy_change] = sup.formatted_value(percentage_change_total_current_liabilities, 2, '%')
    fields["totalCurrentLiabilities"]['value'][ttm] = average_ttm_total_current_liabilities

    # nonCurrentLiabilitiesTotal
    value_mrq_total_non_current_liabilities = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="nonCurrentLiabilitiesTotal", decimal=2)
    value_prev_year_quarter_total_non_current_liabilities = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="nonCurrentLiabilitiesTotal", decimal=2)
    percentage_change_total_non_current_liabilities = sup.safe_multiply(sup.relative_change(value_mrq_total_non_current_liabilities, value_prev_year_quarter_total_non_current_liabilities),100)
    average_ttm_total_non_current_liabilities = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="nonCurrentLiabilitiesTotal", n=5, convert='bln')
    fields["nonCurrentLiabilitiesTotal"]['value'][yoy_change] = sup.formatted_value(percentage_change_total_non_current_liabilities, 2, '%')
    fields["nonCurrentLiabilitiesTotal"]['value'][ttm] = average_ttm_total_non_current_liabilities

    # totalDebt
    value_mrq_total_debt = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="shortLongTermDebtTotal", decimal=2)
    value_prev_year_quarter_total_debt = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="shortLongTermDebtTotal", decimal=2)
    percentage_change_total_debt = sup.safe_multiply(sup.relative_change(value_mrq_total_debt, value_prev_year_quarter_total_debt),100)
    average_ttm_total_debt = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="shortLongTermDebtTotal", n=5, convert='bln')
    fields["totalDebt"]['value'][yoy_change] = sup.formatted_value(percentage_change_total_debt, 2, '%')
    fields["totalDebt"]['value'][ttm] = average_ttm_total_debt

    # LTDebt
    value_mrq_lt_debt = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="longTermDebt", decimal=2)
    value_prev_year_quarter_lt_debt = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="longTermDebt", decimal=2)
    percentage_change_lt_debt = sup.safe_multiply(sup.relative_change(value_mrq_lt_debt, value_prev_year_quarter_lt_debt),100)
    average_ttm_lt_debt = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="longTermDebt", n=5, convert='bln')
    fields["LTDebt"]['value'][yoy_change] = sup.formatted_value(percentage_change_lt_debt, 2, '%')
    fields["LTDebt"]['value'][ttm] = average_ttm_lt_debt

    # ShareholdersEquity
    value_mrq_shareholders_equity = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="totalStockholderEquity", decimal=2)
    value_prev_year_quarter_shareholders_equity = sup.get_statement_value(data["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="totalStockholderEquity", decimal=2)
    percentage_change_shareholders_equity = sup.safe_multiply(sup.relative_change(value_mrq_shareholders_equity, value_prev_year_quarter_shareholders_equity),100)
    average_ttm_shareholders_equity = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="totalStockholderEquity", n=5, convert='bln')
    fields["ShareholdersEquity"]['value'][yoy_change] = sup.formatted_value(percentage_change_shareholders_equity, 2, '%')
    fields["ShareholdersEquity"]['value'][ttm] = average_ttm_shareholders_equity

    # DebtToEquity
    value_mrq_debt_to_equity = sup.safe_divide(value_mrq_total_liabilities,value_mrq_shareholders_equity, 2)
    value_prev_year_quarter_debt_to_equity = sup.safe_divide(value_prev_year_quarter_total_liabilities,value_prev_year_quarter_shareholders_equity, 2)
    percentage_change_debt_to_equity = sup.safe_delta(value_mrq_debt_to_equity,value_prev_year_quarter_debt_to_equity, 2)
    average_ttm_debt_to_equity = sup.safe_divide(average_ttm_total_liabilities,average_ttm_shareholders_equity, 2)

    fields["DebtToEquity"]['value'][mrq_label] = sup.formatted_value(value_mrq_debt_to_equity, 2, 'x')
    fields["DebtToEquity"]['value'][yoy_change] = sup.formatted_value(percentage_change_debt_to_equity,2, 'x')
    fields["DebtToEquity"]['value'][ttm] = sup.formatted_value(average_ttm_debt_to_equity, 2, 'x')

    # LTDebtToEquity
    value_mrq_lt_debt_to_equity = sup.safe_divide(value_mrq_total_non_current_liabilities, value_mrq_shareholders_equity, 2)
    value_prev_year_quarter_lt_debt_to_equity = sup.safe_divide(value_prev_year_quarter_total_non_current_liabilities, value_prev_year_quarter_shareholders_equity, 2)
    percentage_change_lt_debt_to_equity = sup.safe_delta(value_mrq_lt_debt_to_equity, value_prev_year_quarter_lt_debt_to_equity, 2)
    average_ttm_lt_debt_to_equity = sup.safe_divide(average_ttm_total_non_current_liabilities, average_ttm_shareholders_equity,2)

    fields["LTDebtToEquity"]['value'][mrq_label] = sup.formatted_value(value_mrq_lt_debt_to_equity, 2, 'x')
    fields["LTDebtToEquity"]['value'][yoy_change] = sup.formatted_value(percentage_change_lt_debt_to_equity, 2, 'x')
    fields["LTDebtToEquity"]['value'][ttm] = sup.formatted_value(average_ttm_lt_debt_to_equity, 2, 'x')

    # DebtToCapital
    value_mrq_total_capital = sup.safe_addition(value_mrq_total_debt, value_mrq_shareholders_equity,2)
    value_prev_year_quarter_total_capital = sup.safe_addition(value_prev_year_quarter_total_debt, value_prev_year_quarter_shareholders_equity, 2)
    average_ttm_total_capital = sup.safe_addition(average_ttm_total_debt, average_ttm_shareholders_equity,2)
    value_mrq_debt_to_capital = sup.safe_divide(value_mrq_total_debt, value_mrq_total_capital, 2)
    value_prev_year_quarter_debt_to_capital = sup.safe_divide(value_prev_year_quarter_total_debt, value_prev_year_quarter_total_capital, 2)
    percentage_change_debt_to_capital = sup.safe_delta(value_mrq_debt_to_capital, value_prev_year_quarter_debt_to_capital,2)
    average_ttm_debt_to_capital = sup.safe_divide(average_ttm_total_debt, average_ttm_total_capital, 2)

    fields["DebtToCapital"]['value'][mrq_label] = sup.formatted_value(value_mrq_debt_to_capital, 2, 'x')
    fields["DebtToCapital"]['value'][yoy_change] = sup.formatted_value(percentage_change_debt_to_capital, 2, 'x')
    fields["DebtToCapital"]['value'][ttm] = sup.formatted_value(average_ttm_debt_to_capital, 2, 'x')

    # LTDebtToCapital
    value_mrq_total_lt_capital = sup.safe_addition(value_mrq_lt_debt, value_mrq_shareholders_equity, 2)
    value_prev_year_quarter_total_lt_capital = sup.safe_addition(value_prev_year_quarter_lt_debt,  value_prev_year_quarter_shareholders_equity, 2)
    average_ttm_total_lt_capital = sup.safe_addition(average_ttm_lt_debt, average_ttm_shareholders_equity, 2)
    value_mrq_lt_debt_to_lt_capital = sup.safe_divide(value_mrq_lt_debt, value_mrq_total_lt_capital, 2)
    value_prev_year_quarter_lt_debt_to_lt_capital = sup.safe_divide(value_prev_year_quarter_lt_debt, value_prev_year_quarter_total_lt_capital, 2)
    percentage_change_lt_debt_to_lt_capital = sup.safe_delta(value_mrq_lt_debt_to_lt_capital, value_prev_year_quarter_lt_debt_to_lt_capital, 2)
    average_ttm_lt_debt_to_lt_capital = sup.safe_divide(average_ttm_lt_debt, average_ttm_total_lt_capital, 2)

    fields["LTDebtToCapital"]['value'][mrq_label] = sup.formatted_value(value_mrq_lt_debt_to_lt_capital, 2, 'x')
    fields["LTDebtToCapital"]['value'][yoy_change] = sup.formatted_value(percentage_change_lt_debt_to_lt_capital, 2, 'x')
    fields["LTDebtToCapital"]['value'][ttm] = sup.formatted_value(average_ttm_lt_debt_to_lt_capital, 2, 'x')

    # ProfitMarginDupont
    value_mrq_net_profit = sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=mrq, field="netIncome", decimal=2)
    value_prev_year_quarter_net_profit = sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=prev_year_mrq, field="netIncome", decimal=2)

    value_mrq_revenues = sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=mrq, field="totalRevenue", decimal=2)
    value_prev_year_quarter_revenues = sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=prev_year_mrq, field="totalRevenue", decimal=2)

    value_mrq_net_profit_margin = sup.safe_multiply(sup.safe_divide(value_mrq_net_profit, value_mrq_revenues, 4),100)
    value_prev_year_quarter_net_profit_margin = sup.safe_multiply(sup.safe_divide(value_prev_year_quarter_net_profit, value_prev_year_quarter_revenues, 4),100)

    percentage_change_net_profit_margin = sup.safe_delta(value_mrq_net_profit_margin,value_prev_year_quarter_net_profit_margin, 2)

    if check_mrq:
        ttm_net_profit = sup.get_statement_value(data["Financials"]["Income_Statement"]["yearly"], key=mrq, field="netIncome", decimal=2)
        ttm_total_revenue = sup.get_statement_value(data["Financials"]["Income_Statement"]["yearly"], key=mrq, field="totalRevenue", decimal=2)
    else:
        ttm_net_profit = sup.sum_n_consecutive(data["Financials"]["Income_Statement"]["quarterly"], start_key=mrq, field="netIncome", decimal=2, n=4)
        ttm_total_revenue = sup.sum_n_consecutive(data["Financials"]["Income_Statement"]["quarterly"], start_key=mrq, field="totalRevenue", decimal=2, n=4)

    ttm_net_profit_margin = sup.safe_multiply(sup.safe_divide(ttm_net_profit, ttm_total_revenue, 4),100)

    fields["ProfitMarginDupont"]['value'][mrq_label] = sup.formatted_value(value_mrq_net_profit_margin, 2, '%')
    fields["ProfitMarginDupont"]['value'][yoy_change] = sup.formatted_value(percentage_change_net_profit_margin, 2, 'p.p')
    fields["ProfitMarginDupont"]['value'][ttm] = sup.formatted_value(ttm_net_profit_margin, 2, '%')

    # AssetTurnoverDupont
    average_last_two_q_total_assets = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,field="totalAssets", n=2)
    value_mrq_asset_turnover = sup.safe_divide(value_mrq_revenues, average_last_two_q_total_assets, 2)
    average_last_two_q_prev_year_total_assets = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=prev_year_mrq,field="totalAssets", n=2)
    value_prev_year_quarter_asset_turnover = sup.safe_divide(value_prev_year_quarter_revenues, average_last_two_q_prev_year_total_assets, 2)

    percentage_change_asset_turnover = sup.safe_delta(value_mrq_asset_turnover, value_prev_year_quarter_asset_turnover, 2)

    average_ttm_total_assets = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="totalAssets", n=5)  # Computed again, but ths time without conversion to bln

    ttm_asset_turnover = sup.safe_divide(ttm_total_revenue, average_ttm_total_assets, 2)

    fields["AssetTurnoverDupont"]['value'][mrq_label] = sup.formatted_value(value_mrq_asset_turnover, 2, 'x')
    fields["AssetTurnoverDupont"]['value'][yoy_change] = sup.formatted_value(percentage_change_asset_turnover, 2, 'x')
    fields["AssetTurnoverDupont"]['value'][ttm] = sup.formatted_value(ttm_asset_turnover, 2, 'x')

    # LeverageDupont
    average_last_two_q_shareholders_equity = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,field="totalStockholderEquity", n=2)
    value_mrq_leverage = sup.safe_divide(average_last_two_q_total_assets, average_last_two_q_shareholders_equity, 2)
    average_last_two_q_prev_year_shareholders_equity = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=prev_year_mrq,field="totalStockholderEquity", n=2)
    value_prev_year_quarter_leverage = sup.safe_divide(average_last_two_q_prev_year_total_assets, average_last_two_q_prev_year_shareholders_equity, 2)

    percentage_change_leverage = sup.safe_delta(value_mrq_leverage, value_prev_year_quarter_leverage, 2)

    average_ttm_shareholders_equity = sup.avg_n_consecutive(data["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="totalStockholderEquity", n=5)  # Computed again, but ths time without conversion to bln

    ttm_leverage = sup.safe_divide(average_ttm_total_assets, average_ttm_shareholders_equity, 2)

    fields["LeverageDupont"]['value'][mrq_label] = sup.formatted_value(value_mrq_leverage, 2, 'x')
    fields["LeverageDupont"]['value'][yoy_change] = sup.formatted_value(percentage_change_leverage, 2, 'x')
    fields["LeverageDupont"]['value'][ttm] = sup.formatted_value(ttm_leverage, 2, 'x')

    # InterestCoverageRatio
    value_mrq_operating_income = sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=mrq, field="operatingIncome", decimal=2)
    value_mrq_interest_expense = sup.safe_abs(sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=mrq, field="interestExpense", decimal=2))
    value_mrq_interest_coverage_ratio = sup.safe_divide(value_mrq_operating_income, value_mrq_interest_expense, 2)

    value_prev_year_quarter_operating_income = sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=prev_year_mrq, field="operatingIncome", decimal=2)
    value_prev_year_quarter_interest_expense = sup.safe_abs(sup.get_statement_value(data["Financials"]["Income_Statement"]["quarterly"], key=prev_year_mrq, field="interestExpense", decimal=2))
    value_prev_year_quarter_interest_coverage_ratio = sup.safe_divide(value_prev_year_quarter_operating_income, value_prev_year_quarter_interest_expense, 2)

    percentage_interest_coverage_ratio = sup.safe_delta(value_mrq_interest_coverage_ratio, value_prev_year_quarter_interest_coverage_ratio, 2)

    if check_mrq:
        ttm_operating_income = sup.get_statement_value(data["Financials"]["Income_Statement"]["yearly"], key=mrq, field="operatingIncome", decimal=2)
        ttm_interest_expense= sup.get_statement_value(data["Financials"]["Income_Statement"]["yearly"], key=mrq, field="interestExpense", decimal=2)
    else:
        ttm_operating_income = sup.sum_n_consecutive(data["Financials"]["Income_Statement"]["quarterly"], start_key=mrq, field="operatingIncome", decimal=2, n=4)
        ttm_interest_expense = sup.sum_n_consecutive(data["Financials"]["Income_Statement"]["quarterly"], start_key=mrq, field="interestExpense", decimal=2, n=4)

    ttm_interest_coverage_ratio = sup.safe_divide(ttm_operating_income, ttm_interest_expense, 2)

    fields["InterestCoverageRatio"]['value'][mrq_label] = sup.formatted_value(value_mrq_interest_coverage_ratio, 2, 'x')
    fields["InterestCoverageRatio"]['value'][yoy_change] = sup.formatted_value(percentage_interest_coverage_ratio, 2, 'x')
    fields["InterestCoverageRatio"]['value'][ttm] = sup.formatted_value(ttm_interest_coverage_ratio, 2, 'x')

    # Build a dictionary containing capital structure ratios. To be used by the ratio endpoint
    capital_ratios = {"DebtToEquity": average_ttm_debt_to_equity, "LTDebtToEquity": average_ttm_lt_debt_to_equity, "DebtToCapital": average_ttm_debt_to_capital,
                      "LTDebtToCapital": average_ttm_lt_debt_to_lt_capital, "InterestCoverageRatio": ttm_interest_coverage_ratio}

    result = {"capital_structure_indicators": fields, "capital_ratios": capital_ratios}

    return result




