

import my_func_support as sup
import my_func as f
import requests
import os
from datetime import datetime


api_key = os.getenv('eodhd')


# Function defined for capital structure indicators
def cashflow(symbol, market='US', convert='bln', raw_data=None):

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
    mrq_formatted = datetime.strptime(mrq, '%Y-%m-%d').strftime('%b-%y') # Formatted as mmm-yy, ex. Jun-23

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
    convert_suffix=f'{convert}'

    data_set_balance_sheet = data["Financials"]["Balance_Sheet"]
    data_set_income_statement = data["Financials"]["Income_Statement"]
    data_set_cash_flow = data["Financials"]["Cash_Flow"]

    # Create dictionary with fields
    fields = {
        "MostRecentQuarter": {'label': "MostRecentQuarter", 'type': [], 'value': {mrq_label: mrq}},
        "totalCashFromOperatingActivities": {'label': f"Operating CF ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "capitalExpenditures": {'label': f"Capex ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "freeCashFlow": {'label': f"Free CashFlow ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "totalCashFromFinancingActivities": {'label': f"Financing CF ({convert})", 'type': [1],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "issuanceOfCapitalStock": {'label': f"Stock Issuance ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "netBorrowings": {'label': f"Net borrowings ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "otherCashflowsFromFinancingActivities": {'label': f"Other Financing CF ({convert})", 'type': [1],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "dividendsPaid": {'label': f"Dividends paid ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "salePurchaseOfStock": {'label': f"Stock repurchases ({convert})", 'type': [1], 'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
        "changeInCash": {'label': f"Change In Cash Balance ({convert})", 'type': [], 'value': {}, 'comment': ''},
        "endPeriodCashFlow": {'label': f"Cash @ end {mrq_formatted} ({convert})", 'type': [1], 'value': {}, 'comment': ''},
        "shortTermInvestments": {'label': f"Short-term investments ({convert})", 'type': [1], 'value': {}, 'comment': ''},
        "currentRatio": {'label': f"Current Ratio @ end {mrq_formatted}", 'type': [1, 2], 'value': {}, 'comment': ''}
    }

    # totalCashFromOperatingActivities
    api_field = "totalCashFromOperatingActivities"
    op_cf = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(op_cf['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(op_cf['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(op_cf['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(op_cf['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(op_cf['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # capitalExpenditures
    api_field = "capitalExpenditures"
    capex = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(capex['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(capex['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(capex['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(capex['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(capex['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # freeCashFlow
    api_field = "freeCashFlow"
    fcf = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(fcf['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(fcf['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(fcf['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(fcf['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(fcf['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    # freeCashFlow
    api_field = "freeCashFlow"
    fcf = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(fcf['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(fcf['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(fcf['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(fcf['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(fcf['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    # totalCashFromFinancingActivities
    api_field = "totalCashFromFinancingActivities"
    fin_cf = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(fin_cf['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(fin_cf['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(fin_cf['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(fin_cf['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(fin_cf['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # issuanceOfCapitalStock
    api_field = "issuanceOfCapitalStock"
    stock_issuance = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(stock_issuance['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(stock_issuance['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(stock_issuance['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(stock_issuance['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(stock_issuance['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # netBorrowings
    api_field = "netBorrowings"
    net_borrowings = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(net_borrowings['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(net_borrowings['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(net_borrowings['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(net_borrowings['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(net_borrowings['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # otherCashflowsFromFinancingActivities
    api_field = "otherCashflowsFromFinancingActivities"
    other_fin_cf = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(other_fin_cf['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(other_fin_cf['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(other_fin_cf['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(other_fin_cf['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(other_fin_cf['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # dividendsPaid
    api_field = "dividendsPaid"
    dividends = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(dividends['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(dividends['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(dividends['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(dividends['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(dividends['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # salePurchaseOfStock
    api_field = "salePurchaseOfStock"
    stock_repurchase = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(stock_repurchase['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(stock_repurchase['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(stock_repurchase['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(stock_repurchase['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(stock_repurchase['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # changeInCash
    api_field = "changeInCash"
    change_cash = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(change_cash['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(change_cash['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(change_cash['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(change_cash['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(change_cash['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # endPeriodCashFlow
    api_field = "endPeriodCashFlow"
    end_cash = sup.populate_statement_field(data_set_cash_flow, quarter_info, statement_type='snapshot', api_field=api_field,
                              decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(end_cash['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(end_cash['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(end_cash['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(end_cash['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(end_cash['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)

    # shortTermInvestments
    api_field = "shortTermInvestments"
    st_inv = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot',
                                            api_field=api_field,
                                            decimal=decimal)
    fields[api_field]['value'][mrq_label] = sup.formatted_value(st_inv['value_mrq'], decimal=decimal, convert=convert)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(st_inv['mrq_yoy_percentage_change'],
                                                                 decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(st_inv['mrq_qoq_percentage_change'],
                                                                 decimal=decimal, suffix=perc_suffix)
    fields[api_field]['value'][ttm] = sup.formatted_value(st_inv['value_ttm'], decimal=decimal, convert=convert)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(st_inv['ttm_yoy_percentage_change'],
                                                                 decimal=decimal, suffix=perc_suffix)

    # currentRatio
    api_field = "currentRatio"
    current_assets = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field="totalCurrentAssets",
                              decimal=decimal)
    current_liabilities = sup.populate_statement_field(data_set_balance_sheet, quarter_info, statement_type='snapshot', api_field="totalCurrentLiabilities",
                              decimal=decimal)

    value_mrq_current_ratio = sup.safe_divide(current_assets['value_mrq'], current_liabilities['value_mrq'], decimal=decimal)
    value_prev_year_mrq_current_ratio = sup.safe_divide(current_assets['value_prev_year_mrq'], current_liabilities['value_prev_year_mrq'], decimal=decimal)
    value_prev_q_current_ratio = sup.safe_divide(current_assets['value_prev_q'], current_liabilities['value_prev_q'], decimal=decimal)
    yoy_delta_change_current_ratio = sup.safe_delta(value_mrq_current_ratio, value_prev_year_mrq_current_ratio, decimal=decimal)
    qoq_delta_change_current_ratio = sup.safe_delta(value_mrq_current_ratio, value_prev_q_current_ratio, decimal=decimal)

    ttm_mrq_current_ratio = sup.safe_divide(current_assets['value_ttm'], current_liabilities['value_ttm'], decimal=decimal)
    ttm_prev_year_mrq_current_ratio = sup.safe_divide(current_assets['value_prev_year_ttm'], current_liabilities['value_prev_year_ttm'], decimal=decimal)
    yoy_delta_change_ttm_current_ratio = sup.safe_delta(ttm_mrq_current_ratio, ttm_prev_year_mrq_current_ratio, decimal=decimal)

    fields[api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_current_ratio, decimal=decimal, suffix=None)
    fields[api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_current_ratio, decimal=decimal, suffix=None)
    fields[api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_current_ratio, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_current_ratio, decimal=decimal, suffix=None)
    fields[api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_current_ratio, decimal=decimal,suffix=None)

    return fields
