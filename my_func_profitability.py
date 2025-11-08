

import my_func_support as sup
import my_func as f
import requests
import os
from datetime import datetime, timedelta
import my_func_eod as eod


api_key = os.getenv('eodhd')


# Function defined for capital structure indicators
def profitability(symbol, market='US', convert='bln', earnings_time=5, raw_data=None):

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

    data_set_income_statement = data["Financials"]["Income_Statement"]
    data_set_earnings = data["Earnings"]["History"]

    # Create dictionary with fields
    fields = {
      "profitability": {
            "MostRecentQuarter": {'label': "MostRecentQuarter", 'type': [], 'value': {mrq_label: mrq}},
            "totalRevenue": {'label': f"Revenues ({convert})", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "grossProfit": {'label': f"GrossIncome ({convert})", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "ebitda": {'label': f"EBITDA ({convert})", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "operatingIncome": {'label': f"OperatingIncome ({convert})", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "netIncome": {'label': f"NetIncome ({convert})", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "epsActual": {'label': f"Non-GAAP EPS ({currency})", 'type': [1,2],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "grossMargin": {'label': f"GrossMargin (%)", 'type': [1,3],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "ebitdaMargin": {'label': f"EBITDAMargin (%)", 'type': [1,3],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "operatingMargin": {'label': f"OperatingMargin (%)", 'type': [1,3],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''},
            "netMargin": {'label': f"NetMargin (%)", 'type': [1,3],'value': {mrq_label: None, yoy_change: None, qoq_change: None, ttm: None, ttm_change: None}, 'comment': ''}
        },
        "earnings": {
            "epsActual": {'label': f"epsActual ({currency})", 'type': [1,4], 'value': {}, 'comment': ''},
            "epsEstimate": {'label': f"epsEstimate ({currency})", 'type': [1,4], 'value': {}, 'comment': ''},
            "surprisePercent": {'label': f"surprisePercent (%)", 'type': [1,4], 'value': {}, 'comment': ''},
            "reportDate": {'label': "Report Date", 'type': [1,4], 'value': {}, 'comment': ''},
            "priceChangePrev5Days": {'label': "Price Change -5 Days (%)", 'type': [1,4], 'value': {}, 'comment': ''},
            "priceChangePost5Days": {'label': "Price Change +5 Days (%)", 'type': [1,4], 'value': {}, 'comment': ''}
        }
    }

    # Revenues
    api_field = "totalRevenue"
    revenues = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(revenues['value_mrq'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(revenues['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(revenues['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(revenues['value_ttm'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(revenues['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)


    # grossProfit
    api_field = "grossProfit"
    grossProfit = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(grossProfit['value_mrq'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(grossProfit['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(grossProfit['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(grossProfit['value_ttm'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(grossProfit['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)


    # ebitda
    api_field = "ebitda"
    ebitda = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field=api_field,
                              decimal=decimal)
    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(ebitda['value_mrq'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(ebitda['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(ebitda['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(ebitda['value_ttm'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(ebitda['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)


    # operatingIncome
    api_field = "operatingIncome"
    operatingIncome = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field=api_field,
                            decimal=decimal)
    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(operatingIncome['value_mrq'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(operatingIncome['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(operatingIncome['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(operatingIncome['value_ttm'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(operatingIncome['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)


    # netIncome
    api_field = "netIncome"
    netIncome = sup.populate_statement_field(data_set_income_statement, quarter_info, statement_type='period', api_field=api_field,
                            decimal=decimal)
    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(netIncome['value_mrq'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(netIncome['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(netIncome['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(netIncome['value_ttm'], decimal=decimal, convert=convert)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(netIncome['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)


    # epsActual
    api_field = "epsActual"
    epsActual = sup.populate_earnings_field(data_set_earnings, quarter_info, api_field=api_field, decimal=decimal)
    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(epsActual['value_mrq'], decimal=decimal, convert=None)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(epsActual['mrq_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(epsActual['mrq_qoq_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(epsActual['value_ttm'], decimal, convert=None)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(epsActual['ttm_yoy_percentage_change'],
                                                                                decimal=decimal, suffix=perc_suffix)


    # grossMargin
    api_field = "grossMargin"
    value_mrq_gross_margin = sup.safe_multiply(sup.safe_divide(grossProfit['value_mrq'],revenues['value_mrq'], decimal=decimal_mg), 100)
    value_prev_year_mrq_gross_margin = sup.safe_multiply(sup.safe_divide(grossProfit['value_prev_year_mrq'],revenues['value_prev_year_mrq'], decimal=decimal_mg), 100)
    value_prev_q_gross_margin = sup.safe_multiply(sup.safe_divide(grossProfit['value_prev_q'],revenues['value_prev_q'], decimal=decimal_mg), 100)
    yoy_delta_change_gross_margin = sup.safe_delta(value_mrq_gross_margin, value_prev_year_mrq_gross_margin, decimal=decimal)
    qoq_delta_change_gross_margin = sup.safe_delta(value_mrq_gross_margin, value_prev_q_gross_margin, decimal=decimal)

    ttm_mrq_gross_margin = sup.safe_multiply(sup.safe_divide(grossProfit['value_ttm'],revenues['value_ttm'],  decimal=decimal_mg), 100)
    ttm_prev_year_mrq_gross_margin = sup.safe_multiply(sup.safe_divide(grossProfit['value_prev_year_ttm'],revenues['value_prev_year_ttm'], decimal=decimal_mg), 100)
    yoy_delta_change_ttm_gross_margin = sup.safe_delta(ttm_mrq_gross_margin, ttm_prev_year_mrq_gross_margin, decimal=decimal)

    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_gross_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_gross_margin, decimal=decimal, suffix=pp_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_gross_margin, decimal=decimal,suffix=pp_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_gross_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_gross_margin, decimal=decimal, suffix=pp_suffix)

    # ebitdaMargin
    api_field = "ebitdaMargin"
    value_mrq_ebitda_margin = sup.safe_multiply(sup.safe_divide(ebitda['value_mrq'],revenues['value_mrq'], decimal=decimal_mg), 100)
    value_prev_year_mrq_ebitda_margin = sup.safe_multiply(sup.safe_divide(ebitda['value_prev_year_mrq'],revenues['value_prev_year_mrq'], decimal=decimal_mg), 100)
    value_prev_q_ebitda_margin = sup.safe_multiply(sup.safe_divide(ebitda['value_prev_q'],revenues['value_prev_q'], decimal=decimal_mg), 100)
    yoy_delta_change_ebitda_margin = sup.safe_delta(value_mrq_ebitda_margin, value_prev_year_mrq_ebitda_margin, decimal=decimal)
    qoq_delta_change_ebitda_margin = sup.safe_delta(value_mrq_ebitda_margin, value_prev_q_ebitda_margin, decimal=decimal)

    ttm_mrq_ebitda_margin = sup.safe_multiply(sup.safe_divide(ebitda['value_ttm'],revenues['value_ttm'],  decimal=decimal_mg), 100)
    ttm_prev_year_mrq_ebitda_margin = sup.safe_multiply(sup.safe_divide(ebitda['value_prev_year_ttm'],revenues['value_prev_year_ttm'], decimal=decimal_mg), 100)
    yoy_delta_change_ttm_ebitda_margin = sup.safe_delta(ttm_mrq_ebitda_margin, ttm_prev_year_mrq_ebitda_margin, 2)

    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_ebitda_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_ebitda_margin, decimal=decimal, suffix=pp_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_ebitda_margin, decimal=decimal,suffix=pp_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_ebitda_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_ebitda_margin, decimal=decimal, suffix=pp_suffix)

    # operatingMargin
    api_field = "operatingMargin"
    value_mrq_op_margin = sup.safe_multiply(sup.safe_divide(operatingIncome['value_mrq'],revenues['value_mrq'], decimal=decimal_mg), 100)
    value_prev_year_mrq_op_margin = sup.safe_multiply(sup.safe_divide(operatingIncome['value_prev_year_mrq'],revenues['value_prev_year_mrq'], decimal=decimal_mg), 100)
    value_prev_q_op_margin = sup.safe_multiply(sup.safe_divide(operatingIncome['value_prev_q'],revenues['value_prev_q'], decimal=decimal_mg), 100)
    yoy_delta_change_op_margin = sup.safe_delta(value_mrq_op_margin, value_prev_year_mrq_op_margin, decimal=decimal)
    qoq_delta_change_op_margin = sup.safe_delta(value_mrq_op_margin, value_prev_q_op_margin, decimal=decimal)

    ttm_mrq_op_margin = sup.safe_multiply(sup.safe_divide(operatingIncome['value_ttm'],revenues['value_ttm'],  decimal=decimal_mg), 100)
    ttm_prev_year_mrq_op_margin = sup.safe_multiply(sup.safe_divide(operatingIncome['value_prev_year_ttm'],revenues['value_prev_year_ttm'], decimal=decimal_mg), 100)
    yoy_delta_change_ttm_op_margin = sup.safe_delta(ttm_mrq_op_margin, ttm_prev_year_mrq_op_margin, 2)

    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_op_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_op_margin, decimal=decimal, suffix=pp_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_op_margin, decimal=decimal,suffix=pp_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_op_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_op_margin, decimal=decimal, suffix=pp_suffix)

    # netMargin
    api_field = "netMargin"
    value_mrq_net_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_mrq'],revenues['value_mrq'], decimal=decimal_mg), 100)
    value_prev_year_mrq_net_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_prev_year_mrq'],revenues['value_prev_year_mrq'], decimal=decimal_mg), 100)
    value_prev_q_net_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_prev_q'],revenues['value_prev_q'], decimal=decimal_mg), 100)
    yoy_delta_change_net_margin = sup.safe_delta(value_mrq_net_margin, value_prev_year_mrq_net_margin, decimal=decimal)
    qoq_delta_change_net_margin = sup.safe_delta(value_mrq_net_margin, value_prev_q_net_margin, decimal=decimal)

    ttm_mrq_net_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_ttm'],revenues['value_ttm'],  decimal=decimal_mg), 100)
    ttm_prev_year_mrq_net_margin = sup.safe_multiply(sup.safe_divide(netIncome['value_prev_year_ttm'],revenues['value_prev_year_ttm'], decimal=decimal_mg), 100)
    yoy_delta_change_ttm_net_margin = sup.safe_delta(ttm_mrq_net_margin, ttm_prev_year_mrq_net_margin, 2)

    fields["profitability"][api_field]['value'][mrq_label] = sup.formatted_value(value_mrq_net_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][yoy_change] = sup.formatted_value(yoy_delta_change_net_margin, decimal=decimal, suffix=pp_suffix)
    fields["profitability"][api_field]['value'][qoq_change] = sup.formatted_value(qoq_delta_change_net_margin, decimal=decimal,suffix=pp_suffix)
    fields["profitability"][api_field]['value'][ttm] = sup.formatted_value(ttm_mrq_net_margin, decimal=decimal)
    fields["profitability"][api_field]['value'][ttm_change] = sup.formatted_value(yoy_delta_change_ttm_net_margin, decimal=decimal, suffix=pp_suffix)

    # Populate earnings fields
    # Step 1: Fetch all EOD data for 1000 days
    try:
        # Fetch prices for the last 1000 trading days to cover all necessary dates
        eod_data = eod.fetch_eod(symbol=symbol, market=market, interval='1000d', frequency='d')
    except ValueError as e:
        raise ValueError(f"Error fetching price data: {e}")

    # Convert the EOD data to a list of sorted trading dates and a dictionary for quick lookup
    eod_data_dict = {entry['date']: entry for entry in eod_data}
    trading_dates = sorted(eod_data_dict.keys())  # Sort dates for index-based access

    # Step 2: Iterate over earnings quarters and calculate price changes
    earnings_quarters = sorted(data_set_earnings.keys(), reverse=True)
    mrq_index = earnings_quarters.index(mrq)
    earnings_quarters = earnings_quarters[mrq_index:mrq_index + earnings_time]

    for quarter in earnings_quarters:
        quarter_formatted = datetime.strptime(quarter, '%Y-%m-%d').strftime('%b-%y')
        earnings_data = data_set_earnings[quarter]
        report_date = earnings_data['reportDate']  # Retrieve the report date

        # Add earnings data
        fields["earnings"]["epsActual"]['value'][quarter_formatted] = earnings_data["epsActual"]
        fields["earnings"]["epsEstimate"]['value'][quarter_formatted] = earnings_data["epsEstimate"]
        fields["earnings"]["surprisePercent"]['value'][quarter_formatted] = sup.formatted_value(earnings_data["surprisePercent"], decimal=decimal)
        fields["earnings"]["reportDate"]['value'][quarter_formatted] = report_date

        # Convert the report date to datetime
        report_date_dt = datetime.strptime(report_date, '%Y-%m-%d')

        # Step 3: Find the closing prices for 5 trading days before and after the report date
        price_before = sup.get_eod(report_date_dt, eod_data_dict, trading_dates, -5)
        price_after = sup.get_eod(report_date_dt, eod_data_dict, trading_dates, 5)

        # Price on report date
        price_on_report = eod_data_dict.get(report_date_dt.strftime('%Y-%m-%d'), {}).get('close')

        # Step 4: Calculate the price changes if the prices are found
        if price_before is not None and price_on_report is not None:
            price_change_prev = sup.safe_multiply(sup.safe_divide(sup.safe_delta(price_on_report, price_before, 2), price_before, 4), 100)
        else:
            price_change_prev = None

        if price_after is not None and price_on_report is not None:
            price_change_post = sup.safe_multiply(sup.safe_divide(sup.safe_delta(price_after, price_on_report, 2), price_on_report, 4), 100)
        else:
            price_change_post = None

        # Step 5: Store the price change values in the fields
        fields["earnings"]["priceChangePrev5Days"]['value'][quarter_formatted] = sup.formatted_value(price_change_prev,
                                                                                                     decimal) if price_change_prev is not None else None
        fields["earnings"]["priceChangePost5Days"]['value'][quarter_formatted] = sup.formatted_value(price_change_post,
                                                                                                     decimal) if price_change_post is not None else None

    return fields

