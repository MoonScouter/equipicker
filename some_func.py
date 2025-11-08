import my_func_support as sup

api_field = "totalRevenue"
decimal=2
data_set_income_statement = data["Financials"]["Income_Statement"]
data_set_balance_sheet = data["Financials"]["Balance_Sheet"]

quarter_info = sup.get_quarter_info(data)

# Populate performance fields (similar logic for other fields)
def populate_field(data, quarter_info:dict, api_field, statement_type = 'period', decimal=2):
    values ={}

    mrq = quarter_info['mrq']
    prev_year_mrq = quarter_info['prev_year_mrq']
    prev_q = quarter_info['prev_q']
    check_mrq = quarter_info['check_mrq']

    # Get field values and the y/y change
    value_mrq = sup.get_statement_value(data["quarterly"], key=mrq, field=api_field, decimal=decimal)
    value_prev_year_mrq = sup.get_statement_value(data["quarterly"], key=prev_year_mrq, field=api_field, decimal=decimal)
    mrq_yoy_percentage_change = sup.safe_multiply(sup.relative_change(value_mrq, value_prev_year_mrq), 100)

    # Get field values and the q/q change
    value_prev_q = sup.get_statement_value(data["quarterly"], key=prev_q, field=api_field, decimal=decimal)
    mrq_qoq_percentage_change = sup.safe_multiply(sup.relative_change(value_mrq, value_prev_q), 100)

    # Compute TTM values for mrq
    if check_mrq:
        value_ttm = sup.get_statement_value(data["yearly"], key=mrq, field=api_field, decimal=decimal)
    else:
        if statement_type == 'period':
            value_ttm = sup.sum_n_consecutive(data["quarterly"], start_key=mrq, field=api_field, decimal=decimal, n=4)
        elif statement_type == 'snapshot':
            value_ttm = sup.avg_n_consecutive(data["quarterly"], start_key=mrq, field=api_field, decimal=decimal, n=4)

    # Compute previous year TTM values for the equivalent mrq
    if check_mrq:
        value_prev_year_ttm = sup.get_statement_value(data["yearly"], key=prev_year_mrq, field=api_field, decimal=decimal)
    else:
        if statement_type == 'period':
            value_prev_year_ttm = sup.sum_n_consecutive(data["quarterly"], start_key=prev_year_mrq, field=api_field, decimal=decimal, n=4)
        elif statement_type == 'snapshot':
            value_prev_year_ttm = sup.avg_n_consecutive(data["quarterly"], start_key=prev_year_mrq, field=api_field, decimal=decimal, n=4)

    ttm_yoy_percentage_change = sup.safe_multiply(sup.relative_change(value_ttm, value_prev_year_ttm), 100)

    values['value_mrq'] = value_mrq
    values['value_prev_year_mrq'] = value_prev_year_mrq
    values['mrq_yoy_percentage_change'] = mrq_yoy_percentage_change
    values['value_prev_q'] = value_prev_q
    values['mrq_qoq_percentage_change'] = mrq_qoq_percentage_change
    values['value_ttm'] = value_ttm
    values['value_prev_year_ttm'] = value_prev_year_ttm
    values['ttm_yoy_percentage_change'] = ttm_yoy_percentage_change

    return values

revenues = populate_field(data_set_income_statement, quarter_info, statement_type='period', api_field=api_field, decimal=decimal)

fields["performance"][api_field]['value'][mrq_label] = sup.formatted_value(revenues['value_mrq'], decimal, convert)
fields["performance"][api_field]['value'][yoy_change] = sup.formatted_value(revenues['mrq_yoy_percentage_change'], 2, '%')
fields["performance"][api_field]['value'][qoq_change] = sup.formatted_value(revenues['mrq_qoq_percentage_change'], 2, '%')
fields["performance"][api_field]['value'][ttm] = sup.formatted_value(revenues['value_ttm'], decimal, convert)
fields["performance"][api_field]['value'][ttm_change] = sup.formatted_value(revenues['ttm_yoy_percentage_change'], 2, '%')
