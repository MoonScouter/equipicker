
from datetime import datetime, timedelta
import calendar

# Function to extract keys from a JSON response
def extract_keys(obj, path=None):
    """Recursive function to extract all keys from a nested JSON."""
    if path is None:
        path = []

    # If obj is a dictionary, iterate over its keys
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = path + [k]
            yield new_path
            yield from extract_keys(v, new_path)

    # If obj is a list, iterate over its elements
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            yield from extract_keys(item, path + [str(idx)])


# Function to get unique exchanges from a dataset of tickers extracted from specific country - works with Exchanges API
def get_unique_exchanges(stocks):
    return list({stock['OperatingMIC'] for stock in stocks})


# Function to get stocks from an exchange from a dataset of tickers from a specific country - works with Exchanges API
def get_exchange_stocks(stocks, code):
    return [
        {"Code": stock["Code"], "Name": stock["Name"], "Exchange": stock["Exchange"]}
        for stock in stocks if stock["Exchange"] == code
    ]


#  Function to get summary of tickers per exchange - works with Exchanges API
def get_exchange_summary(stocks, exchanges):
    summary = []

    for exchange in exchanges:
        exchange_stocks = get_exchange_stocks(stocks, exchange)
        exchange_info = {
            "Exchange": exchange,
            "Count": len(exchange_stocks),
            # "Tickers": exchange_stocks
        }
        summary.append(exchange_info)

    return summary


def convert_float(value, convert=None, decimal=2):
    if value is not None:
        try:
            value = float(value)
        except ValueError:
            return 'NA'

        if convert is None:
            return round(value, decimal)
        elif convert == 'th':
            return round(value / 1e3, decimal)
        elif convert == 'mln':
            return round(value / 1e6, decimal)
        elif convert == 'bln':
            return round(value / 1e9, decimal)
    return 'NA'


def safe_multiply(element1, element2):
    # Check if either element is None
    if element1 is None or element2 is None:
        return 'NA'

    # Check if either element is a string
    if isinstance(element1, str) or isinstance(element2, str):
        return 'NA'

    # If both checks are passed, multiply the elements
    return element1 * element2


def safe_divide(numerator, denominator, decimal=2):
    # Check if either value is None or denominator is zero
    if numerator is None or denominator is None or denominator == 0:
        return 'NA'

    # Check if either element is a string
    if isinstance(numerator, str) or isinstance(denominator, str):
        return 'NA'

    # Perform the division and round to the specified decimals
    return round(numerator / denominator, decimal)


def relative_change(numerator, denominator, decimal=6):
    # Check if either value is None or denominator is zero
    if numerator is None or denominator is None or denominator == 0:
        return 'NA'

    # Check if either element is a string
    if isinstance(numerator, str) or isinstance(denominator, str):
        return 'NA'

    # Ensure both numerator and denominator are integers (or floats)
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return 'NA'

    # Check if numerator and denominator have different signs
    if (numerator < 0 < denominator) or (numerator > 0 > denominator):
        return 'NA'

    # Perform the division and round to the specified decimals
    return round((numerator / denominator) - 1, decimal)


def safe_delta(element1, element2, decimal=2):
    # Check if either value is None
    if element1 is None or element2 is None:
        return 'NA'

    # Check if either element is a string
    if isinstance(element1, str) or isinstance(element2, str):
        return 'NA'

    # Perform the subtraction and round to the specified decimals
    return round(element1 - element2, decimal)


def safe_addition(element1, element2, decimal=2):
    # Check if either value is None
    if element1 is None or element2 is None:
        return 'NA'

    # Check if either element is a string
    if isinstance(element1, str) or isinstance(element2, str):
        return 'NA'

    # Perform the subtraction and round to the specified decimals
    return round(element1 + element2, decimal)


def safe_abs(element):
    # Check if either value is None
    if element is None:
        return 'NA'

    # Check if either element is a string
    if isinstance(element, str):
        return 'NA'

    # Perform the subtraction and round to the specified decimals
    return round(abs(element))


def calculate_mean(row):
    numeric_values = [v for v in row if isinstance(v, (int, float)) and v != 0.0]
    return round(sum(numeric_values) / len(numeric_values), 2) if numeric_values else 'NA'


def get_next_key(dictionary, current_key):
    keys = list(dictionary.keys())
    current_index = keys.index(current_key)
    # Check if current_key is not the last one
    if current_index + 1 < len(keys):
        return keys[current_index + 1]
    else:
        return None


def check_mrq(data):
    try:
        # Retrieve the most recent quarter date
        # mrq = data["Highlights"].get("MostRecentQuarter")
        mrq = list(data['Financials']['Income_Statement']['quarterly'].keys())[0]
        if not mrq:
            raise ValueError("Most Recent Quarter data is missing.")

        # Retrieve the fiscal year-end month
        fiscal_year_end = data["General"].get("FiscalYearEnd")
        if not fiscal_year_end:
            raise ValueError("Fiscal Year End data is missing.")

        # Convert MRQ string to datetime object
        quarter_date = datetime.strptime(mrq, '%Y-%m-%d')

        # Get the month name of the MRQ
        mrq_month = quarter_date.strftime('%B')

        # Check if MRQ month is the same as fiscal year-end month
        return mrq_month == fiscal_year_end

    except Exception as e:
        print(f"Error occurred: {e}")
        return False


def get_mrq(data):
    try:
        return list(data['Financials']['Income_Statement']['quarterly'].keys())[0]
    except ValueError as e:
        print(f"Error occurred: {e}")
        return None


def get_previous_year_mrq(data):
    try:
        # Get all quarters in chronological order
        quarters = sorted(data["Financials"]["Income_Statement"]["quarterly"].keys())

        # Get most recent quarter from data
        mrq = get_mrq(data)

        # Find the index of the MRQ in the sorted list
        mrq_index = quarters.index(mrq)

        # Calculate the index of the same quarter one year ago (4 quarters back)
        previous_year_index = mrq_index - 4

        # Retrieve the quarter from a year ago
        if previous_year_index >= 0:
            return quarters[previous_year_index]
        else:
            raise ValueError("No data available for the quarter from a year ago.")
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def get_prev_q(data):
    try:
        # Get all quarters in chronological order
        quarters = sorted(data["Financials"]["Income_Statement"]["quarterly"].keys())

        # Get most recent quarter from data
        mrq = get_mrq(data)

        # Find the index of the MRQ in the sorted list
        mrq_index = quarters.index(mrq)

        # Calculate the index of the previous quarter (1 quarter back)
        prev_q_index = mrq_index - 1

        # Retrieve the previous quarter
        if prev_q_index >= 0:
            return quarters[prev_q_index]
        else:
            raise ValueError("No data available for the previous quarter.")
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def avg_n_consecutive(data, start_key, field, n=2, convert=None, decimal=2):
    keys = list(data.keys())
    if start_key not in keys:
        raise ValueError(f"Start key '{start_key}' not found in dictionary.")

    start_index = keys.index(start_key)
    values = []

    # Collect values from n consecutive entries
    for i in range(start_index, min(start_index + n, len(keys))):
        if field in data[keys[i]].keys():
            value = data[keys[i]].get(field)
            if value is not None:  # Ensure the value is not None
                value = convert_float(value, convert=None, decimal=decimal)
                values.append(value)
            else:
                print(f'{field} is None')

    # Compute the average for the available data
    if len(values) > 0:
        avg_value = sum(values) / len(values)
        return convert_float(avg_value, convert=convert, decimal=decimal)
    else:
        return 'NA'  # Not enough values to compute the average


def sum_n_consecutive(data, start_key, field, n=2, convert=None, decimal=2):
    keys = list(data.keys())
    if start_key not in keys:
        raise ValueError(f"Start key '{start_key}' not found in dictionary.")

    start_index = keys.index(start_key)
    values = []

    # Collect values from n consecutive entries
    for i in range(start_index, min(start_index + n, len(keys))):
        if field in data[keys[i]].keys():
            value = data[keys[i]].get(field)
            if value is not None:  # Ensure the value is not None
                value = convert_float(value, convert=None, decimal=decimal)
                values.append(value)
            else:
                print(f'{field} is None')

    # Sum the values if we have enough values
    if len(values) == n:
        total_sum = sum(values)
        return convert_float(total_sum, convert=convert, decimal=decimal)
    else:
        return 'NA'  # Not enough values to compute the sum


def get_statement_value(dict, key, field, convert=None, decimal=2):
    return convert_float(dict[key].get(field), convert=convert, decimal=decimal)


def formatted_value(value, decimal=None, suffix=None, convert=None):
    # Apply conversion of value if required
    converted_value = convert_float(value, decimal=decimal, convert=convert)

    if converted_value == 'NA':
        return 'NA'

    # Format the value according to the specified number of decimals
    formatted_str = f"{converted_value:.{decimal}f}" if decimal is not None else str(converted_value)

    # Append the suffix if provided
    if suffix:
        formatted_str += f" {suffix}"

    return formatted_str


def get_quarter_info(data):
    try:
        # Get all quarters in chronological order from Income_statement(newest first)
        quarters = sorted(data["Financials"]["Income_Statement"]["quarterly"].keys(), reverse=True)

        # Retrieve the most recent quarter date
        mrq = quarters[0] if quarters else None
        if not mrq:
            raise ValueError("Most Recent Quarter data is missing from Income_Statement.")

        # Get all quarters in chronological order from Balance_sheet (newest first) to add a logging
        quarters_bs = sorted(data["Financials"]["Balance_Sheet"]["quarterly"].keys(), reverse=True)

        # Retrieve the most recent quarter date based on Balance_sheet
        mrq_bs = quarters_bs[0] if quarters_bs else None
        if not mrq_bs:
            raise ValueError("Most Recent Quarter data is missing from Balance_Sheet.")
        elif mrq_bs != mrq:
            raise ValueError("Most Recent Quarter data is different between Income_Statement and Balance_Sheet.")

        # Calculate previous quarter (1 quarter back)
        prev_q_index = 1 if len(quarters) > 1 else None
        prev_q = quarters[prev_q_index] if prev_q_index is not None else None
        if not prev_q:
            raise ValueError("Previous Quarter data is missing.")

        # Calculate previous year's most recent quarter (4 quarters back)
        prev_year_index = 4 if len(quarters) > 4 else None
        prev_year_mrq = quarters[prev_year_index] if prev_year_index is not None else None
        if not prev_year_mrq:
            raise ValueError("Previous Year Most Recent Quarter data is missing.")

        # Calculate previous year's quarter preceding most recent quarter (5 quarters back)
        prev_year_prev_q_index = 5 if len(quarters) > 5 else None

        if not prev_year_prev_q_index:
            raise ValueError("Previous Year Quarter preceding Most Recent Quarter data is missing.")

        prev_year_prev_q = quarters[prev_year_prev_q_index] if prev_year_prev_q_index is not None else None

        # Retrieve the fiscal year-end month
        fiscal_year_end = data["General"].get("FiscalYearEnd")
        if not fiscal_year_end:
            raise ValueError("Fiscal Year End data is missing.")

        # Convert MRQ string to datetime object
        quarter_date = datetime.strptime(mrq, '%Y-%m-%d')

        # Get the month name of the MRQ
        mrq_month = quarter_date.strftime('%B')

        # Check if MRQ month is the same as fiscal year-end month
        check_mrq = (mrq_month == fiscal_year_end)

        return {
            "mrq": mrq,
            "prev_q": prev_q,
            "prev_year_mrq": prev_year_mrq,
            "prev_year_prev_q": prev_year_prev_q,
            "check_mrq": check_mrq
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        return {
            "mrq": None,
            "prev_q": None,
            "prev_year_mrq": None,
            "prev_year_prev_q": None,
            "check_mrq": False
        }


def populate_statement_field(data, quarter_info: dict, api_field, statement_type='period', decimal=2):
    values = {}

    mrq = quarter_info['mrq']
    prev_year_mrq = quarter_info['prev_year_mrq']
    prev_q = quarter_info['prev_q']
    check_mrq_flag = quarter_info['check_mrq']

    # Get field values and the y/y change
    value_mrq = get_statement_value(data["quarterly"], key=mrq, field=api_field, decimal=decimal)
    value_prev_year_mrq = get_statement_value(data["quarterly"], key=prev_year_mrq, field=api_field,decimal=decimal)
    mrq_yoy_percentage_change = safe_multiply(relative_change(value_mrq, value_prev_year_mrq), 100)

    # Get field values and the q/q change
    value_prev_q = get_statement_value(data["quarterly"], key=prev_q, field=api_field, decimal=decimal)
    mrq_qoq_percentage_change = safe_multiply(relative_change(value_mrq, value_prev_q), 100)

    # Initialize TTM values
    value_ttm = None
    value_prev_year_ttm = None

    # Compute TTM values for mrq
    if check_mrq_flag:
        value_ttm = get_statement_value(data["yearly"], key=mrq, field=api_field, decimal=decimal)
    else:
        if statement_type == 'period':
            value_ttm = sum_n_consecutive(data["quarterly"], start_key=mrq, field=api_field, decimal=decimal, n=4)
        elif statement_type == 'snapshot':
            value_ttm = avg_n_consecutive(data["quarterly"], start_key=mrq, field=api_field, decimal=decimal, n=4)

    # Compute previous year TTM values for the equivalent mrq
    if check_mrq_flag:
        value_prev_year_ttm = get_statement_value(data["yearly"], key=prev_year_mrq, field=api_field, decimal=decimal)
    else:
        if statement_type == 'period':
            value_prev_year_ttm = sum_n_consecutive(data["quarterly"], start_key=prev_year_mrq, field=api_field, decimal=decimal, n=4)
        elif statement_type == 'snapshot':
            value_prev_year_ttm = avg_n_consecutive(data["quarterly"], start_key=prev_year_mrq, field=api_field, decimal=decimal, n=4)

    ttm_yoy_percentage_change = safe_multiply(relative_change(value_ttm, value_prev_year_ttm), 100)

    values['value_mrq'] = value_mrq
    values['value_prev_year_mrq'] = value_prev_year_mrq
    values['mrq_yoy_percentage_change'] = mrq_yoy_percentage_change
    values['value_prev_q'] = value_prev_q
    values['mrq_qoq_percentage_change'] = mrq_qoq_percentage_change
    values['value_ttm'] = value_ttm
    values['value_prev_year_ttm'] = value_prev_year_ttm
    values['ttm_yoy_percentage_change'] = ttm_yoy_percentage_change

    return values


def populate_earnings_field(data, quarter_info: dict, api_field, decimal=2):
    values = {}

    mrq = quarter_info['mrq']
    prev_year_mrq = quarter_info['prev_year_mrq']
    prev_q = quarter_info['prev_q']

    # Get field values and the y/y change
    value_mrq = get_statement_value(data, key=mrq, field=api_field, decimal=decimal)
    value_prev_year_mrq = get_statement_value(data, key=prev_year_mrq, field=api_field,decimal=decimal)
    mrq_yoy_percentage_change = safe_multiply(relative_change(value_mrq, value_prev_year_mrq), 100)

    # Get field values and the q/q change
    value_prev_q = get_statement_value(data, key=prev_q, field=api_field, decimal=decimal)
    mrq_qoq_percentage_change = safe_multiply(relative_change(value_mrq, value_prev_q), 100)

    # Compute TTM values for mrq
    value_ttm = sum_n_consecutive(data, start_key=mrq, field=api_field, decimal=decimal, n=4)

    # Compute previous year TTM values for the equivalent mrq
    value_prev_year_ttm = sum_n_consecutive(data, start_key=prev_year_mrq, field=api_field, decimal=decimal, n=4)

    ttm_yoy_percentage_change = safe_multiply(relative_change(value_ttm, value_prev_year_ttm), 100)

    values['value_mrq'] = value_mrq
    values['value_prev_year_mrq'] = value_prev_year_mrq
    values['mrq_yoy_percentage_change'] = mrq_yoy_percentage_change
    values['value_prev_q'] = value_prev_q
    values['mrq_qoq_percentage_change'] = mrq_qoq_percentage_change
    values['value_ttm'] = value_ttm
    values['value_prev_year_ttm'] = value_prev_year_ttm
    values['ttm_yoy_percentage_change'] = ttm_yoy_percentage_change

    return values


def parse_period(period):
    # Handling periods like '0d', '7d', '1m', '2m', 'may-2024, jan-2024, january-2024, Jan-2024 etd'
    try:
        # Strip whitespace and convert to lowercase for handling 'd' and 'm'
        period = period.strip().lower()

        # Handle periods ending with 'd' (e.g., '1d', '7d', '1D', '7D')
        if period.endswith('d'):
            days = int(period[:-1])  # Extract the number of days
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')  # Today is the end date

        # Handle periods ending with 'm' (e.g., '1m', '1M', '2m', '2M')
        elif period.endswith('m'):
            months = int(period[:-1])  # Extract the number of months
            start_date = (datetime.now() - timedelta(days=months * 30)).strftime(
                '%Y-%m-%d')  # Approximate by 30 days per month
            end_date = datetime.now().strftime('%Y-%m-%d')  # Today is the end date

        # Handle full or abbreviated month-year formats (e.g., 'May-2024', 'may-2024', 'January-2024', 'january-2024')
        else:
            period = period.title()  # Normalize to title case (e.g., 'january-2024' -> 'January-2024')
            try:
                # Try full month name first
                start_date = datetime.strptime(period, "%B-%Y").strftime('%Y-%m-%d')  # First day of the month
                month_num = datetime.strptime(period, "%B-%Y").month
            except ValueError:
                # If that fails, try abbreviated month name
                start_date = datetime.strptime(period, "%b-%Y").strftime('%Y-%m-%d')  # First day of the month
                month_num = datetime.strptime(period, "%b-%Y").month

            year = int(period[-4:])  # Extract year
            last_day_of_month = calendar.monthrange(year, month_num)[1]  # Get last day of the month
            end_date = datetime.strptime(f'{year}-{month_num}', '%Y-%m').replace(day=last_day_of_month).strftime(
                '%Y-%m-%d')
    except ValueError:
        raise ValueError(
            "Invalid period format. Use formats like '7d', '1m', 'May-2024', 'Jan-2024', or 'January-2024'")

    return start_date, end_date


# Helper function to find the price 5 trading days before or after the report date
def get_eod(report_date, eod_data_dict, trading_dates, offset):

    # Ensure report_date is a valid trading day or choose the closest earlier trading day
    while report_date.strftime('%Y-%m-%d') not in eod_data_dict and report_date.strftime('%Y-%m-%d') > trading_dates[0]:
        report_date -= timedelta(days=1)  # Move to previous day until a trading day is found

    # Convert the report_date back to string to match the format of trading_dates
    report_date_str = report_date.strftime('%Y-%m-%d')

    # Get the index of the report date in the trading_dates list
    if report_date_str in trading_dates:
        report_index = trading_dates.index(report_date_str)
    else:
        return None  # If no valid trading day is found

    # Calculate the target index for 5 trading days before/after
    target_index = report_index + offset

    # Ensure the target index is within bounds
    if 0 <= target_index < len(trading_dates):
        target_date = trading_dates[target_index]
        return eod_data_dict[target_date]['close']
    else:
        return None






