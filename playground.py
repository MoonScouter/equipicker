
import my_func_profitability as profit
import my_func as f
import my_func_support as sup
import my_func_ratios_old as ratio
import my_func_ratios as ratio_new
import requests
import importlib
import my_func_capital_old as cap
import my_func_capital as cap_new
importlib.reload(f)
importlib.reload(cap_new)
importlib.reload(ratio_new)
importlib.reload(sup)
importlib.reload(cap)
importlib.reload(ratio)
import os
from datetime import datetime
import my_func_cashflow as cash


api_key = os.getenv('eodhd')

all = f.one_shot_analysis(symbol='CAT', peers=['PCAR', 'GPC'])
cashflow = f.cashflow('DIS')
cashflow.to_csv('cash_dis.csv')
print(cashflow)

capital = f.capital('DIS')
capital_new = f.capital_new('DIS')
filename_cap='cap_dis.csv'
filename_cap_new='cap_new_dis.csv'
capital.to_csv(filename_cap)
capital_new.to_csv(filename_cap_new)


symbol = 'CAT'
profit = f.profitability(symbol=symbol, type='overview', convert=None, earnings_time=5)
print(profit)
ratio_new = ratio_new.get_ratios(symbol=symbol)
ratio = f.ratios(symbols=[symbol], type = 'overview')
ratio_new_2 = f.ratios_new(symbols=[symbol], type = 'overview')
print(ratio)
print(ratio_new_2)

capital_new = cap_new.capital_structure(symbol='DIS')

filename1='profit_dis_indicators.csv'
filename2='profit_dis_eps.csv'
profit['profit_indicators_and_margins'].to_csv(filename1)
profit['earnings_surprise'].to_csv(filename2)

symbol = ['NVDA']
ratios.to_csv(filename)

ratios = f.ratios(symbols=symbols, type='capital')
# print(ratios)
ratios.to_csv(filename)

business = f.business('PATH')
ratios = f.ratios(symbols=['MET','AIG','PFG','CNO','LNC'])

capital = f.capital('UBER')
filename_capital='capital_RIOT.csv'
capital.to_csv(filename_capital)
print(capital)

endpoint_debug = f.generate_url('fundamentals', ticker=f'UBER.US')

params_debug = {
    'api_token': api_key,
    'fmt': 'json'
}

response_debug = requests.get(endpoint_debug, params=params_debug)

if response_debug.status_code == 200:
    data_debug = response_debug.json()

mrq = data_debug["Highlights"].get("MostRecentQuarter")
mrq_formatted = datetime.strptime(mrq, '%Y-%m-%d').strftime('%b-%y')  # Formatted as mmm-yy, ex. Jun-23
check_mrq = sup.check_mrq(data_debug)
prev_year_mrq = sup.get_previous_year_mrq(data_debug)



value_mrq_total_assets = sup.get_statement_value(data_debug["Financials"]["Balance_Sheet"]["quarterly"], key=mrq, field="totalAssets", decimal=2)
value_prev_year_quarter_total_assets = sup.get_statement_value(data_debug["Financials"]["Balance_Sheet"]["quarterly"], key=prev_year_mrq, field="totalAssets", decimal=2)
percentage_change_total_assets = sup.relative_change(value_mrq_total_assets, value_prev_year_quarter_total_assets) * 100
average_ttm_total_assets = sup.avg_n_consecutive(data_debug["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq, field="totalAssets", n=5, convert='bln')


value_mrq_net_profit = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"], key=mrq,
                                               field="netIncome", decimal=2)
value_prev_year_quarter_net_profit = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"],
                                                             key=prev_year_mrq, field="netIncome", decimal=2)

value_mrq_revenues = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"], key=mrq,
                                             field="totalRevenue", decimal=2)
value_prev_year_quarter_revenues = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"],
                                                           key=prev_year_mrq, field="totalRevenue", decimal=2)

value_mrq_net_profit_margin = sup.safe_divide(value_mrq_net_profit * 100, value_mrq_revenues, 2)
value_prev_year_quarter_net_profit_margin = sup.safe_divide(value_prev_year_quarter_net_profit * 100,
                                                            value_prev_year_quarter_revenues, 2)

percentage_change_net_profit_margin = sup.delta_change(value_mrq_net_profit_margin,
                                                       value_prev_year_quarter_net_profit_margin)

if check_mrq:
    ttm_net_profit = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["yearly"], key=mrq,
                                             field="netIncome", decimal=2)
    ttm_total_revenue = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["yearly"], key=mrq,
                                                field="totalRevenue", decimal=2)
else:
    ttm_net_profit = sup.sum_n_consecutive(data_debug["Financials"]["Income_Statement"]["quarterly"], start_key=mrq,
                                           field="netIncome", decimal=2, n=4)
    ttm_total_revenue = sup.sum_n_consecutive(data_debug["Financials"]["Income_Statement"]["quarterly"], start_key=mrq,
                                              field="totalRevenue", decimal=2, n=4)

ttm_net_profit_margin = sup.safe_divide(ttm_net_profit * 100, ttm_total_revenue, 2)

# AssetTurnoverDupont
average_last_two_q_total_assets = sup.avg_n_consecutive(data_debug["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,
                                                        field="totalAssets", n=2)
value_mrq_asset_turnover = sup.safe_divide(value_mrq_revenues, average_last_two_q_total_assets, 2)
average_last_two_q_prev_year_total_assets = sup.avg_n_consecutive(data_debug["Financials"]["Balance_Sheet"]["quarterly"],
                                                                  start_key=prev_year_mrq, field="totalAssets", n=2,
                                                                  )
value_prev_year_quarter_asset_turnover = sup.safe_divide(value_prev_year_quarter_revenues,
                                                         average_last_two_q_prev_year_total_assets, 2)

percentage_change_asset_turnover = sup.delta_change(value_mrq_asset_turnover, value_prev_year_quarter_asset_turnover)

ttm_asset_turnover = sup.safe_divide(ttm_total_revenue, average_ttm_total_assets, 2)


# LeverageDupont
average_last_two_q_shareholders_equity = sup.avg_n_consecutive(data_debug["Financials"]["Balance_Sheet"]["quarterly"],
                                                               start_key=mrq, field="totalStockholderEquity", n=2,
                                                               )
value_mrq_leverage = sup.safe_divide(average_last_two_q_total_assets, average_last_two_q_shareholders_equity, 2)
average_last_two_q_prev_year_shareholders_equity = sup.avg_n_consecutive(
    data_debug["Financials"]["Balance_Sheet"]["quarterly"], start_key=prev_year_mrq, field="totalStockholderEquity", n=2,
    )
value_prev_year_quarter_leverage = sup.safe_divide(average_last_two_q_prev_year_total_assets,
                                                   average_last_two_q_prev_year_shareholders_equity, 2)

percentage_change_leverage = sup.delta_change(value_mrq_leverage, value_prev_year_quarter_leverage)

average_ttm_shareholders_equity = sup.avg_n_consecutive(data_debug["Financials"]["Balance_Sheet"]["quarterly"], start_key=mrq,
                                                        field="totalStockholderEquity", n=5)

ttm_leverage = sup.safe_divide(average_ttm_total_assets, average_ttm_shareholders_equity, 2)



# InterestCoverageRatio
value_mrq_operating_income = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"], key=mrq,
                                                     field="operatingIncome", decimal=2)
value_mrq_interest_expense = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"], key=mrq,
                                                     field="interestExpense", decimal=2)
value_mrq_interest_coverage_ratio = sup.safe_divide(value_mrq_operating_income, value_mrq_interest_expense, 2)

value_prev_year_quarter_operating_income = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"],
                                                                   key=prev_year_mrq, field="operatingIncome",
                                                                   decimal=2)
value_prev_year_quarter_interest_expense = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["quarterly"],
                                                                   key=prev_year_mrq, field="interestExpense",
                                                                   decimal=2)
value_prev_year_quarter_interest_coverage_ratio = sup.safe_divide(value_prev_year_quarter_operating_income,
                                                                  value_prev_year_quarter_interest_expense, 2)

percentage_interest_coverage_ratio = sup.delta_change(value_mrq_interest_coverage_ratio,
                                                      value_prev_year_quarter_interest_coverage_ratio)

if check_mrq:
    ttm_operating_income = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["yearly"], key=mrq,
                                                   field="operatingIncome", decimal=2)
    ttm_interest_expense = sup.get_statement_value(data_debug["Financials"]["Income_Statement"]["yearly"], key=mrq,
                                                   field="interestExpense", decimal=2)
else:
    ttm_operating_income = sup.sum_n_consecutive(data_debug["Financials"]["Income_Statement"]["quarterly"], start_key=mrq,
                                                 field="operatingIncome", decimal=2, n=4)
    ttm_interest_expense = sup.sum_n_consecutive(data_debug["Financials"]["Income_Statement"]["quarterly"], start_key=mrq,
                                                 field="interestExpense", decimal=2, n=4)

ttm_interest_coverage_ratio = sup.safe_divide(ttm_operating_income, ttm_interest_expense, 2)






