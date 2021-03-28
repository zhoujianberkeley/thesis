#%%
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
from pathlib import Path
from datetime import datetime, date
from tqdm import tqdm
import qgrid
import matplotlib
import matplotlib.pyplot as plt

# Tushare + wind API 能用tushare就tushare 实在不行上windAPI 不用csmar
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
import tushare as ts
ts.set_token(token)
pro = ts.pro_api()

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

s_path = Path("_saved_factors")
if not os.path.exists(s_path):
    os.mkdir(s_path)

import utils
from utils import todate, save_f, cleandf, date_list
# %%
# load data
_load = True
if _load:
    Quarter_data = pd.read_csv(Path('data', 'QuarterFactorPrcd.csv'),encoding='gbk')
    Quarter_data = todate(Quarter_data, 'end_date', format='%Y-%m-%d')
    # s_date = date(2009, 12, 31)
    # Quarter_data = Quarter_data[Quarter_data['end_date'] > pd.to_datetime(s_date)]
    Quarter_data.index = range(len(Quarter_data))
    Quarter_data.to_pickle(Path('data', 'QuarterFactorPrcd.pkl'))
else:
    Quarter_data = pd.read_pickle(Path('data', 'QuarterFactorPrcd.pkl'))

# %%
'''
1.acc -Annual:
Annual income before extraordinary items(ib) minus operating cash flow(oancf) dividended by average total asset(at)
If oancf is missing, then set to change in act - change in che - change in lct + change in dlc + change in txp-dp

che: cash and equivalent
act: 
data: tushare API balancesheet_vip cashflow_vip income_vip
    average total asset (at=(t0+t1)/2 ?):  total_assets 
    annual income before extraordinary items: revenue
    operating cash flow: n_cashflow_act
'''
fn = 'Factor01_acc'
f1 = (Quarter_data['ebit_x'] - Quarter_data['n_cashflow_act'])/Quarter_data['total_assets']
Quarter_data[fn] = f1
save_f(f1, fn)


# %%
'''
2.absacc
Absolute value of acc
'''
fn = 'Factor02_absacc'
f2 = Quarter_data['Factor01_acc'].abs()
Quarter_data[fn] = f2
save_f(f2, fn)

# %%
'''
4.age -Annual:
Number of years since tushare first coverage

current_time_period(yr) - list_date(yr)

data: 
API - stock_basic: list_date
In the final data label you transform list_date to age
'''
fn = 'Factor04_age'
list_date = pro.stock_basic(exchange='',list_status='L')
list_date['list_date'] = list_date['list_date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
Quarter_data = Quarter_data.merge(list_date[['ts_code', 'list_date']], on='ts_code')
f4 = Quarter_data['end_date'] - Quarter_data['list_date']
Quarter_data[fn] = f4

save_f(f4, fn)

# %%
'''
6. analyst_score: Analyst forecast
Most recently available analyst forecasted overall score
wind 综合评分
'''
fn = 'Factor06_analyst_score'
f6 = Quarter_data['rank']
Quarter_data[fn] = f6
save_f(f6, fn)
# %%
'''
9.bm - Annual
Book value of equity(ceq) divided by end of fiscal year-end market capitalization 
Tushare:
book_value-->total_hldr_eqy_inc_min_int
market_value :wind定义2 - total_share
'''
fn = 'Factor09_bm'
Quarter_data['market_value'] = Quarter_data['total_share']
f9 = Quarter_data['total_hldr_eqy_inc_min_int']/Quarter_data['market_value']
Quarter_data['Factor09_bm'] = f9
save_f(f9, fn)

#%%
'''
10.bm_ia - Annual
Industry adjusted book-to-market ratio(subtract the industry mean)
'''
fn = 'Factor10_bm'
f10 = Quarter_data['Factor09_bm']-Quarter_data['Industry_total_eqy']/Quarter_data['Industry_market_value']
Quarter_data[fn] = f10
save_f(f10, fn)

#%%
'''
11.cash - annual
Cash and cash equivalents divided by end of fiscal year-end market capitalization
ca_to_assets
'''
fn = 'Factor11_cash'
f11 = Quarter_data['ca_to_assets']
Quarter_data[fn] = f11
save_f(f11, fn)
#%%

'''
12.cashdebt - annual -cash flow to debt
Earning before depreciation and extraordinary items(ib+dp) divided by average total liabilities(lt)

ebitda
Alternate: op_to_debt
We import EBIT indicator from both income and fina_indicator, so its variable name change to _x
'''
fn = 'Factor12_cashdebt'
f12 = Quarter_data['ebitda_x']/Quarter_data['total_liab']
Quarter_data[fn] = f12
save_f(f12, fn)
#%%
'''
13.cashpr - annual
Fiscal year-end market capitalization plus long-term debt(dltt) minus total assets(at) divided by cash and equivalents(che)

(总市值+长期债务-总资产)/流动现金及等价物

total_ncl 长期债务 (非流动负债)  
market capitalization - total_share
total_assets
c_cash_equ_end_period
'''
Quarter_data['Factor13_cashpr']=(Quarter_data['market_value']+Quarter_data['total_ncl']-Quarter_data['total_assets'])/Quarter_data['c_cash_equ_end_period']

# %%
'''
14.cfp - annual   cash flow to price ratio 
operating cash flows divided by fiscal-year-end market capitalization
operating cash flow?> n_cashflow_act经营活动产生的现金流量
price- market capitalization 
'''
Quarter_data['Factor14_cfp'] = Quarter_data['n_cashflow_act'] / Quarter_data['market_value']
#%%
'''
15.cfp_ia Industry adjusted CFP
'''
Quarter_data['Factor15_cfp_ia'] = Quarter_data['Factor14_cfp'] - Quarter_data['Industry_opcash']/Quarter_data['Industry_opcash']
#%%
'''
16. chatoia: annual  industry-adjusted change in asset turnover
 2-digit SIC-fiscal-year mean-adjusted change in sales(sale) divided by total assets(at)
行业 SW_L1
total_assets
'''
Quarter_data['Factor16_chatoia'] = Quarter_data['assets_turn']-Quarter_data['Industry_asset_turn']

#%%
'''
18.chinv: Change in inventory - Annual
Change in inventory(inv) scaled by average total asset(at) 
change in total inventory , deflated by average total assets.
'''
Quarter_data['Factor18_chinv'] = Quarter_data.groupby(Quarter_data['ts_code'])['inventories'].diff()/Quarter_data['total_assets']
#%%
'''
23.cinvest:Corporate investment -Quarterly
Change over one quarter in net PP&E(ppentq) divided by sales(saleq) - average of this variable for prior 3 quarters;
if saleq=0, then scale by 0.01

PP&E - fix_assets  BS
sales - c_fr_sale_sg  IS
'''
def rollmean(df, valus_col, winsize):
    _df = df.pivot_table(values=valus_col, index='end_date', columns='ts_code').rolling(winsize).mean().melt(ignore_index=False)
    return _df

Quarter_data['c_fr_sale_sg_1'] = Quarter_data['c_fr_sale_sg'].fillna(value=0.01)

Quarter_data['F23_1'] = Quarter_data['fix_assets']/Quarter_data['c_fr_sale_sg_1']
Quarter_data['F23_2'] = Quarter_data['fix_assets']/Quarter_data['c_fr_sale_sg_1']
Quarter_data['F23_3'] = (Quarter_data['F23_2']).groupby(Quarter_data['ts_code']).rolling(3).mean().values
Quarter_data['Factor23_cinvest'] = Quarter_data['F23_1'] - Quarter_data['F23_3']
#%%
'''
25.currat: current ratio- Annual
current asset/current liab

BS,fina_indicator
'''
Quarter_data['Factor25_currat'] = Quarter_data['current_ratio']
#%%
'''
26.depr: -Annual
Deprecation divided by PP&E
BS CF
没有固定资产折旧，这里我用了固定资产+油气资产+生物性生物资产
'''
Quarter_data['Factor26_depr'] = Quarter_data['depr_fa_coga_dpba'] / Quarter_data['fix_assets']
#%%
'''
27.divi :Dividend initiation -Annual
An indicator variable equal to 1 if company pays dividends but did not in prior year
'''


f27_dir = Path('data', 'divi')
f27_pt = Path('data', 'factor27_divi.csv')

if not os.path.exists(f27_pt):
    '''下载divi数据'''
    from utils import load_divi
    load_divi(f27_dir)

    dfs = pd.DataFrame()
    for f in tqdm(os.listdir(f27_dir)):
        divi = pd.read_csv(f27_dir/f)
        dfs = pd.concat([dfs, divi], axis=0)

    dfs = cleandf(dfs)
    dfs.to_csv(f27_pt, index=False)

# %%
divi = pd.read_csv(f27_pt)
divi['end_date'] = divi['end_date'].astype(int).astype(str)
divi = divi[divi['end_date'].isin(date_list)]

divi['pay_01'] = divi['pay_date']*0+1
divi['pay_01'].fillna(value=0,inplace=True)

divi = divi.sort_values(by=['ts_code','end_date'],ascending=[True,True])
divi['raw_divi'] = divi.groupby(['ts_code'])['pay_01'].diff().fillna(0)

#%%
divi_tm = divi[['ts_code', 'end_date', 'raw_divi']]
divi_tm['Factor27_divi'] = divi_tm['raw_divi'].replace({-1:0})
divi_tm['end_date'] = divi['end_date'].apply(lambda x: datetime.strptime(x, "%Y%m%d"))
Quarter_data = Quarter_data.merge(divi_tm,  on=["ts_code", "end_date"])

# %%
'''
28. divo -annual
An indicator variable equal to 1 if company did not pays dividends but did in prior year
Use divi data
'''
divi_tm = divi[['ts_code', 'end_date', 'raw_divi']]
divi_tm['Factor28_divo'] = divi_tm['raw_divi'].replace({1:0})
divi_tm['Factor28_divo'] = divi_tm['raw_divi'].replace({-1:1})
divi_tm['end_date'] = divi_tm['end_date'].apply(lambda x: datetime.strptime(x, "%Y%m%d"))
Quarter_data = Quarter_data.merge(divi_tm,  on=["ts_code", "end_date"])

# %%
'''
30. dy:dividend to price -Annual
Total dividends(dvt) divided by market capitalization at fiscal year-end

c_pay_dist_dpcp_int_exp:分配股利、利润或偿付利息支付的现金 在CF里
Book/market value  - total share总股本
'''
Quarter_data['Factor30_dy'] = Quarter_data['c_pay_dist_dpcp_int_exp'] / Quarter_data['market_value']

# %%
'''
32. egr: Growth in common shareholder equity -Annual
Annual percent change in book value of equity(ceq)

total_hldr_eqy_inc_min_int 股东权益合计(含少数股东权益)
'''
Quarter_data['Factor32_egr'] = Quarter_data['total_hldr_eqy_inc_min_int'].groupby(Quarter_data['ts_code']).diff()
#%%

'''
33. ep: Earnings to price Annual
Annual income before extraordinary items(ib) divided by end of fiscal year market cap
Annual income before extraordinary items(ib)-- ebit
market cap用什么口径 market_value:total_share
'''
Quarter_data['Factor33_ep'] = Quarter_data['ebit_x']/Quarter_data['market_value']

#%%

'''
34. gma: gross profitability Annual
Revenues(revt) - cost of goods sold(cogs) divided by lagged total assets(at)
COGS主营业务成本
:income total_cogs,revenue
:BS total_assets
'''
Quarter_data['Factor34_gma'] = (Quarter_data['revenue']-Quarter_data['total_cogs'])/(Quarter_data['total_assets'].groupby(Quarter_data['ts_code']).shift(1))

#%%

'''
35. grCAPX: growth in capital expenditure annual
Percent change in capital expenditures from year t-2 to year t
#Wind -投资活动支出的现金流
资本性支出可以从资产负债表的期末和期初变化情况计算出来。资本性支出 = 购置各种长期资产的支出 - 无息长期负债的差额其中：长期资产包括长期投资、固定资产、无形资产、其他长期资产，再其中，固定资产支出 =固定资产净值变动 + 折旧；其他长期资产支出= 其他长期资产增加 + 摊销
链接：https://www.zhihu.com/question/28300709/answer/241691621
资本性支出近似等于：“投资活动现金流量”上的“购置固定资产、无形资产、生物资产所支付的现金”减去“处置固定资产、无形资产、生物资产所收到的现金”即可。
grCAPX:
-CF
n_recp_disp_fiolta 处置固定资产、无形资产和其他长期资产收回的现金净额
c_pay_acq_const_fiolta 购建固定资产、无形资产和其他长期资产支付的现金
'''
Quarter_data['CAPX'] = Quarter_data['n_recp_disp_fiolta'] - Quarter_data['c_pay_acq_const_fiolta']
Quarter_data['Factor35_grCAPX'] = Quarter_data['CAPX'] - Quarter_data.groupby(Quarter_data['ts_code'])['CAPX'].shift(2)

#%%
'''
36. grltnoa:  annual
Glowth in long term net operating assets
净经营长期资产 是BS几个财务口径的加总 ___我们用非流动资产代替
'''
Quarter_data['Factor36_gltrnoa'] = Quarter_data.groupby(Quarter_data['ts_code'])['total_nca'].pct_change()
#%%

'''
37. herf: industrial sales concentration -Annual
2-digit SIC-fiscal-year sales concentration(sum of squared percent of sales in industry for each company)
** See the original paper
这个是对应行业/时间的集中度，需要收集指数成分股来做
SW_L1:收集申万L1
'''
from utils import load_SW
# load_SW()

SWL1_classify = pd.read_excel(Path('data', 'SWL1申万一级行业组成.xlsx'))
map_dict3 = SWL1_classify[['con_code','industry']].set_index('con_code').to_dict()
Quarter_data['industry'] = Quarter_data['ts_code'].map(map_dict3['industry'])

map_dict4 = SWL1_classify[['con_code','industry_sin']].set_index('con_code').to_dict()
Quarter_data['industry_sin'] = Quarter_data['ts_code'].map(map_dict4['industry_sin'])

Quarter_data['Factor37_herf'] = 0

industry_list = Quarter_data.industry.unique()
from utils import date_list
for ii in tqdm(range(len(industry_list))):
    for tt in range(len(date_list)):
        INDEX = Quarter_data[(Quarter_data['end_date']==date_list[tt])&(Quarter_data['industry']==industry_list[ii])].index
        herf = ((Quarter_data.loc[INDEX,'revenue']/(Quarter_data.loc[INDEX,'revenue'].sum()))**2).sum()
        Quarter_data.loc[INDEX,'Factor37_herf'] = herf

#%%

'''
38. hire: Employee growth rate - Annual
Percent change in number of employees(emp),we use payment to replace it
--好像财报数据也不披露员工总人数，我们用应付职工薪酬代替吧
:stock_company  payroll_payable
'''
Quarter_data['Factor38_hire'] = Quarter_data['payroll_payable'].groupby(Quarter_data['ts_code']).diff()
#df1 = pro.stock_company(exchange=['SZSE','SSE'], fields='ts_code,chairman,manager,secretary,reg_capital,setup_date,province')

#%%
'''
41. IPO  annual
An indicator variable equal to 1 if first year available on CRSP monthly stock file
我们采用季度数据
如果公司是本季度新上市(在数据库中)则IPO=1
'''
Quarter_data['Factor41_IPO'] = Quarter_data['is_beginning']

#%%

'''
42. invest:Capital expenditures and inventory -Annual
Annual change in gross property, plant, and equipment(ppegt) + annual change in inventories(invt) all scaled by lagged total assets
ppegt 净固定资产 ,我这里先用固定资产了？是不是应该再改一下？看看口径怎么调整
BS - asset
BS - inventories(invt)
'''
Quarter_data['Factor42_invest'] = (Quarter_data['inventories'].groupby(Quarter_data['ts_code']).diff(1) + Quarter_data['fix_assets'].groupby(Quarter_data['ts_code']).diff(1)) / Quarter_data['total_assets'].groupby(Quarter_data['ts_code']).diff(1)

#%%

'''
43. lev: Leverage -Annual
Total liabilities(lt) divided by fiscal year-end market capitalization
total_liab - BS
MC - market_value
'''
Quarter_data['Factor43_lev'] = Quarter_data['total_liab']/ Quarter_data['market_value']

#%%

'''
44. lgr: Growth in long-term debt - Annual
Annual percent change in total liabilities
total_liab - BS
'''
Quarter_data['Factor44_lgr'] = Quarter_data['total_ncl'].groupby(Quarter_data['ts_code']).diff()/(Quarter_data['total_ncl'].groupby(Quarter_data['ts_code']).shift(1))

#%%
'''
50. ms: Financial Statement Score
Sum of 8 indicator variables for fundamental performance - developed by some characteristics
sample: BM<=20% pctl?, we ignore this condtion and let the factor works among all samples.

Indusrty Table:
# names = ["营业成本", "经营活动现金流", "净利润", "营业收入",
#           "投资活动现金流", "ROA", "资产总计","应付职工薪酬","所有者权益合计",
#           "投资活动现金流出小计","实收资本(或股本)","总资产周转率",
#           "现金及现金等价物增加净额","营业利润","处置固定资产、无形资产和其他长期资产收回的现金净额",
#           "收盘价","总市值"]
# names = ['Industry_cost', 'Industry_opcash', 'Industry_net_income', 'Industry_revenue', 
#        'Industry_inv_cash','Industry_roa', 'Industry_total_assets', 'Industry_payroll_payable', 'Industry_total_eqy',
#       'Industry_inv_cash_out','Industry_total_share','Industry_asset_turn', 
#       'Industry_incr_cash_cash_equ', 'Industry_oper_profit', 'Industry_n_recp_disp_fiolta',
#       'Industry_close', 'Industry_market_value']

G1: 1 if ROA(NI/total_asset) > median of industry
G2: 1 if Cash flow ROA(cf/total_asset) > median of industry
G3: 1 if Operating Cash flow > NetIncome
G4: 1 if Firms earning variability < median of industry
        Barth etal(1999)
         EVAR = variance of past 6 yrs percentage change in earnings (NI_t - NI_t-1) / abs(NI_t-1)
G5: 1 if a firm's sale growth variability < median of industry
##G6: 1 if R&D > median of industry # We do not have industry R&D, skip...
G7: 1 if capital expenditure > median of industry
G8: 1 if advertising intensity > median of industry
'''
G1 = Quarter_data['roa'] > Quarter_data['Industry_roa']
G2 = (Quarter_data['n_cashflow_act'] / Quarter_data['total_assets']) > (
            Quarter_data['Industry_opcash'] / Quarter_data['Industry_total_assets'])
G3 = Quarter_data['n_cashflow_act'] > Quarter_data['n_income']


def EVAR(Series):
    return (Series.diff() / Series.shift(1).abs()).rolling(6).std()


G4 = EVAR(Quarter_data['n_income'].groupby(Quarter_data['ts_code'])) < EVAR(
    Quarter_data['Industry_net_income'].groupby(Quarter_data['ts_code']))
G5 = EVAR(Quarter_data['revenue'].groupby(Quarter_data['ts_code'])) < EVAR(
    Quarter_data['Industry_oper_profit'].groupby(Quarter_data['ts_code']))

# G6 deleted
Quarter_data['CAPX'] = Quarter_data['n_recp_disp_fiolta'] - Quarter_data['c_pay_acq_const_fiolta']
G7 = Quarter_data['CAPX'] > Quarter_data['Industry_inv_cash_out']
G8 = Quarter_data['oper_cost'] > Quarter_data['Industry_cost']
# 用营业费用代替下属二级广告费用
Quarter_data['Factor50_ms'] = (
            G1.astype(int) + G2.astype(int) + G3.astype(int) + G4.astype(int) + G5.astype(int) + G7.astype(
        int) + G8.astype(int))
# As we do not have R&D indice, Factor50 will range within 0-7
#%%
'''
52. mve_ia: 2-digit SIC industry-adjusted fiscal year-end market capitalization - Annual
'''
Quarter_data['Factor52_mve_ia'] = Quarter_data['market_value']-Quarter_data['Industry_market_value']
#%%
'''
53. nincr: Number of consecutive quarters (up to eight quarters) with an increase in earnings (ibq) over same quarter in the prior year 
Numbers of earning increase (<=8), I ignore the<=8 condition.
earnings 采用净利润口径 n_income
'''
Quarter_data['Factor53_nincr'] = 0
Quarter_data = Quarter_data.sort_values(['ts_code', 'end_date'])
nincr = (Quarter_data.groupby(Quarter_data['ts_code'])['n_income'].diff(4)>0)

value = 0
for ii in tqdm(range(len(nincr))):
    if nincr[ii] == False:
        value = 0
    elif nincr[ii] == True:
        value = value + 1
        value = min(8, value)
        Quarter_data.loc[ii,'Factor53_nincr'] = value

#%%
'''
54. operprof: operating profitability Annual
Revenue minus cost of Goods sold -SG&A expense- interest expense divided by lagged common shareholders' equity
SG&A =total_cogs + sell_exp + admin_exp

revenue
total_cogs
sell_exp
admin_exp
fin_exp
int_exp
total_hldr_eqy_inc_min_int  股东权益合计（含少数股东权益）
'''
Quarter_data['Factor54_operprof'] = (Quarter_data['revenue'] - Quarter_data['total_cogs'] - Quarter_data['sell_exp'] - Quarter_data['admin_exp'] - Quarter_data['fin_exp'] - Quarter_data['int_exp'])/ Quarter_data['total_hldr_eqy_inc_min_int'].groupby(Quarter_data['ts_code']).shift(1)

#%%
'''
55. orgcap: organizational capital -Annual
Capitalized SG&A expenses 
符合资本化条件的支出:无形资产下属二级科目，用无形资产代替
'''
Quarter_data['Factor55_orgcap'] = Quarter_data['intan_assets']

#%%
'''
56. pchcapx_ia: 2-digit SIC industry - fiscal-year mean-adjusted percent change in capital expenditures (capx)
'''
Quarter_data['CAPX'] = Quarter_data['n_recp_disp_fiolta'] - Quarter_data['c_pay_acq_const_fiolta']
Quarter_data['Factor56_pchcapx_ia'] = Quarter_data['CAPX'].groupby(Quarter_data['ts_code']).pct_change() - Quarter_data['Industry_inv_cash_out'].groupby(Quarter_data['ts_code']).pct_change()
#%%
'''
57. pchcurrat: Percent change in current ratio. 
current_ratio - fina_indicator
'''
Quarter_data['Factor57_pchcurrat'] = Quarter_data['current_ratio'].groupby(Quarter_data['ts_code']).diff() / Quarter_data['current_ratio'].groupby(Quarter_data['ts_code']).shift(1)

#%%
'''
58. pchdepr: Percent change in depreciation
CF - depr_fa_coga_dpba
'''
Quarter_data['Factor58_pchdepr'] = Quarter_data['depr_fa_coga_dpba'].groupby(Quarter_data['ts_code']).pct_change()

#%%
'''
59. pchgm_pchsale: Percent change in gross margin (sale-cogs) minus percent change in sales (sale)
Gross margin is a company's net sales revenue minus its cost of goods sold (COGS)
营业收入revenue  （sales revenue）营业总成本total_cogs
'''
Quarter_data['gross_margin'] = Quarter_data['revenue']-Quarter_data['total_cogs']
Quarter_data['Factor59_pchgm_pchsale'] = (Quarter_data['gross_margin']) - Quarter_data['revenue'].groupby(Quarter_data['ts_code']).pct_change()

#%%
'''
60. pchquick: Percent change in quick 
'''
Quarter_data['Factor60_pchquick'] = Quarter_data['quick_ratio'].groupby(Quarter_data['ts_code']).pct_change()

#%%
'''
61. pchsale_pchinvt: Annual percent change in sales (sale) minus annual percent change in inventory (invt).  
销售收入 -按revenue计算
inventories
'''
Quarter_data['Factor61_pchsale_pchinvt'] = Quarter_data.groupby(Quarter_data['ts_code'])['revenue'].pct_change() - Quarter_data.groupby(Quarter_data['ts_code'])['inventories'].pct_change()

#%%
'''
62. pchsale_pchrect: Annual percent change in sales (sale) minus annual percent change in receivables (rect) 
revenue
accounts_receiv
'''
Quarter_data['Factor62_pchsale_pchrect'] = Quarter_data.groupby(Quarter_data['ts_code'])['revenue'].pct_change() - Quarter_data.groupby(Quarter_data['ts_code'])['accounts_receiv'].pct_change()

#%%

'''
63. pchsale_pchxsga: Annual percent change in sales (sale) minus annual percent change in SG&A (xsga) 
'''
Quarter_data['Factor63_pchsale_pchxsga'] = Quarter_data['revenue'].groupby(Quarter_data['ts_code']).pct_change() - (Quarter_data['sell_exp']+Quarter_data['admin_exp']+Quarter_data['fin_exp']).groupby(Quarter_data['ts_code']).pct_change()

#%%

'''
80. saleinv 
Annual sales divided by total inventory 
'''
Quarter_data['Factor80_saleinv'] = Quarter_data['revenue'] / Quarter_data['inventories']
#%%

'''
64. pchsaleinv: Percent change in saleinv (Factor 80)
'''
Quarter_data['Factor64_pchsaleinv'] = Quarter_data.groupby(Quarter_data['ts_code'])['Factor80_saleinv'].pct_change()

#%%

'''
65. pctacc: Same as acc except that the numerator is divided by the absolute value of ib; if ib= 0 then ib set to 0.01 for denominator 
'''
Quarter_data['Factor65_pctacc'] = (Quarter_data['revenue']-Quarter_data['n_cashflow_act'])/Quarter_data['ebit_x']

# %%

'''
67. ps: Financial Statement Score
Sum of 9 indicator 0-1 variables to form fundamental health score 

F_Score = F_ROA + F_△ROA + F_CFO + F_ACCRUAL + F_△MARGIN + F_△TURN + F_△LEVER + F_△LIQUID + EQ_OFFER

F_ROA        :  1 if ROA = NI/Assets >0
F_△ROA       :  1 if ROA_t - ROA_t-1 >0
F_CFO        :  1 if Cash flow of operation营运现金流/total_asset >0  ocf:n_cashflow_act
F_ACCRUAL    :  1 if CFO > ROA
F_△MARGIN    :  1 if △MARGIN >0.  MARGIN = growth margin ratio = gross margin/total sales
                                 △MARGIN = MARGIN_t - MARGIN_t-1                               
F_△TURN      :  1 if △TURN >0, TURN = turnover ratio = total sales/ total assets
                                 △TURN = TURN_t - TURN_t-1
F_△LEVER     :  1 if △LEVER >0, LEVERAGE = total long term debt/ average total assets
                                 △LEVER = LEVER_t - LEVER_t-1
F_△LIQUID    :  1 if △LIQUID >0, LIQUID = current ratio
                                 △LIQUID = LIQUID_t - LIQUID_t-1
EQ_OFFER     :  1 if no nwe equity offered by companies.无新股发行

caliberation:
in this article the asset are using beginning of the year. We just use this year to maintain period simiarity. 
'''
F1 = Quarter_data['roa'] > 0
F2 = Quarter_data['roa'].groupby(Quarter_data['ts_code']).diff(1) > 0
F3 = Quarter_data['n_cashflow_act'] / Quarter_data['total_assets'] > 0
F4 = Quarter_data['n_cashflow_act'] / Quarter_data['total_assets'] > Quarter_data['roa']
F5 = (Quarter_data['gross_margin'] / Quarter_data['revenue']).groupby(Quarter_data['ts_code']).diff(1) > 0
F6 = Quarter_data['assets_turn'].groupby(Quarter_data['ts_code']).diff(1) > 0
F7 = (Quarter_data['total_ncl'] / Quarter_data['total_assets']).groupby(Quarter_data['ts_code']).diff(1) > 0
F8 = Quarter_data['current_ratio'].groupby(Quarter_data['ts_code']).diff(1) > 0

industry_code = pd.read_excel(Path('data', 'Industry.xlsx'), sheet_name='行业总览和增发')
industry_code = industry_code.rename({'证券代码':"ts_code"}, axis=1).set_index('ts_code')

Quarter_data = Quarter_data.set_index(['ts_code', 'end_date'])
Quarter_data['EQ_offer'] = 1
from utils import start_year, end_year
for ii in tqdm(range(start_year, end_year+1)):
    tmp = industry_code[[f"增发上市日\n[年度] {ii}"]].dropna()
    tmp["end_date"] = tmp.loc[:, f"增发上市日\n[年度] {ii}"] + pd.offsets.QuarterEnd(0)
    tmp = tmp.set_index("end_date", append=True)

    id = Quarter_data.merge(tmp, on=['ts_code', "end_date"], how="inner").index
    Quarter_data.loc[id, "EQ_offer"] = 0

Quarter_data = Quarter_data.reset_index()
Quarter_data['Factor67_ps'] = F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8 + Quarter_data['EQ_offer']

#%%

'''
68. quick: (current assets - inventory) / current liabilities
fina_indicator:quick_ratio
'''
Quarter_data['Factor68_quick'] = Quarter_data['quick_ratio']

#%%

'''
69. rd: An indicator variable equal to 1 if R&D expense as a percentage of total assets has an increase greater than 5%. 
'''
Quarter_data['Factor69_rd'] = ((Quarter_data['r_and_d'] / Quarter_data['total_assets']).groupby(Quarter_data['ts_code']).pct_change()>0.05)

#%%

'''
70. rd_mve: R&D expense divided by end-of-fiscal-year market capitalization 
'''
Quarter_data['Factor70_rd_mve'] = Quarter_data['r_and_d'] / Quarter_data['market_value']

#%%

'''
71. rd_sale: R&D expense divided by sales (xrd/sale) 
r_and_d
revenue
'''
Quarter_data['Factor71_rd_sale'] = Quarter_data['r_and_d'] / Quarter_data['revenue']

#%%

'''
72. realestate : Real estate holdings
Buildings and capitalized leases divided by gross PP&E 
invest_real_estate
fix_assets
'''
Quarter_data['Factor72_realestate'] = Quarter_data['invest_real_estate'] / Quarter_data['fix_assets']

#%%

'''
74. roaq
Income before extraordinary items (ibq) divided by one quarter lagged total assets (atq) 
'''
Quarter_data['Factor74_roaq'] = Quarter_data['ebit_x'] / Quarter_data['total_assets'].groupby(Quarter_data['ts_code']).shift(1)

#%%

'''
75. roavol:Earning volatility
Standard deviation for 16 quarters of income before extraordinary items (ibq) divided by average total assets (atq) 
average total assets? I use total asset
'''
Quarter_data['Factor75_roavol'] = (Quarter_data['ebit_x']/ Quarter_data['total_assets']).groupby(Quarter_data['ts_code']).rolling(16).std().values

#%%

'''
76. roeq
Earnings before extraordinary items divided by lagged common shareholders’ equity 
'''
Quarter_data['Factor76_roeq'] = Quarter_data['ebit_x'] / Quarter_data['total_hldr_eqy_inc_min_int'].groupby(Quarter_data['ts_code']).shift(1)

#%%

'''
77. roic: return on invested capital
Annual earnings before interest and taxes minus nonoperating income (nopi) 
divided by non-cash enterprise value (ceq+lt-che) 
lt Total liabilities
ceq book value to equity
che cash and equivalent
'''
Quarter_data['Factor77_roic'] = (Quarter_data['ebit_x'] - Quarter_data['non_oper_income'])/(Quarter_data['total_liab']+Quarter_data['total_hldr_eqy_inc_min_int']-Quarter_data['c_cash_equ_end_period'])

#%%

'''
78. rsup: revenue surprise
Sales from quarter t minus sales from quarter t-4 (saleq) divided by fiscal-quarter-end market capitalization (cshoq * prccq, or total_share) 
'''
Quarter_data['Factor78_rsup'] = (Quarter_data['revenue']-Quarter_data['revenue'].groupby(Quarter_data['ts_code']).shift(4)) / Quarter_data['total_share']

#%%

'''
79. salecash
Annual sales divided by cash and cash equivalents 
'''
Quarter_data['Factor79_saleinv'] = Quarter_data['revenue'] / Quarter_data['c_cash_equ_end_period']

#%%

'''
81. salerec
Annual sales divided by accounts receivable 
'''
Quarter_data['Factor81_salerec'] = Quarter_data['revenue'] / Quarter_data['accounts_receiv']

#%%

'''
82. secured :secured debt
Total liability scaled secured debt担保债务负债总额
Replace with mortgage_financing我们用抵押融资来代替
'''
Quarter_data['Factor82_secured_debt'] = Quarter_data['mortgage_financing']

#%%

'''
83. securedind 
An indicator equal to 1 if company has secured debt obligations(CDO)
Replace with if company had mortgage_financing
'''
Quarter_data['Factor83_securedind'] = Quarter_data['mortgage_financing']*0+1

#%%

'''
84. sfe 
Analysts mean annual earnings forecast for nearest upcoming fiscal year from most recent month available prior to month of portfolio formation from I/B/E/S summary files scaled by price per share at fiscal quarter end 
-->We use the analyse institutions number for giving a rank. As the institution investors always give 'maintain' status for simplicity, so we didn't distinguish high or low.
Continue to run Factor20_nanalyst
'''
Quarter_data['Factor84_sfe'] = Quarter_data['rank_num']
#%%

'''
85. sgr
Annual percent change in sales (sale)
'''
Quarter_data['Factor85_sgr'] = Quarter_data.groupby('ts_code')['revenue'].pct_change()

#%%

'''
86. sin
An indicator variable equal to 1 if a company’s primary industry classification 
is in smoke or tobacco, beer or alcohol, or gaming 
Need to run Factor37 before
'''
#是否涉及不良产业
#调出每个公司的申万一级，主观打分:是否为国家和人民做出贡献，是增量还是内卷
Quarter_data['Factor86_sin'] = Quarter_data['industry_sin']

#%%

'''
87. sp
Annual revenue (sale) divided by fiscal year-end market capitalization 
'''
Quarter_data['Factor87_sp'] = Quarter_data['revenue'] / Quarter_data['market_value']

#%%

'''
90. stdacc 
Standard deviation for 16 quarters of accruals (acc measured with quarterly Compustat) scaled by sales;    
acc-factor1
'''
Quarter_data['Factor90_stdacc'] = Quarter_data.groupby(Quarter_data['ts_code'],as_index=False)['Factor01_acc'].rolling(16).std().values

#%%

'''
91. stdcf 
Standard deviation for 16 quarters of cash flows divided by sales (saleq); if saleq= 0, then scale by 0.01. 
Cash flows defined as ibq minus quarterly accruals 
#16 quarter is too much for our sample! Let's use 8
cf: ebit - acc
'''
Quarter_data['Factor91_stdcf'] = ((Quarter_data['ebit_x']-Quarter_data['Factor01_acc'])/Quarter_data['revenue']).groupby(Quarter_data['ts_code']).rolling(7).std().values

#%%

'''
92. sue
Unexpected quarterly earnings divided by fiscal-quarter-end market cap. 
Unexpected earnings is I/B/E/S actual earnings minus median forecasted earnings if available, 
else it is the seasonally differenced quarterly earnings before extraordinary items from Compustat quarterly file 
扣除非经常性项目后的季度收益差异
'''
Quarter_data['Factor92_sue'] = Quarter_data.groupby(Quarter_data['ts_code'])['ebit_x'].diff(4)

#%%

'''
93. tang: debt capacity/firm tangibility
Cash holdings + 0.715 × receivables +0.547 × inventory + 0.535 × PPE/total assets 
'''
Quarter_data['Factor93_tang'] = Quarter_data['cash_reser_cb'] + 0.715*Quarter_data['accounts_receiv'] + 0.547*Quarter_data['inventories'] + 0.535*Quarter_data['fix_assets']/Quarter_data['total_assets']

#%%

'''
94. tb :Tax income to book income -Annual
Tax income, calculated from current tax expense divided by maximum federal tax rate, 
divided by income before extraordinary items 
Tax income = tax_expense/(maximum_tax_rate*ebit)

(Tax income > 1: Too much tax, bad
Tax income < 1: Avoid some tax)

We replace it with tax_return(by government) sign. I cannot find maximum_tax_rate in wind for different company
'''
Quarter_data['Factor94_tb'] = Quarter_data['taxes_payable']/(0.25*Quarter_data['ebit_x'])
# %%
Out_df = utils.extrct_fctr(Quarter_data)
Out_df.to_csv(Path('_saved_factors', 'QrtFactor.csv'))
#%%