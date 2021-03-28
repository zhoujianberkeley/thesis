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

# %%
# load data
_load = True
if _load:
    Monthly_Quotation = utils.cleandf(pd.read_csv(Path('data', 'buffer', 'MonFactorPrcd.csv')))
    Monthly_Quotation = utils.todate(Monthly_Quotation, 'end_date', format='%Y-%m-%d')
    Monthly_Quotation.index = range(len(Monthly_Quotation))
    Monthly_Quotation.to_pickle(Path('data', 'buffer', 'MonFactorPrcd.pkl'))
else:
    Monthly_Quotation = pd.read_pickle(Path('data', 'buffer', 'MonFactorPrcd.pkl'))

# %%
'''
7.beta - monthly
estimate market beta from weekly returns and equal weighted market returns
for 3 years ending month t-1 with at least 52 weeks of returns
--We use daily returns for more accuracy.
'''
Beta = pd.read_excel(Path('data', '月度beta.xlsx'))
Beta = Beta.rename({'证券代码':'ts_code'},axis=1).drop('证券简称',axis=1).set_index('ts_code')

Beta1 = Beta.melt(ignore_index=False)
Beta1 = Beta1.rename({'variable':'end_date','value':'Factor07_beta'},axis=1)
Monthly_Quotation = Monthly_Quotation.merge(Beta1, on=['ts_code','end_date'], how='left')

#%%
'''
8.betasq - monthly
market beta(7) squared 
'''
Monthly_Quotation['Factor08_betasq'] = Monthly_Quotation['Factor07_beta']**2
#%%
'''
19.chmom: -monthly 行情
Cumulative returns from month t-6 to t-1 minus month t-12 to t-7
Note: shift(1) will be done together later
Input: ts_code  trade_date  start_date  end_date
Output: ts_code  trade_date  ohlc  pct_chg 
'''
dfs_ = Monthly_Quotation.pivot_table(values='change', index='end_date', columns='ts_code').rolling(6).sum().diff(6)
dfs = dfs_.melt(ignore_index=False).reset_index().rename(columns={'value':'Factor19_chmom'})
Monthly_Quotation = Monthly_Quotation.merge(dfs,on=['ts_code','end_date'],how='left')
# %%
'''
29. dolvol : Dollar trading volume - Monthly 行情 (RMB trading volume?)
Natural log of trading volume times price per share from month t-2

Use trading amount instead (月成交额)
'''
#拿到每月标签
Monthly_Quotation['Factor29_dolvol'] = np.log(Monthly_Quotation['amount'].groupby(Monthly_Quotation['ts_code']).shift(2))
#%%

'''
46. mom12m: 12m momentum - Monthly
11 month cumulative returns ending one month before month end
月度行情，累计回报
'''
Monthly_Quotation['Factor46_mom12m'] = Monthly_Quotation['change'].groupby(Monthly_Quotation['ts_code']).rolling(11).sum().values

#%%

'''
47. mom1m:1 month momentom -Monthly
1-month cumulative return
'''
Monthly_Quotation['Factor47_mom1m'] = Monthly_Quotation['change'].groupby(Monthly_Quotation['ts_code']).shift(0)

#%%

'''
48. mom36m: Cumulative returns from monthst-36 to t-13
'''
Monthly_Quotation['Factor48_mom36m'] = Monthly_Quotation['change'].groupby(Monthly_Quotation['ts_code']).rolling(23).sum().shift(12).values

#%%

'''
49. mom6m: 5-month cumulative returns ending one month before month end
'''
Monthly_Quotation['Factor49_mom6m'] = Monthly_Quotation['change'].groupby(Monthly_Quotation['ts_code']).rolling(5).sum().values

#%%

'''
51. mve: Size - monthly
Natural log of market capitalization at end of month t-1 
'''
Monthly_Quotation['Factor51_mve'] = np.log(Monthly_Quotation['market_value'].groupby(Monthly_Quotation['ts_code']).shift(0))

#%%
'''
95. turn: Share turnover -Monthly
Average monthly trading volume for most recent 3 months scaled by number of shares outstanding in current month 
最近三个月的平均每月交易量vol除以本月已发行的股份数目
月行情 -wind
'''
Monthly_Quotation['Factor95_turn'] = Monthly_Quotation['vol'].groupby(Monthly_Quotation['ts_code']).rolling(3).mean().values /Monthly_Quotation['total_share']
# %%
'''
96. zerotrade - Monthly
Turnover weighted number of zero trading days for most recent 1 month
最近1个月的换手率加权零交易日数
* 看原文 
daily行情
zerotrade date Liu 2006 JFE: I define the liquidity measure of a security,LMx,
as the standardized turnover-adjusted number of zero daily trading volumes 
over the prior x months(x=1,6,12),that is,
LMx=
[Num_of_zero_daily_volumes_in_prior_x_months + (1/x-month_turnover)/Deflator]*21x/NoTD

Where:
x-month turnover, set x=1 here
x-month turnover is turnover over the prior x months, calculated as the sum of daily turnover over the prior x months, 
月换手率
daily turnover is the ratio of the number of shares traded on a day to the number of shares outstanding at the end of the day,

NoTD is the total number of trading days in the market over the prior x months,
市场前x个月交易日总数，来自于交易日个数的合并

Deflator is chosen such that 0< (1/x-month_turnover)/Deflator <1  for all stock
选择一个平减指数，以便所有股票的0 <（1/x-month_turnover）/ Deflator <1
(0-1 压缩) 选x-month_turnover的max吧

日频数据、月频数据

Zero trade days - 停牌日

LMx = [Num_of_zero_daily_volumes_in_prior_x_months + (1/x-month_turnover)/Deflator]*21x/NoTD

As x=1 , the indicator equals to:
LM1 = Num_of_zero_daily_volumes_in_prior_1_months + (1/monthly_turnover)/Deflator

解释:这个合成指标更多是描绘停盘和换手率的交互。在这个合成指标中，由于大部分停盘都是0-1，所以在turnover后面加入了调参权重，我们取每个个股最大的1/turnover_rate值
Num_of_zero_daily_volumes_in_prior_1_months与该指标呈1:1的正线性关系，monthly_turnover与该指标呈1/Deflator的负相关
更多的是突出前面一个指标的贡献
'''
spnd_dir = Path('data', 'spnd')
spnd_pt = Path('data', 'factor96_zerotrade_spnd.csv')
# get suspend data
if not os.path.exists(spnd_pt):
    '''下载停牌数据'''
    import utils
    utils.load_spnd(spnd_dir)

    dfs = pd.DataFrame()
    for f in tqdm(os.listdir(spnd_dir)):
        divi = pd.read_csv(spnd_dir/f)
        dfs = pd.concat([dfs, divi], axis=0)

    dfs = utils.cleandf(dfs)
    dfs.to_csv(spnd_pt, index=False)

# %%
df = pd.read_csv(spnd_pt)
df = df.sort_values(by=['ts_code','suspend_date'],ascending=[True,True]).drop_duplicates()
#历史遗留到2010-01-01的我们不再仔细追究，选取停牌日开始在2010年之后的
df['year'] = df.suspend_date.astype(str).str[:4].astype(int)
df = df[(df['year'] <= utils.end_year) & (df['year'] >= utils.start_year)]
df.index = range(len(df))

#转为月末
date = [datetime.strptime(str(i),'%Y%m%d') for i in df.suspend_date.values]
df.loc[:,'end_date'] = date
df['end_date'] = df['end_date'] + pd.offsets.MonthEnd(0)

df_aggregate = df.pivot_table(index=['ts_code','end_date'],values=["suspend_date"],aggfunc=len)
df_aggregate_1 = df_aggregate.reset_index()

Monthly_Quotation = Monthly_Quotation.merge(df_aggregate_1,how='left', on=['ts_code', 'end_date'])
Monthly_Quotation['suspend_date'] = Monthly_Quotation['suspend_date'].fillna(0)
Deflator = 1/Monthly_Quotation.turnover.min()
Monthly_Quotation['Factor96_zerotrade'] = Monthly_Quotation.suspend_date + 1/Monthly_Quotation.turnover /Deflator
# %%
Out_df = utils.extrct_fctr(Monthly_Quotation)
Out_df.to_csv(Path('_saved_factors', 'MonFactor.csv'))
# %%