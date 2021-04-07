#%%
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
from pathlib import Path
from datetime import datetime, date, timedelta
# from tqdm import tqdm
from tqdm.auto import tqdm
tqdm.pandas() # Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`

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
 #%%
# load data
_load = False
if _load:
    Daily_Quotation = utils.cleandf(pd.read_csv(Path('data', 'buffer', 'DayFactorPrcd.csv')))
    Daily_Quotation = Daily_Quotation.rename(columns={'代码': 'ts_code', '日期': 'trade_date', '成交量(股)': 'volume'})
    Daily_Quotation['trade_date'] = pd.to_datetime(Daily_Quotation['trade_date'])
    Daily_Quotation['end_date'] = Daily_Quotation['trade_date'] + pd.offsets.MonthEnd(0)
    Daily_Quotation.index = range(len(Daily_Quotation))
    Daily_Quotation = Daily_Quotation.sort_values(by=['ts_code', 'trade_date'])
    Daily_Quotation.to_pickle(Path('data', 'buffer', 'DayFactorPrcd.pkl'))
else:
    Daily_Quotation = pd.read_pickle(Path('data', 'buffer', 'DayFactorPrcd.pkl'))

Daily_vol = Daily_Quotation[['ts_code', 'trade_date', 'volume']] #数据原始太大了 拆分做

# %%
'''
3.aeavol -Quarterly:
Average daily trading volume(vol) for 3 days around(?before) earnings announcement -
average daily volume for 1-month ending 2 weeks before earnings announcement
divided by 1-month average daily volume
(财报发布日前三天的平均vol - 财报发布前一个月到前2周的vol) / 财报发布前一个月的平均日vol.
#出于信息时效性，非交易日则不计算，不再顺延?? - 我们采用前三个交易日
?精确的一个月应该怎么做

注意这个指标最后计入下一个季度内
为了指标的稳定性，我们对其作出改动，也方便编程 一个月假定有21个交易日，2周假定有10个交易日
Change 2 weeks-->10 trade date;  
1 month-->21 trade date

--daily vol + forecast/express/disclosure_date 业绩预告/业绩快报/财报披露日期 
data: 
stock_basic: get all stock list
daily: volume
财报披露日disclosure_date: actual_date

#有的公司发布日期在周末，这里需要做交易日处理！
'''
# todo ignore factor 3 and factor 31 for now because it's too tedious
# %%
# download finanical statement announcement data
FSdate_pt = Path('data', 'stock_financial_statement_date.xlsx')
FSdate_dir = Path('data', 'fsdate')
if not os.path.exists(FSdate_pt):
    utils.load_fsdate(FSdate_dir, utils.date_list)

    dfs = pd.DataFrame()
    for f in tqdm(os.listdir(FSdate_dir)):
        divi = pd.read_csv(FSdate_dir/f)
        dfs = pd.concat([dfs, divi], axis=0)

    dfs = utils.cleandf(dfs)
    dfs.to_csv(FSdate_pt, index=False)

FSdate = utils.cleandf(pd.read_excel(FSdate_pt))
FSdate = FSdate[~(FSdate.actual_date.isna()==True)]
FSdate['actual_date'] = FSdate['actual_date'].astype(int)
FSdate = FSdate.sort_values(by=['ts_code','end_date'])
FSdate.to_csv(Path('data', 'buffer', 'FSdate.csv'), index=False)

# %%
FSdate = pd.read_csv(Path('data', 'buffer', 'FSdate.csv'), parse_dates=['actual_date'])
FSdate['is_FS_Date']=1
# FSdate.loc[:,'actual_date_datetime64'] = [datetime.strptime(str(i), "%Y%m%d")  for i in FSdate.actual_date.values]
# trade_date是实际发布日期，为了合并merge而改名

#%%
df_fct3 = pd.merge(FSdate[['ts_code', 'end_date', 'actual_date', 'is_FS_Date']], Daily_vol, how='outer',
                   left_on=['ts_code', 'actual_date'], right_on=['ts_code', 'trade_date'])
df_fct3 = df_fct3.sort_values(by=['ts_code', 'trade_date'])
df_fct3.index = range(len(df_fct3))

#%%
def get_slice(_df, cur, start, end):
    return _df[(_df.trade_date >= (cur - timedelta(start))) & (_df.trade_date <= (cur - timedelta(end)))]

def cal_fct3(_df):
    # (财报发布日前三天的平均vol - 财报发布前一个月到前2周的vol) / 财报发布前一个月的平均日vol.
    dfs = []
    for _, row in _df.iterrows():
        if row['is_FS_Date'] == 1:
            cur = row['trade_date']
            slice1 = get_slice(_df, cur, 4, 1) # [ii - 4:ii - 1, 'volume'].mean()
            slice2 = get_slice(_df, cur, 21, 10)
            slice3 = get_slice(_df, cur, 21, 1)
            res = (slice1['volume'].mean() - slice2['volume'].mean())/slice3['volume'].mean()
            df = pd.DataFrame({"Factor3_aeavol":res}, index=pd.MultiIndex.from_tuples([(row['ts_code'], row['trade_date'])],
                                                                                 names=['ts_code', 'trade_date']))
            dfs.append(df)
    return pd.concat(dfs)

df_factor3 = df_fct3.head(10000).groupby('ts_code').progress_apply(lambda x: cal_fct3(x))
# %%
'''
40. ill: Illiquidity - Monthly
The illiquidity measure employed here, called ILLIQ; is the daily ratio of
absolute stock return to its dollar volume, averaged over some period. It can be
interpreted as the daily price response associated with one dollar of trading
volume, thus serving as a rough measure of price impact.

Average of daily(absolute return divided by dollar volume) 
月中的(日涨跌幅绝对值 除以 成交额) 的平均值
'''
#%% md
# $$ ILLIQ_{iy} = 1/D_{iy}\sum_{d=1}^{D_{iy}}|R_{iyd}|/VOID_{iyd} $$
#
# https://d1wqtxts1xzle7.cloudfront.net/34946331/amihud.pdf?1412125425=&response-content-disposition=inline%3B+filename%3DAmihud.pdf&Expires=1614543405&Signature=Cr8gwEv6LAyl2Eiow6VJzw5iWxpP3-ty6r0i3Mu5-3vmqJnpuJzBWleoixnyz5Uu3UxgwtO0DyFOaDvWWYocm-o0ZeqmJX6plW2DpH-pSA8wycvbmoSy~6LFo5XXQ4Zr1dYFOWNsaGnVqoui1NFLAQ85N8d8H~o6MydMNZFjfJuXmr8-7rzsMaxXKI~62JUZsMIyKTxJGnmKhxYVT7-1xdve2UcyO2JUKG2uyMTEbqwQv8-eTbBKR8cHYWlp2NoMDjSDFYm~TiyT6BJsAKa8S5zohJgGS5Lw-sADlVKC10NP3m3vXtsocef1XB6dcOPPZVbmR9S-pqNZBLH5zzTGdQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
#%%
Daily_Quotation['illiq'] = Daily_Quotation['涨跌(元)'].abs()/Daily_Quotation['成交金额(元)']
Daily_Quotation_F40 = Daily_Quotation.groupby(['ts_code','end_date'],as_index=False)['illiq'].mean()
Daily_Quotation_F40 = Daily_Quotation_F40.rename({'illiq':'Factor40_ill'},axis=1)
Daily_Quotation = Daily_Quotation.merge(Daily_Quotation_F40,on=['ts_code','end_date'],how='left')

# %%
'''
45. maxret:maximum daily return -Monthly
maximum daily return from returns during calendar month t-1
上月内日最高涨幅
注：因子最后统一给一个t-1期滞后。shift(1)不必现在做
'''
Daily_Quotation_F45 = Daily_Quotation.groupby(['ts_code','end_date'],as_index=False)['涨跌幅(%)'].max()
Daily_Quotation_F45 = Daily_Quotation_F45.rename(columns={'涨跌幅(%)':'Factor45_maxret'})
Daily_Quotation = Daily_Quotation.merge(Daily_Quotation_F45,on=['ts_code','end_date'],how='left')

# %%
'''
73. retvol
Standard deviation of daily returns from month t-1 
上个月日收益的标准差，月内日行情汇总指标
'''
Daily_Quotation_F73 = Daily_Quotation.groupby(['ts_code','end_date'],as_index=False)['涨跌幅(%)'].var()
Daily_Quotation_F73['涨跌幅(%)'] = Daily_Quotation_F73['涨跌幅(%)'].apply(np.sqrt)
Daily_Quotation_F73 = Daily_Quotation_F73.rename({'涨跌幅(%)':'Factor73_retvol'},axis=1)
Daily_Quotation = Daily_Quotation.merge(Daily_Quotation_F73,on=['ts_code','end_date'],how='left')

#%%
'''
88. std_dolvol : volatility of liquidity(dollar trading volume)
Monthly standard deviation of daily dollar trading volume
每个月中日交易金额的波动率 Wind - 月波动率
'''
Daily_Quotation_F88 = Daily_Quotation.groupby(['ts_code','end_date'],as_index=False)['成交金额(元)'].var()
Daily_Quotation_F88['成交金额(元)'] = Daily_Quotation_F88['成交金额(元)'].apply(np.sqrt)
Daily_Quotation_F88 = Daily_Quotation_F88.rename({'成交金额(元)':'Factor88_std_dolvol'},axis=1)
Daily_Quotation = Daily_Quotation.merge(Daily_Quotation_F88, on=['ts_code', 'end_date'], how='left')

# %%
'''
89. std_turn : volatility of liquidity(share turnover)
Monthly standard deviation of daily share turnover 
每个月中日交易量的平均换手率 - 月平均换手率
'''
Daily_Quotation_F89 = Daily_Quotation.groupby(['ts_code','end_date'],as_index=False)['换手率(%)'].var()
Daily_Quotation_F89['换手率(%)'] = Daily_Quotation_F89['换手率(%)'].apply(np.sqrt)
Daily_Quotation_F89 = Daily_Quotation_F89.rename({'换手率(%)':'Factor89_std_turn'},axis=1)
Daily_Quotation = Daily_Quotation.merge(Daily_Quotation_F89, on=['ts_code', 'end_date'], how='left')
# %%
Out_df_ = utils.extrct_fctr(Daily_Quotation)
Out_df = Out_df_.groupby(['ts_code', 'end_date'], as_index=False).mean()
utils.check(Out_df)
Out_df.to_csv(Path('_saved_factors', 'DayFactor.csv'), index=False)
# %%