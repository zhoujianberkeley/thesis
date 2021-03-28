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


FSdate = utils.cleandf(pd.read_excel(Path('data', 'stock_financial_statement_date.xlsx')))
FSdate = FSdate[~(FSdate.actual_date.isna()==True)]
FSdate['actual_date'] = FSdate['actual_date'].astype(int)
date = [datetime.strptime(str(i), "%Y%m%d")  for i in FSdate.actual_date.values]
FSdate.loc[:,'actual_date_datetime64'] = date
FSdate = FSdate.sort_values(by=['ts_code','end_date'])
FSdate.to_csv(Path('data', 'buffer', 'FSdate.csv'), index=False)


FSdate = pd.read_csv(Path('data', 'buffer', 'FSdate.csv'))
FSdate['is_FS_Date']=1
FSdate_short = FSdate[['ts_code','end_date','actual_date_datetime64','is_FS_Date']]
FSdate_short = FSdate_short.rename({'actual_date_datetime64':'trade_date'},axis=1)
# trade_date是实际发布日期，为了合并merge而改名


#%%

idx = pd.date_range(start='2010-01-04',end='2020-12-31',freq='D')
Daily_vol_slice = Daily_vol.set_index(idx)
Daily_vol_slice['newdate']=Daily_vol.index
# 由于股票数据并非面板，有增有减，所以不能直接用大的index-it，会增加不必要的维度

# PriceDate Column - Does not contain Saturday and Sunday stock entries
data['PriceDate'] =  pd.to_datetime(data['PriceDate'], format='%m/%d/%Y')
Daily_vol2 = data.sort_index(by=['PriceDate'], ascending=[True])
# Starting date is Aug 25 2004
idx = pd.date_range(start='01-01-2004',end='31-12-2020',freq='D')

Daily_vol2 = Daily_vol2.set_index(idx)
Daily_vol2['newdate']=Daily_vol2.index
newdate=Daily_vol2['newdate'].values   # Create a time series column
Daily_vol2 = pd.merge(newdate, Daily_vol2, on='trade_date', how='outer')
date = (datetime.datetime.strptime(date_end, "%Y-%m-%d")-datetime.timedelta(days=1)).strftime('%Y-%m-%d') 补齐日期序列缺陷

# datetime64 to object：不用再转化成csv做了
FSdate['actual_date_datetime64'] = FSdate['actual_date_datetime64'].apply(lambda x: x.strftime('%Y-%m-%d'))

# %%
