#%%
# Tushare + wind API 能用tushare就tushare 实在不行上windAPI 不用csmar
from functools import reduce
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tushare as ts
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
ts.set_token(token)
pro = ts.pro_api()

pd.set_option('display.max_rows', 100)
_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")
#%%

'''
Use Tushare API to get Quarterly Financial Statement Data
'''

from utils import date_list
date_list = date_list

print("-----------BalanceSheet-----------")
BalanceSheet = pro.balancesheet_vip(period=date_list[0],fields='')
for tt in tqdm(range(1,len(date_list))):
    try:
        df1 = pro.balancesheet_vip(period=date_list[tt],fields='')
        BalanceSheet = pd.concat([BalanceSheet,df1],axis=0)
    except:
        print(f"fail to load {date_list[tt]}")
        continue
BalanceSheet.to_csv('data/BS.csv')

print("-----------CashFlow-----------")
CashFlow = pro.cashflow_vip(period=date_list[0],fields='')
for tt in tqdm(range(1,len(date_list))):
    try:
        df1 = pro.cashflow_vip(period=date_list[tt],fields='')
        CashFlow = pd.concat([CashFlow,df1],axis=0)
    except:
        print(f"fail to load {date_list[tt]}")
        continue
CashFlow.to_csv('data/CF.csv')

print("-----------Income-----------")
Income = pro.income_vip(period=date_list[0],fields='')
for tt in tqdm(range(1,len(date_list))):
    try:
        df1 = pro.income_vip(period=date_list[tt],fields='')
        Income = pd.concat([Income,df1],axis=0)
    except:
        print(f"fail to load {date_list[tt]}")
        continue
Income.to_csv('data/IS.csv')

print("-----------FinaIndicator-----------")
FinaIndicator =  pro.fina_indicator_vip(period=date_list[0],fields='') # 描述：获取上市公司财务指标数据，
for tt in tqdm(range(1,len(date_list))):
    try:
        df1 = pro.fina_indicator_vip(period=date_list[tt],fields='')
        FinaIndicator = pd.concat([FinaIndicator,df1],axis=0)
    except:
        print(f"fail to load {date_list[tt]}")
        continue
FinaIndicator.to_csv('data/FinaIndicator.csv')

# %%
dfs = [BalanceSheet, CashFlow, Income, FinaIndicator]
df_final = reduce(lambda left,right:pd.merge(left,right,how='outer',on=['ts_code','ann_date','end_date']),dfs)

QuarterFactorRaw = df_final.drop_duplicates()

QuarterFactorRaw.to_csv(Path('data', 'buffer', 'QuarterFactorRaw.csv'), index=False)

# %%