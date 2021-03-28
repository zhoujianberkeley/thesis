#%%
import os
import pandas as pd
pd.set_option('display.max_rows', 100)
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import qgrid

# Tushare + wind API 能用tushare就tushare 实在不行上windAPI 不用csmar
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
import tushare as ts
ts.set_token(token)
pro = ts.pro_api()

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

# %%
'''
周行情 Weekly Quotation
I have already runned factor 39 and 66 and saved
'''
Weekly_Quotation_1 = pd.read_csv(Path('data', 'Weekly_quotation2010-2015.csv'),encoding='gbk')
Weekly_Quotation_2 = pd.read_csv(Path('data', 'Weekly_quotation2016-2020.csv'),encoding='gbk')
#最后转入其他jupyter时，上面的文件夹目录记得修改

Weekly_Quotation = pd.concat([Weekly_Quotation_1,Weekly_Quotation_2],axis=0)

Weekly_Quotation.drop('Unnamed: 20',axis=1,inplace=True)

Weekly_Quotation = Weekly_Quotation.sort_values(by=['代码','日期']) #,ascending=[True,True]

# sannity check
assert Weekly_Quotation.drop_duplicates(['代码','日期']).shape == Weekly_Quotation.shape

Weekly_Quotation.to_csv(Path('data', 'buffer', 'WeekFactorPrcd.csv'), index=False, encoding='gbk')

# %%