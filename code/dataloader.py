# %%
import datetime
import pandas as pd
import numpy as np
import os
from pathlib import Path

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

loadXiu = 0
loadYu = 0
#%%
p = Path('.') /'factors' / 'china factors' / '_saved_factors' / 'full_factors.h5'
raw_data = pd.read_hdf(p, key='data')
# data = raw_data.set_index(["ts_code", "end_date"])
data = raw_data
data.index = data.index.set_names(["ticker",'date'])
data = data.rename({"monthly_return":"Y"}, axis=1)

# change datetime format
data = data.reset_index("date")
data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d"))
data = data.set_index('date', append=True).sort_index()
data = data.fillna(0)
data.to_hdf(Path('.') / 'data' / 'data.h5', key='data')

# %% Xiu's data
if loadXiu:
    p = Path('.') / 'data' / 'xiu_datashare' / 'datashare.csv'
    raw_data = pd.read_csv(p)
    data = raw_data.set_index(["permno", "DATE"]).sort_index()
    data.index = data.index.set_names(["ticker",'date'])
    data.dropna()
    # change datetime format
    data = data.reset_index("date")
    data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d"))
    data = data.set_index('date', append=True).sort_index()

    data.to_pickle(Path('.') / 'data' / 'xiu_datashare' / 'datashare.pkl')

# %% Yu's data
if loadYu:
    raw = pd.read_hdf(Path("./data/yu_data/dataset.h5"), key='data')
    yudata = raw.copy()
    # y 是 Mretwd

    data = yudata.set_index(["Stkcd", "Trdmnt"]).drop('Mretnd', axis=1)
    data.index = data.index.set_names(["ticker",'date'])
    # change datetime format
    data = data.reset_index("date")
    data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m"))
    data = data.set_index('date', append=True).sort_index()

    # shift return by 1
    data = data.rename({"Mretwd":"pct_chg"}, axis=1)
    data.loc[:, 'pct_chg'] = data.groupby('ticker')['pct_chg'].shift(-1)
    data = data.dropna()
    data.to_pickle(Path('.') / 'data' / "yu_data"/ "yu_factor.pkl")