# %%
import datetime
import pandas as pd
import numpy as np
import os
from pathlib import Path
# %%
def ffill(df, col_stk, col_time): # two columns: stock code, time
    stkList = df[col_stk].unique()
    df = df.sort_values(by = col_time) # data should be sorted ascending by time
    for stk in stkList:
        df[df[col_stk] == stk] = df[df[col_stk] == stk].fillna(method='ffill')
    return df
#%%
p = Path('.') / 'factors' / 'Final_Data_V1.csv'
raw_data = pd.read_csv(p)
data = raw_data.set_index(["ts_code", "end_date"]).drop('Unnamed: 0', axis=1)
data.index = data.index.set_names(["ticker",'date'])
# change datetime format
data = data.reset_index("date")
data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d"))
data = data.set_index('date', append=True).sort_index()

s = data['Factor47_mom1m']
ss = ((s.rank()/s.count() - 0.5)*2)

timeList = data.index.get_level_values('date').unique().sort_values()

i = 0
for t in timeList:
    tdf = data.loc[pd.IndexSlice[:, t], :]

    for Z in Zlist:
        fillArr = dataset_x[Z].loc[t, :].fillna(np.nanmedian(dataset_x[Z].loc[t, :]))
        fillArr = 2*(fillArr.rank()/len(fillArr)-0.5)
        dataset_x[Z].loc[t, :] = fillArr
    print(i)
    i = i+1
    print(t)
# %%
def rank():
    fs = []
    for f in fs:
        pass
    pass

def rank_():
    pass




# %%
# get frequency
p = Path('.') / 'factors' / 'Indicator_Table.xlsx'
fdf = pd.read_excel(p)

def chg_frq(freq):
    if freq in ['Daily', 'Weekly']:
        return 'Monthly'
    elif freq in ['Annual']:
        return 'Quarterly'
    else:
        return freq

for i, row in fdf.iterrows():
    name = row['indicator'].split(" ")
    fdf.loc[fdf.index[i], "num"] = name[0]
    fdf.loc[fdf.index[i], "factor"] = name[1]
    fdf.loc[fdf.index[i], "freq"] = chg_frq(name[-1])

fdf_dir = Path('.') / 'factors' / 'Indicator_Table_Clean.csv'
fdf.to_csv(fdf_dir, index=False)

# %%

# shift features
fdf_dir = Path('.') / 'factors' / 'Indicator_Table_Clean1.csv'
fdf = pd.read_csv(fdf_dir)

from utils import shift_

for id, row in fdf.iterrows():
    num = "{:02d}".format(int(row['num']))
    freq = row['freq']
    for f in [i for i in data.columns if i.startswith(f'Factor{num}')]:

        data.groupby(['ticker'])[f].shift(4)

# shift return by 1
data.loc[:, 'pct_chg'] = data.groupby('ticker')['Y'].shift(-1)
data = data.dropna()
data.to_pickle(Path('.') / 'data' / "factor.pkl")
# %% Xiu's data
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
raw = pd.read_hdf(Path("./data/csmar/dataset.h5"), key='data')
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
data.to_pickle(Path('.') / 'data' / "yu_factor.pkl")