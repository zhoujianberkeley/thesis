# %%
import datetime
import pandas as pd
import numpy as np
import os
from pathlib import Path
# %%
p = Path('.') / 'data' / 'Monthly_Quotation.csv'
raw_data = pd.read_csv(p)

data = raw_data.set_index(["ts_code", "trade_date"]).drop('Unnamed: 0', axis=1)
data.index = data.index.set_names(["ticker",'date'])
# change datetime format
data = data.reset_index("date")
data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d"))
data = data.set_index('date', append=True).sort_index()

# shift return by 1
data.loc[:, 'pct_chg'] = data.groupby('ticker')['pct_chg'].shift(-1)
data = data.dropna()
data.to_pickle(Path('.') / 'data' / "factor.pkl")
# %% Xiu's data
p = Path('.') / 'data' / 'xiu_datashare' / 'xiu_datashare.csv'
raw_data = pd.read_csv(p)
raw_data.to_pickle(Path('.') / 'data' / 'xiu_datashare' / 'xiu_datashare.pkl')