# %%
import pandas as pd
import numpy as np
import os
from pathlib import Path
# %%

p = Path('.') / 'data' / 'daily_ohlc_vol_amount.csv'

raw_data = pd.read_csv(p)
# data = raw_data.copy()
data = raw_data.set_index(["ts_code", "trade_date"]).drop('Unnamed: 0', axis=1)
data.to_pickle(Path('.') / 'data' / "factor.pkl")
# %% Xiu's data

p = Path('.') / 'data' / 'xiu_datashare' / 'xiu_datashare.csv'
raw_data = pd.read_csv(p)
raw_data.to_pickle(Path('.') / 'data' / 'xiu_datashare' / 'xiu_datashare.pkl')