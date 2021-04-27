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

import utils
utils.setdir_fctr()

s_path = Path("_saved_factors")
if not os.path.exists(s_path):
    os.mkdir(s_path)
# %%
# use split-adjusted share prices
Monthly_Quotation_sa = utils.cleandf(pd.read_csv(Path('data', 'buffer', 'MonFactorPrcd_sa.csv')))
Monthly_Quotation_sa = utils.todate(Monthly_Quotation_sa, 'end_date', format='%Y-%m-%d')

Monthly_Quotation_sa = Monthly_Quotation_sa.set_index(["ts_code", 'end_date']).sort_index()
Monthly_Quotation_sa['monthly_return'] = Monthly_Quotation_sa.groupby(['ts_code'])['close'].pct_change()

# load risk free rate
rf = pd.read_csv(Path('_saved_factors', 'MacroFactor.csv'), index_col=0, parse_dates=['end_date'])[['RiskFreeRate']]
rf["Mon_rfr"] = (1+rf['RiskFreeRate']/100)**(1/12)-1
rf = rf.sort_index().shift(1)
Monthly_Quotation_sa = Monthly_Quotation_sa.merge(rf, how="left", left_on="end_date", right_index=True)

Monthly_Quotation_sa['Mon_rfr'] = Monthly_Quotation_sa['Mon_rfr'].fillna(0)
Monthly_Quotation_sa['excess_return'] = Monthly_Quotation_sa['monthly_return'] - Monthly_Quotation_sa['Mon_rfr']

# %%
Monthly_Quotation_sa = Monthly_Quotation_sa.reset_index()
Monthly_Quotation_sa[['ts_code', 'end_date', 'excess_return']].to_csv(Path('_saved_factors', 'MonY.csv'), index=False)
Monthly_Quotation_sa[['ts_code', 'end_date', 'close']].to_csv(Path('_saved_factors', 'MonClose.csv'), index=False)
# %%
