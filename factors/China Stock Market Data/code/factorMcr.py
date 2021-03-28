import numpy as np
import os
import pandas as pd
from pathlib import Path
import datetime

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

s_path = Path("_saved_factors")
if not os.path.exists(s_path):
    os.mkdir(s_path)

import utils
# %%
dfds = pd.read_excel(Path('data', 'macro', 'default_spread.xls'))
ds = dfds.groupby(['year', 'month']).mean()['default spread']
ds.name = "DefaultSpread"
# %%
dfsvar = pd.read_excel(Path('data', 'macro', 'SHCI.xlsx'))
dfsvar['pct_chg'] = np.square(dfsvar['close'].pct_change())
svar = dfsvar.groupby(['year', 'month'])['pct_chg'].sum()
svar.name = 'svar'
# %%
df_factors = pd.read_excel(Path('data', 'macro', 'macro.xlsx')).set_index(['year', 'month'])
cols = ['RiskFreeRate', 'TermSpread', 'DividendPriceRatio', 'NetEquityExpansion', 'PE', 'PB']
factors = df_factors[cols]
# %%
res = pd.concat([factors, ds, svar], axis=1)
res.index = res.index.map(lambda x: datetime.date(int(x[0]), int(x[1]), 1) + pd.offsets.MonthEnd(0))
res.index.name = 'end_date'

Out_df = res.loc["2010":"2020", :].reset_index()

Out_df.to_csv(Path('_saved_factors', 'MacroFactor.csv'))
# %%