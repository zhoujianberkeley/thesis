#%%
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from utils_stra import setwd
from HFBacktest import HFBacktest, HFSingleFactorBacktest
setwd()

# %%
p = Path('.') / 'data' / "data.h5"
data = pd.read_hdf(p, key="data")

ind_ftr = [i for i in data.columns if i.startswith('Ind_')]
mcr_ftr = [i for i in data.columns if i.startswith('Macro_')]
data = data[list(data.iloc[:, :88].columns) + ind_ftr + mcr_ftr + ["Y"]]
#%%
close = pd.read_hdf(Path('factors', 'china factors', '_saved_factors', "close.h5"), key='data')
close = close['close'].unstack(level=0)
close.index = close.index.astype(str)
close.iloc[20:]
# %%
model_name = "RF"
ml_fctr = pd.read_csv(Path('code') / model_name / "predictions.csv").set_index(["ticker", "date"])['predict'].unstack(level=0)
ml_fctr.index = ml_fctr.index.astype(str)
ml_fctr = ml_fctr.replace(np.inf, 10000000)
ml_fctr = ml_fctr.replace(-np.inf, -10000000)
# ml_fctr = ml_fctr.sub(np.nanmean(ml_fctr, axis=1), axis=0).divide(np.nanstd(ml_fctr), axis=0)
ml_fctr.iloc[20:]
# %%
analysis = HFBacktest(close)
analysis.addFactor('ml_factor', ml_fctr)
analysis.orthogonalizeFactors()
analysis.analyzeSingleFactor()
#%%