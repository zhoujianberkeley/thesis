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
with open(Path('code') / model_name / f"predictions.pkl", "rb+") as f:
    years_dict = pickle.load(f)
key1 = 'ytest'
key2 = 'ytest_hat'

ytrues, yhats= [], []
years_r2 = {}
for key, value in years_dict.items():
    ytrue = value[key1]
    yhat = value[key2]
    # years_r2[key] = cal_r2(ytrue, yhat)
    ytrues.append(ytrue)
    yhats.append(yhat)
ytrues, yhats = np.concatenate(ytrues, axis=0), np.concatenate(yhats, axis=0)
assert ytrues.shape == yhats.shape
# %%
ml_fctr = pd.read_hdf(Path('factors', 'china factors', '_saved_factors', "close.h5"), key='data')['close'].unstack(level=0)
# ml_fctr = ml_fctr.iloc[20:]
ml_fctr.index = ml_fctr.index.astype(str)
ml_fctr = ml_fctr.replace(np.inf, 10000000)
ml_fctr = ml_fctr.replace(-np.inf, -10000000)
# ml_fctr = ml_fctr.sub(np.nanmean(ml_fctr, axis=1), axis=0).divide(np.nanstd(ml_fctr), axis=0)

# %%
analysis = HFBacktest(close)
analysis.addFactor('ml_factor', ml_fctr)
analysis.orthogonalizeFactors()
analysis.analyzeSingleFactor()
#%%