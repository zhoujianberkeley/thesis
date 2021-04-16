#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from utils_stra import setwd
# from HFBacktest1 import HFBacktest, HFSingleFactorBacktest
setwd()
#%%
model_name = "NN3"

close = pd.read_hdf(Path('factors', 'china factors', '_saved_factors', "close.h5"), key='data')
close.index.names = ["ticker", "date"]
ml_fctr = pd.read_csv(Path('code') / model_name / "predictions.csv").set_index(["ticker", "date"])
ml_fctr = ml_fctr.dropna(how='all')
bt_df = close.merge(ml_fctr, on=["ticker", "date"], how="right")

close = bt_df['close'].unstack(level=0)
change = close.pct_change()
# close.index = close.index.astype(str)

ml_fctr = bt_df['predict'].unstack(level=0)
# ml_fctr.index = ml_fctr.index.astype(str)
# ml_fctr = ml_fctr.replace(np.inf, 10000000)
# ml_fctr = ml_fctr.replace(-np.inf, -10000000)
# ml_fctr = ml_fctr.sub(np.nanmean(ml_fctr, axis=1), axis=0).divide(np.nanstd(ml_fctr), axis=0)


# %%
def decile10(date_df, rtrn_df):
    date_df = date_df.T.dropna()
    date = date_df.columns[0]
    date_df = date_df.sort_values(by=[date],ascending=False)
    thrsd = date_df.shape[0]//10

    id = date_df.iloc[:thrsd, ].index
    d10 = rtrn_df.T.loc[id, date].mean()

    return d10

def decile1(date_df, rtrn_df):
    date_df = date_df.T.dropna()
    date = date_df.columns[0]
    date_df = date_df.sort_values(by=[date], ascending=False)
    thrsd = date_df.shape[0] // 10

    id = date_df.iloc[-thrsd:, ].index
    d1 = rtrn_df.T.loc[id, date].mean()
    return d1
    # return pd.DataFrame.from_dict({"decile 10":[d10],
    #                                 "decile 1":[d1]}, orient="index", columns=[date])

bt_res = pd.DataFrame()
bt_res["decile 10"] = ml_fctr.groupby('date').apply(lambda x: decile10(x, change)).fillna(0)
bt_res["decile 1"] = ml_fctr.groupby('date').apply(lambda x: decile1(x, change)).fillna(0)


bt_res[["d10 rt", "d1 rt"]] = (bt_res[["decile 10", "decile 1"]]+1).cumprod()
bt_res["ls"] = bt_res["decile 10"] - bt_res["decile 1"]
bt_res["ls_return"] = (bt_res["ls"]+1).cumprod()

bt_res[['d10 rt', 'd1 rt', 'ls_return']].plot()
# bt_res[['d10 rt', 'd1 rt']].plot()
plt.show()

rtrn = bt_res['ls']

print(f"sharpe ratio {model_name}", round(np.sqrt(12)*rtrn.mean()/rtrn.std(), 2))


# %%
# analysis = HFBacktest(close, )
# analysis.addFactor('ml_factor', ml_fctr)
# analysis.addFactor()
# analysis.orthogonalizeFactors()
# analysis.analyzeSingleFactor()
#%%
# bm = pd.DataFrame(index=ml_fctr.index, columns=ml_fctr.columns).fillna(0)
# analysis2 = HFSingleFactorBacktest('ml_factor',ml_fctr, close, bm)
# analysis2.analyze()
#%%