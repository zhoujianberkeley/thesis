#%%
import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from utils_stra import setwd
# from HFBacktest1 import HFBacktest, HFSingleFactorBacktest
setwd()
#%%
# ml_fctr.index = ml_fctr.index.astype(str)
# ml_fctr = ml_fctr.replace(np.inf, 10000000)
# ml_fctr = ml_fctr.replace(-np.inf, -10000000)
# ml_fctr = ml_fctr.sub(np.nanmean(ml_fctr, axis=1), axis=0).divide(np.nanstd(ml_fctr), axis=0)

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


def backtest(model_name, factor, change_df):
    bt_res = pd.DataFrame()
    bt_res[f"decile 10"] = factor.groupby('date').apply(lambda x: decile10(x, change_df)).fillna(0)
    bt_res["decile 1"] = factor.groupby('date').apply(lambda x: decile1(x, change_df)).fillna(0)


    bt_res[f"{model_name} d10 rt"] = (bt_res["decile 10"]+1).cumprod() - 1

    bt_res[f"{model_name} d1 rt"] = -((-bt_res["decile 1"] + 1).cumprod()) + 1

    bt_res["ls"] = bt_res["decile 10"] - bt_res["decile 1"]
    bt_res[f"{model_name} ls_return"] = (bt_res["ls"]+1).cumprod() - 1

    rtrn = bt_res['ls']
    print(f"sharpe ratio {model_name}", round(np.sqrt(12)*rtrn.mean()/rtrn.std(), 2))

    bt_res_slice = bt_res[[f"{model_name} d10 rt", f"{model_name} d1 rt", f'{model_name} ls_return']]
    bt_res_slice.plot()
    plt.show()

    return bt_res_slice
# %%
close_raw = pd.read_hdf(Path('factors', 'china factors', '_saved_factors', "close.h5"), key='data')
close_raw.index.names = ["ticker", "date"]

res = []
for model_name in ["NN1", "RF", "ENET"]:
    ml_fctr = pd.read_csv(Path('code') / model_name / "predictions.csv").set_index(["ticker", "date"])
    ml_fctr = ml_fctr.dropna(how='all')
    bt_df = close_raw.merge(ml_fctr, on=["ticker", "date"], how="right")
    close = bt_df['close'].unstack(level=0)
    change = close.pct_change()


    ml_fctr = bt_df['predict'].unstack(level=0)

    res.append(backtest(model_name, ml_fctr, change))
res_df = pd.concat(res, axis=1)

# add benchmark
bm = pd.read_excel(Path("data")/"SHCI.xlsx",usecols=range(6)).set_index('date')
bm.index = pd.to_datetime(bm.index) + pd.tseries.offsets.MonthEnd(0)
bm.index = bm.index.astype(str)
bm = bm.reindex(res_df.index)
res_df["SHCI"] = (bm[['pct_change']].fillna(0) + 1).cumprod()-1

res_df[[i for i in res_df.columns if "ls_return" in i] + ["SHCI"]].plot()
plt.show()

res_df[[i for i in res_df.columns if "ls_return" not in i]].plot()
plt.show()

save_dir = Path("thesis")
if not os.path.exist(save_dir):
    os.mkdir(save_dir)
res_df.to_excel(save_dir / "backtest.xlsx")
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