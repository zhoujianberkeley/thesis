# %%
import os
import pandas as pd
from pathlib import Path
import platform
import logging
from multiprocessing import cpu_count
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import re
import time
from tqdm import tqdm
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import joblib

from utils_stra import split, cal_r2, cal_normal_r2, cal_model_r2
from utils_stra import save_arrays, save_res, save_year_res, stream, setwd
from utils_stra import add_months, gen_model_pt, save_model
from strategy_func import tree_model, tree_model_fast, genNNmodel, _loss_fn
from model_func import runModel, runFeatureImportance

setwd()
# create logger with 'spam_application'
logger = logging.getLogger('records')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('records.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# %%
p = Path('.') / 'data' / "data.h5"
data = pd.read_hdf(p, key="data")

ind_ftr = [i for i in data.columns if i.startswith('Ind_')]
mcr_ftr = [i for i in data.columns if i.startswith('Macro_')]
data = data[list(data.iloc[:, :89].columns) + ind_ftr + mcr_ftr + ["Y"]]

#%%
runGPU = 0
retrain = 0

# train30% validation20% test50% split
def initConfig():
    config = {"runOLS3":0,
              'runOLS3+H':0,
              'runOLS5':1,
              'runOLS5+H': 1,
                "runOLS":0,
                "runOLSH":0,
                "runENET":1,
                "runPLS":0,
                "runPCR":0,
                "runNN1":0,
                "runNN2":1,
                "runNN3":1,
                "runNN4":0,
                "runNN5": 0,
                "runNN6": 0,
                "runRF": 0,
                "runGBRT": 0,
                "runGBRT2": 0
              }
    return config


config = initConfig()

def cal_importance(data, config, method):
    runNN = sum([config[i] for i in [i for i in config.keys() if re.match("runNN[0-9]", i)]])
    res = pd.DataFrame(columns=['factor', 'r2'])

    model_name, container = runFeatureImportance(data, config, runNN)
    r2oos, r2oos_df = cal_model_r2(container, model_name, set_type="oos")
    print(f"{model_name} rm all facotr R2: ", "{0:.3%}".format(r2oos))
    res = res.append({"factor":"all factor", "r2":r2oos}, ignore_index=True)

    for fctr in tqdm([i for i in data.columns if i.startswith("Factor")]):
        databk = data.copy()
        if method == "zero":
            databk.loc[:, fctr] = 0
        elif method == "permutation":
            databk.loc[:, fctr] = databk.loc[:, fctr].shuffle
        else:
            raise NotImplementedError()

        model_name, container = runFeatureImportance(databk, config, runNN)

        r2oos, r2oos_df = cal_model_r2(container, model_name, set_type="oos")
        print(f"{model_name} remove {fctr} R2: ", "{0:.3%}".format(r2oos))
        res = res.append({"factor":fctr, "r2":r2oos}, ignore_index=True)
    return res, model_name

for config_key in config.keys():
    if config[config_key] == 0:
        continue
    print(f"running feature importance for {config_key}")

    ftr_imp, model_name = cal_importance(data, config)
    ftr_imp = ftr_imp.set_index('factor')
    ftr_imp['r2 reduction'] = ftr_imp.loc["all factor", "r2"] - ftr_imp["r2"]
    ftr_imp['r2 reduct max'] = np.maximum(ftr_imp['r2 reduction'], 0)
    ftr_imp['r2 reduction pct'] = ftr_imp['r2 reduction']/ftr_imp['r2 reduct max'].sum()
    ftr_imp = ftr_imp.sort_values(by='r2 reduction pct', ascending=False)
    ftr_imp.to_csv(Path("code", f"{model_name}")/f"feature_importance_{model_name}.csv")
    print(ftr_imp)

    config[config_key] = 0