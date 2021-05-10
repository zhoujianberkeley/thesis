# %%
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np

import re
from tqdm import tqdm

from utils_stra import cal_model_r2, setwd
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
              'runOLS5':0,
              'runOLS5+H': 0,
                "runOLS":0,
                "runOLSH":0,
                "runENET":0,
                "runPLS":0,
                "runPCR":0,
                "runNN1":1,
                "runNN2":0,
                "runNN3":0,
                "runNN4":0,
                "runNN5": 0,
                "runNN6": 0,
                "runRF": 0,
                "runGBRT": 0,
                "runGBRT2": 0
              }
    return config


config = initConfig()

def cal_importance(data, config, frequency, pre_dir, method):
    runNN = sum([config[i] for i in [i for i in config.keys() if re.match("runNN[0-9]", i)]])
    res = pd.DataFrame(columns=['factor', 'r2'])

    model_name, container = runFeatureImportance(data, config, runNN, frequency, pre_dir)
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

        model_name, container = runFeatureImportance(databk, config, runNN, frequency, pre_dir)

        r2oos, r2oos_df = cal_model_r2(container, model_name, set_type="oos")
        print(f"{model_name} remove {fctr} R2: ", "{0:.3%}".format(r2oos))
        res = res.append({"factor":fctr, "r2":r2oos}, ignore_index=True)
    return res, model_name

pre_dir = "Filter IPO"
frequency = "Y"

for config_key in config.keys():
    if config[config_key] == 0:
        continue
    print(f"running feature importance for {config_key}")

    ftr_imp, model_name = cal_importance(data, config, frequency, pre_dir, method='zero')
    ftr_imp = ftr_imp.set_index('factor')
    ftr_imp['r2 reduction'] = ftr_imp.loc["all factor", "r2"] - ftr_imp["r2"]
    ftr_imp['r2 reduct max'] = np.maximum(ftr_imp['r2 reduction'], 0)
    ftr_imp['r2 reduction pct'] = ftr_imp['r2 reduction']/ftr_imp['r2 reduct max'].sum()
    ftr_imp = ftr_imp.sort_values(by='r2 reduction pct', ascending=False)
    ftr_imp.to_csv(Path("code", pre_dir ,f"{model_name}")/f"feature_importance_{model_name}.csv")
    print(ftr_imp)
    config[config_key] = 0