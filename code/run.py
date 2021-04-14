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
from model_func import runModel

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
data = data[list(data.iloc[:, :88].columns) + ind_ftr + mcr_ftr + ["Y"]]

# %%
runGPU = 0
retrain = 0

# train30% validation20% test50% split
def intiConfig():
    config = {"runOLS3":0,
              'runOLS3+H':0,
                "runOLS":0,
                "runOLSH":0,
                "runENET":0,
                "runPLS":0,
                "runPCR":0,
                "runNN1":0,
                "runNN2":0,
                "runNN3":0,
                "runNN4":0,
                "runNN5": 0,
                "runNN6": 0,
                "runRF": 1,
                "runGBRT": 0,
                "runGBRT2": 0
              }
    return config

config = intiConfig()


for config_key in config.keys():
    if config[config_key] == 0:
        continue
    print(f"running model {config_key}")
    # data index should be ticker date
    runNN = sum([config[i] for i in [i for i in config.keys() if re.match("runNN[0-9]", i)]])
    if runNN and runGPU:
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0' and platform.system() == 'linus':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    # for year in tqdm(range(2013, 2019)):
    #     p_t = [str(1900), str(year)] # period of training
    #     # p_t = [str(year-3), str(year)]
    #     p_v = [str(year+1), str(year + 1)] # period of valiation
    #     p_test = [str(year + 2), str(year+2)]
    if runNN:
        model_name, bcktst_df, container, nn_valid_r2, nn_oos_r2, model_dir = runModel(data, config, retrain, runGPU, runNN)

        r2v, r2v_df = cal_model_r2(container, model_name, set_type="valid")
        r2is = r2v
        print(f"{model_name} valid R2: ", "{0:.3%}".format(r2v))
        c = np.array(nn_valid_r2)
        d = np.array(nn_oos_r2)
        print("the correlation coefficients is :", np.corrcoef(c, d))
        plt.scatter(c, d)
        plt.xlabel("validation scores")
        plt.ylabel("test scores")
        plt.title(f"{model_name}")
        plt.show()
        with open(model_dir / "validation_score.pkl", "wb") as f:
            pickle.dump(c, f)
        with open(model_dir / "testing_score.pkl", "wb") as f:
            pickle.dump(d, f)
    else:
        model_name, bcktst_df, container = runModel(data, config, retrain, runGPU, runNN)

        r2is, r2is_df = cal_model_r2(container, model_name, set_type="is")
        print(f"{model_name} IS R2: ", "{0:.3%}".format(r2is))

    r2oos, r2oos_df = cal_model_r2(container, model_name, set_type="oos")
    print(f"{model_name} R2: ", "{0:.3%}".format(r2oos))
    # nr2oos = cal_model_r2(container, model_name, normal=True)
    # print(f"{model_name} Normal R2: ", "{0:.3%}".format(nr2oos))

    # nr2is = cal_model_r2(container, model_name, oos=False, normal=True)
    # print(f"{model_name} ISN R2: ", "{0:.3%}".format(nr2is))

    save_res(model_name, r2is, r2oos, nr2is=0, nr2oos=0)

    if not os.path.exists(Path('code') / model_name):
        os.mkdir(Path('code') / model_name)
    with open(Path('code') / model_name / f"predictions.pkl", "wb+") as f:
        pickle.dump(container[model_name], f)

    config[config_key] = 0

#%%
# combine = np.concatenate([c.reshape(-1,1),d.reshape(-1,1)], axis=1)
# slice = combine[combine[:,0].argsort()][:-100]
# np.corrcoef(slice[:,0], slice[:,1])
#%%
# c = np.array(nn_valid_r2)
# c.sort()
# c[-len(nn_valid_r2)//10]

# %%