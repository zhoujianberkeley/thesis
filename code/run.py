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
XX, yy = split(data)

print(XX.shape)
print(yy.shape)
# %%
runGPU = 1
retrain = 0

# train30% validation20% test50% split
def intiConfig():
    config = {"runOLS3":0,
              'runOLS3+H':0,
                "runOLS":1,
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
              "runRF": 0,
              "runGBRT": 0,
              }
    return config

config = intiConfig()
for config_key in config.keys():
    if config[config_key] == 0:
        continue
    print(f"running model {config_key}")
    # data index should be ticker date
    runNN = sum([config[i] for i in [i for i in config.keys() if re.match("runNN[0-9]", i)]])
    if runNN:
        if runGPU:
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
                if platform.system() == 'linus':
                    raise SystemError('GPU device not found')
                else:
                    pass
            print('Found GPU at: {}'.format(device_name))

    container = {}
    # for year in tqdm(range(2013, 2019)):
    #     p_t = [str(1900), str(year)] # period of training
    #     # p_t = [str(year-3), str(year)]
    #     p_v = [str(year+1), str(year + 1)] # period of valiation
    #     p_test = [str(year + 2), str(year+2)]
    #
    for year in tqdm(pd.date_range('20131231', '20200831', freq='M')):
        year = datetime.datetime.strftime(year, "%Y-%m")

        p_t = ['1900-01', str(year)] # period of training
        p_v = [add_months(year, 1), add_months(year, 3)] # period of valiation
        p_test = [add_months(year, 4), add_months(year, 4)]

        _Xt, _yt = split(data.loc(axis=0)[:, p_t[0]:p_t[1]].sample(frac=1, random_state=0))
        _Xv, _yv = split(data.loc(axis=0)[:, p_v[0]:p_v[1]].sample(frac=1, random_state=0))
        _Xtest, _ytest = split(data.loc(axis=0)[:, p_test[0]:p_test[1]])
        #OLS
        if config['runOLS3']:
            model_name = "OLS3"
            data_ols3 = data[['Factor46_mom12m', 'Factor51_mve', 'Factor09_bm', 'Y']]

            _Xt, _yt = split(data_ols3.loc(axis=0)[:, p_t[0]:p_t[1]])
            _Xv, _yv = split(data_ols3.loc(axis=0)[:, p_v[0]:p_v[1]])
            _Xtest, _ytest = split(data_ols3.loc(axis=0)[:, p_test[0]:p_test[1]])

            Xt = np.vstack((_Xt, _Xv))
            yt = np.vstack((_yt, _yv))
            Xtest, ytest = _Xtest, _ytest
            model_fit = LinearRegression().fit(Xt, yt)
        elif config['runOLS3+H']:
            from sklearn.linear_model import HuberRegressor
            model_name = "OLS3+H"
            data_ols3 = data[['Factor46_mom12m', 'Factor51_mve', 'Factor09_bm', 'Y']]

            _Xt, _yt = split(data_ols3.loc(axis=0)[:, p_t[0]:p_t[1]])
            _Xv, _yv = split(data_ols3.loc(axis=0)[:, p_v[0]:p_v[1]])
            _Xtest, _ytest = split(data_ols3.loc(axis=0)[:, p_test[0]:p_test[1]])

            Xt = np.vstack((_Xt, _Xv))
            yt = np.vstack((_yt, _yv))
            Xtest, ytest = _Xtest, _ytest
            model_fit = HuberRegressor(epsilon=3).fit(Xt, yt.reshape(-1, ))

        elif config['runOLS']:
            model_name= "OLS"
            Xt = np.vstack((_Xt, _Xv))
            yt = np.vstack((_yt, _yv))
            Xtest, ytest = _Xtest, _ytest
            model_fit = LinearRegression(n_jobs=-1).fit(Xt, yt.reshape(-1, ))

        elif config['runOLSH']:  # OLS + H
            model_name = "OLSH"
            Xt = np.vstack((_Xt, _Xv))
            yt = np.vstack((_yt, _yv))
            Xtest, ytest = _Xtest, _ytest
            model_fit = HuberRegressor().fit(Xt, yt.reshape(-1, ))

        elif config['runENET']:
            from sklearn.linear_model import ElasticNet
            model_name = "ENET"
            Xt, yt = _Xt, _yt
            Xv, yv = _Xv, _yv
            Xtest, ytest = _Xtest, _ytest

            lambda_ = [0.1, 0.01, 0.001, 0.0001]
            params = [{'lambda': i} for i in lambda_]

            out_cv = []
            for p in tqdm(params):
                model_fit = ElasticNet(alpha=p['lambda'], l1_ratio=0.5, random_state=0)
                model_fit.fit(Xt, yt.reshape(-1, ))

                yv_hat = model_fit.predict(Xv).reshape(-1, 1)
                perfor = cal_r2(yv, yv_hat)
                out_cv.append(perfor)
                # print('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
                logger.info('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
            # tic = time.time()
            # print(f"{model} train time: ", tic - tis)
            best_p = params[np.argmax(out_cv)]
            print("best p", best_p)
            logger.info(f"{model_name} {year} {params} best hyperparamer ", best_p)
            model_fit = ElasticNet(alpha=best_p['lambda'], l1_ratio=0.5, random_state=0)
            model_fit.fit(Xt, yt)
            ytest_hat = model_fit.predict(Xtest).reshape(-1, 1)
            best_perfor = cal_r2(ytest, ytest_hat)
            print(f"{model_name} oss r2:", best_perfor)

        elif config['runPLS']:
            from sklearn.cross_decomposition import PLSRegression
            model_name = "PLS"
            Xt, yt = _Xt, _yt
            Xv, yv = _Xv, _yv
            Xtest, ytest = _Xtest, _ytest

            maxk = min(30, Xt.shape[1])
            ks = np.arange(1, maxk, 2)
            params = [{'k': i} for i in ks]

            out_cv = []
            for p in tqdm(params):
                pls = PLSRegression(n_components=p['k'])
                model_fit = pls.fit(Xt, yt)

                yv_hat = model_fit.predict(Xv)
                perfor = cal_r2(yv, yv_hat)
                out_cv.append(perfor)
                print('params: ' + str(p) + '. CV r2-validation:' + "{0:.3%}".format(perfor))
                logging.info('params: ' + str(p) + '. CV r2-validation:' + "{0:.3%}".format(perfor))

            best_p = params[np.argmax(out_cv)]
            print("best hyper-parameter", best_p)

            pls = PLSRegression(n_components=best_p['k'])
            model_fit = pls.fit(Xt, yt)

            ytest_hat = model_fit.predict(Xtest)
            best_perfor = cal_r2(ytest, ytest_hat)
            print(f"{model_name} oss r2 in {year}:", best_perfor)

        elif config['runPCR']:
            model_name = "PCR"
            mtrain = np.mean(_yt)
            Xt, yt = _Xt, _yt
            Xv, yv = _Xv, _yv
            Xtest, ytest = _Xtest, _ytest

            # # prepare for PCR running
            # XTX = np.dot(Xt.T, Xt)  # X=xtrain.'*xtrain;
            # _pca_val, _pca_vec = np.linalg.eig(XTX)  # X*pca_vec = pca_vec*pca_val
            # idx = _pca_val.argsort()[::-1]
            # pca_val = _pca_val[idx]
            # pca_vec = _pca_vec[:, idx]

            # p1 = pca_vec[:, :maxk-5]  # 选出最大的30个
            # Z = np.dot(Xt, p1)

            # hyper-parameter
            maxk = min(30, Xt.shape[1])
            ks = np.arange(1, maxk, 2)
            params = [{'k': i} for i in ks]

            out_cv = []
            for p in tqdm(params):
                # xx = Z[:, :p['k']]
                # b = np.linalg.inv(xx.T@xx) @ (xx.T@yt)  # b = (inv(xx.'*xx)*xx.') * Y;
                # bf = p1[:, :p['k']]@b  #b = p1(:, 1: j)*b;
                #
                # yv_hat = Xv@bf + mtrain  # yhatbig1 = xtest * b + mtrain;
                pca = PCA(n_components=p['k'])
                X_reduced = pca.fit_transform(Xt)
                model_fit = LinearRegression()
                model_fit = model_fit.fit(X_reduced, yt)

                xv_r = pca.transform(Xv)
                yv_hat = model_fit.predict(xv_r)
                perfor = cal_r2(yv, yv_hat)
                out_cv.append(perfor)
                print('params: ' + str(p) + '. CV r2-validation:' + "{0:.3%}".format(perfor))
                logging.info('params: ' + str(p) + '. CV r2-validation:' + "{0:.3%}".format(perfor))

            best_p = params[np.argmax(out_cv)]
            print("best hyper-parameter", best_p)
            # xx = Z[:, :best_p['k']]
            # b = np.linalg.inv(xx.T @ xx) @ (xx.T @ yt)
            # bf = p1[:, :best_p['k']] @ b
            # ytest_hat = (Xtest @ bf + mtrain).reshape(-1, 1)
            pca = PCA(n_components=best_p['k'])
            Xt = pca.fit_transform(Xt)
            model_fit = LinearRegression()
            model_fit = model_fit.fit(Xt, yt)

            Xtest = pca.transform(Xtest)
            ytest_hat = model_fit.predict(Xtest)
            best_perfor = cal_r2(ytest, ytest_hat)
            print(f"{model_name} oss r2 in {year}:", best_perfor)

        elif runNN:
            import tensorflow as tf
            import tensorflow.keras as keras
            from keras.models import Sequential
            from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
            from strategy_func import genNNmodel, _loss_fn

            if config["runNN1"]:
                i = 1
            elif config["runNN2"]:
                i = 2
            elif config["runNN3"]:
                i = 3
            elif config["runNN4"]:
                i = 4
            elif config["runNN5"]:
                i = 5
            elif config["runNN6"]:
                i = 6

            model_name = f"NN{i}"

            nn_is_preds = []
            nn_valid_preds = []
            nn_oos_preds = []

            nn_valid_r2 = []
            nn_oos_r2 = []

            model_cntn = []
            for model_num in range(5):
                model_dir = Path("code", f"{model_name}")
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                model_pt = model_dir / f"{year}_iter{model_num}_bm.hdf5"

                if retrain:
                    tf.random.set_seed(model_num)

                    _Xt, _yt = split(data.loc(axis=0)[:, p_t[0]:p_t[1]].sample(frac=1, random_state=model_num))
                    _Xv, _yv = split(data.loc(axis=0)[:, p_v[0]:p_v[1]].sample(frac=1, random_state=model_num+1))

                    Xt, yt = _Xt, _yt
                    Xv, yv = _Xv, _yv
                    Xtest, ytest = _Xtest, _ytest

                    model_fit = genNNmodel(XX.shape[1], i)
                    earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=33,
                                                                 verbose=0, mode="min", baseline=None,
                                                                 restore_best_weights=True)

                    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_pt, monitor='val_loss', verbose=0,
                                                                    save_best_only=True, mode='min')
                    cb_list = [earlystop, checkpoint]
                    loss_fn = _loss_fn
                    # loss_fn = tf.losses.MeanSquaredError()
                    opt = keras.optimizers.Adam(clipvalue=0.5)
                    if runGPU:
                        with tf.device('/device:GPU:0'):
                            print("running on GPU")
                            model_fit.compile(loss=loss_fn, optimizer=opt, metrics=['mse'])
                            # fit the keras model on the dataset
                            model_fit.fit(Xt, yt, epochs=1000, batch_size=2560, verbose=0,
                                          callbacks=cb_list, validation_data=(Xv, yv), validation_freq=1)
                    else:
                        model_fit.compile(loss=loss_fn, optimizer=opt, metrics=['mse'])
                        model_fit.fit(Xt, yt, epochs=2, batch_size=2560, verbose=1,
                                      callbacks=cb_list, validation_data=(Xv, yv), validation_freq=1)
                    plt.plot(model_fit.history.history['loss'], label='train')
                    plt.plot(model_fit.history.history['val_loss'], label='validation')
                    plt.legend()
                    plt.show()

                else:
                    print("loading models")
                    Xt, yt = _Xt, _yt
                    _Xv, _yv = split(data.loc(axis=0)[:, p_v[0]:p_v[1]].sample(frac=1, random_state=model_num+1))
                    Xv, yv = _Xv, _yv
                    Xtest, ytest = _Xtest, _ytest
                    model_fit = tf.keras.models.load_model(model_pt, custom_objects={'_loss_fn': _loss_fn})

                model_cntn.append(model_fit)

                # is_predictions = model_fit.predict(Xt)
                valid_pred = model_fit.predict(Xv)
                oos_pred = model_fit.predict(Xtest)
                # r2is = cal_r2(yt, is_predictions)
                r2valid = cal_r2(yv, valid_pred)
                r2oos = cal_r2(ytest, oos_pred)
                # nr2oos = cal_normal_r2(ytest, predictions)

                # print(f"model{model_num} train r2", "{0:.3%}".format(r2is))
                print(f"model{model_num} valid r2", "{0:.3%}".format(r2valid))
                print(f"model{model_num} test r2", "{0:.3%}".format(r2oos))

                # nn_is_preds.append(is_predictions)
                # nn_valid_preds.append(valid_predictions)
                nn_valid_r2.append(r2valid)
                nn_oos_r2.append(r2oos)
                # if r2valid < 0.1:
                #   nn_oos_preds.append(oos_pred)
                nn_valid_preds.append(valid_pred)
                nn_oos_preds.append(oos_pred)

        elif config['runRF']:
            logger.info(year)
            model_name = "RF"
            Xt, yt = _Xt, _yt
            Xv, yv = _Xv, _yv
            Xtest, ytest = _Xtest, _ytest
            if not retrain:
                model_fit = tree_model_fast(model_name, year, Xt, yt, Xv, yv, runRF=True, runGBRT=False)
            else:
                model_fit = tree_model(Xt, yt, Xv, yv, runRF=True, runGBRT=False)
            save_model(model_name, year, model_fit)
        elif config['runGBRT']:
            model_name = "GBRT"
            Xt, yt = _Xt, _yt
            Xv, yv = _Xv, _yv
            Xtest, ytest = _Xtest, _ytest
            if not retrain:
                model_fit = tree_model_fast(model_name, year, Xt, yt, Xv, yv, runRF=False, runGBRT=True)
            else:
                model_fit = tree_model(Xt, yt, Xv, yv, runRF=False, runGBRT=True)
            # Don't use pickle or joblib as that may introduces dependencies on xgboost version.
            # The canonical way to save and restore models is by load_model and save_model.
            model_pt = gen_model_pt(model_name, year)
            model_fit.save_model(model_pt)

        if runNN:
            # yt_hat = np.mean(np.concatenate(nn_is_preds, axis=1), axis=1).reshape(-1, 1)
            yv_hat = np.mean(np.concatenate(nn_valid_preds, axis=1), axis=1).reshape(-1, 1)
            ytest_hat = np.mean(np.concatenate(nn_oos_preds, axis=1), axis=1).reshape(-1, 1)
            print(f"mean r2 in {year}among models", "{0:.3%}".format(np.mean(nn_oos_r2)))

            save_arrays(container, model_name, year, ytest_hat, savekey='ytest_hat')
            save_arrays(container, model_name, year, ytest, savekey='ytest')

            save_arrays(container, model_name, year, yv_hat, savekey='yv_hat')
            save_arrays(container, model_name, year, yv, savekey='yv')

            save_year_res(model_name, year, 0, cal_r2(ytest, ytest_hat))
        else:
            yt_hat = model_fit.predict(Xt).reshape(-1, 1)
            ytest_hat = model_fit.predict(Xtest).reshape(-1, 1)

            save_arrays(container, model_name, year, yt_hat, savekey='yt_hat')
            save_arrays(container, model_name, year, yt, savekey='yt')

            save_arrays(container, model_name, year, ytest_hat, savekey='ytest_hat')
            save_arrays(container, model_name, year, ytest, savekey='ytest')

            save_year_res(model_name, year, cal_r2(yt, yt_hat), cal_r2(ytest, ytest_hat))

    if runNN:
        r2v, r2v_df = cal_model_r2(container, model_name, set_type="valid")
        print(f"{model_name} valid R2: ", "{0:.3%}".format(r2v))
    else:
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