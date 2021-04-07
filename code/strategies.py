# %%
import numpy as np
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import time
from tqdm import tqdm
import datetime

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

# %%
def split(data):
    features = data.columns.to_list()
    features.remove("Y")
    # features.remove("change")
    # features.remove("name")

    X = data[features].values
    y = data['Y'].values.reshape(-1, 1)
    return X, y

def cal_r2(true, hat):
    if len(hat.shape) == 1 and true.shape[0] == hat.shape[0]:
        hat = hat.reshape(-1, 1)
    assert true.shape == hat.shape, "shape not match"
    a = np.square(true-hat).sum()
    b = np.square(true).sum()
    return 1 - a/b

def cal_normal_r2(true, hat):
    if len(hat.shape) == 1 and true.shape[0] == hat.shape[0]:
        hat = hat.reshape(-1, 1)
    assert true.shape == hat.shape, "shape not match"
    a = np.square(true-hat).sum()
    b = np.square(true-np.mean(true)).sum()
    return 1 - a/b

def cal_model_r2(container, model, oos=True, normal=False):
    if oos:
        ytrue = container[model]['true']
        yhat = container[model]['hat']
    else:
        ytrue = container[model]['yt']
        yhat = container[model]['yt_hat']

    if normal:
        return cal_normal_r2(ytrue, yhat)
    else:
        return cal_r2(ytrue, yhat)

def save_hat(container, model, to_save, savehat=None, savekey=None):
    if model not in container.keys():
        container[model] = {}
        container[model]['hat'] = np.array([]).reshape(-1, 1)
        container[model]['true'] = np.array([]).reshape(-1, 1)
        return save_hat(container, model, to_save, savehat, savekey)
    elif savekey is not None:
        if savekey not in container[model].keys():
            container[model][savekey] = np.array([]).reshape(-1, 1)
            return save_hat(container, model, to_save, savehat, savekey)
        else:
            container[model][savekey] = np.vstack((container[model][savekey], to_save))
            return
    elif savehat is True:
        container[model]['hat'] = np.vstack((container[model]['hat'], to_save))
        return
    elif savehat is False:
        container[model]['true'] = np.vstack((container[model]['true'], to_save))
        return
    else:
        raise NotImplementedError()

def save_res(r2oos, nr2oos, r2is, nr2is):
    dir = Path("code", "model_result.csv")
    if not os.path.exists(dir):
        pd.DataFrame(columns=["r2oos"]).to_csv(dir, index=False)

    res = pd.read_csv(dir)
    res.loc[model_name, "r2oos" ] = r2oos
    res.loc[model_name, "nr2oos"] = nr2oos
    res.loc[model_name, "r2is"] = r2is
    res.loc[model_name, "nr2is"] = nr2is
    res.to_csv(dir, index=False)

def stream(data):
    for year in range(2013, 2018):
        p_t = [str(1900), str(year)]
        p_v = [str(year + 1), str(year + 2)]
        p_test = [str(year + 3), str(year + 3)]

        _Xt, _yt = split(data.loc(axis=0)[:, p_t[0]:p_t[1]])
        _Xv, _yv = split(data.loc(axis=0)[:, p_v[0]:p_v[1]])
        _Xtest, _ytest = split(data.loc(axis=0)[:, p_test[0]:p_test[1]])
        yield _Xt, _yt, _Xv, _yv, _Xtest, _ytest

def tree_model(Xt, yt, Xv, yv, Xtest, ytest, runRF, runGBRT):
    assert runRF + runGBRT == 1
    if runRF:
        model_name = "Random Forest"
        # max_depth = np.arange(1, 7)
        # max_features = [3, 5, 10, 20, 30, 50]
        max_depth = np.arange(1, 3)
        max_features = [3, 5, 10]
        params = [{'max_dep': i, 'max_fea': j} for i in max_depth for j in max_features]
    elif runGBRT:
        model_name = "Boosting Trees"
        # boosting params
        num_trees = (np.arange(2, 12, 4)) * 100
        learning_rate = [0.01, 0.1]
        # subsample = [0.5, 0.6, 0.7, 0.8]
        subsample = [1]
        loss = ['huber']

        # cart tree params
        max_depth = [1, 2]

        params = []
        for t in num_trees:
            for d in max_depth:
                for l in learning_rate:
                    for s in subsample:
                        for o in loss:
                            params.append({'num_trees': t, 'max_dep': d, 'lr': l, 'subsample': s, 'loss': o})

    tis = time.time()
    out_cv = []
    for p in tqdm(params):
        if runRF:
            tree_m = RandomForestRegressor(n_estimators=300, max_depth=p['max_dep'], max_features=p['max_fea'],
                                           min_samples_split=10, random_state=0, n_jobs=-1)
        elif runGBRT:
            tree_m = GradientBoostingRegressor(max_depth=p['max_dep'], n_estimators=p['num_trees'],
                                               learning_rate=p['lr'], max_features=p['max_dep'],
                                               min_samples_split=10, loss=p['loss'], min_samples_leaf=10,
                                               subsample=p['subsample'], random_state=0)
        tree_m.fit(Xt, yt.reshape(-1, ))
        yv_hat = tree_m.predict(Xv).reshape(-1, 1)
        perfor = cal_r2(yv, yv_hat)
        out_cv.append(perfor)

        print('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
        logging.info('params: ' + str(p) + '. CV r2-validation:' + str(perfor))

        # model.fit(train_X, train_y)
        # pred_cv = model.predict(cv_X)
        # perfor = r2_oos(cv_y, pred_cv)
    tic = time.time()
    print(f"{model_name} train time: ", tic - tis)

    best_p = params[np.argmax(out_cv)]
    if runRF:
        tree_m = RandomForestRegressor(n_estimators=300, max_depth=best_p['max_dep'], max_features=best_p['max_fea'],
                                       min_samples_split=10, random_state=0, n_jobs=-1)
    if runGBRT:
        tree_m = GradientBoostingRegressor(max_depth=best_p['max_dep'], n_estimators=best_p['num_trees'],
                                           learning_rate=best_p['lr'], max_features=best_p['max_dep'],
                                           min_samples_split=10, loss=best_p['loss'], min_samples_leaf=10,
                                           subsample=p['subsample'], random_state=0)
    tree_m.fit(Xt, yt.reshape(-1, ))
    return tree_m
# %%
p = Path('.') / 'data' / "data.h5"
data = pd.read_hdf(p, key="data")

XX, yy = split(data)

print(XX.shape)
print(yy.shape)
# %% train30% validation20% test50% split
config = {"runOLS":0,
            "runOLS3":0,
            "runOLSH":0,
            "runENET":0,
            "runPCR":0,
            "runNN1":0,
            "runNN2":0,
            "runNN3":0,
            "runNN4":0,
            "runNN5":0,
            "runNN6":0,
          }
runOLS = config['runOLS']
runOLS3 = config['runOLS3']
runOLSH = config['runOLSH']
runENET = config['runENET']
runPCR = config['runPCR']
runNN1 = config['runNN1']
runNN2 = config['runNN2']
runNN3 = config['runNN3']
runNN4 = config['runNN4']
runNN5 = config['runNN5']
runNN6 = config['runNN6']
runNN = runNN1 + runNN2 + runNN3 + runNN4 + runNN5 + runNN6

runOLS3 = 0
runRF = 0
runGBRT = 1
# data index should be ticker date
container = {}
for year in tqdm(range(2013, 2018)):
    p_t = [str(1900), str(year)] # period of training
    p_v = [str(year+1), str(year + 2)] # period of valiation
    p_test = [str(year + 3), str(year+3)]

    _Xt, _yt = split(data.loc(axis=0)[:, p_t[0]:p_t[1]])
    _Xv, _yv = split(data.loc(axis=0)[:, p_v[0]:p_v[1]])
    _Xtest, _ytest = split(data.loc(axis=0)[:, p_test[0]:p_test[1]])

    #OLS
    if runOLS:
        model_name= "OLS"
        Xt = np.vstack((_Xt, _Xv))
        yt = np.vstack((_yt, _yv))
        Xtest, ytest = _Xtest, _ytest
        model_fit = LinearRegression(n_jobs=-1).fit(Xt, yt)

    if runOLS3:
        model_name = "OLS3"
        data_ols3 = data[['Factor46_mom12m', 'Factor51_mve', 'Factor09_bm', 'Y']]

        _Xt, _yt = split(data_ols3.loc(axis=0)[:, p_t[0]:p_t[1]])
        _Xv, _yv = split(data_ols3.loc(axis=0)[:, p_v[0]:p_v[1]])
        _Xtest, _ytest = split(data_ols3.loc(axis=0)[:, p_test[0]:p_test[1]])

        Xt = np.vstack((_Xt, _Xv))
        yt = np.vstack((_yt, _yv))
        Xtest, ytest = _Xtest, _ytest
        model_fit = LinearRegression().fit(Xt, yt)

    if runOLSH:  # OLS + H
        model_name = "OLSH"
        Xt = np.vstack((_Xt, _Xv))
        yt = np.vstack((_yt, _yv))
        Xtest, ytest = _Xtest, _ytest
        model_fit = HuberRegressor().fit(Xt, yt.reshape(-1, ))
        ytest_hat = model_fit.predict(Xtest).reshape(-1, 1)

        save_hat(container, model_name, ytest_hat, savehat=True)
        save_hat(container, model_name, ytest, savehat=False)

    if runENET:
        from sklearn.linear_model import ElasticNet
        model_name = "ENET"
        Xt, yt = _Xt, _yt
        Xv, yv = _Xv, _yv
        Xtest, ytest = _Xtest, _ytest

        lambda_ = [0.1, 0.0001]
        params = [{'lambda': i} for i in lambda_]

        out_cv = []
        for p in tqdm(params):
            model_fit = ElasticNet(alpha=p['lambda'], l1_ratio=0.5, random_state=0)
            model_fit.fit(Xt, yt.reshape(-1, ))

            yv_hat = model_fit.predict(Xv).reshape(-1, 1)
            perfor = cal_r2(yv, yv_hat)
            out_cv.append(perfor)
            print('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
            logging.info('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
        # tic = time.time()
        # print(f"{model} train time: ", tic - tis)
        best_p = params[np.argmax(out_cv)]
        print("best p", best_p)
        model_fit = ElasticNet(alpha=best_p['lambda'], l1_ratio=0.5, random_state=0)
        model_fit.fit(Xt, yt)
        ytest_hat = model_fit.predict(Xtest).reshape(-1, 1)
        best_perfor = cal_r2(ytest, ytest_hat)
        print(f"{model_name} oss r2:", best_perfor)

    if runPCR:
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
        ks = np.arange(1, maxk)
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

    if runNN:
        from keras.models import Sequential
        from keras.layers import Dense
        import keras

        Xt = np.vstack((_Xt, _Xv))
        yt = np.vstack((_yt, _yv))
        Xtest, ytest = _Xtest, _ytest

        layers = [
            Dense(32, input_dim=Xt.shape[1], activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(2, activation='relu')
        ]

        if runNN:
            def genNNmodel(i):
                layers_ = layers[:i].copy()
                layers_.append(Dense(1, name="output_layer"))
                # define the keras model
                model_fit = Sequential(
                    # keras.layers.BatchNormalization()
                    layers_)
                return model_fit

            if runNN1:
                i = 1
            elif runNN2:
                i = 2
            elif runNN3:
                i = 3
            elif runNN4:
                i = 4
            elif runNN5:
                i = 5
            elif runNN6:
                i = 6

            model_name = f"NN{i}"
            model_fit = genNNmodel(i)

            # compile the keras model
            opt = keras.optimizers.Adam(clipvalue=0.5)
            model_fit.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt, metrics=['mse'])
            # fit the keras model on the dataset
            model_fit.fit(Xt, yt, epochs=8, batch_size=640, verbose=1, workers=16)
            # make  predictions with the model
    if runRF:
        model_name = "RF"
        print("runing RF")
        Xt, yt = _Xt, _yt
        Xv, yv = _Xv, _yv
        Xtest, ytest = _Xtest, _ytest
        model_fit = tree_model(Xt, yt, Xv, yv, Xtest, ytest, runRF=True, runGBRT=False)

    if runGBRT:
        model_name = "GBRT"
        print("runing GBRT")
        Xt, yt = _Xt, _yt
        Xv, yv = _Xv, _yv
        Xtest, ytest = _Xtest, _ytest
        model_fit = tree_model(Xt, yt, Xv, yv, Xtest, ytest, runRF=False, runGBRT=True)

    yt_hat = model_fit.predict(Xt).reshape(-1, 1)
    save_hat(container, model_name, yt_hat, savekey='yt_hat')
    save_hat(container, model_name, yt, savekey='yt')

    ytest_hat = model_fit.predict(Xtest).reshape(-1, 1)
    save_hat(container, model_name, ytest_hat, savehat=True)
    save_hat(container, model_name, ytest, savehat=False)


r2oos = cal_model_r2(container, model_name)
print(f"{model_name} R2: ", "{0:.3%}".format(r2oos))
nr2oos = cal_model_r2(container, model_name, normal=True)
print(f"{model_name} Normal R2: ", "{0:.3%}".format(nr2oos))
r2is = cal_model_r2(container, model_name, oos=False, normal=False)
print(f"{model_name} IS R2: ", "{0:.3%}".format(r2is))
nr2is = cal_model_r2(container, model_name, oos=False, normal=True)
print(f"{model_name} ISN R2: ", "{0:.3%}".format(nr2is))

save_res(r2oos, nr2oos, r2is, nr2is)


# %% strategy ols
model_fit = LinearRegression().fit(X, y)

# %% strategy ols + H

model_fit = HuberRegressor(epsilon=3).fit(XX, yy.reshape(-1, ))
model_fit.score(XX, yy)

# %% enet
from sklearn.linear_model import ElasticNet
model_fit = ElasticNet(random_state=0)
model_fit.fit(XX, yy)
model_fit.predict(XX)
# %%
import pandas as pd
xdic={'X': {11: 300, 12: 170, 13: 288, 14: 360, 15: 319, 16: 330, 17: 520, 18: 345, 19: 399, 20: 479}}
ydic={'y': {11: 305000, 12: 270000, 13: 360000, 14: 370000, 15: 379000, 16: 405000, 17: 407500, 18: 450000, 19: 450000, 20: 485000}}
X=pd.DataFrame.from_dict(xdic)
y=pd.DataFrame.from_dict(ydic)
import numpy as np
X_seq = np.linspace(X.min(),X.max(),300).reshape(-1,1)
# %% GLM
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

import numpy as np

degree=9
poly = PolynomialFeatures(degree)
pl_XX = poly.fit_transform(XX)

# poly = PolynomialFeatures(interaction_only=True)
# polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
model_fit = LinearRegression()
model_fit.fit(pl_XX, yy)

# %% PLS and PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

runPLS = True

data = pd.read_pickle(Path('.') / 'data' / "yu_factor.pkl")
r2 = np.zeros((2, min(30, XX.shape[1])))
for k in tqdm(range(1, min(30, XX.shape[1]-1))):
    print(k)
    ytest_hats, yv_hats = np.array([]).reshape(-1, 1), np.array([]).reshape(-1, 1)
    yvs, ytests = np.array([]).reshape(-1, 1), np.array([]).reshape(-1, 1)

    data_pls = stream(data)
    for slice in data_pls:
        _Xt, _yt, _Xv, _yv, _Xtest, _ytest = slice
        Xt, yt = _Xt, _yt
        Xv, yv = _Xv, _yv
        Xtest, ytest = _Xtest, _ytest

        if runPLS:
            pls2 = PLSRegression(n_components=k).fit(Xt, yt)
            yv_hat = pls2.predict(Xv)

        ytest_hat = pls2.predict(Xtest)
        ytest_hats = np.vstack((ytest_hats, ytest_hat))
        ytests = np.vstack((ytests, ytest))

    print(cal_r2(yvs, yv_hats))
    r2[0, k] = cal_r2(yvs, yv_hats)
    print(cal_r2(ytests, ytest_hats))
    r2[1, k] = cal_r2(ytests, ytest_hats)

r2
# %% strategy PLS
from sklearn.cross_decomposition import PLSRegression
# XX = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
# Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
pls2 = PLSRegression(n_components=30)
pls2.fit(Xt, yt)
ytest_hat = pls2.predict(Xtest)
cal_r2(ytest, ytest_hat)
# %% strategy PCR
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pca = PCA(n_components=30)
X_reduced = pca.fit_transform(Xt)
model_fit = LinearRegression()
model_fit = model_fit.fit(X_reduced, yt)

# xtest_r = pca.transform(Xtest)
# ytest_hat = linear.predict(xtest_r)
# cal_r2(ytest, ytest_hat)

xtest_r = pca.transform(Xt)
ytest_hat = model_fit.predict(xtest_r)
cal_r2(yt, ytest_hat)

# %% RF and GBRT
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

runRF = 1
runGBRT = 0


if runRF:
    model_name = "Random Forest"
    # max_depth = np.arange(1, 7)
    # max_features = [3, 5, 10, 20, 30, 50]
    max_depth = np.arange(1, 3)
    max_features = [3, 5, 10]
    params = [{'max_dep': i, 'max_fea': j} for i in max_depth for j in max_features]
if runGBRT:
    model_name = "Boosting Trees"
    # boosting params
    num_trees = (np.arange(2, 12, 4)) * 100
    learning_rate = [0.01, 0.1]
    # subsample = [0.5, 0.6, 0.7, 0.8]
    subsample = [1]
    loss = ['huber']

    # cart tree params
    max_depth = [1, 2]

    params = []
    for t in num_trees:
        for d in max_depth:
            for l in learning_rate:
                for s in subsample:
                    for o in loss:
                        params.append({'num_trees': t, 'max_dep': d, 'lr': l, 'subsample': s, 'loss': o})

tis = time.time()
out_cv = []
for p in tqdm(params):
    if runRF:
        tree_m = RandomForestRegressor(n_estimators=300, max_depth=p['max_dep'], max_features=p['max_fea'],
                                       min_samples_split=10, random_state=0, n_jobs=-1)
    if runGBRT:
        tree_m = GradientBoostingRegressor(max_depth=p['max_dep'], n_estimators=p['num_trees'],
                                          learning_rate=p['lr'], max_features=p['max_dep'],
                                          min_samples_split=10, loss=p['loss'], min_samples_leaf=10,
                                          subsample=p['subsample'], random_state=0)
    tree_m.fit(Xt, yt.reshape(-1, ))
    yv_hat = tree_m.predict(Xv).reshape(-1, 1)
    perfor = cal_r2(yv, yv_hat)
    out_cv.append(perfor)

    print('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
    logging.info('params: ' + str(p) + '. CV r2-validation:' + str(perfor))

    # model.fit(train_X, train_y)
    # pred_cv = model.predict(cv_X)
    # perfor = r2_oos(cv_y, pred_cv)
tic = time.time()
print(f"{model_name} train time: ", tic - tis)

best_p = params[np.argmax(out_cv)]
if runRF:
    tree_m = RandomForestRegressor(n_estimators=300, max_depth=best_p['max_dep'], max_features=best_p['max_fea'],
                               min_samples_split=10, random_state=0, n_jobs=-1)
if runGBRT:
    tree_m = GradientBoostingRegressor(max_depth=best_p['max_dep'], n_estimators=best_p['num_trees'],
                                          learning_rate=best_p['lr'], max_features=best_p['max_dep'],
                                          min_samples_split=10, loss=best_p['loss'], min_samples_leaf=10,
                                          subsample=p['subsample'], random_state=0)
tree_m.fit(Xt, yt.reshape(-1, ))
ytest_hat = tree_m.predict(Xtest).reshape(-1, 1)
best_perfor = cal_r2(ytest, ytest_hat)

print(f"{model_name} oss r2:", cal_r2(ytest, ytest_hat))

# %% RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
model_fit = RandomForestRegressor(max_depth=2, random_state=0)
model_fit.fit(X, y)
# %% GBRT
from sklearn.ensemble import GradientBoostingRegressor
tis = time.time()
gbrt = GradientBoostingRegressor(n_estimators=300, max_depth=2, max_features=5,  random_state=0)
gbrt.fit(XX, yy.reshape(-1, ))
ytest_hat = gbrt.predict(XX).reshape(-1, 1)
tic = time.time()

print("IS gbrt r2:", cal_r2(yy, ytest_hat))
print("GBRT train time: ", tic - tis)

# %% NN
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization


train_val_split = int(XX.shape[0]*0.6)
val_test_split = train_val_split + int(XX.shape[0]*0.2)
Xt, yt = XX[:train_val_split, :], yy[:train_val_split, :]
Xv, yv = XX[train_val_split:val_test_split, :], yy[train_val_split:val_test_split, :]
Xtest, ytest = XX[val_test_split:, :], yy[val_test_split:, :]

import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout




i = 5
runGPU = 0
model_name = f"NN{i}"

model_cntn = []
for model_num in tqdm(range(5)):
    tf.random.set_seed(model_num)
    model_fit = genNNmodel(i)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=30,
                                                 verbose=1, mode="min", baseline=None, restore_best_weights=True)
    model_dir = Path("code", f"{model_name}")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_pt = model_dir / f"{model_name}_bm.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_pt, monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min')
    cb_list = [earlystop, checkpoint]


    def _loss_fn(true, hat):
        a = tf.math.reduce_sum(tf.math.square(true - hat))
        b = tf.math.reduce_sum(tf.math.square(true - tf.math.reduce_mean(true)))
        return -(1 - a / b)

    loss_fn = _loss_fn
    # loss_fn = tf.losses.MeanSquaredError()
    # compile the keras model
    opt = keras.optimizers.Adam(clipvalue=0.5)
    if runGPU:
        with tf.device('/device:GPU:0'):
            model_fit.compile(loss=loss_fn, optimizer=opt, metrics=['mse'])
            # fit the keras model on the dataset
            model_fit.fit(Xt, yt, epochs=100, batch_size=2560, verbose=0,
                          callbacks=cb_list, validation_data=(Xv, yv), validation_freq=2)
    else:
        model_fit.compile(loss=loss_fn, optimizer=opt, metrics=['mse'])
        # fit the keras model on the dataset
        model_fit.fit(Xt, yt, epochs=100, batch_size=2560, verbose=0,
                      callbacks=cb_list, validation_data=(Xv, yv), validation_freq=1)
    model_cntn.append(model_fit)

    import matplotlib.pyplot as plt

    plt.plot(model_fit.history.history['loss'], label='train')
    plt.plot(model_fit.history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()

    predictions = model_fit.predict(Xtest)
    cal_r2(ytest, predictions)



# %%
