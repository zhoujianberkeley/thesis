# %%
import numpy as np
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
import time
from tqdm import tqdm
import datetime

# %%
def split(data):
    features = data.columns.to_list()
    features.remove("pct_chg")
    # features.remove("change")
    # features.remove("name")

    X = data[features].values
    # X = data[['pre_close', 'open', 'high', 'low', 'close',
    #           'avg_price', 'turnover', 'PE', 'PB', 'PS', 'PCF']].values
    y = data['pct_chg'].values.reshape(-1, 1)
    return X, y

def cal_r2(true, hat):
    if len(hat.shape) == 1 and true.shape[0] == hat.shape[0]:
        hat = hat.reshape(-1, 1)
    assert true.shape == hat.shape, "shape not match"
    a = np.square(true-hat).sum()
    b = np.square(true).sum()
    return 1 - a/b

def cal_model_r2(container, model):
    ytrue = container[model]['true']
    yhat = container[model]['hat']
    return cal_r2(ytrue, yhat)

def save_hat(container, model, to_save, savehat):
    if model not in container.keys():
        container[model] = {}
        container[model]['hat'] = np.array([]).reshape(-1, 1)
        container[model]['true'] = np.array([]).reshape(-1, 1)
    if savehat:
        container[model]['hat'] = np.vstack((container[model]['hat'], to_save))
    else:
        container[model]['true'] = np.vstack((container[model]['true'], to_save))

def stream(data):
    for year in range(2013, 2018):
        p_t = [str(1900), str(year)]
        p_v = [str(year + 1), str(year + 2)]
        p_test = [str(year + 3), str(year + 3)]

        _Xt, _yt = split(data.loc(axis=0)[:, p_t[0]:p_t[1]])
        _Xv, _yv = split(data.loc(axis=0)[:, p_v[0]:p_v[1]])
        _Xtest, _ytest = split(data.loc(axis=0)[:, p_test[0]:p_test[1]])
        yield _Xt, _yt, _Xv, _yv, _Xtest, _ytest

# %%
p = Path('.') / 'data' / "yu_factor.pkl"
data = pd.read_pickle(p)

XX, yy = split(data)

print(XX.shape)
print(yy.shape)
# %% train30% validation20% test50% split

runOLS = 0
runOLSH = 0
runENET = 0
runPCR = 0
runNN = 1

data = pd.read_pickle(Path('.') / 'data' / "yu_factor.pkl")
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
        model="OLS"
        Xt = np.vstack((_Xt, _Xv))
        yt = np.vstack((_yt, _yv))
        Xtest, ytest = _Xtest, _ytest
        linear = LinearRegression().fit(Xt, yt)
        ytest_hat = linear.predict(Xtest)

        save_hat(container, model, ytest_hat, savehat=True)
        save_hat(container, model, ytest, savehat=False)

    if runOLSH:  # OLS + H
        model = "OLSH"
        Xt = np.vstack((_Xt, _Xv))
        yt = np.vstack((_yt, _yv))
        Xtest, ytest = _Xtest, _ytest
        huber = HuberRegressor().fit(Xt, yt.reshape(-1, ))
        ytest_hat = huber.predict(Xtest).reshape(-1, 1)

        save_hat(container, model, ytest_hat, savehat=True)
        save_hat(container, model, ytest, savehat=False)

    if runENET:
        from sklearn.linear_model import ElasticNet
        model = "ENET"
        Xt, yt = _Xt, _yt
        Xv, yv = _Xv, _yv
        Xtest, ytest = _Xtest, _ytest

        lambda_ = [0.1, 0.0001]
        params = [{'lambda': i} for i in lambda_]

        out_cv = []
        for p in tqdm(params):
            regr = ElasticNet(alpha=p['lambda'], l1_ratio=0.5, random_state=0)
            regr.fit(Xt, yt.reshape(-1, ))

            yv_hat = regr.predict(Xv).reshape(-1, 1)
            perfor = cal_r2(yv, yv_hat)
            out_cv.append(perfor)
            print('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
            logging.info('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
        # tic = time.time()
        # print(f"{model} train time: ", tic - tis)
        best_p = params[np.argmax(out_cv)]
        regr = ElasticNet(alpha=best_p['lambda'], l1_ratio=0.5, random_state=0)
        regr.fit(Xt, yt)
        ytest_hat = regr.predict(Xtest).reshape(-1, 1)
        best_perfor = cal_r2(ytest, ytest_hat)
        print(f"{model} oss r2:", best_perfor)

        save_hat(container, model, ytest_hat, savehat=True)
        save_hat(container, model, ytest, savehat=False)
    if runPCR:
        model = "PCR"
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
            regr = LinearRegression()
            linear = regr.fit(X_reduced, yt)

            xv_r = pca.transform(Xv)
            yv_hat = linear.predict(xv_r)
            perfor = cal_r2(yv, yv_hat)
            out_cv.append(perfor)
            print('params: ' + str(p) + '. CV r2-validation:' + str(perfor))
            logging.info('params: ' + str(p) + '. CV r2-validation:' + str(perfor))

        best_p = params[np.argmax(out_cv)]
        # xx = Z[:, :best_p['k']]
        # b = np.linalg.inv(xx.T @ xx) @ (xx.T @ yt)
        # bf = p1[:, :best_p['k']] @ b
        # ytest_hat = (Xtest @ bf + mtrain).reshape(-1, 1)
        pca = PCA(n_components=best_p['k'])
        X_reduced = pca.fit_transform(Xt)
        regr = LinearRegression()
        linear = regr.fit(X_reduced, yt)

        xtest_r = pca.transform(Xtest)
        ytest_hat = linear.predict(xtest_r)
        best_perfor = cal_r2(ytest, ytest_hat)
        print(f"{model} oss r2:", best_perfor)

        save_hat(container, model, ytest_hat, savehat=True)
        save_hat(container, model, ytest, savehat=False)

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

        for i in range(1, 6):
            modelname = f"NN{i}"
            print(modelname)

            layers_ = layers[:i].copy()
            layers_.append(Dense(1, name="output_layer"))
            # define the keras model
            model = Sequential(
                # keras.layers.BatchNormalization()
                layers_
            )

            # compile the keras model
            opt = keras.optimizers.Adam(clipvalue=0.5)
            model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt, metrics=['mse'])
            # fit the keras model on the dataset
            model.fit(Xt, yt, epochs=2, batch_size=640, verbose=1, workers=16)
            # make class predictions with the model
            ytest_hat = model.predict(Xtest, verbose=1)

            save_hat(container, modelname, ytest_hat, savehat=True)
            save_hat(container, modelname, ytest, savehat=False)

if runNN:
    for i in range(1, 6):
        modelname = f"NN{i}"
        r2oos = cal_model_r2(container, modelname)
        print(f"{modelname} R2: ", r2oos)
if runOLS:
    r2oos = cal_model_r2(container, "OLS")
    print("OLS R2: ", r2oos)
if runOLSH:
    r2oos = cal_model_r2(container, "OLSH")
    print("OLS + H R2: ", r2oos)
if runENET:
    r2oos = cal_model_r2(container, "ENET")
    print("Elastic Net R2: ", r2oos)
if runPCR:
    r2oos = cal_model_r2(container, "PCR")
    print("PCR R2: ", r2oos)

# %% strategy ols
linear = LinearRegression().fit(X, y)

# %% strategy ols + H
huber = HuberRegressor(epsilon=1.1, max_iter=100).fit(XX, yy.reshape(-1, ))
huber.score(XX, yy)

# %% enet
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0)
regr.fit(XX, yy)
regr.predict(XX)
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
linear = LinearRegression()
linear.fit(pl_XX,yy)

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
regr = LinearRegression()
linear = regr.fit(X_reduced, yt)

# xtest_r = pca.transform(Xtest)
# ytest_hat = linear.predict(xtest_r)
# cal_r2(ytest, ytest_hat)

xtest_r = pca.transform(Xt)
ytest_hat = linear.predict(xtest_r)
cal_r2(yt, ytest_hat)

# %% RF and GBRT
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

runRF = 1
runGBRT = 0

if runRF:
    model = "Random Forest"
    # max_depth = np.arange(1, 7)
    # max_features = [3, 5, 10, 20, 30, 50]
    max_depth = np.arange(1, 3)
    max_features = [3, 5, 10]
    params = [{'max_dep': i, 'max_fea': j} for i in max_depth for j in max_features]
if runGBRT:
    model = "Boosting Trees"
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
print(f"{model} train time: ", tic - tis)

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

print(f"{model} oss r2:", cal_r2(ytest, ytest_hat))

# %% RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
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
# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
# load the dataset
# split into input (X) and output (y) variables
X = XX
y = yy
# define the keras model

layers = [
    Dense(32, input_dim=Xt.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(2, activation='relu')
]
layers_ = layers
layers_.append(Dense(1, name="output_layer"))
# model.add(Dense(1))
model = Sequential(layers_)
opt = keras.optimizers.Adam(clipvalue=1)
# compile the keras model
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
# fit the keras model on the dataset
model.fit(X, y, epochs=5, batch_size=640, verbose=1, workers=16)
# make class predictions with the model
predictions = model.predict(X, verbose=1)
# summarize the first 5 cases
# for i in range(5):
#     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
cal_r2(y, predictions)

# %%
# monthly out-of-sample R2

# annual out-of-sample R2

# model complexity

# feature importance


