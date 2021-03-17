# %%
import numpy as np
import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.datasets import make_regression
from tqdm import tqdm
import datetime

# %%
p = Path('.') / 'data' / "factor.pkl"
data = pd.read_pickle(p)

XX, yy = split(data)

print(XX.shape)
print(yy.shape)
# %%
def split(data):
    features = data.columns.to_list()
    features.remove("pct_chg")
    features.remove("change")
    features.remove("name")

    X = data[features].values
    # X = data[['pre_close', 'open', 'high', 'low', 'close',
    #           'avg_price', 'turnover', 'PE', 'PB', 'PS', 'PCF']].values
    y = data['pct_chg'].values.reshape(-1, 1)
    return X, y

def cal_r2(true, hat):
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

# %% train30% validation20% test50% split

runOLS = False
runOLSH = False
runENET = True

data = pd.read_pickle(Path('.') / 'data' / "factor.pkl")
container = {}
for year in tqdm(range(2013, 2018)):
    p_t = [str(1900), str(year)]
    p_v = [str(year+1), str(year + 2)]
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
        Xt = np.vstack((_Xt, _Xv))
        yt = np.vstack((_yt, _yv))
        Xtest, ytest = _Xtest, _ytest

        regr = ElasticNet(random_state=0)
        regr.fit(Xt, yt)
        ytest_hat = regr.predict(Xtest).reshape(-1, 1)

        save_hat(container, model, ytest_hat, savehat=True)
        save_hat(container, model, ytest, savehat=False)

if runOLS:
    r2oos = cal_model_r2(container, "OLS")
    print("OLS R2: ", r2oos)
if runOLSH:
    r2oos = cal_model_r2(container, "OLSH")
    print("OLS + H R2: ", r2oos)
if runENET:
    r2oos = cal_model_r2(container, "ENET")
    print("Elastic Net R2: ", r2oos)
# %% strategy ols
linear = LinearRegression().fit(X, y)

# %% strategy ols + H
huber = HuberRegressor().fit(X, y)
huber.score(X, y)

# %% strategy PLS
from sklearn.cross_decomposition import PLSRegression

X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
pls2 = PLSRegression(n_components=5)
pls2.fit(Xt, yt)
Y_pred = pls2.predict(Xtest)

# %% PLS and PCR
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

runPLS = False
runPCR = True
data = pd.read_pickle(Path('.') / 'data' / "factor.pkl")

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
        if runPCR:
            pca = PCA(n_components=k)
            X_reduced = pca.fit_transform(scale(Xt))
            regr = LinearRegression()
            linear = LinearRegression().fit(X_reduced, yt)
            # todo finish PCA

        yv_hats = np.vstack((yv_hats, yv_hat))
        yvs = np.vstack((yvs, yv))

        ytest_hat = pls2.predict(Xtest)
        ytest_hats = np.vstack((ytest_hats, ytest_hat))
        ytests = np.vstack((ytests, ytest))

    print(cal_r2(yvs, yv_hats))
    r2[0, k] = cal_r2(yvs, yv_hats)
    print(cal_r2(ytests, ytest_hats))
    r2[1, k] = cal_r2(ytests, ytest_hats)

r2

# %% strategy PCR

pca = PCA()
X_reduced = pca.fit_transform(scale(X))
regr = LinearRegression()

LinearRegression().fit(X_reduced, y)


# %% enet
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0)
regr.fit(XX, yy)
regr.predict(XX)

# %% GLM

# %% RF
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=300, max_depth=2, max_features=50, random_state=0)
regr.fit(X, y)
print(regr.predict([[0, 0, 0, 0]]))

# %% GBRT
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)

# %% NN

# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


# %%
# monthly out-of-sample R2

# annual out-of-sample R2

# model complexity

# feature importance


