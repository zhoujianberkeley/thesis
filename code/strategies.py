# %%
import numpy as np
import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.datasets import make_regression

# %%

p = Path('.') / 'data' / "factor.pkl"
data = pd.read_pickle(p).dropna()

y = data['pct_chg'].values
X = data[['open', 'high', 'low', 'close', 'pre_close', 'change', 'vol', 'amount']].values
print(y.shape)
X.shape
# %% train validation test split
for year in range(2008, 2020):
    train = range(2001, year)
    validation = range(year, year+18)

Xtrain
Xvalid
Xtest
# %% strategy ols

linear = LinearRegression().fit(X, y)

# %% strategy ols + H
huber = HuberRegressor().fit(X, y)
huber.score(X, y)

# %% strategy PLS
from sklearn.cross_decomposition import PLSRegression
X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
pls2 = PLSRegression(n_components=2)
pls2.fit(X, Y)

Y_pred = pls2.predict(X)


# %% strategy PCR
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

pca = PCA()
X_reduced = pca.fit_transform(scale(X))
regr = LinearRegression()

LinearRegression().fit(X_reduced, y)


# %% enet
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0)
regr.fit(X, y)


# %% GLM

# %% RF
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
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


