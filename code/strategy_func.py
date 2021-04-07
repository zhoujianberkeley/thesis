from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import time
from tqdm import tqdm
import numpy as np
import logging

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout

from utils_stra import setwd, cal_r2
setwd()

# create logger with 'spam_application'
logger = logging.getLogger('records')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('records.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
# %%
def tree_model(Xt, yt, Xv, yv, runRF, runGBRT):
    assert runRF + runGBRT == 1
    if runRF:
        model_name = "Random Forest"
        max_depth = np.arange(2, 7, 2)
        max_features = [Xt.shape[1]//3]
        # max_depth = np.arange(1, 3)
        # max_features = [3, 5, 10]
        params = [{'max_dep': i, 'max_fea': j} for i in max_depth for j in max_features]
    elif runGBRT:
        model_name = "Boosting Trees"
        # boosting params
        num_trees = (np.arange(2, 12, 4)) * 100
        learning_rate = [0.01, 0.1]
        # loss = ['huber']
        # cart tree params
        max_depth = [1, 2]
        from sklearn.model_selection import ParameterGrid
        param_grid = {'num_trees': num_trees,
                      'max_depth': max_depth,
                      'lr':learning_rate,}
        params = list(ParameterGrid(param_grid))

        # params.append({'num_trees': t, 'max_dep': d, 'lr': l, 'subsample': s, 'loss': o})

    tis = time.time()
    out_cv = []
    for p in tqdm(params):
        if runRF:
            tree_m = RandomForestRegressor(n_estimators=300, max_depth=p['max_dep'], max_features=p['max_fea'],
                                           min_samples_split=10, random_state=0, n_jobs=-1)
        elif runGBRT:
            tree_m = xgb.XGBRegressor(n_estimators=p['num_trees'], max_depth=p['max_dep'], learning_rate=p['lr'],
                                      objective='reg:pseudohubererror')
            tree_m.fit(Xt, yt.reshape(-1, ))


            tree_m = GradientBoostingRegressor(max_depth=p['max_dep'], n_estimators=p['num_trees'],
                                               learning_rate=p['lr'],
                                               min_samples_split=10, loss=p['loss'], min_samples_leaf=10,
                                               subsample=p['subsample'], random_state=0)
        tree_m.fit(Xt, yt.reshape(-1, ))
        yv_hat = tree_m.predict(Xv).reshape(-1, 1)
        perfor = cal_r2(yv, yv_hat)
        out_cv.append(perfor)
        print('params: ' + str(p) + '. CV r2-validation:' + str(perfor))

    tic = time.time()
    print(f"{model_name} train time: ", tic - tis)

    best_p = params[np.argmax(out_cv)]
    logger.critical('best params for rf: ' + str(best_p))

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


def genNNmodel(dim, layers_num, dropout=False):
    if dropout:
        # now add a ReLU layer explicitly:
        layers_ = [
            Dropout(0.2),
            BatchNormalization(input_dim=dim),
            Dense(64),
            LeakyReLU(alpha=0.01),
            Dropout(0.5),

            BatchNormalization(),
            Dense(32),
            LeakyReLU(alpha=0.01),
            Dropout(0.5),

            BatchNormalization(),
            Dense(16),
            LeakyReLU(alpha=0.01),
            Dropout(0.5),

            BatchNormalization(),
            Dense(8),
            LeakyReLU(alpha=0.01),
            Dropout(0.5),

            BatchNormalization(),
            Dense(4),
            LeakyReLU(alpha=0.01),
            Dropout(0.5),
        ]
    elif layers_num == 6:
        layers_ = [
        Dense(64, input_dim=dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu')
        ]
    else:
        layers_ = [
            BatchNormalization(input_dim=dim),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(2, activation='relu')
        ]
        layers_ = layers_[:layers_num].copy()

    layers_.append(BatchNormalization())
    layers_.append(Dense(1, name="output_layer"))
    model_fit = Sequential(
        layers_)
    return model_fit


def _loss_fn(true, hat):
    a = tf.math.reduce_sum(tf.math.square(true - hat))
    b = tf.math.reduce_sum(tf.math.square(true - tf.math.reduce_mean(true)))
    return -(1 - a / b)
