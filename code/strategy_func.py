import time
from tqdm import tqdm
from multiprocessing import cpu_count
import numpy as np
import joblib
import os
import logging

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout

from utils_stra import setwd, cal_r2, gen_model_pt
setwd()

# create logger with 'spam_application'
logger = logging.getLogger('records')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('records.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
# %%
def tree_model_fast(model_name, year, Xt, yt, Xv, yv, runRF, runGBRT):
    assert runRF + runGBRT == 1
    model_pt = gen_model_pt(model_name, year)
    if not os.path.exists(model_pt):
        print(f"can't find trained model {model_name} {year}, retraining")
        return tree_model(Xt, yt, Xv, yv, runRF, runGBRT)
    else:
        print(f"load model from {model_pt}")
        if runRF:
            tree_m = joblib.load(model_pt)
        elif runGBRT:
            tree_m = xgb.XGBRegressor()
            tree_m.load_model(model_pt)
        return tree_m

def tree_model(Xt, yt, Xv, yv, runRF, runGBRT):
    assert runRF + runGBRT == 1
    if runRF:
        model_name = "Random Forest"
        max_depth = np.arange(2, 7, 2)
        max_features = [Xt.shape[1]//3]
        # max_depth = np.arange(1, 3)
        # max_features = [3, 5, 10]
        param_grid = {'max_dep': max_depth,
                      'max_fea':max_features}
        params = list(ParameterGrid(param_grid))
    elif runGBRT:
        model_name = "Boosting Trees"
        # boosting params
        num_trees = [1000]
        learning_rate = [0.01, 0.1]
        # loss = ['huber']
        # cart tree params
        max_depth = [1, 2]
        param_grid = {'num_trees': num_trees,
                      'max_dep': max_depth,
                      'lr':learning_rate}
        params = list(ParameterGrid(param_grid))

    tis = time.time()
    scores = []
    model_list = []
    for p in tqdm(params):
        print(p)
        if runRF:
            tree_m = RandomForestRegressor(n_estimators=300, max_depth=p['max_dep'], max_features=p['max_fea'],
                                           min_samples_split=10, random_state=0, n_jobs=cpu_count()-2)
            tree_m.fit(Xt, yt.reshape(-1, ))
        elif runGBRT:
            tree_m = xgb.XGBRegressor(n_estimators=p['num_trees'], max_depth=p['max_dep'], learning_rate=p['lr'],
                                      objective='reg:pseudohubererror', random_state=0, n_jobs=cpu_count()-2)

            tree_m.fit(Xt, yt.reshape(-1, ), early_stopping_rounds=0.1*p['num_trees'],
                                      eval_set=[(Xv, yv.reshape(-1, ))], verbose=False)
            print(f"gbrt best iter {tree_m.best_iteration}   best score {tree_m.best_score}")
            # tree_m = GradientBoostingRegressor(max_depth=p['max_dep'], n_estimators=p['num_trees'],
            #                                    learning_rate=p['lr'],
            #                                    min_samples_split=10, loss=p['loss'], min_samples_leaf=10,
            #                                    subsample=p['subsample'], random_state=0)
        yv_hat = tree_m.predict(Xv).reshape(-1, 1)
        score = cal_r2(yv, yv_hat)
        print('params: ' + str(p) + '. CV r2-validation:' + "{0:.3%}".format(str(score)))
        scores.append(score)
        model_list.append(tree_m)
    tic = time.time()
    print(f"{model_name} train time: ", tic - tis)

    best_p = params[np.argmax(scores)]
    best_model = model_list[np.argmax(scores)]
    logger.info('best params for rf: ' + str(best_p))
    # if runRF:
    #     tree_m = RandomForestRegressor(n_estimators=300, max_depth=best_p['max_dep'], max_features=best_p['max_fea'],
    #                                    min_samples_split=10, random_state=0, n_jobs=-1)
    # elif runGBRT:
    #     tree_m = xgb.XGBRegressor(n_estimators=best_p['num_trees'], max_depth=best_p['max_dep'], learning_rate=best_p['lr'],
    #                               objective='reg:pseudohubererror', random_state=0, n_jobs=-1)
    # tree_m.fit(Xt, yt.reshape(-1, ))
    tree_m = best_model
    return tree_m


def genNNmodel(dim, layers_num):
    assert layers_num in range(1, 7)
    if layers_num<=5:
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
        layers_ = layers_[:1+4*layers_num].copy()
    elif layers_num == 6:
        layers_ = [
            Dropout(0.2),

            BatchNormalization(input_dim=dim),
            Dense(128),
            LeakyReLU(alpha=0.01),
            Dropout(0.5),

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

    layers_.append(BatchNormalization())
    layers_.append(Dense(1, name="output_layer"))
    model_fit = Sequential(
        layers_)
    return model_fit


def _loss_fn(true, hat):
    a = tf.math.reduce_sum(tf.math.square(true - hat))
    b = tf.math.reduce_sum(tf.math.square(true - tf.math.reduce_mean(true)))
    return -(1 - a / b)
