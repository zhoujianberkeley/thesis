import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import joblib


def setwd():
    _paths = os.path.split(os.getcwd())
    if _paths[-1] == "code":
        os.chdir("..")
    elif os.getcwd() == '/content': # dir for colab
        os.chdir(Path("drive")/ "MyDrive"/"Colab Notebooks")
    print("working directory:", os.getcwd())

setwd()
# create logger with 'spam_application'
logger = logging.getLogger('records')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('records.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
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


def _cal_model_r2(years_dict, set_type):
    if set_type == 'oos':
        key1 = 'ytest'
        key2 = 'ytest_hat'
    elif set_type == 'valid':
        key1 = 'yv'
        key2 = 'yv_hat'
    elif set_type == 'is':
        key1 = 'yt'
        key2 = 'yt_hat'
    else:
        raise NotImplementedError()

    ytrues, yhats= [], []
    years_r2 = {}
    for key, value in years_dict.items():
        ytrue = value[key1]
        yhat = value[key2]
        years_r2[key] = cal_r2(ytrue, yhat)
        ytrues.append(ytrue)
        yhats.append(yhat)
    ytrues, yhats = np.concatenate(ytrues, axis=0), np.concatenate(yhats, axis=0)
    df = pd.DataFrame.from_dict(years_r2, orient='index', columns=['r2']).sort_values(by='r2', ascending=True)
    print(df)
    return cal_r2(ytrues, yhats), df


def cal_model_r2(container, model, set_type, normal=False):
    r2, r2df = _cal_model_r2(container[model], set_type)
    return r2, r2df


def save_arrays(container, model, year, to_save, savekey):
    if model not in container.keys():
        container[model] = {}
        return save_arrays(container, model, year, to_save, savekey)
    elif year not in container[model].keys():
        container[model][year] = {}
        return save_arrays(container, model, year, to_save, savekey)
    else:
        container[model][year][savekey] = to_save



def save_res(model_name, r2is, r2oos, nr2is=None, nr2oos=None):
    dir = Path("code", "model_result.csv")
    if not os.path.exists(dir):
        tmp = pd.DataFrame(columns=["r2oos"])
        tmp.index.name = "model"
        tmp.to_csv(dir, index=False)

    res = pd.read_csv(dir, index_col=0)
    res.index.name = "model"
    res.loc[model_name, "oos r2" ] = "{0:.2%}".format(r2oos)
    res.loc[model_name, "is r2"] = "{0:.2%}".format(r2is)
    if nr2is:   res.loc[model_name, "demean oos r2"] = "{0:.2%}".format(nr2oos)
    if nr2oos:  res.loc[model_name, "demean is r2"] = "{0:.2%}".format(nr2is)
    res.to_csv(dir)


def save_year_res(model_name, year, r2is, r2oos):
    dir = Path("code", "model_result_year.csv")
    if not os.path.exists(dir):
        tmp = pd.DataFrame(columns=["oos"])
        tmp.index.name = "model"
        tmp.to_csv(dir, index=False)

    res = pd.read_csv(dir, index_col=0)
    res.index.name = "model"

    res.loc[model_name + " is", str(year) ] = "{0:.2%}".format(r2is)
    res.loc[model_name + " is", "oos"] = 0

    res.loc[model_name + " oos", str(year)] = "{0:.2%}".format(r2oos)
    res.loc[model_name + " oos", "oos"] = 1

    res.to_csv(dir)


def gen_model_pt(name, yr):
    model_dir = Path("code", f"{name}")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_pt = model_dir / f"{yr}_model.joblib"
    return model_pt


def save_model(name, yr, to_save):
    model_pt = gen_model_pt(name, yr)
    joblib.dump(to_save, model_pt)


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
def _add_one_month(orig_date):
    # orig_date = '2015-02'
    # advance year and month by one month
    new_year, new_month = list(map(int, orig_date.split('-')))
    new_month += 1
    # note: in datetime.date, months go from 1 to 12
    if new_month > 12:
        new_year += 1
        new_month -= 12
    return f"{new_year}-{str(new_month).zfill(2)}"

def add_months(orig_date, n):
    for _ in range(n):
        orig_date = _add_one_month(orig_date)
    return orig_date

_add_one_month("2015-12")
add_months("2015-12", 13)
# %%

# def save_arrays(container, model, year, to_save, savehat=None, savekey=None):
#     if model not in container.keys():
#         container[model] = {}
#         container[model]['hat'] = np.array([]).reshape(-1, 1)
#         container[model]['true'] = np.array([]).reshape(-1, 1)
#         return save_arrays(container, model, year, to_save, savehat, savekey)
#     elif year not in container[model].keys():
#         container[model][year] = {}
#         return save_arrays(container, model, year, to_save, savehat, savekey)
#
#     elif savekey is not None:
#         if savekey not in container[model].keys():
#             container[model][savekey] = np.array([]).reshape(-1, 1)
#             return save_arrays(container, model, to_save, savehat, savekey)
#         else:
#             container[model][savekey] = np.vstack((container[model][savekey], to_save))
#             return
#     elif savehat is True:
#         container[model]['hat'] = np.vstack((container[model]['hat'], to_save))
#         return
#     elif savehat is False:
#         container[model]['true'] = np.vstack((container[model]['true'], to_save))
#         return
#     else:
#         raise NotImplementedError()