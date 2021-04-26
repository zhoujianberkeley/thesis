import datetime
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import joblib
from functools import reduce


def setwd():
    _paths = os.path.split(os.getcwd())
    if _paths[-1] == "code":
        os.chdir("..")
    elif os.getcwd() == '/content': # dir for colab
        os.chdir(Path("drive")/ "MyDrive"/"Colab Notebooks")
    # print("working directory:", os.getcwd())

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
    print("*****", set_type, "*****")
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


def gen_model_pt(name, yr, pre_dir, **kwargs):
    if pre_dir == "":
        model_dir = Path("code", f"{name}")
    else:
        model_dir = Path("code", pre_dir, f"{name}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if "runNN" in kwargs and kwargs["runNN"]:
        model_num = kwargs["model_num"]
        model_pt = model_dir / f"{yr}_iter{model_num}_bm.hdf5"
    else:
        model_pt = model_dir / f"{yr}_model.joblib"
    return model_pt

def _load_model(name, yr, pre_dir):
    model_pt = gen_model_pt(name, yr, pre_dir)
    return joblib.load(model_pt)

def _save_model(name, yr, to_save, pre_dir):
    model_pt = gen_model_pt(name, yr, pre_dir)
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

def gen_filterIPO(data, file_pt):
    if not os.path.exists(file_pt):
        token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        list_date = pro.stock_basic(exchange='',list_status='L')
        list_date['list_date'] = pd.to_datetime(list_date['list_date'])
        list_date = list_date.rename({"ts_code":"ticker"}, axis=1).set_index("ticker")[["list_date"]]
        list_date.to_csv(file_pt)
    else:
        list_date = pd.read_csv(file_pt, index_col=0, parse_dates=['list_date'], infer_datetime_format=True)

    tmp = data.merge(list_date, how="left", left_on="ticker", right_index=True)
    tmp["age"] = tmp.index.get_level_values('date') - tmp['list_date']
    filter_ipo = tmp["age"] > datetime.timedelta(days=180)
    return filter_ipo

def filter_data(input, fltr_lst):
    res = []
    for fltr_type in fltr_lst:
        if fltr_type == "IPO":
            filter_IPO = gen_filterIPO(input, file_pt=Path('.') / 'data' / "list_date.csv")
            res.append(filter_IPO)
        else:
            raise NotImplementedError
    filter = reduce(lambda x,y: x&y, res)
    output = input[filter]
    return output
# %%