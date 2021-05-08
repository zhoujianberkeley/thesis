# %%
import datetime
import os
import pickle
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
import jqdatasdk as jq
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils_stra import setwd, filter_data, cal_r2

setwd()

def connect_jq():
    jq.auth('13670286853', 'xiaojianbao1996')
    print(jq.get_query_count())

connect_jq()
# %%
def load_weight(index_name, reload=False):
    save_dir = Path('data') / 'buffer' / f"{index_name}_weight.pkl"
    if not reload:
        if os.path.exists(save_dir):
            with open(save_dir, "rb") as f:
                weight_dict = pickle.load(f)
            return weight_dict
        else:
            return load_weight(index_name, reload=True)
    else:
        weight_dict = {}
        for tp in tqdm(pd.date_range('20131231', '20200831', freq='M')):
            print(tp)
            tp = tp.strftime("%Y-%m-%d")
            weight = jq.get_index_weights(index_name, date=tp)
            weight_dict[tp] = weight
        with open(save_dir, "wb") as f:
            pickle.dump(weight_dict, f)
        return weight_dict


def load_price(index_name, reload=False):
    save_dir = Path('data') / 'buffer' / f"{index_name}_return.csv"
    if not reload:
        if os.path.exists(save_dir):
            price = pd.read_csv(save_dir, index_col=0, parse_dates=[0], infer_datetime_format=True)
            return price
        else:
            return load_price(index_name, reload=True)
    else:
        weight_dict = load_weight(index_name)
        price = pd.DataFrame()
        for dt, weight in tqdm(weight_dict.items()):
            end_dt = datetime.datetime.strptime(dt, '%Y-%m-%d') + relativedelta(months=+1) + pd.offsets.MonthEnd(0)
            price_df = jq.get_bars(weight.index.to_list(),2,unit='1M',fields=['date','close'],end_dt=end_dt, include_now=True)
            change_df = pd.DataFrame(price_df.groupby(level=0)['close'].pct_change()).loc[pd.IndexSlice[:, 1], pd.IndexSlice[:]].droplevel(1)
            change_df = change_df.rename({'close':'pct_change'}, axis=1)

            df = weight.merge(change_df, left_index=True, right_index=True)
            assert df.shape[0] == 300
            price.loc[end_dt, 'predict_return'] = (df['weight']*df['pct_change']).sum()
        price.to_csv(save_dir)
        return price


def exact_index(index_name):
    price = load_price(index_name)
    bm = jq.get_bars(index_name,82,unit='1M',fields=['date','close'],end_dt = '2020-09-30', include_now=True).set_index('date')
    bm = bm.pct_change().iloc[1:, :].rename({'close':'index_return'}, axis=1)
    bm.index = bm.index + pd.offsets.MonthEnd(0)
    combine = price.merge(bm, left_index=True, right_index=True)
    combine['predict_return'] = combine['predict_return']/100
    combine.plot()
    plt.show()
    return price

# %% machine learning
def renamer(str_list):
    def _renamer(ticker):
        if ticker[-2:] == 'SZ':
            return ticker[0:6] + '.XSHE'
        elif ticker[-2:] == 'SH':
            return ticker[0:6] + '.XSHG'
        else:
            raise NotImplementedError()
    return list(map(_renamer, str_list))

def rep_index(index_name, factor_df, model):
    weight_dict = load_weight(index_name)

    res = pd.DataFrame()
    for dt, weight in tqdm(weight_dict.items()):
        weight = weight_dict[dt]
        end_dt = datetime.datetime.strptime(dt, '%Y-%m-%d') + relativedelta(months=+1) + pd.offsets.MonthEnd(0)
        if end_dt < factor_df.index.get_level_values('date').min():
            continue
        factor = factor_df.loc[pd.IndexSlice[:, end_dt.strftime('%Y-%m-%d')], pd.IndexSlice[:]].reset_index(level=1)
        factor.index = renamer(factor.index.to_list())
        df = weight.merge(factor, how="left", left_index=True, right_index=True, suffixes=("_w", "_f"))
        # print("missing value in prediciton:", df.shape[0] - df.predict.count())
        res.loc[end_dt, 'predict_return'] = (df['weight'] * df['predict']).sum()

    bm = jq.get_bars(index_name, 82, unit='1M', fields=['date','close'],end_dt = '2020-09-30', include_now=True).set_index('date')
    bm = bm.pct_change().iloc[1:, :].rename({'close':'true_return'}, axis=1)
    bm.index = bm.index + pd.offsets.MonthEnd(0)
    combine = res.merge(bm, left_index=True, right_index=True)
    combine['predict_return'] = combine['predict_return']/100
    combine.plot()
    plt.title(f"{model}")
    plt.show()
    stat, stat_df = cal_index_stat(combine, model)
    return combine, stat, stat_df

def cal_index_stat(df, model):
    stat = {}
    r2 = cal_r2(df['true_return'].values, df['predict_return'].values)
    stat['r2'] = "{0:.3%}".format(r2)
    mean_r2 = -15.119/100
    stat['r2 mean'] = "{0:.3%}".format(1 - (1 - r2)/(1 - mean_r2))
    stat_df = pd.DataFrame.from_dict(stat, orient='index').T
    stat_df.index = [model]
    print(stat_df)
    return stat, stat_df


# exact_index("000300.XSHG")

pre_dir = "Filter IPO"
stat_lt = []
names = ["NN1 M", "MEAN"]
for model_name in names:

    ml_fctr = pd.read_csv(Path('code') / pre_dir /model_name / "predictions.csv", parse_dates=["date"], infer_datetime_format=True).set_index(["ticker", "date"])
    ml_fctr = ml_fctr.dropna(how='all')
    ml_fctr = filter_data(ml_fctr, ["IPO"])
    combine, stat, stat_df = rep_index("000300.XSHG", ml_fctr, model_name)
    stat_lt.append(stat_df)

print(pd.concat(stat_lt, axis=0))

# %%
