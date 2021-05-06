# %%
import datetime
import pickle
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
import jqdatasdk as jq
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils_stra import setwd, filter_data
setwd()

# %%
def connect_jq():
    jq.auth('13670286853', 'xiaojianbao1996')
    print(jq.get_query_count())

connect_jq()
# %%
def load_weight(index_name):
    weight_dict = {}
    for tp in tqdm(pd.date_range('20131231', '20200831', freq='M')):
        print(tp)
        tp = tp.strftime("%Y-%m-%d")
        weight = jq.get_index_weights(index_name, date=tp)
        # weight.reset_index().set_index('date')
        weight_dict[tp] = weight

    with open(Path('data') / 'buffer' / f"{index_name}_weight.pkl", "wb") as f:
        pickle.dump(weight_dict, f)
    return weight_dict

# get weights for SH300 and ZZ800 index
index_name = '000300.XSHG'
weight_dict = load_weight(index_name)

# %%
with open(Path('data') / 'buffer' / f"{index_name}_weight.pkl", "rb") as f:
    weight_dict = pickle.load(f)

res = pd.DataFrame()
for dt, weight in tqdm(weight_dict.items()):
    # weight = weight_dict[dt]
    end_dt = datetime.datetime.strptime(dt, '%Y-%m-%d') + relativedelta(months=+1) + pd.offsets.MonthEnd(0)
    price_df = jq.get_bars(weight.index.to_list(),2,unit='1M',fields=['date','close'],end_dt=end_dt, include_now=True)
    change_df = pd.DataFrame(price_df.groupby(level=0)['close'].pct_change()).loc[pd.IndexSlice[:, 1], pd.IndexSlice[:]].droplevel(1)
    change_df = change_df.rename({'close':'pct_change'}, axis=1)

    df = weight.merge(change_df, left_index=True, right_index=True)
    assert df.shape[0] == 300
    res.loc[end_dt, 'return'] = (df['weight']*df['pct_change']).sum()

res.to_csv(Path('data') / 'buffer' / f"{index_name}_return.csv")
# %%
res = pd.read_csv(Path('data') / 'buffer' / f"{index_name}_return.csv", index_col=0, parse_dates=[0], infer_datetime_format=True)
bm = jq.get_bars('000300.XSHG',82,unit='1M',fields=['date','close'],end_dt = '2020-09-30', include_now=True).set_index('date')
bm = bm.pct_change().iloc[1:, :].rename({'close':'index_return'}, axis=1)
bm.index = bm.index + pd.offsets.MonthEnd(0)
combine = res.merge(bm, left_index=True, right_index=True)
combine['return'] = combine/100
combine.plot()
plt.show()

# %% machine learning
pre_dir = "Filter IPO"

# model_name = "NN1 M"
model_name = "PCR M"
ml_fctr = pd.read_csv(Path('code') / pre_dir /model_name / "predictions.csv", parse_dates=["date"], infer_datetime_format=True).set_index(["ticker", "date"])
ml_fctr = ml_fctr.dropna(how='all')
ml_fctr = filter_data(ml_fctr, ["IPO"])

def renamer(str_list):
    def _renamer(ticker):
        if ticker[-2:] == 'SZ':
            return ticker[0:6] + '.XSHE'
        elif ticker[-2:] == 'SH':
            return ticker[0:6] + '.XSHG'
        else:
            raise NotImplementedError()
    return list(map(_renamer, str_list))


with open(Path('data') / 'buffer' / f"{index_name}_weight.pkl", "rb") as f:
    weight_dict = pickle.load(f)

res = pd.DataFrame()
for dt, weight in tqdm(weight_dict.items()):
    weight = weight_dict[dt]
    end_dt = datetime.datetime.strptime(dt, '%Y-%m-%d') + relativedelta(months=+1) + pd.offsets.MonthEnd(0)
    if end_dt < ml_fctr.index.get_level_values('date').min():
        continue
    factor = ml_fctr.loc[pd.IndexSlice[:, end_dt.strftime('%Y-%m-%d')], pd.IndexSlice[:]].reset_index(level=1)
    factor.index = renamer(factor.index.to_list())
    df = weight.merge(factor, how="left", left_index=True, right_index=True, suffixes=("_w", "_f"))
    # print("missing value in prediciton:", df.shape[0] - df.predict.count())
    res.loc[end_dt, 'return'] = (df['weight'] * df['predict']).sum()


bm = jq.get_bars('000300.XSHG',82,unit='1M',fields=['date','close'],end_dt = '2020-09-30', include_now=True).set_index('date')
bm = bm.pct_change().iloc[1:, :].rename({'close':'index_return'}, axis=1)
bm.index = bm.index + pd.offsets.MonthEnd(0)
combine = res.merge(bm, left_index=True, right_index=True)
combine['return'] = combine/100
combine.plot()
plt.show()
from utils_stra import cal_r2

print("r2", cal_r2(combine['index_return'].values, combine['return'].values))

# %%