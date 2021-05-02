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
# get weights for SH300 and ZZ800 index
index_name = '000300.XSHG'
weight_dict = {}
for tp in tqdm(pd.date_range('20131231', '20200831', freq='M')):
    print(tp)
    tp = tp.strftime("%Y-%m-%d")
    weight = jq.get_index_weights(index_name, date=tp)
    # weight.reset_index().set_index('date')
    weight_dict[tp] = weight

with open(Path('data') / 'buffer' / f"{index_name}_weight.pkl", "wb") as f:
    pickle.dump(weight_dict, f)

# %%
with open(Path('data') / 'buffer' / f"{index_name}_weight.pkl", "rb") as f:
    weight_dict = pickle.load(f)

res = pd.DataFrame()
for dt, weight in tqdm(weight_dict.items()):
    # weight = weight_dict[dt]
    end_dt = datetime.datetime.strptime(dt, '%Y-%m-%d') + relativedelta(months=+1)
    price_df = jq.get_bars(weight.index.to_list(),2,unit='1M',fields=['date','close'],end_dt=end_dt, include_now=True)
    change_df = pd.DataFrame(price_df.groupby(level=0)['close'].pct_change()).loc[pd.IndexSlice[:, 1], pd.IndexSlice[:]].droplevel(1)
    change_df = change_df.rename({'close':'pct_change'}, axis=1)

    df = weight.merge(change_df, left_index=True, right_index=True)
    assert df.shape[0] == 300
    res.loc[dt, 'return'] = (df['weight']*df['pct_change']).sum()

res.to_csv(Path('data') / 'buffer' / f"{index_name}_return.csv")
# %%
res = pd.read_csv(Path('data') / 'buffer' / f"{index_name}_return.csv", index_col=0, parse_dates=[0], infer_datetime_format=True)
bm = jq.get_bars('000300.XSHG',82,unit='1M',fields=['date','close'],end_dt = '2020-09-30', include_now=True).set_index('date')
bm = bm.pct_change().shift(-1).iloc[:-1, :].rename({'close':'index_return'}, axis=1)

combine = res.merge(bm, left_index=True, right_index=True)
combine['return'] = combine/100
combine.plot()
plt.show()

# %% machine learning
pre_dir = "Filter IPO"

model_name = "NN1 M"
ml_fctr = pd.read_csv(Path('code') / pre_dir /model_name / "predictions.csv", parse_dates=["date"], infer_datetime_format=True).set_index(["ticker", "date"])
ml_fctr = ml_fctr.dropna(how='all')
ml_fctr = filter_data(ml_fctr, ["IPO"])


with open(Path('data') / 'buffer' / f"{index_name}_weight.pkl", "rb") as f:
    weight_dict = pickle.load(f)
bt_df = close_raw.merge(ml_fctr, on=["ticker", "date"], how="right")

close = bt_df['close'].unstack(level=0)
change = close.pct_change()

ml_fctr = bt_df['predict'].unstack(level=0)

res.append(backtest(model_name, ml_fctr, change))