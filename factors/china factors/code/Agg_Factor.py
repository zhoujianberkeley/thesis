import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import utils

utils.setdir_fctr()

def norm(df):
    df = (df.rank(axis=0, pct=True) - 0.5) * 2
    df = df.fillna(0)
    return df

def process(df, n):
    df = df.set_index(['ts_code', 'end_date'], verify_integrity=True).sort_index()
    df = norm(df).groupby('ts_code').shift(n)
    return df

# %%
dfs_lst = []
for f in tqdm(['DayFactor.csv', 'WeekFactor.csv', 'MonFactor.csv']):
    raw = pd.read_csv(Path('_saved_factors', f))
    df = process(raw, 1)
    dfs_lst.append(df)
fctr_df = pd.concat(dfs_lst, axis=1, join='inner')

print(fctr_df.describe().T.sort_values(['count'], ascending=True))
print("shape: ", fctr_df.shape)

fctr_qrt = pd.read_csv(Path('_saved_factors', 'QrtFactor.csv'))
fctr_qrt = fctr_qrt.set_index(['ts_code', 'end_date'], verify_integrity=True).sort_index()
fctr_qrt = norm(fctr_qrt)
fctr_qrt = fctr_qrt.reindex(fctr_df.index, method='ffill').sort_index()
fctr_qrt = fctr_qrt.groupby('ts_code').shift(4)

fctr_df = fctr_df.merge(fctr_qrt, left_index=True, right_index=True)

print(fctr_df.describe().T.sort_values(['count'], ascending=True))
print("shape: ", fctr_df.shape)

# %%
# macro
macro = pd.read_csv(Path('_saved_factors', 'MacroFactor.csv')).set_index(['end_date'], verify_integrity=True)
macro = macro.drop('DividendPriceRatio', axis=1)
macro['svar'] = macro['svar']*300 # enlarge svar a little bit

macro.columns = ["Macro_" + i for i in macro.columns]
macro_ftr = macro.columns.to_list()

fctr_df = fctr_df.sort_index()
_macro = fctr_df.reset_index().merge(macro, left_on='end_date', right_index=True)#[macro_ftr]
_macro = _macro.set_index(['ts_code', 'end_date'], verify_integrity=True).sort_index()
dfs = [fctr_df]
for col in tqdm(macro_ftr):
    tmp = fctr_df.multiply(_macro[col], axis=0)
    tmp.columns = [i + f"_{col}" for i in tmp.columns]
    dfs.append(tmp)

full_factors = pd.concat(dfs, axis=1)
full_factors = full_factors.reset_index().merge(macro, left_on='end_date', right_index=True).set_index(['ts_code', 'end_date'], verify_integrity=True)

# %%
Ind_fctr = pd.read_csv(Path('_saved_factors', 'IndFactor.csv')).set_index(['ts_code', 'end_date'], verify_integrity=True)
full_factors = full_factors.merge(Ind_fctr, on=['ts_code', 'end_date'])

# %%
y = pd.read_csv(Path('_saved_factors', 'MonY.csv')).set_index(['ts_code', 'end_date'], verify_integrity=True)
full_factors_y = full_factors.merge(y, on=['ts_code', 'end_date'])
full_factors_y = full_factors_y.dropna(subset=['monthly_return'])
full_factors_y.to_hdf(Path('_saved_factors', "full_factors.h5"), key='data')

close = pd.read_csv(Path('_saved_factors', 'MonClose.csv')).set_index(['ts_code', 'end_date'], verify_integrity=True)
close = full_factors_y.merge(close, on=['ts_code', 'end_date'])[['close']]
close.to_hdf(Path('_saved_factors', "close.h5"), key='data')


print(full_factors_y.describe().T.sort_values(['count'], ascending=True))
print("shape: ", full_factors_y.shape)
# %%