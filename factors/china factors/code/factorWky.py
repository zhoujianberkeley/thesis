#%%
from datetime import datetime, date
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
from pathlib import Path
from tqdm.auto import tqdm
import statsmodels.api as sm
import ray
import time
from multiprocessing import cpu_count


os.environ['NUMEXPR_MAX_THREADS'] = '16'
tqdm.pandas()
pd.set_option('display.max_r',10)

# Tushare + wind API 能用tushare就tushare 实在不行上windAPI 不用csmar
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
import tushare as ts
ts.set_token(token)
pro = ts.pro_api()

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

s_path = Path("_saved_factors")
if not os.path.exists(s_path):
    os.mkdir(s_path)

import utils

#%%
# load data
_load = False
if _load:
    Weekly_Quotation = utils.cleandf(pd.read_csv(Path('data', 'buffer', 'WeekFactorPrcd.csv'), encoding='gbk'))
    Weekly_Quotation = Weekly_Quotation.rename({'代码': 'ts_code', '日期': 'trade_date'}, axis=1)
    Weekly_Quotation = Weekly_Quotation.sort_values(by=['ts_code', 'trade_date'])
    Weekly_Quotation.index = range(len(Weekly_Quotation))
    Weekly_Quotation.to_pickle(Path('data', 'buffer', 'WeekFactorPrcd.pkl'))
else:
    Weekly_Quotation = pd.read_pickle(Path('data', 'buffer', 'WeekFactorPrcd.pkl'))

# %%
'''
39. idiovol: idiosyncratic return volatility - monthly
Standard deviation of residuals of weekly returns on weekly equal weighted market returns for 3 year prior to month end
月底 前半年的 周超额收益 标准差
3 yr requires too much data, change it to 0.5 yr == 26week.
# 周数据
1. Stocks - Weekly quotation
2. Index - IC500 Weekly quotation -> excess residuals
3. get residuals VOL(SD)
-index weekly
'''

#获取指数000001.SH收益作为benchmark
df = pro.index_weekly(ts_code='000001.SH', start_date=f"{utils.start_year}0101", end_date=f'{utils.end_year}1231', fields='')
map_dict = df[['trade_date','pct_chg']].set_index('trade_date').to_dict()
# 出str格式用于map
date = [str(i).replace('-','') for i in Weekly_Quotation.trade_date.values]
Weekly_Quotation.loc[:,'trade_date_str'] = date


Weekly_Quotation['benchmark_return'] = Weekly_Quotation['trade_date_str'].map(map_dict['pct_chg'])
Weekly_Quotation['excess_return'] = Weekly_Quotation['涨跌幅(%)'] - Weekly_Quotation['benchmark_return']
Weekly_Quotation['Factor39_idivol'] = Weekly_Quotation.groupby(Weekly_Quotation['ts_code'],as_index=False)['excess_return'].rolling(26).std().values

#%%
'''
66. pricedelay: The proportion of variation in weekly returns for 36 months ending in month t 
                explained by 4 lags of weekly market returns incremental to contemporaneous market return
Monthly average weekly data
截至t个月的36个月内，周收益的变化比例由每周市场收益向同期市场收益递增的4个滞后解释
We shrink it by using 52 week, 2 period lag of weekly market returns and lag y.

Rolling window 52 weeks data
Averaging process was done in the final merge part
'''
#%% md
# Original Paper:
#
# $$r_{jt} = α_j + β_j R_{mt} +\sum_{n=1}^4 δ^{(-n)}_j R_{m,t-n} + \sum_{n=1}^4 γ^{(-n)}_j r_{j,t-n}+ ε_{jt} $$ (1)
#
# where
# $δ_{j^(-n)}$ is only a notion of regression param, not (-n)power
# rj,t is the return on stock j
# Rm,t is the return on the CRSP value-weighted market index at time t
#
# Using the estimated coefficients from this regression, we compute three measures of price delay
# for each firm at the end of June of each year.
# We use the first measure: the fraction of variation of contemporaneous returns explained by the lagged regressors.
#
# This is simply one minus the ratio of the R2 from regression (1) assuming $δ_j^{(−n)} = 0$ and $γ_j^{(−n)}= 0, ∀n ∈ [1, 4]$, over the R2 from regression (1) with no restrictions.
#
# $$ D1 = 1-\frac{R^2_{δ^{(−n)}_j=0,γ^{(−n)}_j=0}}{R^2} $$
#
# This is similar to an F-test on the joint significance of the lagged variables scaled by the amount of total variation explained contemporaneously.
# The larger this number, the more return variation is captured by lagged returns, and hence the stronger is the delay in response to return innovations.

# %%
# 周行情，股票 要出 Rolling window = 52的回归，每个回归的样本数是52
# Y个股收益  自变量X为同期的市场，滞后2期的市场，和滞后两期的个股收益
# 回归得到R2.之后再回归，尝试共同F检验值，或直接按照表达式计算

@ray.remote
def ray_reg(data:pd.DataFrame) -> pd.Series:
    res = []
    for ii in tqdm(data.index):
        if data.loc[ii,'52_week_indicator']==1:
            Y = data.loc[ii-51:ii,'涨跌幅(%)']

            R = data.loc[ii-51:ii,'benchmark_return']
            R_1 = data.loc[ii-52:ii-1,'benchmark_return']
            R_2 = data.loc[ii-53:ii-2,'benchmark_return']
            r_1 = data.loc[ii-52:ii-1,'涨跌幅(%)']
            r_2 = data.loc[ii-53:ii-2,'涨跌幅(%)']

            X1 = np.column_stack((R,R_1,R_2,r_1,r_2))
            X2 = np.column_stack((R)).T
            X1 = sm.add_constant(X1)
            X2 = sm.add_constant(X2)

            #results.adj_rsquared
            #dir(result) to check the whole parameters for regression
            model = sm.OLS(Y, X1)
            results = model.fit()
            R2_U = results.rsquared
            model = sm.OLS(Y, X2)
            results = model.fit()
            R2_R = results.rsquared
            # data.loc[ii,'pricedelay']=1-R2_U/R2_R
            tmp = pd.DataFrame({
                "value": 1-R2_U/R2_R
            }, index=pd.MultiIndex.from_tuples([(data.loc[ii, "ts_code"], data.loc[ii, "trade_date"])], names=["ts_code", "trade_date"]))

            res.append(tmp)
    return pd.concat(res)

# apply multiprocessing
# ray is faster because it does not need to pickle the data
# however it seems to require python 3.5 or more
def ray_apply(df, func, workers = None):
    if workers == None:
        workers = cpu_count()
    ray.init(num_cpus = workers, ignore_reinit_error=True)
    # df = reg_df.reset_index()
    codes = df['ts_code'].drop_duplicates()
    mapping = pd.Series(np.arange(0, len(codes)), index=codes)
    workerid = df['ts_code'].map(mapping) % (workers)
    groupeds = df.groupby(workerid.values)
    res_list = ray.get([func.remote(g) for _,g in list(groupeds)])
    ray.shutdown()
    return pd.concat(res_list)

# %%
# run rolling with ray apply
#保证回归能跑出来，有两期滞后项，并用该指标代替groupby
Weekly_Quotation['52_week_indicator'] = Weekly_Quotation.groupby('ts_code',as_index=False)['excess_return'].rolling(54).std().values
Weekly_Quotation['52_week_indicator'] = Weekly_Quotation['52_week_indicator']*0+1

fct66_pt = Path('data', "factor66_pricedelay.csv")
if os.path.exists(fct66_pt):
    res_df = utils.cleandf(pd.read_csv(fct66_pt))
else:
    tic = time.time()
    res_df = ray_apply(Weekly_Quotation, ray_reg)
    toc = time.time()
    print("calculation uses time : ", toc - tic)
    res_df = res_df.rename({'value':"Factor66_pricedelay"}, axis=1).reset_index()
    res_df.to_csv(Path('data', "factor66_pricedelay.csv"), index=False)

Weekly_Quotation = Weekly_Quotation.merge(res_df, on=['ts_code', 'trade_date'], how='left')
# %% output
Weekly_Quotation['trade_date'] = pd.to_datetime(Weekly_Quotation['trade_date'])
Weekly_Quotation['end_date'] = Weekly_Quotation['trade_date'] + pd.offsets.MonthEnd(0)


Out_df = utils.extrct_fctr(Weekly_Quotation)
Out_df = Out_df.groupby(['ts_code', 'end_date'], as_index=False).mean()
utils.check(Out_df)
Out_df.to_csv(Path('_saved_factors', 'WeekFactor.csv'), index=False)
# %%