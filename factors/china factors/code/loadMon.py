#%%
import os
import pandas as pd
pd.set_option('display.max_rows', 100)
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import qgrid

# Tushare + wind API 能用tushare就tushare 实在不行上windAPI 不用csmar
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
import tushare as ts
ts.set_token(token)
pro = ts.pro_api()


_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

# %%
'''
Use wind to get Monthly Quotation Data
出于数据源严谨性，决定仅采用wind版本数据
'''
Monthly_Quotation = pd.read_csv(Path('data', '月行情.csv'),encoding='gbk')
Monthly_Quotation.rename(columns={'代码':'ts_code','简称':'name','日期':'trade_date', '前收盘价(元)':'pre_close', '开盘价(元)':'open', '最高价(元)':'high', '最低价(元)':'low',
      '收盘价(元)':'close', '成交量(股)':'vol', '成交金额(元)':'amount', '涨跌(元)':'change', '涨跌幅(%)':'pct_chg', '均价(元)':'avg_price', '换手率(%)':'turnover',
      '总市值(元)':'market_value', '总股本(股)':'total_share', '市盈率':'PE', '市净率':'PB', '市销率':'PS', '市现率':'PCF',}, inplace=True)
# %%
# split-adjusted price
Monthly_Quotation_sa = pd.read_csv(Path('data', '月行情后复权.csv'),encoding='utf_8_sig')
Monthly_Quotation_sa.rename(columns={'代码':'ts_code','简称':'name','日期':'trade_date', '前收盘价(元)':'pre_close', '开盘价(元)':'open', '最高价(元)':'high', '最低价(元)':'low',
      '收盘价(元)':'close', '成交量(股)':'vol', '成交金额(元)':'amount', '涨跌(元)':'change', '涨跌幅(%)':'pct_chg', '均价(元)':'avg_price', '换手率(%)':'turnover',
      '总市值(元)':'market_value', '总股本(股)':'total_share', '市盈率':'PE', '市净率':'PB', '市销率':'PS', '市现率':'PCF',}, inplace=True)

Monthly_Quotation_sa = Monthly_Quotation_sa[~(Monthly_Quotation_sa.trade_date.isna()==True)]
date = [datetime.strptime(i, "%Y/%m/%d")  for i in Monthly_Quotation_sa.trade_date.values]
Monthly_Quotation_sa.loc[:, 'trade_date'] = date

Monthly_Quotation_sa['end_date']=0  #把月份中最后一天交易日统一转化为月末 Turn the time stamp to the end of the month
for ii in tqdm(range(len(Monthly_Quotation_sa))):
    Monthly_Quotation_sa.loc[ii,'trade_date'].to_period('M').to_timestamp('M')
Monthly_Quotation_sa['end_date'] = Monthly_Quotation_sa.trade_date + pd.offsets.MonthEnd(0)


assert Monthly_Quotation_sa.drop_duplicates(['ts_code','end_date']).shape == Monthly_Quotation_sa.shape
Monthly_Quotation_sa = Monthly_Quotation_sa.sort_values(['ts_code','end_date'])
Monthly_Quotation_sa.to_csv(Path('data', 'buffer', "MonFactorPrcd_sa.csv"), index=False)

# %%
'''
Monthly Quotation 合并行业
'''
Ind_dir = Path("data", "Industry.xlsx")
industry_code = pd.read_excel(Ind_dir,sheet_name='行业总览和增发')
industry_code.set_index('证券代码')
map_dict = industry_code[['证券代码','所属申万行业名称\n[行业级别] 一级行业↓']].set_index('证券代码').to_dict()

Monthly_Quotation['industry'] = Monthly_Quotation['ts_code'].map(map_dict['所属申万行业名称\n[行业级别] 一级行业↓'])
Monthly_Quotation = Monthly_Quotation.sort_values(by=['ts_code','trade_date'])
Monthly_Quotation.index = range(len(Monthly_Quotation))
Monthly_Quotation = Monthly_Quotation[~(Monthly_Quotation.trade_date.isna()==True)]
#Monthly_Quotation['trade_date'] = Monthly_Quotation['trade_date'].astype(int)
date = [datetime.strptime(i, "%Y-%m-%d")  for i in Monthly_Quotation.trade_date.values]
Monthly_Quotation.loc[:, 'trade_date'] = date

Monthly_Quotation['end_date']=0  #把月份中最后一天交易日统一转化为月末 Turn the time stamp to the end of the month
for ii in tqdm(range(len(Monthly_Quotation))):
    Monthly_Quotation.loc[ii,'trade_date'].to_period('M').to_timestamp('M')
Monthly_Quotation['end_date'] = Monthly_Quotation.trade_date + pd.offsets.MonthEnd(0)

# %%
Industry_close = pd.read_excel(Ind_dir, sheet_name="收盘价") #月度行情，其他为季度
Industry_market_value = pd.read_excel(Ind_dir, sheet_name="总市值") #月度行情，其他为季度
dfs = [Industry_close, Industry_market_value]
names = ['Industry_close', 'Industry_market_value']

def selfmapMon(result, tomap, col_name):
    '''
    将面板i*t排序的数据map到it*1上
    result：it*1顺序排列的面板，也是最后我们需要的result
    tomap: i*t 的面板，注意设置index,需要从index和columns中遍历
    col_name: map到result之后新的列名称
    '''
    tomap_ = tomap.rename({"时间":"end_date"}, axis=1).set_index("end_date")
    tomap_ = tomap_.melt(var_name="industry", value_name=col_name, ignore_index=False).reset_index()
    # tomap_.loc[:, "end_date"] = tomap_.end_date.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    result = result.merge(tomap_, how='left', on=['industry', 'end_date'])
    return result

for df, name in tqdm(zip(dfs, names)):
    print(name)
    Monthly_Quotation = selfmapMon(Monthly_Quotation, df, name)
    assert Monthly_Quotation.groupby(['industry', 'end_date'])[name].diff().sum() == 0

# %%
assert Monthly_Quotation.drop_duplicates(['ts_code','end_date']).shape == Monthly_Quotation.shape
Monthly_Quotation.to_csv(Path('data', 'buffer', "MonFactorPrcd.csv"), index=False)