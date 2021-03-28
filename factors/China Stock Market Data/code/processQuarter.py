#%%
import os
import pandas as pd
pd.set_option('display.max_rows', 20)
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
Quarter_data
去重、查看
'''
from utils import cleandf

Quarter_data = cleandf(pd.read_csv(Path('data', 'buffer', 'QuarterFactorRaw.csv')))
Quarter_data = Quarter_data.sort_values(by=['ts_code','end_date'],ascending=[True,True])

Quarter_data = Quarter_data[~(Quarter_data.end_date.isna())]
Quarter_data = Quarter_data.drop_duplicates()

Quarter_data['end_date'] = Quarter_data['end_date'].astype(int)
date = [datetime.strptime(str(i), "%Y%m%d")  for i in Quarter_data.end_date.values]
Quarter_data.loc[:, 'end_date'] = date

Quarter_data = Quarter_data.reindex(range(len(Quarter_data)))
#qgrid.show_grid(Quarter_data.loc[:,['ts_code','end_date','ann_date']])

#所有差分、roll的函数加一个.groupby，如果是if Quarter_data['is_beginning'] = 1, 则自动为Nan
Quarter_data['is_beginning'] = 0
Quarter_data = Quarter_data.sort_values(['ts_code', 'end_date'])
Quarter_data.loc[0,'is_beginning'] = 1

for ii in tqdm(range(1,len(Quarter_data))):
    if Quarter_data.loc[ii,'ts_code']!= Quarter_data.loc[ii-1,'ts_code']:
        Quarter_data.loc[ii,'is_beginning'] = 1
    else:
        pass

# #所有.shift .rolling .diff 的，都要注意.groupby(Quarter_data['ts_code'])
# #可不可以分组 Rolling
# %%
'''
collection of industry-adjusted data
采用申万一级行业分类
Quarter_data：合并
'''
def selfmap1(result, tomap, col_name):
    # tomap = Industry_cost.copy()
    # col_name = 1
    # result = Quarter_data.copy()
    tomap_ = tomap.rename({"时间":"end_date"}, axis=1).set_index("end_date")
    tomap_ = tomap_.melt(var_name="industry", value_name=col_name, ignore_index=False).reset_index()

    # tomap_.loc[:, "end_date"] = tomap_.end_date.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    result = result.merge(tomap_, how='left', on=['industry', 'end_date'])
    return result

Ind_dir = Path("data", "Industry.xlsx")
industry_code = pd.read_excel(Ind_dir,sheet_name='行业总览和增发')
map_dict = industry_code[['证券代码','所属申万行业名称\n[行业级别] 一级行业↓']].set_index('证券代码').to_dict()
Quarter_data['industry'] = Quarter_data['ts_code'].map(map_dict['所属申万行业名称\n[行业级别] 一级行业↓'])
#Quarter_data
# date = [datetime.datetime.strptime(str(i), "%Y%m%d")  for i in factor.end_date.values]
# factor.loc[:, 'end_date'] = date

Industry_cost = pd.read_excel(Ind_dir, sheet_name="营业成本")
Industry_opcash = pd.read_excel(Ind_dir, sheet_name="经营活动现金流")
Industry_net_income = pd.read_excel(Ind_dir, sheet_name="净利润")
Industry_revenue = pd.read_excel(Ind_dir, sheet_name="营业收入")
Industry_inv_cash = pd.read_excel(Ind_dir, sheet_name="投资活动现金流")
Industry_roa = pd.read_excel(Ind_dir, sheet_name="ROA")
Industry_total_assets = pd.read_excel(Ind_dir, sheet_name="资产总计")

Industry_payroll_payable = pd.read_excel(Ind_dir, sheet_name="应付职工薪酬")
Industry_total_eqy = pd.read_excel(Ind_dir, sheet_name="所有者权益合计")
Industry_inv_cash_out = pd.read_excel(Ind_dir, sheet_name="投资活动现金流出小计")
Industry_total_share = pd.read_excel(Ind_dir, sheet_name="实收资本(或股本)")
Industry_asset_turn = pd.read_excel(Ind_dir, sheet_name="总资产周转率")
Industry_incr_cash_cash_equ = pd.read_excel(Ind_dir, sheet_name="现金及现金等价物增加净额")
Industry_oper_profit = pd.read_excel(Ind_dir, sheet_name="营业利润")
Industry_n_recp_disp_fiolta = pd.read_excel(Ind_dir, sheet_name="处置固定资产、无形资产和其他长期资产收回的现金净额")
Industry_close = pd.read_excel(Ind_dir, sheet_name="收盘价") #月度行情，其他为季度
Industry_market_value = pd.read_excel(Ind_dir, sheet_name="总市值") #月度行情，其他为季度

dfs = [Industry_cost, Industry_opcash, Industry_net_income, Industry_revenue,
       Industry_inv_cash, Industry_roa, Industry_total_assets, Industry_payroll_payable, Industry_total_eqy,
      Industry_inv_cash_out,Industry_total_share,Industry_asset_turn,
      Industry_incr_cash_cash_equ, Industry_oper_profit, Industry_n_recp_disp_fiolta,
      Industry_close, Industry_market_value]
#names = ["营业成本", "经营活动现金流", "净利润", "营业收入",
#          "投资活动现金流", "ROA", "资产总计","应付职工薪酬","所有者权益合计",
#          "投资活动现金流出小计","实收资本(或股本)","总资产周转率",
#          "现金及现金等价物增加净额","营业利润","处置固定资产、无形资产和其他长期资产收回的现金净额",
#          "收盘价","总市值"]
names = ['Industry_cost', 'Industry_opcash', 'Industry_net_income', 'Industry_revenue',
       'Industry_inv_cash','Industry_roa', 'Industry_total_assets', 'Industry_payroll_payable', 'Industry_total_eqy',
      'Industry_inv_cash_out','Industry_total_share','Industry_asset_turn',
      'Industry_incr_cash_cash_equ', 'Industry_oper_profit', 'Industry_n_recp_disp_fiolta',
      'Industry_close', 'Industry_market_value']



for df, name in tqdm(zip(dfs, names)):
    Quarter_data = selfmap1(Quarter_data, df, name)
    print(name)
Quarter_data.to_csv(Path("data", 'buffer', "QuarterFactorPrcd_Ins.csv"), index=False)

# %%
'''
map从wind来的几个额外指标
'''
def selfmap2(result, tomap, col_name):
    '''
    将面板i*t排序的数据map到it*1上
    result：it*1顺序排列的面板，也是最后我们需要的result
    tomap: i*t 的面板，注意设置index,需要从index和columns中遍历
    col_name: map到result之后新的列名称
    '''
    tomap_ = tomap.rename({"证券代码":"ts_code"}, axis=1).set_index("ts_code")
    tomap_ = tomap_.melt(var_name="end_date", value_name=col_name, ignore_index=False).reset_index()
    result = result.merge(tomap_, how='left', on=['ts_code', 'end_date'])
    return result

Quarter_data = pd.read_csv(Path("data", 'buffer', "QuarterFactorPrcd_Ins.csv"))
Qrt_ftrs = Path('data', '季度指标.xlsx')
Quarter_data_add_guarantee_amount = pd.read_excel(Qrt_ftrs,sheet_name='担保发生额合计.季度')
Quarter_data_add_rank = pd.read_excel(Qrt_ftrs,sheet_name='综合评级.季度')
Quarter_data_add_tax_return = pd.read_excel(Qrt_ftrs,sheet_name='税收返还、减免.季度')
Quarter_data_add_mortgage_financing = pd.read_excel(Qrt_ftrs,sheet_name='担保抵押融资.季度')

dfs = [Quarter_data_add_guarantee_amount, Quarter_data_add_rank, Quarter_data_add_tax_return, Quarter_data_add_mortgage_financing]
names = ['guarantee_amount', 'rank', 'tax_return', 'mortgage_financing']
for df, name in tqdm(zip(dfs, names)):
    print(name)
    Quarter_data = selfmap2(Quarter_data, df, name)
#睡前代码，千万记得保存
Quarter_data.to_csv(Path('data', 'buffer', 'QuarterFactorPrcd_季度指标.csv'), index=False)

# %%
Quarter_data = pd.read_csv(Path('data', 'buffer', 'QuarterFactorPrcd_季度指标.csv'))
Qrt_rtgs = Path('data', '季度评级.xlsx')
Quarter_data_add_rank_num = pd.read_excel(Qrt_rtgs,sheet_name='评级机构家数')
Quarter_data_add_rank_num_high = pd.read_excel(Qrt_rtgs,sheet_name='评级调高家数')

dfs = [Quarter_data_add_rank_num, Quarter_data_add_rank_num_high]
names = ['rank_num', 'rank_num_high']
for df, name in tqdm(zip(dfs, names)):
    print(name)
    Quarter_data = selfmap2(Quarter_data, df, name)

Quarter_data.to_csv(Path('data', 'buffer', 'QuarterFactorPrcd_季度评级.csv'), index=False)

# %% 去重
Quarter_data = pd.read_csv(Path('data', 'buffer', 'QuarterFactorPrcd_季度评级.csv'))

flag_sum = Quarter_data[[i for i in Quarter_data.columns if i.startswith('update_flag')]].sum(axis=1, skipna=True)
Quarter_data.loc[:, "update_flag_sum"] = flag_sum

Quarter_data_ = Quarter_data.sort_values(['ts_code','end_date', 'update_flag_sum'], ascending=[True, True, True])
Quarter_data_sr = Quarter_data_.drop_duplicates(['ts_code','end_date'],keep='last')

assert Quarter_data_sr.drop_duplicates(['ts_code','end_date']).shape == Quarter_data_sr.shape
Quarter_data_sr.to_csv(Path('data', 'buffer', 'QuarterFactorPrcd.csv'), index=False, encoding='gbk')

# %%