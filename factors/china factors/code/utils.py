import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
from pathlib import Path
from datetime import datetime, date
from tqdm import tqdm

# Tushare + wind API 能用tushare就tushare 实在不行上windAPI 不用csmar
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
import tushare as ts
ts.set_token(token)
pro = ts.pro_api()

def setdir_fctr():
    _paths = os.getcwd().split('/')
    if _paths[-1] == "code":
        os.chdir("..")
    elif _paths[-1] == "EmpAstPricing":
        os.chdir(Path("factors") / "china factors")
setdir_fctr()

def gen_dl(sy, ey):
    date_list = []
    for y in range(sy, ey):
        dl = [f"{y}0331", f"{y}0630", f"{y}0930", f"{y}1231"]
        date_list.extend(dl)
    return date_list

start_year = 2010
end_year = 2020
date_list = gen_dl(start_year, end_year+1)

def cleandf(df):
    to_drop = [i for i in df.columns if i.startswith("Unnamed")]
    df = df.drop(to_drop, axis=1)
    return df

def todate(df, col, format='%Y-%m-%d'):
    if df[col].dtype == np.dtype(np.object):
        df[col] = df[col].apply(lambda x: datetime.strptime(x, format))
    else:
        pass
    return df

def save_f(tosave, fn):
    s_path = Path("_saved_factors")
    if not os.path.exists(s_path):
        os.mkdir(s_path)
    tosave.to_csv(s_path / f"{fn}.csv")

def extrct_fctr(df):
    out_col = ['ts_code', 'end_date']
    df = df.sort_values(out_col)
    out_col.extend([i for i in df.columns if i.startswith("Factor")])
    res = df[out_col]
    res = res[(res.end_date >= pd.to_datetime(date(start_year, 1, 1))) & (res.end_date <= pd.to_datetime(date(end_year+1, 1, 1)))]
    return res


def check(df):
    dates = df.end_date.unique()
    dates.sort()
    print(dates)
    print(df.describe().T.sort_values(['count'], ascending=True))
    print("shape: ", df.shape)


def load_divi(divi_dir):
    if not os.path.exists(divi_dir):
        os.mkdir(divi_dir)

    stocks = pro.stock_basic(exchange='', list_status='L', fields='')
    stock_pool = stocks['ts_code'].tolist()

    fail_list = []
    while True:
        for ii in tqdm(stock_pool):
            tmp = divi_dir / (ii + ".csv")
            if os.path.exists(tmp):
                continue
            try:
                divi = pro.dividend(ts_code=ii,
                                    fields='ts_code,end_date,div_proc,stk_div,record_date,ex_date,pay_date,div_listdate')
                divi.to_csv(tmp)
            except:
                print(ii, " fail")
                fail_list.append(ii)
                continue
        if fail_list == []:
            break

def load_SW():
    SW_L1 = pro.index_classify(level='L1', src='SW')
    SW_indexlist = SW_L1.index_code.tolist()
    SWL1_classify = pro.index_member(index_code=SW_L1.index_code[0])

    for ii in tqdm(range(1,len(SW_indexlist))):
        df1 = pro.index_member(index_code=SW_L1.index_code[ii])
        SWL1_classify = pd.concat([SWL1_classify,df1],axis=0)

    #国民经济分三个产业，从这个角度来出发 -- 增量产业；对社会发展有增量贡献为0，存量内卷、加工主要以满足消费欲望为主、社会可以缺少的行业为1
    SW_L1['Industry_type']=0
    SW_L1.loc[[5,10,11,13,15,16,20,23,25,26],'Industry_type'] = 1

    #SWL1_classify['out_date'].count()
    #没有上市公司被移出指标
    SWL1_classify['industry_name'] = 0

    map_dict = SW_L1[['index_code','industry_name']].set_index('index_code').to_dict()
    SWL1_classify['industry'] = SWL1_classify['index_code'].map(map_dict['industry_name'])

    map_dict2 = SW_L1[['index_code','Industry_type']].set_index('index_code').to_dict()
    SWL1_classify['industry_sin'] = SWL1_classify['index_code'].map(map_dict2['Industry_type'])

    SWL1_classify.to_excel(Path('data', 'SWL1申万一级行业组成.xlsx'))


def load_spnd(spnd_dir):
    if not os.path.exists(spnd_dir):
        os.mkdir(spnd_dir)

    stocks = pro.stock_basic(exchange='', list_status='L', fields='')
    stock_pool = stocks['ts_code'].tolist()

    fail_list = []
    while True:
        for ii in tqdm(stock_pool):
            tmp = spnd_dir / (ii + ".csv")
            if os.path.exists(tmp):
                continue
            try:
                df = pro.suspend(ts_code=ii, suspend_date='', resume_date='', fields='')
                df.to_csv(tmp)
            except:
                print(ii, " fail")
                fail_list.append(ii)
                continue
        if fail_list == []:
            break



def load_fsdate(fs_dir, dates):
    if not os.path.exists(fs_dir):
        os.mkdir(fs_dir)

    stocks = pro.stock_basic(exchange='', list_status='L', fields='')
    stock_pool = stocks['ts_code'].tolist()

    fail_list = []
    while True:
        for ii in tqdm(stock_pool):
            tmp_pt = fs_dir / (ii + ".csv")
            dfs = []
            if os.path.exists(tmp_pt):
                continue
            for d in dates:
                try:
                    df = pro.disclosure_date(ts_code=ii, end_date=d)
                    dfs.append(df)
                except:
                    print(ii, " fail")
                    fail_list.append(ii)
                    break
            if dfs == []:
                print(ii, "No objects to concatenate")
            else:
                pd.concat(dfs).to_csv(tmp_pt)
        if fail_list == []:
            break

def load_FSdate():
    stocks = pro.stock_basic(exchange='', list_status='L', fields='')
    stock_pool = stocks['ts_code'].tolist()

    FSdate = pro.disclosure_date(ts_code=stock_pool[0],
                                 end_date=date_list[0])

    for ii in range(0, 4163):
        for tt in range(22):
            try:
                df1 = pro.disclosure_date(ts_code=stock_pool[ii],
                                          end_date=date_list[tt])
                FSdate = pd.concat([FSdate, df1], axis=0)
            except:
                continue

    FSdate.to_excel('Quotation\stock_financial_statement_date.xlsx')


# #运行前
# Monthly_Quotation = pd.read_csv(Path('_saved_factors', 'MonFactor.csv'))
# Monthly_Quotation = Monthly_Quotation.set_index(['ts_code','end_date'])
#
# Fault = Monthly_Quotation.reset_index(1)['end_date'].diff()
#
# Lack_list = Fault[Fault>timedelta(32)].index
# # 最长的月是31 days，超过31 days 说明有异常缺失值
# #Monthly_Quotation.index.get_level_values('end_date')[2] - Monthly_Quotation.index.get_level_values('end_date')[1]
#
# for '000155.SZ' in Lack_list:
#
#     F = Monthly_Quotation[Monthly_Quotation.index.get_level_values('ts_code')=='000155.SZ'].index.get_level_values('end_date')
#     #如果index对不上，尝试.tail(1).values[0]
#     idx = pd.date_range(start=F[0],end=F[-1],freq='M')
#
#     idx2 = pd.MultiIndex.from_product([['000155.SZ'],idx], names=['ts_code', 'end_date'])
#
#     G = Monthly_Quotation[Monthly_Quotation.index.get_level_values('ts_code')=='000155.SZ'].reindex(index=idx2, fill_value=0)
#
#     Monthly_Quotation = pd.concat(
#         [Monthly_Quotation[~(Monthly_Quotation.index.get_level_values('ts_code')=='000155.SZ')],G]
#     )
#
# #运行后拆开
# Monthly_Quotation = Monthly_Quotation.reset_index()
#
#
#
# def Filldays(Monthly_Quotation, freq):
#     Monthly_Quotation = Monthly_Quotation.set_index(['ts_code', 'end_date'])
#
#     Fault = Monthly_Quotation.reset_index(1)['end_date'].diff()
#
#     Lack_list = Fault[Fault > timedelta(32)].index
#     # 最长的月是31 days，超过31 days 说明有异常缺失值
#     # Monthly_Quotation.index.get_level_values('end_date')[2] - Monthly_Quotation.index.get_level_values('end_date')[1]
#
#     for ts_code in Lack_list:
#         F = Monthly_Quotation[Monthly_Quotation.index.get_level_values('ts_code') == ts_code].index.get_level_values(
#             'end_date')
#         # 如果index对不上，尝试.tail(1).values[0]
#         idx = pd.date_range(start=F[0], end=F[-1], freq=freq)
#
#         idx2 = pd.MultiIndex.from_product([[ts_code], idx], names=['ts_code', 'end_date'])
#
#         G = Monthly_Quotation[Monthly_Quotation.index.get_level_values('ts_code') == ts_code].reindex(index=idx2,
#                                                                                                       fill_value=0)
#
#         Monthly_Quotation = pd.concat(
#             [Monthly_Quotation[~(Monthly_Quotation.index.get_level_values('ts_code') == ts_code)], G]
#         )
#
#     Monthly_Quotation = Monthly_Quotation.reset_index()