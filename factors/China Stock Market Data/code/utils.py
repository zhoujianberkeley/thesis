import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
from pathlib import Path
from datetime import datetime, date
from tqdm import tqdm
import qgrid
import matplotlib
import matplotlib.pyplot as plt

# Tushare + wind API 能用tushare就tushare 实在不行上windAPI 不用csmar
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
import tushare as ts
ts.set_token(token)
pro = ts.pro_api()

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

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
    return df[out_col]

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