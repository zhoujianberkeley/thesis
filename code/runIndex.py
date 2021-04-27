# %%
token = '654d36bf9bb086cb8c973e0f259e38c3efe24975386b7922e88a4cf2'
import tushare as ts
ts.set_token(token)
pro = ts.pro_api()

# %%
df = pro.index_weight(index_code='399300.SZ', start_date='20180901', end_date='20180930')

pro.index_weight(index_code='399300.SZ', start_date='20180901', end_date='20180930')

df = pro.index_weight(index_code='399300.SZ', start_date='20170101', end_date='20170130')


tmp = pro.index_weight(index_code='399300.SZ', start_date='20100201', end_date='20100230')
tmp.sort_values(['trade_date'])['trade_date'].unique()[0]

tmp = pro.index_weight(index_code='399300.SZ', start_date='20100201', end_date='20100201')


# %%