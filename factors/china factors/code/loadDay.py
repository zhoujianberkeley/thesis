import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import utils

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

dfs = pd.DataFrame()
for yr in tqdm(range(utils.start_year, utils.end_year+1)):
    df = pd.read_csv(Path("data", "日行情", f"日行情{yr}.csv"),encoding='gbk')
    dfs = pd.concat([dfs, df], axis=0)

dfs.to_csv(Path('data', 'buffer', 'DayFactorPrcd.csv'), index=False)
