# %%
import multiprocessing
import ray
import os
import tqdm
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count


os.environ['NUMEXPR_MAX_THREADS'] = '16'
tqdm.pandas()
pd.set_option('display.max_r',10)

# %%
@ray.remote
def cal_daily(data:pd.DataFrame) -> pd.DataFrame:
    # todo fill the function
    return pd.DataFrame()

#%% apply multiprocessing
# ray is faster because it does not need to pickle the data
# however it seems to require python 3.5 or more
def ray_apply(df, func, workers = None):
    if workers == None:
        workers = cpu_count()
    ray.init(num_cpus = workers, ignore_reinit_error=True)
    codes = df.index.get_level_values('ticker').drop_duplicates()
    mapping = pd.Series(np.arange(0, len(codes)), index=codes)
    workerid = df.index.get_level_values('ticker').map(mapping) % (workers)
    groupeds = df.groupby(workerid.values)
    res_list = ray.get([func.remote(g) for _,g in list(groupeds)])
    ray.shutdown()
    return pd.concat(res_list)

#%%
# run rolling with ray apply
tic = time()
res5 = ray_apply(data, cal_daily)
toc = time()
print("calculation uses time : ", toc - tic)
