import os
import time
import utils

utils.setdir_fctr()

print("process quarter factor")
tos = time.time()
os.system('python code/factorQr.py')
toe = time.time()
print("process quarter data : ", toe-tos)

print("process month factor")
os.system('python code/factorMcr.py')

print("process week factor")
os.system('python code/factorWky.py')

print("process daily factor")
os.system('python code/loadDay.py')

print("process macro factor")
os.system('python code/factorMcr.py')

print("process Y")
os.system('python code/factorY.py')

print("aggregate all data")
os.system('python code/Agg_Factor.py')
