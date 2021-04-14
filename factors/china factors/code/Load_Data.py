import os
import time

_paths = os.getcwd().split('/')
if _paths[-1] == "code":
    os.chdir("..")

loadQ = True
if loadQ:
    print("load quarter data")
    tos = time.time()
    os.system('python code/loadQuarter.py')
    toe = time.time()
    print("load quarter data : ", toe-tos)

print("process quarter data")
tos = time.time()
os.system('python code/processQuarter.py')
toe = time.time()
print("process quarter data : ", toe-tos)

print("process month data")
os.system('python code/loadMon.py')

print("process week data")
os.system('python code/loadWk.py')

print("process daily data")
os.system('python code/loadDay.py')
