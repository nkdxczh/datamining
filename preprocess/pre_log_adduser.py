import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")
'''
user_f = open("/home/jason/datamining/datasets/user_info.csv","r")
size = 213000

users=[0 for i in range(size)]
i = 0
for line in user_f:
    if i == 0:
        i += 1
        continue
    i += 1
    cnt = i
    if int(cnt / size * 100) != int((cnt - 1) / size * 100):
        print(int((cnt) / size * 100), '%')

    strs = line.split(",")
    users[int(strs[0])]=[int(strs[1]),int(strs[2])]

user_f.close()'''

log_f = open("/home/jason/datamining/datasets/user_log.csv","r")
size = 26258293
merchants = []
i = 0
for line in log_f:
    if i == 0:
        i += 1
        continue
    i += 1
    cnt = i
    if int(cnt / size * 100) != int((cnt - 1) / size * 100):
        print(int((cnt) / size * 100), '%')

    strs = line.split(",")
    merchants.append(int(strs[3]))

log_f.close()

max_m = max(merchants)

print(max_m)

'''
merchants=[[0,0,0] for i in range(max_m+1)]
log_f = open("/home/jason/datamining/datasets/user_log.csv","r")
i = 0
for line in log_f:
    if i == 0:
        i += 1
        continue
    i += 1
    cnt = i
    if int(cnt / size * 100) != int((cnt - 1) / size * 100):
        print(int((cnt) / size * 100), '%')

    strs = line.split(",")
    user = int(strs[0])
    merchant = int(strs[3])
    merchants[merchant][0]+=users[user][0]
    merchants[merchant][1]+=users[user][1]
    merchants[merchant][2]+=1
log_f.close()

log_f = open("/home/jason/datamining/data/merchant_info.csv","w")
for i in merchants:
    if i[2] == 0:
        log_f.write("0,0\n")
        continue
    age = float(i[0])/float(i[2])
    sex = float(i[1])/float(i[2])
    log_f.write(str(age)+","+str(sex)+"\n")
log_f.close()'''
