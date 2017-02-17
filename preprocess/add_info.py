import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

merchants = []
log_f = open("/home/jason/datamining/data/merchant_info.csv","r")
for line in log_f:
    strs = line.split(",")
    age = float(strs[0])
    gender = float(strs[1])
    merchants.append([age,gender])
log_f.close()

put = open("/home/jason/datamining/data/test_predict_combine.csv",'r')
output = open("/home/jason/datamining/data/test_all_combine.csv",'w')
i=0
for line in put:
    if i==0:
        output.write(line.rstrip())
        output.write(",merchant_age,merchant_gender,user_merchant_age,user_merchant_gender\n")
        i+=1
        continue
    strs=line.split(",")
    merchant_age = merchants[int(strs[1])][0]
    merchant_gender = merchants[int(strs[1])][1]
    user_age = float(strs[2])
    user_gender = float(strs[3])
    user_merchant_age = abs(user_age-merchant_age)
    user_merchant_gender = abs(user_gender-merchant_gender)
    output.write(line.rstrip())
    output.write(","+str(merchant_age)+","+str(merchant_gender)+","+str(user_merchant_age)+","+str(user_merchant_gender)+"\n")

output.close()
