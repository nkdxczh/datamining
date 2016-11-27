import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

f = open("/home/jason/datamining/datasets/user_log.csv","r")
#log_chunks = log_chunks.head(100)
size = 26258293
base = time.mktime(time.strptime("20160101","%Y%m%d"))

data = dict()

i = 0
for line in f:
    if i == 0:
        i += 1
        continue
    i += 1
    cnt = i
    if int(cnt / size * 100) != int((cnt - 1) / size * 100):
        print(int((cnt) / size * 100), '%')

    strs = line.split(",")
    user_id = int(strs[0])
    merchant_id = int(strs[3])
    item = int(strs[1])
    cat = int(strs[2])
    brand = int(strs[4])
    timestr = "2016" + str(strs[5])
    timev = time.mktime(time.strptime(timestr,"%Y%m%d")) - base
    action = int(strs[6])
    actions = [0 for z in range(4)]
    actions[action] = 1

    if not (user_id,merchant_id) in data:
        data[(user_id,merchant_id)] = [user_id, merchant_id, [item], [cat], [brand], timev, actions[0], actions[1], actions[2], actions[3]]
    else:
        data[(user_id,merchant_id)][2].append(item)
        data[(user_id,merchant_id)][3].append(cat)
        data[(user_id,merchant_id)][4].append(brand)
        data[(user_id,merchant_id)][5] += timev
        data[(user_id,merchant_id)][6] += actions[0]
        data[(user_id,merchant_id)][7] += actions[1]
        data[(user_id,merchant_id)][8] += actions[2]
        data[(user_id,merchant_id)][9] += actions[3]

f.close()

f = open("/home/jason/datamining/data/user_merchant_log1.csv","w")

f.write('user_id,merchant_id,user_merchant_items,user_merchant_cats,user_merchant_brands,user_merchant_time,user_merchant_action_0,user_merchant_action_1,user_merchant_action_2,user_merchant_action_3\n')
for i in data:
    f.write(str(data[i][0]) + ",")
    f.write(str(data[i][1]) + ",")
    data[i][2] = " ".join(str(x) for x in set(data[i][2]))
    data[i][3] = " ".join(str(x) for x in set(data[i][3]))
    data[i][4] = " ".join(str(x) for x in set(data[i][4]))
    f.write(data[i][2] + ",")
    f.write(data[i][3] + ",")
    f.write(data[i][4] + ",")
    data[i][5] = data[i][5]/(data[i][6] + data[i][7] + data[i][8] + data[i][9])
    f.write(str(data[i][5]) + ",")
    f.write(str(data[i][6]) + ",")
    f.write(str(data[i][7]) + ",")
    f.write(str(data[i][8]) + ",")
    f.write(str(data[i][9]) + "\n")

f.close()

