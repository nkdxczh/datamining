import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

log_chunks = pd.read_csv("/home/jason/datamining/datasets/user_log.csv")
#log_chunks = log_chunks.head(10)
size = len(log_chunks)
base = time.mktime(time.strptime("20160101","%Y%m%d"))

columns = ['user_id', 'merchant_id', 'user_merchant_items', 'user_merchant_cats', 'user_merchant_brands', 'user_merchant_time', 'user_merchant_action_0', 'user_merchant_action_1', 'user_merchant_action_2', 'user_merchant_action_3']

my_log_chunks = pd.DataFrame(columns = columns)

data = dict()

for i in range(len(log_chunks)):
    cnt = i
    if int(cnt / size * 100) != int((cnt - 1) / size * 100):
        print(int((cnt) / size * 100), '%')

    user_id = log_chunks.loc[i,'user_id']
    merchant_id = log_chunks.loc[i,'merchant_id']
    item = log_chunks.loc[i,'item_id']
    cat = log_chunks.loc[i,'cat_id']
    brand = log_chunks.loc[i,'brand_id']
    timestr = "2016" + str(log_chunks.loc[i,'time_stamp'])
    timev = time.mktime(time.strptime(timestr,"%Y%m%d")) - base
    action = log_chunks.loc[i,'action_type']
    actions = [0 for i in range(4)]
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

j = 0
for i in data:
    data[i][5] = data[i][5]/(data[i][6] + data[i][7] + data[i][8] + data[i][9])
    data[i][2] = " ".join(str(x) for x in set(data[i][2]))
    data[i][3] = " ".join(str(x) for x in set(data[i][3]))
    data[i][4] = " ".join(str(x) for x in set(data[i][4]))
    my_log_chunks.loc[j] = data[i]
    j += 1

my_log_chunks.to_csv("/home/jason/datamining/data/user_merchant_log.csv", index = False)
