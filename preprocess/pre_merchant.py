import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

columns = ['merchant_id', 'merchant_items', 'merchant_cats', 'merchant_brands', 'merchant_time', 'merchant_action_0', 'merchant_action_1', 'merchant_action_2', 'merchant_action_3','merchant_users']
merchant_chunks = pd.DataFrame(columns = columns)

log_chunks = pd.read_csv("/home/jason/datamining/datasets/user_log.csv", dtype = np.int32)
size = len(log_chunks)
#log_chunks = log_chunks.head(100)

base = time.mktime(time.strptime("20160101","%Y%m%d"))

data = dict()

for i in range(len(log_chunks)):
    cnt = i
    if int(cnt / size * 100) != int((cnt - 1) / size * 100):
        print(int((cnt) / size * 100), '%')
    user = log_chunks.loc[i,'user_id']
    merchant_id = log_chunks.loc[i,'merchant_id']
    item = log_chunks.loc[i,'item_id']
    cat = log_chunks.loc[i,'cat_id']
    brand = log_chunks.loc[i,'brand_id']
    timestr = "2016" + str(log_chunks.loc[i,'time_stamp'])
    timev = time.mktime(time.strptime(timestr,"%Y%m%d")) - base
    action = log_chunks.loc[i,'action_type']
    actions = [0 for i in range(4)]
    actions[action] = 1

    if not merchant_id in data:
        data[merchant_id] = [merchant_id, [item], [cat], [brand], timev, actions[0], actions[1], actions[2], actions[3], [user]]
    else:
        data[merchant_id][1].append(item)
        data[merchant_id][2].append(cat)
        data[merchant_id][3].append(brand)
        data[merchant_id][4] += timev
        data[merchant_id][5] += actions[0]
        data[merchant_id][6] += actions[1]
        data[merchant_id][7] += actions[2]
        data[merchant_id][8] += actions[3]
        data[merchant_id][9].append(user)

for i in data:
    data[i][4] = data[i][4]/(data[i][5] + data[i][6] + data[i][7] + data[i][8])
    data[i][1] = " ".join(str(x) for x in set(data[i][1]))
    data[i][2] = " ".join(str(x) for x in set(data[i][2]))
    data[i][3] = " ".join(str(x) for x in set(data[i][3]))
    data[i][9] = " ".join(str(x) for x in set(data[i][9]))
    merchant_chunks.loc[i] = data[i]

merchant_chunks.to_csv("/home/jason/datamining/data/merchant_log.csv", index = False)
