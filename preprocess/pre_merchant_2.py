import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

file_name = "./output"
f = open(file_name)
user_size = 212063
buff = ["" for i in range(user_size)]

i = 0
for line in f:
    buff[i] = line
    i += 1

f.close()

user_chunks = pd.read_csv("/home/jason/datamining/datasets/user_info.csv", dtype = np.int32)

user_chunks["user_merchants"] = ""
user_chunks["user_items"] = ""
user_chunks["user_cats"] = ""
user_chunks["user_brands"] = ""
user_chunks["user_time"] = 0
user_chunks["user_action_0"] = 0
user_chunks["user_action_1"] = 0
user_chunks["user_action_2"] = 0
user_chunks["user_action_3"] = 0

base = time.mktime(time.strptime("20160101","%Y%m%d"))

for i in range(len(user_chunks)):
    #print(i/user_size)
    cnt = i
    if int(cnt / user_size * 100) != int((cnt - 1) / user_size * 100):
        print(int((cnt) / user_size * 100), '%')

    user_id = user_chunks.iloc[i,0]
    age_range = user_chunks.iloc[i,1]
    gender = user_chunks.iloc[i,2]
    query_lines = buff[int(user_id)].split(";")

    items = []
    merchants = []
    cats = []
    brands = []
    timev = 0
    actions = [0 for z in range(4)]
    
    for j in range(len(query_lines)):
        #print(i)
        query_items = query_lines[j].split(",")
        if len(query_items) != 7:
            continue
        timestr = "2016" + str(query_items[5])
        timev += time.mktime(time.strptime(timestr,"%Y%m%d")) - base
        actions[int(query_items[6])] += 1 

        #if not merchant in merchants:
        merchants.append(query_items[3])
        #if not item in items:
        items.append(query_items[1])
        #if not cat in cats:
        cats.append(query_items[2])
        #if not brand in brands:
        brands.append(query_items[4])
        '''if not merchant in user_chunks.loc[:,'user_merchants'][i]:
        user_chunks.loc[:,'user_merchants'][i] = " ".join(str(x) for x in merchants)
        user_chunks.loc[:,'user_cats'][i] = " ".join(str(x) for x in cats)
        user_chunks.loc[:,'user_brands'][i] = " ".join(str(x) for x in brands)
        if not item in user_chunks.loc[:,'user_items'][i]:
            user_chunks.loc[:,'user_items'][i].append(item)
        if not cat in user_chunks.loc[:,'user_cats'][i]:
            user_chunks.loc[:,'user_cats'][i].append(cat)
        if not brand in user_chunks.loc[:,'user_merchants'][i]:
            user_chunks.loc[:,'user_brands'][i].append(brand)'''

    user_chunks.iloc[i] = [user_id,age_range,gender," ".join(str(x) for x in set(merchants))," ".join(str(x) for x in set(items))," ".join(str(x) for x in set(cats))," ".join(str(x) for x in set(brands)),timev / sum(actions),actions[0],actions[1],actions[2],actions[3]]

user_chunks.to_csv("/home/jason/datamining/data/user_log.csv", index = False)
