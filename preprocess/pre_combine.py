import numpy as np
import pandas as pd
import collections

user_log = dict()

f = open("/home/jason/datamining/data/user_log.csv","r")

for line in f:
    try:
        strs = line.rstrip().split(",")
        user_id = int(strs[0])
        age_range = int(strs[1])
        gender = int(strs[2])
        user_merchants = [int(i) for i in strs[3].split(" ")]
        user_items = [int(i) for i in strs[4].split(" ")]
        user_cats = [int(i) for i in strs[5].split(" ")]
        user_brands = [int(i) for i in strs[6].split(" ")]
        user_time = float(strs[7])
        user_action_0 = int(strs[8])
        user_action_1 = int(strs[9])
        user_action_2 = int(strs[10])
        user_action_3 = int(strs[11])
        user_log[user_id] = [age_range, gender, user_merchants, user_items, user_cats, user_brands, user_time, user_action_0, user_action_1, user_action_2, user_action_3]
    except:
        continue

f.close()
print("user",len(user_log))

merchant_log = dict()

f = open("/home/jason/datamining/data/merchant_log.csv","r")

for line in f:
    try:
        strs = line.rstrip().split(",")
        merchant_id = int(float(strs[0]))
        merchant_items = [int(i) for i in strs[1].split(" ")]
        merchant_cats = [int(i) for i in strs[2].split(" ")]
        merchant_brands = [int(i) for i in strs[3].split(" ")]
        merchant_time = float(strs[4])
        merchant_action_0 = int(float(strs[5]))
        merchant_action_1 = int(float(strs[6]))
        merchant_action_2 = int(float(strs[7]))
        merchant_action_3 = int(float(strs[8]))
        merchants_user = [int(i) for i in strs[9].split(" ")]
        merchant_log[merchant_id] = [merchants_user, merchant_items, merchant_cats, merchant_brands, merchant_time, merchant_action_0, merchant_action_1, merchant_action_2, merchant_action_3]
    except:
        continue

f.close()
print("merchant",len(merchant_log))

user_merchant_log = dict()

f = open("/home/jason/datamining/data/user_merchant_log1.csv","r")

for line in f:
    try:
        strs = line.rstrip().split(",")
        user_id = int(float(strs[0]))
        merchant_id = int(float(strs[1]))
        user_merchant_items = len([int(i) for i in strs[2].split(" ")])
        user_merchant_cats = len([int(i) for i in strs[3].split(" ")])
        user_merchant_brands = len([int(i) for i in strs[4].split(" ")])
        user_merchant_time = float(strs[5])
        user_merchant_action_0 = int(float(strs[6]))
        user_merchant_action_1 = int(float(strs[7]))
        user_merchant_action_2 = int(float(strs[8]))
        user_merchant_action_3 = int(float(strs[9]))
        user_merchant_log[(user_id,merchant_id)] = [user_merchant_items, user_merchant_cats, user_merchant_brands, user_merchant_time, user_merchant_action_0, user_merchant_action_1, user_merchant_action_2, user_merchant_action_3]
    except:
        continue

f.close()
print("user_merchant",len(user_merchant_log))


combine_log = dict()

columns = ['user_id', 'merchant_id', 'age_range', 'gender', 'user_merchants', 'user_items', 'user_cats', 'user_brands', 'user_time', 'user_action_0', 'user_action_1', 'user_action_2', 'user_action_3', 'merchants_user', 'merchant_items', 'merchant_cats', 'merchant_brands', 'merchant_time', 'merchant_action_0', 'merchant_action_1', 'merchant_action_2', 'merchant_action_3', 'user_merchant_items_overlap', 'user_merchant_cats_overlap', 'user_merchant_brands_overlap', 'user_merchant_items', 'user_merchant_cats', 'user_merchant_brands', 'user_merchant_time', 'user_merchant_action_0', 'user_merchant_action_1', 'user_merchant_action_2', 'user_merchant_action_3']

combine_chunks = pd.DataFrame(columns = columns)

f = open("/home/jason/datamining/data/test_label.csv","r")

out = open("/home/jason/datamining/data/test_combine.csv","w")

out.write('user_id,merchant_id,age_range,gender,user_merchants,user_items,user_cats,user_brands,user_time,user_action_0,user_action_1,user_action_2,user_action_3,merchants_user,merchant_items,merchant_cats,merchant_brands,merchant_time,merchant_action_0,merchant_action_1,merchant_action_2,merchant_action_3,user_merchant_items_overlap,user_merchant_cats_overlap,user_merchant_brands_overlap,user_merchant_items,user_merchant_cats,user_merchant_brands,user_merchant_time,user_merchant_action_0,user_merchant_action_1,user_merchant_action_2,user_merchant_action_3,prob\n')

j = 0
for line in f:
    '''if j > 2:
        break'''
    try:
        strs = line.rstrip().split(",")
        prob = int(strs[0])
        user_id = int(strs[1])
        merchant_id = int(strs[2])

        age_range = user_log[user_id][0]
        gender = user_log[user_id][1]
        user_merchants = len(user_log[user_id][2])
        user_items = len(user_log[user_id][3])
        user_cats  = len(user_log[user_id][4])
        user_brands = len(user_log[user_id][5])
        user_time = user_log[user_id][6]
        user_action_0 = user_log[user_id][7]
        user_action_1 = user_log[user_id][8]
        user_action_2 = user_log[user_id][9]
        user_action_3 = user_log[user_id][10]

        merchants_user = len(merchant_log[merchant_id][0])
        merchant_items = len(merchant_log[merchant_id][1])
        merchant_cats = len(merchant_log[merchant_id][2])
        merchant_brands = len(merchant_log[merchant_id][3])
        merchant_time = merchant_log[merchant_id][4]
        merchant_action_0 = merchant_log[merchant_id][5]
        merchant_action_1 = merchant_log[merchant_id][6]
        merchant_action_2 = merchant_log[merchant_id][7]
        merchant_action_3 = merchant_log[merchant_id][8]

        if (user_id,merchant_id) in user_merchant_log:
            user_merchant_items = user_merchant_log[(user_id,merchant_id)][0]
            user_merchant_cats = user_merchant_log[(user_id,merchant_id)][1]
            user_merchant_brands = user_merchant_log[(user_id,merchant_id)][2]
            user_merchant_time = user_merchant_log[(user_id,merchant_id)][3]
            user_merchant_action_0 = user_merchant_log[(user_id,merchant_id)][4]
            user_merchant_action_1 = user_merchant_log[(user_id,merchant_id)][5]
            user_merchant_action_2 = user_merchant_log[(user_id,merchant_id)][6]
            user_merchant_action_3 = user_merchant_log[(user_id,merchant_id)][7]
        else:
            user_merchant_items = 0
            user_merchant_cats = 0
            user_merchant_brands = 0
            user_merchant_time = 0
            user_merchant_action_0 = 0
            user_merchant_action_1 = 0
            user_merchant_action_2 = 0
            user_merchant_action_3 = 0

        a_multiset = collections.Counter(user_log[user_id][3])
        b_multiset = collections.Counter(merchant_log[merchant_id][1])
        user_merchant_items_overlap = len(list((a_multiset & b_multiset).elements()))

        a_multiset = collections.Counter(user_log[user_id][4])
        b_multiset = collections.Counter(merchant_log[merchant_id][2])
        user_merchant_cats_overlap = len(list((a_multiset & b_multiset).elements()))

        a_multiset = collections.Counter(user_log[user_id][5])
        b_multiset = collections.Counter(merchant_log[merchant_id][3])
        user_merchant_brands_overlap = len(list((a_multiset & b_multiset).elements()))
        
        #combine_log[(user_id,merchant_id)] = [user_id, merchant_id, age_range, gender, user_merchants, user_items, user_cats, user_brands, user_time, user_action_0, user_action_1, user_action_2, user_action_3, merchants_user, merchant_items, merchant_cats, merchant_brands, merchant_time, merchant_action_0, merchant_action_1, merchant_action_2, merchant_action_3, user_merchant_items_overlap, user_merchant_cats_overlap, user_merchant_brands_overlap, user_merchant_items, user_merchant_cats, user_merchant_brands, user_merchant_time, user_merchant_action_0, user_merchant_action_1, user_merchant_action_2, user_merchant_action_3]
        out.write(str(user_id) + "," + str(merchant_id) + "," + str(age_range) + "," + str(gender) + "," + str(user_merchants) + "," + str(user_items) + "," + str(user_cats) + "," + str(user_brands) + "," + str(user_time) + "," + str(user_action_0) + "," + str(user_action_1) + "," + str(user_action_2) + "," + str(user_action_3) + "," + str(merchants_user) + "," + str(merchant_items) + "," + str(merchant_cats) + "," + str(merchant_brands) + "," + str(merchant_time) + "," + str(merchant_action_0) + "," + str(merchant_action_1) + "," + str(merchant_action_2) + "," + str(merchant_action_3) + "," + str(user_merchant_items_overlap) + "," + str(user_merchant_cats_overlap) + "," + str(user_merchant_brands_overlap) + "," + str(user_merchant_items) + "," + str(user_merchant_cats) + "," + str(user_merchant_brands) + "," + str(user_merchant_time) + "," + str(user_merchant_action_0) + "," + str(user_merchant_action_1) + "," + str(user_merchant_action_2) + "," + str(user_merchant_action_3) + "," + str(prob) + "\n")

        #combine_chunks[j] = combine_log[(user_id,merchant_id)]
        j += 1

    except:
        continue

out.close()
f.close()
print("combine_log",j)

#combine_chunks.to_csv("/home/jason/datamining/data/train_combine.csv")
