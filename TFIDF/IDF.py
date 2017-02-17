import math
import warnings

warnings.filterwarnings("ignore")

user_size = 212063
merchant_size = 4995
log_size = 26258293
#log_size = 100

log_file = open("/home/jason/datamining/datasets/user_log.csv", 'r')

item_size = 916774
cat_size = 1582
brand_size = 8162

items = [0.0 for i in range(item_size)]
cats = [0.0 for i in range(cat_size)]
brands = [0.0 for i in range(brand_size)]

for i in range(log_size):
    if i == 0:
        log_file.readline()
        continue
    cnt = i
    if int(cnt / log_size * 100) != int((cnt - 1) / log_size * 100):
        print(int((cnt) / log_size * 100), '%')

    line=log_file.readline()
    strs=line.split(',')

    user_id = int(strs[0])
    item = int(strs[1])
    cat = int(strs[2])
    merchant_id = int(strs[3])
    brand = int(strs[4])

    items[item]+=1
    cats[cat]+=1
    brands[brand]+=1

log_file.close()

items = list(map(lambda x: x/(log_size-1), items))
item_file = open("/home/jason/datamining/data/TFIDF/item.csv", 'w')
for i in items:
    if i==0:
        item_file.write('\n')
        continue
    item_file.write(str(math.log(1/i))+'\n')
item_file.close()

cats = list(map(lambda x: x/(log_size-1), cats))
cat_file = open("/home/jason/datamining/data/TFIDF/cat.csv", 'w')
for i in cats:
    if i==0:
        cat_file.write('\n')
        continue
    cat_file.write(str(math.log(1/i))+'\n')
cat_file.close()

brands = list(map(lambda x: x/(log_size-1), brands))
brand_file = open("/home/jason/datamining/data/TFIDF/brand.csv", 'w')
for i in brands:
    if i==0:
        brand_file.write('\n')
        continue
    brand_file.write(str(math.log(1/i))+'\n')
brand_file.close()
