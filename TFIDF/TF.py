import warnings

warnings.filterwarnings("ignore")

log_file = open("/home/jason/datamining/datasets/user_log.csv", 'r')

user_size = 212063
merchant_size = 4996
log_size = 26258293
#log_size = 100
users = [[dict(),dict(),dict(),0] for i in range(user_size)]
merchants = [[dict(),dict(),dict(),0] for i in range(merchant_size)]

for i in range(log_size):
    if i == 0:
        log_file.readline()
        continue
    cnt = i
    if int(cnt / log_size * 100) != int((cnt - 1) / log_size * 100):
        print(int((cnt) / log_size * 100), '%')

    line=log_file.readline()
    strs=line.split(',')

    user = int(strs[0])
    item = int(strs[1])
    cat = int(strs[2])
    merchant = int(strs[3])
    brand = int(strs[4])

    users[user][3]+=1
    merchants[merchant][3]+=1

    if item not in users[user][0]:
        users[user][0][item]=1
    else:
        users[user][0][item]+=1

    if cat not in users[user][1]:
        users[user][1][cat]=1
    else:
        users[user][1][cat]+=1

    if brand not in users[user][2]:
        users[user][2][brand]=1
    else:
        users[user][2][brand]+=1

    if item not in merchants[merchant][0]:
        merchants[merchant][0][item]=1
    else:
        merchants[merchant][0][item]+=1

    if cat not in merchants[merchant][1]:
        merchants[merchant][1][cat]=1
    else:
        merchants[merchant][1][cat]+=1

    if brand not in merchants[merchant][2]:
        merchants[merchant][2][brand]=1
    else:
        merchants[merchant][2][brand]+=1

log_file.close()

item_size = 916774
cat_size = 1582
brand_size = 8163

items = [0.0 for i in range(item_size)]
cats = [0.0 for i in range(cat_size)]
brands = [0.0 for i in range(brand_size)]

inputf = open("/home/jason/datamining/data/TFIDF/item.csv", 'r')
for i in range(item_size):
    line = inputf.readline()
    if len(line.rstrip()) == 0:
        continue
    items[i] = float(line.rstrip())
inputf.close()

inputf = open("/home/jason/datamining/data/TFIDF/cat.csv", 'r')
for i in range(cat_size):
    line = inputf.readline()
    if len(line.rstrip()) == 0:
        continue
    cats[i] = float(line.rstrip())
inputf.close()

inputf = open("/home/jason/datamining/data/TFIDF/brand.csv", 'r')
for i in range(brand_size):
    line = inputf.readline()
    if len(line.rstrip()) == 0:
        continue
    brands[i] = float(line.rstrip())
inputf.close()

output = open("/home/jason/datamining/data/TFIDF/user.csv", 'w')

ucount = 100;

for i in range(user_size):
    if users[i][3]==0:
        output.write('\n')
        continue
    if users[i][3]<ucount:
        ucount=users[i][3]
    uitems = users[i][0]
    j = 0
    for k,v in uitems.items():
        if j != 0:
            output.write(',')
        output.write(str(k)+':'+str(v/users[i][3])+'|'+str((v/users[i][3])*items[k]))
        j+=1
    output.write(';')

    ucats = users[i][1]
    j = 0
    for k,v in ucats.items():
        if j != 0:
            output.write(',')
        output.write(str(k)+':'+str(v/users[i][3])+'|'+str((v/users[i][3])*cats[k]))
        j+=1
    output.write(';')

    ubrands = users[i][2]
    j = 0
    for k,v in ubrands.items():
        if j != 0:
            output.write(',')
        output.write(str(k)+':'+str(v/users[i][3])+'|'+str((v/users[i][3])*brands[k]))
        j+=1
    output.write('\n')
    
output.close()

output = open("/home/jason/datamining/data/TFIDF/merchant.csv", 'w')
mcount=100
for i in range(merchant_size):
    if merchants[i][3]==0:
        output.write('\n')
        continue
    if merchants[i][3]<mcount:
        mcount=merchants[i][3]
    uitems = merchants[i][0]
    j = 0
    for k,v in uitems.items():
        if j != 0:
            output.write(',')
        output.write(str(k)+':'+str(v/merchants[i][3])+'|'+str((v/merchants[i][3])*items[k]))
        j+=1
    output.write(';')

    ucats = merchants[i][1]
    j = 0
    for k,v in ucats.items():
        if j != 0:
            output.write(',')
        output.write(str(k)+':'+str(v/merchants[i][3])+'|'+str((v/merchants[i][3])*cats[k]))
        j+=1
    output.write(';')

    ubrands = merchants[i][2]
    j = 0
    for k,v in ubrands.items():
        if j != 0:
            output.write(',')
        output.write(str(k)+':'+str(v/merchants[i][3])+'|'+str((v/merchants[i][3])*brands[k]))
        j+=1
    output.write('\n')
    
output.close()

print(ucount,mcount)

