from scipy import spatial

def similarity(x,y):
    xl=[]
    for k,v in x.items():
        xl.append([k,v[0],v[1]])
    xl=sorted(xl,key=lambda x:x[2],reverse=True)

    yl=[]
    for k,v in y.items():
        yl.append([k,v[0],v[1]])
    yl=sorted(yl,key=lambda x:x[2],reverse=True)

    candidate=set()
    for i in range(5):
        if i<len(xl):
            candidate.add(xl[i][0])
        if i<len(yl):
            candidate.add(yl[i][0])

    xv=[]
    for i in candidate:
        if i not in x:
            xv.append(0)
        else:
            xv.append(x[i][0])
    yv=[]
    for i in candidate:
        if i not in y:
            yv.append(0)
        else:
            yv.append(y[i][0])

    return 1-spatial.distance.cosine(xv,yv)

user_size = 212063
merchant_size = 4996
users = [[dict(),dict(),dict()] for i in range(user_size)]
merchants = [[dict(),dict(),dict()] for i in range(merchant_size)]

print('reading user')

inputf = open("/home/jason/datamining/data/TFIDF/user.csv", 'r')

for i in range(user_size):
    if i == 0:
        inputf.readline()
        continue
    cnt = i
    if int(cnt / user_size * 100) != int((cnt - 1) / user_size * 100):
        print(int((cnt) / user_size * 100), '%')

    line=inputf.readline()
    strs=line.rstrip().split(';')

    items=strs[0]
    item_list = items.split(',')
    for j in item_list:
        kv = j.split(':')
        v = kv[1].split('|')
        users[i][0][int(kv[0])]=[float(v[0]),float(v[1])]

    items = strs[1]
    item_list = items.split(',')
    for j in item_list:
        kv = j.split(':')
        v = kv[1].split('|')
        users[i][1][int(kv[0])] = [float(v[0]), float(v[1])]

    items = strs[2]
    item_list = items.split(',')
    for j in item_list:
        kv = j.split(':')
        v = kv[1].split('|')
        users[i][2][int(kv[0])] = [float(v[0]), float(v[1])]

inputf.close()

print('reading merchant')

inputf = open("/home/jason/datamining/data/TFIDF/merchant.csv", 'r')

for i in range(merchant_size):
    if i == 0:
        inputf.readline()
        continue
    cnt = i
    if int(cnt / merchant_size * 100) != int((cnt - 1) / merchant_size * 100):
        print(int((cnt) / merchant_size * 100), '%')

    line=inputf.readline()
    strs=line.rstrip().split(';')

    items=strs[0]
    item_list = items.split(',')
    for j in item_list:
        kv = j.split(':')
        v = kv[1].split('|')
        merchants[i][0][int(kv[0])]=[float(v[0]),float(v[1])]

    items=strs[1]
    item_list = items.split(',')
    for j in item_list:
        kv = j.split(':')
        v = kv[1].split('|')
        merchants[i][1][int(kv[0])]=[float(v[0]),float(v[1])]

    items=strs[2]
    item_list = items.split(',')
    for j in item_list:
        kv = j.split(':')
        v = kv[1].split('|')
        merchants[i][2][int(kv[0])]=[float(v[0]),float(v[1])]

inputf.close()

inputf = open("/home/jason/datamining/datasets/train_label.csv", 'r')
output = open("/home/jason/datamining/data/TFIDF/newfeatures_train.csv", 'w')

test_size=130502
train_size = 130365
miss=0
cnt=0
for i in range(train_size):
    if i == 0:
        line = inputf.readline()
        output.write(line)
        continue
    if cnt%1000 == 0:
        print(cnt)
    cnt+=1

    line = inputf.readline()
    strs = line.rstrip().split(',')
    ids = strs[0].split('#')
    if len(ids)<2:
        continue
    user = int(ids[0])
    merchant = int(ids[1])

    tem_similarity=[]
    tem_similarity.append(similarity(users[user][0],merchants[merchant][0]))
    tem_similarity.append(similarity(users[user][1],merchants[merchant][1]))
    tem_similarity.append(similarity(users[user][2],merchants[merchant][2]))

    output.write(str(tem_similarity[0]) + ',' + str(tem_similarity[1]) + ',' + str(tem_similarity[2]) + '\n')

inputf.close()
output.close()
print(miss)

