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

    #print(xl)
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

x=dict()
x[0]=[5,12]
x[1]=[2,17]
x[2]=[1,12]
x[6]=[3,14]
x[54]=[32,11]
y=dict()
y[8]=[5,12]
y[10]=[2,17]
y[5]=[1,12]
y[7]=[3,14]
y[541]=[32,11]
'''
y[0]=[40,18]
y[7]=[7,13]
y[22]=[60,15]
y[6]=[10,14]
y[54]=[32,15]'''
print(similarity(x,y))

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
'''

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
'''
inputf = open("/home/jason/datamining/data/TFIDF/train.csv", 'r')

train = [dict() for i in range(merchant_size)]

for i in range(merchant_size):
    if i == 0:
        inputf.readline()
        continue
    cnt = i
    if int(cnt / merchant_size * 100) != int((cnt - 1) / merchant_size * 100):
        print(int((cnt) / merchant_size * 100), '%')

    line=inputf.readline()
    strs=line.rstrip().split(';')

    for s in strs:
        mt=s.split(',')
        if len(mt)<2:
            continue
        train[i][int(mt[0])]=int(mt[1])

inputf.close()
print("predict")
inputf = open("/home/jason/datamining/datasets/test_label.csv", 'r')
output = open("/home/jason/datamining/data/TFIDF/result.csv", 'w')

test_size=130502
miss=0
cnt=0
for i in range(test_size):
    if i == 0:
        line = inputf.readline()
        output.write(line)
        continue
    if cnt%1000 == 0:
        print(cnt)
    cnt += 1

    line = inputf.readline()
    strs = line.rstrip().split(',')
    ids = strs[0].split('#')
    if len(ids)<2:
        continue
    user = int(ids[0])
    merchant = int(ids[1])

    if user in train[merchant]:
        output.write(str(user)+'#'+str(merchant)+','+train[merchant][user]+'\n')
        continue

    if len(train[merchant])==0:
        miss+=1
        output.write(str(user)+'#'+str(merchant)+','+'0.06'+'\n')
        continue

    all_similarity=[]
    value=[]
    for k,v in train[merchant].items():
        tem_similarity=[]
        tem_similarity.append(similarity(users[k][0],users[user][0]))
        tem_similarity.append(similarity(users[k][1],users[user][1]))
        tem_similarity.append(similarity(users[k][2],users[user][2]))
        all_similarity.append(sum(tem_similarity)/3)
        value.append(v)

    sums=sum(all_similarity)

    predict = 0
    for z in range(len(all_similarity)):
        predict += (all_similarity[z]/sums)*value[z]

    output.write(str(user) + '#' + str(merchant) + ',' + str(predict) + '\n')

inputf.close()
output.close()
print(miss)