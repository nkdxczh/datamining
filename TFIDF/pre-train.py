inputf = open("/home/jason/datamining/datasets/train_label.csv", 'r')

user_size = 212063

train_size = 130365
#train_size = 10
merchant_size = 4996

users = [dict() for i in range(user_size)]
merchants = [dict() for i in range(merchant_size)]

for i in range(train_size):
    if i == 0:
        inputf.readline()
        continue
    cnt = i
    if int(cnt / train_size * 100) != int((cnt - 1) / train_size * 100):
        print(int((cnt) / train_size * 100), '%')

    line = inputf.readline()
    strs = line.rstrip().split(',')
    ids = strs[0].split('#')
    user = int(ids[0])
    merchant = int(ids[1])

    #users[user][merchant]=int(strs[1])
    merchants[merchant][user]=int(strs[1])
    #print(line,user,merchant,strs[1])

inputf.close()

output = open("/home/jason/datamining/data/TFIDF/train.csv", 'w')

for i in range(merchant_size):
    if len(merchants[i])==0:
        output.write('\n')
        continue

    for k,v in merchants[i].items():
        output.write(str(k)+','+str(v)+';')
    output.write('\n')

output.close()