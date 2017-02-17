f = open("/home/jason/datamining/data/train_combine.csv",'r')
o1 = open("/home/jason/datamining/data/train_combine_nolog.csv",'w')
o2 = open("/home/jason/datamining/data/train_combine_log.csv",'w')
i = 0
i1 = 0
i2 = 0
m = 0
lc = [0 for i in range(21)]
for line in f:
    i += 1
    if i == 1:
        o1.write(line)
        o2.write(line)
        continue

    s = line.split(',')
    c = int(s[32])
    if c > 20:
        tem = 20
    else:
        tem = c
    lc[tem] += 1
    if c <= 0:
        o1.write(line)
        i1 += 1
    else:
        o2.write(line)
        i2 += 1

o1.close()
o2.close()
print(i1,i2)
print(lc)
