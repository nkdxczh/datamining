log_file = open("/home/jason/datamining/datasets/user_log.csv", 'r')

log_size = 26258293

merchant_size=0

for i in range(log_size):
    if i == 0:
        log_file.readline()
        continue
    cnt = i
    if int(cnt / log_size * 100) != int((cnt - 1) / log_size * 100):
        print(int((cnt) / log_size * 100), '%')

    line=log_file.readline()
    strs=line.split(',')

    merchant_id = int(strs[3])

    if merchant_id > merchant_size:
        merchant_size = merchant_id

print(merchant_size)
