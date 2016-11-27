file_name = "/home/jason/datamining/datasets/user_log.csv"
output_name = "./output"
f = open(file_name, 'r')
size = 212063
big_size = 26258293
buff = ["" for i in range(size)]
cnt = 0

for line in f:
    cnt += 1
    if int(cnt / big_size * 100) != int((cnt - 1) / big_size * 100):
        print(int((cnt - 1) / big_size * 100), '%')

    try:
        buff[int(line.split(',')[0])] += line[:-1] + ';';
    except:
        pass

f.close()

f = open(output_name, 'a')
for i in buff:
    f.write(i + '\n')
f.close()

