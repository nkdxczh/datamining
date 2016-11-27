result = open("/home/jason/datamining/data/result_bagging.csv","r")
test = open("/home/jason/datamining/datasets/test_label.csv","r")

output = open("/home/jason/datamining/data/test_result_bagging.csv","w")

first = test.readline().strip()
output.write(first + '\n')

i = 0

while i < 130500:
    i += 1
    test_record = test.readline().strip()
    result_record = result.readline().strip()
    output.write(test_record+result_record+'\n')

output.close()
test.close()
result.close()
