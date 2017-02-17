if __name__ == '__main__':
    raw_data = open("/home/jason/datamining/data/test_combine.csv",'r')
    predict_data = open("/home/jason/datamining/data/predict_test_combine.csv",'r')
    output = open("/home/jason/datamining/data/test_predict_combine.csv",'w')

    l1 = raw_data.readline()
    output.write(l1.rstrip())
    output.write(',LOR,GNB,KNN,DT,ET,RF,GB,LD,QD,NN,XG,VT')
    output.write('\n')

    i = 130501
    l1 = raw_data.readline()
    while i > 1:
        i -= 1
        l2 = predict_data.readline()
        l2 = l2[:len(l2)-2]
        output.write(l1.rstrip())
        output.write(',')
        output.write(l2.rstrip())
        output.write('\n')
        l1 = raw_data.readline()

    output.close()

    raw_data = open("/home/jason/datamining/data/train_combine.csv",'r')
    predict_data = open("/home/jason/datamining/data/predict_train_combine.csv",'r')
    output = open("/home/jason/datamining/data/train_predict_combine.csv",'w')

    l1 = raw_data.readline()
    output.write(l1.rstrip())
    output.write(',LOR,GNB,KNN,DT,ET,RF,GB,LD,QD,NN,XG,VT')
    output.write('\n')

    i = 130365
    l1 = raw_data.readline()
    while i > 1:
        i -= 1
        l2 = predict_data.readline()
        l2 = l2[:len(l2)-2]
        output.write(l1.rstrip())
        output.write(',')
        output.write(l2.rstrip())
        output.write('\n')
        l1 = raw_data.readline()

    output.close()
