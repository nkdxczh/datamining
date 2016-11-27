import math

def eveluate(y, predict):
    print(y.iloc[0],predict[0])
    print(len(y),len(predict))
    loss = 0
    for i in range(len(y)):
        if y.iloc[i] == 1:
            loss += math.log(predict[i][1])
        else:
            loss += math.log(predict[i][0])
    return -loss / len(y)
