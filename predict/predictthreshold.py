import sys
sys.path.insert(0, '/home/jason/datamining/project')

from feature.read import *

from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data_file = "/home/jason/datamining/data/train_combine.csv"
    predict_file = open("/home/jason/datamining/data/result_bagging_train.csv",'r')

    pre = []
    for line in predict_file:
        pre.append(float(line))

    print('reading training and testing data...')    
    X, y = read_data(data_file)

    test = [1 for i in range(len(y))]
    print(min(pre),max(pre))

    thresholds = [i*0.005+0.01 for i in range(100)]
    thresholds = [i*0.001+0.01 for i in range(100)]
    loss = []
    minl = 2
    mint = 0
    for i in range(len(thresholds)):
        count = 0
        predict = []
        threshold = thresholds[i]
        for p in pre:
            '''if p < threshold:
                count += 1
                p = 0'''
            p = threshold
            predict.append(p)
        #print(count)
        m = log_loss(y,predict)
        if m < minl:
            minl = m
            mint = threshold
        loss.append(m)

    print(minl,mint)

    plt.plot(thresholds,loss)
    plt.show()

    
