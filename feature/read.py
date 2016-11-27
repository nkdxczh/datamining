import numpy as np
import pandas as pd

from sklearn import preprocessing

def read_data(data_file):    
    data = pd.read_csv(data_file)  
    y = data.prob  
    x = data.drop(['prob'], axis=1)  
    #x = preprocessing.normalize(x) 
    return x, y
