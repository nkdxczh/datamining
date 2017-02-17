import numpy as np
import pandas as pd

from sklearn import preprocessing

def read_data(data_file):    
    data = pd.read_csv(data_file)  
    y = data.prob  
    x = data.drop(['prob'], axis=1)  
    x = x.drop(['user_time'], axis=1)  
    x = x.drop(['user_id'], axis=1)  
    x = x.drop(['merchant_time'], axis=1)  
    x = x.drop(['user_merchant_time'], axis=1)  
    x = x.drop(['merchant_id'], axis=1)  
    x = x.drop(['LOR'], axis=1)  
    x = x.drop(['GNB'], axis=1) 
    x = x.drop(['KNN'], axis=1) 
    x = x.drop(['DT'], axis=1) 
    x = x.drop(['ET'], axis=1) 
    x = x.drop(['RF'], axis=1) 
    x = x.drop(['GB'], axis=1) 
    x = x.drop(['LD'], axis=1)
    x = x.drop(['QD'], axis=1)
    x = x.drop(['NN'], axis=1)
    x = x.drop(['XG'], axis=1)
    x = x.drop(['VT'], axis=1)
    #x = preprocessing.normalize(x) 
    return x, y, x.columns.values
