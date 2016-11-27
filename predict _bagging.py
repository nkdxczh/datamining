import os
import time
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random
import math
import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score

# Linear Regression Classifier
def linear_regression_classifier(train_x, train_y):
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)
    return model
 
# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    model = MultinomialNB()

    param_grid = {'alpha': [math.pow(10,-i) for i in range(11)]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    
    model = MultinomialNB(alpha = best_parameters['alpha'])  
    model.fit(train_x, train_y)
    return model
 
 
# KNN Classifier
def knn_classifier(train_x, train_y):
    model = KNeighborsClassifier()

    param_grid = {'n_neighbors': list(range(1,21))}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    
    model = KNeighborsClassifier(n_neighbors = best_parameters['n_neighbors'])

    bagging = BaggingClassifier(model, max_samples=0.5, max_features=0.5 )
    bagging.fit(train_x, train_y)
    return bagging
 
 
# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model
 
 
# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    model = RandomForestClassifier()

    param_grid = {'n_estimators': list(range(1,21))}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    
    model = RandomForestClassifier(n_estimators = best_parameters['n_estimators'])
    
    model.fit(train_x, train_y)
    return model
 
 
# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)

    bagging = BaggingClassifier(model, max_samples=0.5, max_features=0.5 )
    bagging.fit(train_x, train_y)
    return bagging
 
 
# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    model = GradientBoostingClassifier()
    
    model = RandomForestClassifier()

    param_grid = {'n_estimators': list(range(100,300,10))}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    
    model = RandomForestClassifier(n_estimators = best_parameters['n_estimators'])

    model.fit(train_x, train_y)
    return model

# SVM Classifier
def svm_classifier(train_x, train_y):
    model = SVC(kernel='linear', probability=True)
    model.fit(train_x, train_y)
    return model
 
# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    model = SVC(kernel='linear', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    #for para, val in best_parameters.items():
        #print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def read_data(data_file):    
    data = pd.read_csv(data_file)  
    train = data 
    test = data[int(len(data)*0.9):]  
    train_y = train.prob  
    train_x = train.drop(['prob'], axis=1)  
    train_x = preprocessing.normalize(train_x)
    #train_x = train.loc[:, ['gender','a1','a2','a3','catsCount']]
    test_y = test.prob  
    test_x = test.drop(['prob'], axis=1)  
    test_x = preprocessing.normalize(test_x)
    #test_x = test.loc[:, ['gender','a1','a2','a3','catsCount']]  
    return train_x, train_y, test_x, test_y
        
def feature_select(x,y):
    clf = ExtraTreesClassifier()
    clf = clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    return model

if __name__ == '__main__':
    data_file = "/home/jason/datamining/data/train_combine.csv"    
    thresh = 0.5    
    model_save_file = "/home/jason/datamining/model/models"     
    model_save = {}
    result_save_file = '/home/jason/datamining/result/results' 
     
    test_classifiers = ['DT']    
    classifiers = { 'DT':decision_tree_classifier
    }
        
    print('reading training and testing data...')    
    train_x, train_y, test_x, test_y = read_data(data_file)
    select_model = feature_select(train_x, train_y)
    train_x = select_model.transform(train_x)
    test_x = select_model.transform(test_x)

    result = []
        
    for classifier in test_classifiers:    
        print('******************* %s ********************' % classifier)    
        start_time = time.time()    
        model = classifiers[classifier](train_x, train_y)    
        print('training took %fs!' % (time.time() - start_time))    
        predict = model.predict(test_x)

        precision = metrics.precision_score(test_y, predict)    
        recall = metrics.recall_score(test_y, predict)    
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))    
        accuracy = metrics.accuracy_score(test_y, predict)    
        print('accuracy: %.2f%%' % (100 * accuracy))

	scores = cross_val_score(model, train_x, train_y)
        print(scores)
    
