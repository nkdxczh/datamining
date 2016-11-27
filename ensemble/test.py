import sys
sys.path.insert(0, '/home/jason/datamining/project')

from feature.select import *
from feature.read import *

import warnings
warnings.filterwarnings("ignore")

# meta-estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import BaggingClassifier

from sklearn.cross_validation import cross_val_score

def bagging(model, train_x, train_y):
    bagging = BaggingClassifier(model, max_samples=0.5, max_features=0.5 )
    bagging.fit(train_x, train_y)
    return bagging

if __name__ == '__main__':
     
    classifiers = {
    'KN': KNeighborsClassifier(3),
    'SVC': SVC(kernel="linear", C=0.025),
    'SVC': SVC(gamma=2, C=1),
    'DT': DecisionTreeClassifier(max_depth=5),
    'RF': RandomForestClassifier(n_estimators=10, max_depth=5, max_features=1),  # clf.feature_importances_
    'ET': ExtraTreesClassifier(n_estimators=10, max_depth=None),  # clf.feature_importances_
    'AB': AdaBoostClassifier(n_estimators=100),
    'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), # clf.feature_importances_
    'GNB': GaussianNB(),
    'LD': LinearDiscriminantAnalysis(),
    'QD': QuadraticDiscriminantAnalysis()}
        
    print('testing gridsearch...')  
    data_file = "/home/jason/datamining/data/train_combine.csv"
    
    print('reading training and testing data...')    
    X, y = read_data(data_file)

    print('selecting features...')   
    select_model = feature_select(X, y)
    X = select_model.transform(X)

    print('calculating...')  
    for name, clf in classifiers.items():
        bagging = bagging(clf, X, y)
        scores = cross_val_score(bagging, X, y)
        print(name,'\t--> ',scores.mean())
