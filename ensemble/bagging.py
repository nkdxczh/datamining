import sys
sys.path.insert(0, '/home/jason/datamining/project')

from feature.select import *
from feature.read import *

from sklearn import decomposition

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from models.models import *

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

from sklearn.cross_validation import cross_val_score

def bagging(model, train_x, train_y):
    from sklearn.ensemble import BaggingClassifier
    bagging = BaggingClassifier(model, max_samples=0.5, max_features=0.5 )
    bagging.fit(train_x, train_y)
    return bagging

def calibration(model, train_x, train_y):
    from sklearn.calibration import CalibratedClassifierCV
    calibration = CalibratedClassifierCV(model, max_samples=0.5, max_features=0.5 )
    calibration.fit(train_x, train_y)
    return calibration

if __name__ == '__main__':
     
    classifiers = {
    'MNB': naive_bayes_classifier()}
        
    print('testing boosting...')  
    data_file = "/home/jason/datamining/data/train_combine.csv"
    
    print('reading training and testing data...')    
    X, y = read_data(data_file)

    print('selecting features...')   
    select_model = feature_select_et(X, y)
    X = select_model.transform(X)

    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    X = scale(X)
    min0 = -min(X[:,0])
    min1 = -min(X[:,1])
    min2 = -min(X[:,2])
    X[:,0] += min0
    X[:,1] += min1
    X[:,2] += min2

    print('calculating...')  
    for name, [clf,para] in classifiers.items():
        clf = bagging(clf, X, y)
        scores = cross_val_score(clf, X, y)
        print(name,'\t--> ',scores.mean())
