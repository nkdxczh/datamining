import sys
sys.path.insert(0, '/home/jason/datamining/project')

from feature.select import *
from feature.read import *

from models.models import *

from sklearn import decomposition

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

import warnings
warnings.filterwarnings("ignore")

# meta-estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.cross_validation import cross_val_score

def boosting(model, train_x, train_y):
    from sklearn.ensemble import AdaBoostClassifier
    boosting = AdaBoostClassifier(model, n_estimators = 200)
    boosting.fit(train_x, train_y)
    return boosting

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
    X[:,0] += -min(X[:,0])
    X[:,1] += -min(X[:,1])
    X[:,2] += -min(X[:,2])

    print('calculating...')  
    for name, [clf, para] in classifiers.items():
        clf = boosting(clf, X, y)
        scores = cross_val_score(clf, X, y)
        print(name,'\t--> ',scores.mean())
