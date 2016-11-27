import sys
sys.path.insert(0, '/home/jason/datamining/project')

from feature.select import *
from feature.read import *

from models.models import *
from ensemble.bagging import *
from evaluation import *
from ensemble.gridsearch import *

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import VotingClassifier

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict

if __name__ == '__main__':
    print('testing bagging...')  
    data_file = "/home/jason/datamining/data/train_combine.csv"
    test_file = "/home/jason/datamining/data/test_combine.csv"
    
    print('reading training and testing data...')    
    X, y = read_data(data_file)
    X = scale(X)
    test_X, test_y = read_data(test_file)
    test_X = normalize(test_X)

    print('selecting features...')   
    select_model = feature_select_et(X, y)
    X = select_model.transform(X)
    test_X = select_model.transform(test_X)

    pca = feature_pca(X)
    X = pca.transform(X)
    #X = normalize(X)
    test_X = pca.transform(test_X)
    #test_X = normalize(test_X)

    '''for i in range(len(X[0])):
        base = -min(X[:,i])
        X[:,i] += base
        test_X[:,i] += base'''

    classifiers = {
        #'LIR': linear_regression_classifier(),
        #'LOR': logistic_regression_classifier(),
        #'GNB': gaussian_bayes_classifier(),
        #'MNB': naive_bayes_classifier(),
        #'KNN': knn_classifier(),
        #'DT': decision_tree_classifier(),
        #'ET': extra_trees_classifier(),
        #'RF': random_forest_classifier(),
        #'SVM': svm_classifier(), 
        #'SVC': svm_cross_classifier(), 
        #'AB': ada_boost_validation(), 
        'GB': gradient_boosting_classifier()
        #'LD': linear_discriminant_analysis(),
        #'QD': quadratic_discriminant_analysis(),
        #'NN' : neural_network_classifier()
    }

    estimators = []
    weights = []

    for name, [clf, grid] in classifiers.items():
        print('calculating %s...' % name)   
        #try:
        #clf = gridsearch(clf, X, y, grid)
        clf = bagging(clf, X, y)
        estimators.append((name,clf))
        scores = cross_val_score(clf, X, y, scoring='log_loss')
        weights.append(-1/scores.mean())
        print(name,'\t--> ',-scores.mean())
        '''except:
            print(name,'failed!')
            if not len(estimators) == len(weights):
                estimators.remove(len(estimators))'''
    
    eclf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    scores = cross_val_score(eclf, X, y, scoring='log_loss')
    print('voting\t--> ',-scores.mean())

    if -scores.mean() > 1/weights[0]:
        eclf = estimators[0][1]

    eclf.fit(X, y)

    joblib.dump(eclf, 'eclf.pkl')

    '''train_pred = eclf.predict_proba(X)
    
    print('writing train...')   
    out = open("/home/jason/datamining/data/result_bagging_train.csv","w")
    for i in train_pred:
        out.write(str(i[1])+"\n")
    out.close()'''

    predict_y = eclf.predict_proba(test_X)
    
    print('writing...')   
    out = open("/home/jason/datamining/data/result_bagging.csv","w")
    for i in predict_y:
        out.write(str(i[1])+"\n")
    out.close()
