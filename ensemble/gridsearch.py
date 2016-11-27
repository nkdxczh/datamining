from test import *

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV

def gridsearch(model, train_x, train_y, param_grid):
    if len(param_grid) == 0:
        return model
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    return grid_search

if __name__ == '__main__':

    classifiers = {
    'KN': KNeighborsClassifier(),
    'SVC1': SVC(kernel="linear"),
    'SVC2': SVC(),
    'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),  # clf.feature_importances_
    'ET': ExtraTreesClassifier(max_depth=None),  # clf.feature_importances_
    'AB': AdaBoostClassifier(),
    'GB': GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)}

    param_grids = {
    'KN': {'n_neighbors': list(range(1,21))},
    'SVC1': {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]},
    'SVC2': {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [1e-1, 1, 10]},
    'DT': {'max_depth' : list(range(1,11))},
    'RF': {'n_estimators' : list(range(1,21)), 'max_depth' : list(range(1,6)), 'max_features' : list(range(1,6))},
    'ET': {'n_estimators' : list(range(1,21))},
    'AB': {'n_estimators' : [50,100,200]},
    'GB': {'n_estimators' : [50,100,200]}
    }
        
    print('testing gridsearch...')  
    data_file = "/home/jason/datamining/data/train_combine.csv"
    
    print('reading training and testing data...')    
    X, y = read_data(data_file)

    print('selecting features...')   
    select_model = feature_select(X, y)
    X = select_model.transform(X)

    print('calculating...')  
    for name, clf in classifiers.items():
        try:
            clf = gridsearch(clf, X, y, param_grids[name])
            scores = cross_val_score(clf, X, y)
            print(name,'\t--> ',scores.mean())
        except:
            print(name,'not gridsearch')
