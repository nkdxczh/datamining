import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt

def draw_hist(feature_names,importances):
    fi = []
    for i in range(len(feature_names)):
        fi.append([feature_names[i],importances[i]])
    fi=sorted(fi,key=lambda x:x[1],reverse=True)

    x=[]
    y=[]
    for i in range(30):
        x.append(fi[i][0])
        y.append(fi[i][1])

    #print(x)
    #print(y)

    ix = [i for i in range(len(x))]

    plt.bar(ix, y)
    plt.xticks(ix, x,rotation=-25, wrap=True, ha='left',va='top')

    #plt.show()
        
def feature_select_et(x,y,feature_names=''):
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier()
    clf = clf.fit(x, y)
    if not len(feature_names) == 0:
        importances = clf.feature_importances_
        draw_hist(feature_names,importances)
    model = SelectFromModel(clf, prefit=True)
    return model

def feature_select_rf(x,y,feature_names=''):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf = clf.fit(x, y)
    if not len(feature_names) == 0:
        importances = clf.feature_importances_
        draw_hist(feature_names,importances)
    model = SelectFromModel(clf, prefit=True)
    return model

def feature_select_ls(x,y,feature_names=''):
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf = clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    return model

def feature_pca(x):
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=3)
    pca.fit(x)
    return pca

def feature_svd(x):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=7, n_iter=7, random_state=42)
    svd.fit(x) 
    return svd
