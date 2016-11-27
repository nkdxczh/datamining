import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
        
def feature_select_et(x,y):
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier()
    clf = clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    return model

def feature_select_rf(x,y):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf = clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    return model

def feature_select_ls(x,y):
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf = clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    return model

def feature_pca(x):
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=5)
    pca.fit(x)
    return pca

