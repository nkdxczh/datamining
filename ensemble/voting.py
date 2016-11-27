import sys
sys.path.insert(0, '/home/jason/datamining/project')

from feature.select import *
from feature.read import *

import warnings
warnings.filterwarnings("ignore")

# meta-estimator
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

data_file = "/home/jason/datamining/data/train_combine.csv"
    
print('reading training and testing data...')    
X, y = read_data(data_file)

print('selecting features...')   
select_model = feature_select(X, y)
X = select_model.transform(X)

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', weights=[1,1,1])

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.5f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
