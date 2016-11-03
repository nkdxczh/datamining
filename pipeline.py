import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

train_label = pd.read_csv("/home/jason/datamining/datasets/train_label.csv")
X_train = train_label.values[:,1]
Y_train = train_label.values[:,:1]

pipe_lr = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))
                    ])
pipe_lr.fit(X_train, Y_train)
#print('Test accuracy: %.3f' % pipe_lr.score(X_test, Y_test))
