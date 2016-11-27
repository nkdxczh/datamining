import numpy as np

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# 定义管道，先降维(pca)，再逻辑回归
pca = decomposition.PCA()
logistic = linear_model.LogisticRegression()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# 把管道再作为grid_search的estimator
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)
estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))

estimator.fit(X_digits, y_digits)

estimator.predict(X_digits)

print(estimator.score(X_digits, y_digits))
