#reading in the dataset directly from the UCI website using pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
#encoding the class labels
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
print(le.transform(['M', 'B']))
#divide the dataset into a separate training dataset  and a separate test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


#Tuning hyperparameters via grid search
#穷尽搜索计算复杂度较高时，可以考虑随机搜索
# from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()),('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},{'clf__C': param_range,'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
#stratified k-fold cross-validation by default
#n_jobs=-1,指利用电脑上所有的cpu
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=1)
"""在windows下面运行 sklearn 中的代码，如果遇到 并行执行 n_jobs > 1 就会出现：
raise ImportError,这个错误解决： 把所有的代码方到 if __name__ == ‘__main__’ 中就可以了"""
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
#在独立的测试集上测试模型泛化误差
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

#Algorithm selection with nested cross-validation
#gives us a good estimate of whatto expect if we tune the hyperparameters of a model and then use it on unseen data
from sklearn.cross_validation import cross_val_score
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=1)
scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=[ {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring='accuracy',cv=5)
scores = cross_val_score(gs,X_train,y_train,scoring='accuracy',cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
