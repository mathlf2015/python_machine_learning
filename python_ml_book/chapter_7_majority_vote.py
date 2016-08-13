#排列组合中的组合数
from scipy.misc import comb
import math
def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    #二项分布，n次中k次发生错误
    probs = [comb(n_classifier, k) *error**k *(1-error)**(n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probs)
print(ensemble_error(n_classifier=11, error=0.25))
#visualize the relationship between ensemble and base errors in a line graph
import numpy as np
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
import matplotlib.pyplot as plt
plt.plot(error_range, ens_errors,label='Ensemble error',linewidth=2)
#base error指的是随机猜，为0.5
plt.plot(error_range, error_range,linestyle='--', label='Base error',linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()

####################################################################
#IRIS数据上使用自己写的VOTE类函数
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.5,random_state=1)
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
#测试各个分类器各自的分类效果
clf1 = LogisticRegression(penalty='l2',C=0.001,random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],['clf', clf1]])
#因为决策树不用scaling,所以不用用到Pipeline,没有pipe2
pipe3 = Pipeline([['sc', StandardScaler()],['clf', clf3]])
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), label))
##########################################################################################
#使用MajorityVoteClassifier ，比较其和各单个分类器的效果
#sklearn已经自带了MajorityVoteClassifier，可以参考chapter_7_plot_voting_decisions.py
from chapter_7_Majority_Vote_Class import MajorityVoteClassifier
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
print('all classifier 10-fold cross validation:\n')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), label))

##################################################################################################
#Evaluating and tuning the ensemble classifier
#绘制各个分类器的ROC曲线
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train,y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,color=clr,linestyle=ls,label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],linestyle='--',color='gray',linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

##########################################################################################
#绘制决策边界
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product
'''for i in product([0, 1], [0, 1]):
    print(i)'''
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2,sharex='col',sharey='row',figsize=(7, 5))
for idx, clf, tt in zip(product([0, 1], [0, 1]),all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],X_train_std[y_train==0, 1],c='blue',marker='^',s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],X_train_std[y_train==1, 1],c='red',marker='o',s=50)
    axarr[idx[0], idx[1]].set_title(tt)
#text中坐标用来调整文字开始的位置以右下角坐标为0,0
plt.text(-3.5, -4.5,s='Sepal width [standardized]',ha='center', va='center', fontsize=12)
plt.text(-11.5, 4.5,s='Petal length [standardized]',ha='center', va='center',fontsize=12, rotation=90)
plt.show()

#tuning param
print(mv_clf.get_params())
from sklearn.grid_search import GridSearchCV
params = {'decisiontreeclassifier__max_depth': [1, 2],'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf,param_grid=params,cv=10,scoring='roc_auc')
grid.fit(X_train, y_train)
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f+/-%0.2f %r"% (mean_score, scores.std() / 2, params))
print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

