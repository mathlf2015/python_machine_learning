import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#得到数据
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())
#数据探索
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5);
plt.show()
#各个特征相关系数热力图，correlation matrix,挑选单变量回归的特征
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=cols,
                xticklabels=cols)
plt.show()


from chapter_10_OLS import LinearRegressionGD
X = df[['RM']].values
y = df['MEDV'].values
#Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19
y = np.array(y).reshape((len(y), 1))
#print(type(X),X.shape,type(y),y.shape)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
#转换为原来的数据形式
y_std = np.array(y_std).reshape((len(y_std), ))
#y = np.array(y).reshape((len(y), ))
#模型建立及误差随迭代次数的变化
lr = LinearRegressionGD()
lr.fit(X_std, y_std)
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None
#拟合情况的可视化
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()

#将预测结果转换为原来的规格
num_rooms_std = sc_x.transform([5.0])
price_std = lr.predict(num_rooms_std)
price_std = np.array(price_std).reshape(1,)
print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))

#查看模型参数
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])



