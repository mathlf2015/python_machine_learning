#创建一个数据框用于操作
import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
print(df)
#查看每列的缺失值个数
print(df.isnull().sum())
#将数据框转化为numpy数组,sklearn中算法输入为numpy数组形式
print(df.values)
#rows with missing values can be easily dropped via the dropna method
print(df.dropna())
#清除列
print(df.dropna(axis=1))
# only drop rows where all columns are NaN
df.dropna(how='all')
# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)
# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])
#插值处理缺失值
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)
#create a new data frame to illustrate the problem
import pandas as pd
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)
size_mapping = {'XL': 3,'L': 2,'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)
#a reverse-mapping dictionary
inv_size_mapping = {v: k for k, v in size_mapping.items()}
#enumerate the class labels starting at 0
import numpy as np
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
#map the converted class labels back to the original string representation
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
#convenient LabelEncoder class directly implemented in scikit-learn to achieve the same
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
#fit_transform method is just a shortcut for calling fit and transform separately
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))
#convenient LabelEncoder class to encode the string labels into integers
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)
#create a new dummy feature for each unique value in the nominal feature column
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
#converted the sparse matrix representation into a regular (dense) NumPy array
print(ohe.fit_transform(X).toarray())
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0],sparse=False)
print(ohe.fit_transform(X))
#create those dummy features via one-hot encoding is to use the get_dummies method implemented in pandas
print(pd.get_dummies(df[['price', 'color', 'size']]))





