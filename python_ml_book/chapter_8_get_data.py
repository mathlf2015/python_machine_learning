import pyprind
import pandas as pd
import os
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
#建立空数据框
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path ='E:/mydata/aclImdb_v1/aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):       #列出当前路径下的文件名
            with open(os.path.join(path, file), 'r',encoding='ISO8859-1') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index)) #随机重排
df.to_csv('E:/mydata/movie_data.csv', index=False)
df = pd.read_csv('E:/mydata/movie_data.csv',encoding='ISO8859-1')
print(df.head(3))
