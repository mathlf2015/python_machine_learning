# -*-coding: utf-8 -*-
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model=Sequential() # 模块初始化
model.add(Dense(64, input_dim=20, init='uniform')) # 添加输入层（20节点），第一隐藏层（64节点）的连接
model.add(Activation('tanh')) # 第一隐藏层用tanh作为激活函数
model.add(Dropout(0.5)) # 使用Dropout防止过拟合
model.add(Dense(64, init='uniform'))  # 添加第一隐藏层（64节点），第二隐藏层（64节点)的连接
model.add(Activation('tanh')) # 第二隐藏层用tanh作为激活函数
model.add(Dropout(0.5)) # 使用Dropout防止过拟合
model.add(Dense(64, input_dim=1, init='uniform'))  # 添加输入层（64节点），第一隐藏层（1节点）的连接
model.add(Activation('sigmoid')) # 输出层用sigmoid作为激活函数

sgd=SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True) # 定义求解算法
model.compile(loss='mean_squared_error',optimizer=sgd) # 编译生成模型，损失函数为平方均误差平方和

model.fit(X_train, y_train, nb_epoch=20, batch_size=16) # 训练模型
score=model.evaluate(X_test, y_test, batch_size=16) # 测试模型
