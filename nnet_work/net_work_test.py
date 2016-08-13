# -*- coding: utf-8 -*-
import mnist_loader
import net_work_sgd
training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
#得到的是zip对象，需要转化成list
test_data=list(test_data)
training_data=list(training_data)
validation_data=list(validation_data)
#print(test_data[:1])
#测试数据目标值和训练集目标值不一样，是一个数值
#图像像素为28*28=784，30为迭代次数，10为每次的批量
net = net_work_sgd.Network([784, 30, 10])
#3.0为学习速率，test_data默认为0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
