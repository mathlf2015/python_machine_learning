from chapter_12_loaddata import load_mnist
from chapter_12_neuralnet import NeuralNetMLP
import sys
import numpy as np
from chapter_12_gradient_check import MLPGradientCheck
X_train, y_train = load_mnist('E:/mydata/shuzi_shibie/', kind='train')
X_test,y_test = load_mnist('E:/mydata/shuzi_shibie/', kind='t10k')
nn_check = MLPGradientCheck(n_output=10,
                            n_features=X_train.shape[1],
                            n_hidden=10,
                            l2=0.0,
                            l1=0.0,
                            epochs=10,
                            eta=0.001,
                            alpha=0.0,
                            decrease_const=0.0,
                            minibatches=1,
                            shuffle=False,
                            random_state=1)
nn_check.fit(X_train[:5], y_train[:5], print_progress=False)