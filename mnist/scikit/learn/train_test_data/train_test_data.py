#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 下午9:23
# @Author  : Aries
# @Site    :
# @File    : train_test_data.py
# @Software: PyCharm
'''
任何机器学习需要区分测试集  当然这个也不例外
这个机器学习项目
'''
import numpy as np

from mnist.scikit.learn.utils.DataFetchUtils import get_mnist_data_and_target, get_mnist
from mnist.scikit.learn.utils.DataUtils import serialize_data, get_serialize_data

mnist = get_mnist()
X, Y = get_mnist_data_and_target()


def get_train_and_test():
    X_train, X_test, Y_train, Y_test = X[:60000], X[10000:], Y[:60000], Y[10000:]
    shuffle_train_index = np.random.permutation(60000)
    shuffle_test_index = np.random.permutation(10000)
    try:
        if get_serialize_data('X_train') == None and get_serialize_data('Y_train'):
            print('x is none')
            X_train, Y_train = X_train[shuffle_train_index], Y_train[shuffle_train_index]
            serialize_data(X_train, 'X_train')
            serialize_data(Y_train, 'Y_train')
        else:
            print('x is not none')
            X_train, Y_train = get_serialize_data('X_train'), get_serialize_data('Y_train')

        if get_serialize_data('X_test') == None and get_serialize_data('Y_test'):
            print('y is none')
            X_test, Y_test = X_test[shuffle_test_index], Y_test[shuffle_test_index]
            serialize_data(X_test, 'X_test')
            serialize_data(Y_test, 'Y_test')
        else:
            print('y is not none')
            X_test, Y_test = get_serialize_data('X_test'), get_serialize_data('Y_test')
    except FileNotFoundError as e:
        X_train, Y_train = X_train[shuffle_train_index], Y_train[shuffle_train_index]
        X_test, Y_test = X_test[shuffle_test_index], Y_test[shuffle_test_index]
        serialize_data(X_train, 'X_train')
        serialize_data(Y_train, 'Y_train')
        serialize_data(X_test, 'X_test')
        serialize_data(Y_test, 'Y_test')

    return X_train, Y_train, X_test, Y_test


x_train, y_train, x_test, y_test = get_train_and_test()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
