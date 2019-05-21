#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/27 下午11:25
# @Author  : Aries
# @Site    :
# @File    : SGDMultiClassifier.py
# @Software: PyCharm
'''
最后的一种分类任务是 多数出-多类别分类
简而言之 就是多标签分类泛化

图片降噪的分类器
'''
import matplotlib
import numpy.random.mtrand as rnd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from mnist.scikit.learn.utils.DataUtils import get_serialize_data, plot_digits,plot_digits_angle

'''
先从训练集和测试集开始 
'''
x_train = get_serialize_data('X_train')
x_test = get_serialize_data('x_test')

noise_train = rnd.randint(0, 100, (len(x_train), 784))
noise_test = rnd.randint(0, 100, (len(x_test), 784))
'''
重新整理 x and y
'''
x_train_mod = x_train + noise_train
x_test_mod = x_test + noise_test
y_train_mod = x_train
y_test_mod = x_test

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_mod,y_train_mod)
clean_digit = knn_clf.predict([x_test_mod[6000]])

plt.figure(figsize=(10, 10))
plt.subplot(211)
plot_digits_angle(x_test_mod[6000])
plt.subplot(212)
plot_digits_angle(clean_digit)

plt.show()


