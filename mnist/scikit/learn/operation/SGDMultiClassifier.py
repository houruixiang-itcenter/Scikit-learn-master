#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/27 下午11:25
# @Author  : Aries
# @Site    :
# @File    : SGDMultiClassifier.py
# @Software: PyCharm
'''
多类别分类器
OvA  一对多策略 一个图片遍历所有 哪个分数高就使用哪个
OvO 一对一  一对数字 为基本单位 图片数为n  则需要的二分分类器为 n * (n-1) / 2
SGD 会自动使用OvA
'''
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier

from mnist.scikit.learn.utils.DataFetchUtils import get_mnist_data_and_target
from mnist.scikit.learn.utils.DataUtils import get_serialize_data, serialize_data
import numpy as np

# 忽略 警告warnings
warnings.filterwarnings('ignore')

print('----------------------------------------SGD--------------------------------------------')

x, y = get_mnist_data_and_target()
y_train = get_serialize_data('Y_train')
x_train = get_serialize_data('X_train')
sgd_clf = get_serialize_data('sgd_clf')  # type: SGDClassifier
sgd_clf.fit(x_train, y_train)
val = sgd_clf.predict([x_train[36000]])
print('predict is %s and except is %s' % (val, y_train[36000]))
'''
这样一个 初步的多类别分类器就ok了
在scikit-learn内部实际上训练了10个二元分类器
每次预测时候 获取每个二元分类器的决策分数 然后选择决策分数最高的那个
'''

print('----------------------------------------SGD返回当前多类别分类器的决策分数--------------------------------------------')
some_digit_scores = sgd_clf.decision_function([x_train[36000]])
print(some_digit_scores)

'''
获取分数最高的index
根据index 来找出对应的数据  再训练分类器时候目标类别会存储在预测起的classes_中
'''
print(np.argmax(some_digit_scores))
print(sgd_clf.classes_[np.argmax(some_digit_scores)])

print(
    '---------------------------------OneVsOneClassifier and OneVsRestClassifier-------------------------------------')
'''
如果要强制Scikit-Learn使用OvO  或者Ovr  可以分别使用 OneVsOneClassifier and OneVsRestClassifier

eg: OvO
'''
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(x_train, y_train)
val = ovo_clf.predict([x_train[36000]])
print(val)
print(len(ovo_clf.estimators_))

serialize_data(sgd_clf, 'sgd-multi')
