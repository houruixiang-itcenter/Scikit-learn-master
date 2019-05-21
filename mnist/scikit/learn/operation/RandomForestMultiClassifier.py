#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/27 下午11:25
# @Author  : Aries
# @Site    :
# @File    : SGDMultiClassifier.py
# @Software: PyCharm
'''
多类别分类器
OvA  一对多策略 <每个数字一个> 一个图片遍历所有 哪个分数高就使用哪个
OvO 一对一  一对数字 为基本单位 图片数为n  则需要的二分分类器为 n * (n-1) / 2
SGD 会自动使用OvA
'''
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier

from mnist.scikit.learn.utils.DataFetchUtils import get_mnist_data_and_target
from mnist.scikit.learn.utils.DataUtils import get_serialize_data, serialize_data
import numpy as np

# 忽略 警告warnings
warnings.filterwarnings('ignore')

print('----------------------------------------RandomForest--------------------------------------------')

x, y = get_mnist_data_and_target()
y_train = get_serialize_data('Y_train')
x_train = get_serialize_data('X_train')
forest_clf = get_serialize_data('forest_clf')  # type: RandomForestClassifier
forest_clf.fit(x_train, y_train)
val = forest_clf.predict([x_train[36000]])
print('predict is %s and except is %s' % (val, y_train[36000]))

'''
对于RandomForest Scikit-Learn没必要选择OvO 或者 Ova
因为随机森林  predict_proba会获得分类器将每个实例分类为每个实例的概率列表  
所以 不需要自己构建分类器
'''
some_digit_forest_scores = forest_clf.predict_proba([x_train[36000]])
index = np.argmax(some_digit_forest_scores)
types = forest_clf.classes_
predict_result = forest_clf.classes_[index]
print(some_digit_forest_scores)
print(types)
print(predict_result)

serialize_data(forest_clf, 'rf-multi')
