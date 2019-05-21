#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/5 下午5:55
# @Author  : Aries
# @Site    : 
# @File    : softmax.py
# @Software: PyCharm
'''
softmax回归

逻辑回归模型经过推广,可以直接支持多个类别,而不需要训练并组合多个二元分类器
这就是softmax回归,或者叫多元逻辑回归
'''
import warnings

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

'''
加载数据
'''
iris = datasets.load_iris()
x = iris['data'][:, (2, 3)]
y = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
softmax_reg.fit(x, y)
print(softmax_reg.classes_)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))
