#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/30 下午6:36
# @Author  : Aries
# @Site    :
# @File    : PCAPE.py
# @Software: PyCharm
'''
使用网格搜索选择和函数和调整超参数
'''
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
import numpy as np

from command.DataUtils import get_serialize_data

'''
由于kPCA是一种无监督式的学习算法,所以没有明确的性能指标来帮你选择最佳的和函数和超参数值
而降维通常是监督式学习任务(例如分类)的准备步骤,
所以可以使用网格搜索来找到使任务性能最佳的核和超参数

eg:找到最佳的核函数和gamma的值
'''
X = get_serialize_data('X', 5)
Y = get_serialize_data('Y', 5)
clf = Pipeline([
    ('kpca', KernelPCA(n_components=154)),
    ('log_reg', LogisticRegression())
])

param_grid = [{
    'kpca__gamma': np.linspace(0.03, 0.05, 10),
    'kpca__kernel': ['rbf', 'sigmoid']
}]

gird_search = GridSearchCV(clf,param_grid,cv=3)
gird_search.fit(X,Y)

print(gird_search.best_params_)
