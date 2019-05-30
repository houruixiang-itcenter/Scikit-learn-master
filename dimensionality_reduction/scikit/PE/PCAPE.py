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
from sklearn.metrics import mean_squared_error

print('---------------------------------------------------监督式-------------------------------------------------------')
'''
监督式

由于kPCA是一种无监督式的学习算法,所以没有明确的性能指标来帮你选择最佳的和函数和超参数值
而降维通常是监督式学习任务(例如分类)的准备步骤,
所以可以使用网格搜索来找到使任务性能最佳的核和超参数

eg:找到最佳的核函数和gamma的值
'''
# X = get_serialize_data('X', 5)
# Y = get_serialize_data('Y', 5)
X_train = get_serialize_data('X_train', 5)
Y_train = get_serialize_data('Y_train', 5)
X_train = X_train[:1000, :]
Y_train = Y_train[:1000]
clf = Pipeline([
	('kpca', KernelPCA(n_components=154)),
	('log_reg', LogisticRegression())
])

param_grid = [{
	'kpca__gamma': np.linspace(0.03, 0.05, 10),
	'kpca__kernel': ['rbf', 'sigmoid']
}]

gird_search = GridSearchCV(clf, param_grid, cv=3)
gird_search.fit(X_train, Y_train)

print(gird_search.best_params_)

print('--------------------------------------------------非监督式------------------------------------------------------')
'''
还有一种完全不受监督方法,就是选择重建误差最低的核和超参数
由于RBF的加入,他不再想PCA解压缩那么容易

那么如何重建呢 ?
方法一:
训练一个监督式的回归模型:
以投影后的训练集作为训练集(训练数据),并以原始实例作为目标(标签)


如果你设置  fit_inverse_transform = True,Scikit-Learn会自动执行该操作
'''
rbf_pca = KernelPCA(n_components=154, kernel='rbf', gamma=0.0433, fit_inverse_transform=True)
x_reduced = rbf_pca.fit_transform(X_train)
x_preimage = rbf_pca.inverse_transform(x_reduced)


# param_grid = [{
# 	'kpca__gamma': np.linspace(0.03, 0.05, 10),
# 	'kpca__kernel': ['rbf', 'sigmoid']
# }]
#
# gird_search = GridSearchCV(rbf_pca, param_grid, cv=3)
# gird_search.inverse_(x_preimage, X_train)
#
# print(gird_search.best_params_)

'''
计算重建原像的误差
'''
print(mean_squared_error(X_train,x_preimage))

'''
树上这样说
现在你可以使用交叉验证的网格搜索,来寻找这个原像重建误差最小的核和超参数

但是我觉得不可以
'''

print('-----------------------------------------------局部线性嵌入----------------------------------------------------')
'''
局部线性嵌入(LLE)是一种非常强大的非线性降维技术,不再是像之前的算法那样依赖于投影,他是一种流形学习技术

LLE首先测量每一个算法如何与最近的另据线性相关,然后训练集寻找一个能最大程度保留这些局部关系的地位表示,所以他特别擅长展开弯曲的流形
,特别是没有太多噪音
'''
