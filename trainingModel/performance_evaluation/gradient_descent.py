#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 下午4:41
# @Author  : Aries
# @Site    :
# @File    : gradient_descent.py
# @Software: PyCharm
'''
之前的线性模型
特征数量翻倍 消耗时间也会翻倍
所幸方程式线性的 只要内存足够 还是很快的


所以我们接着来看几种优化算法
// ???? 可以解决特征数或者训练实例多到内存无法存储的场景

'''
import time
import warnings

import numpy as np
from sklearn.linear_model import SGDRegressor
warnings.filterwarnings('ignore')

'''
梯度下降
就是一步步的调整特征值 然后最后趋于成本函数的最小值 


---
类似于爬山/下山一样
线性函数的成本函数式MSE 所幸只有一个最小值 不会有局部最小和全局最小之间的混淆错误


但是需要注意的是如果特征值之间的量级不同 如 特征1:0~1 特征2: 100~10000
这样 会让成本函数 这个碗状的函数 从起点到最小值的时间特别长
所以有很大必要进行特征值的特征缩放  StandardScaler


****************
由此可见  训练模型就是寻求成本函数最小的过程 
'''
# todo  由此可见  训练模型就是寻求成本函数最小的过程

print('----------------------------------------------批量梯度下降------------------------------------------------------')

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)

'''
python验证上面的标准方程
np.ones---按照参数
'''
x_b = np.c_[np.ones((100, 1)), x]  # add x0

'''
首先来看这个算法的快速实现 
'''
eta = 0.1  # learning rate
n_iterations = 1000
m = 100

theta = np.random.rand(2, 1)  # random initialization
t1 = time.time()
for iteration in range(n_iterations):
    gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradients
print(theta)
t2 = time.time()
print(t2 - t1)
'''
这样就求出一堆标准方程

'''

print('----------------------------------------------随机梯度下降------------------------------------------------------')
'''
随机热梯度下降 较之于批量梯度下降要快的多 

'''
n_epochs = 50

t0, t1 = 5, 50  # learning schedule hyperparameters


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.rand(2, 1)  # 初始的特征值
l1 = time.time()
for epoch in range(n_epochs):
    # 每100次是一轮
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
print(theta)
l2 = time.time()
print(l2 - l1)
'''
批量梯度下降 需要训练1000次  --- 步数是平均的  --- 基于全部实例
随机梯度下降 需要训练50次就可以获得一个不错的结果  --- 不是平均的 --- 基于单个实例 
**********注意:随机梯度  并不是基于全部实例计算梯度,而是随机选择实例进行训练
数据量大的时候 随机梯度下降的优势会出来

-------
每次迭代 在训练集中随机选择一个实例,进行梯度的计算 

由于算法随机的性质 成本函数将不再是缓缓下降  然后最后到达最低值  而是不断的上上下下,但是总体的趋势还是下降的 


对于不规则的成本函数 随机梯度下降可以有效帮助算法跳出局部的最小值 
所以相比批量梯度下降,它对找到全局最小值更有优势

随机梯度下降的好处是可以逃离局部最优,但是缺点是永远定位不出最小值

要解决这个问题 就是逐步降低学习率:
随着时间的推移,步长越来越小,让算法尽量靠近全局最小值,这个过程叫做:  模拟退火 
确定每一个迭代学习的函数叫做:  学习计划

'''
print('----------------------------------------------SGD随机梯度下降------------------------------------------------------')
'''
使用scikit-learn来实现随机的梯度下降
'''
'''
n_iter -- 迭代次数 
penalty -- 无任何正则化函数 
eta0 -- 初始学习率


---
numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能，区别： 
ravel()：如果没有必要，不会产生源数据的副本 
flatten()：返回源数据的副本 
squeeze()：只能对维数为1的维度降维
'''
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(x, y.ravel())

'''
intercept_: 偏置项  这里是x0的特征权重,因为x0恒定为1 所以其为偏置项
coef_:特征权重
'''
print(sgd_reg.intercept_, sgd_reg.coef_)

print('----------------------------------------------小批量梯度下降------------------------------------------------------')
'''
批量梯度下降 :基于全部实例
随机梯度下降 :基于单个实例
小批量梯度下降  :基于一小部分实例,也就是小批量
'''

