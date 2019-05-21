#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/2 下午10:28
# @Author  : Aries
# @Site    : 
# @File    : polynomial_reg.py
# @Software: PyCharm
'''
多项式回归
有的数据的拟合并不是简单的线性回归

比如简单的二次方程

----
下面来看一个简单的二次方程
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.rand(m, 1)

'''
这就是 一个简单的二元一次方程
显然直线永远无法拟合这样的数据 
使用scikit-learn将每个特征的平方加入训练集<这个例子中只有一个特征>

如果这样 一次的x怎么办???
'''
poly_fearures = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_fearures.fit_transform(x)
'''
然后使用线性回归 进行训练模型
'''
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
print(lin_reg.intercept_,lin_reg.coef_)
'''
PolynomialFeatures是比较重要的一个 转换器吧<自定义> 
作用就是将多项式做为组合形式添加到训练模型中 供线性回归模型训练 
例如 一个3阶的方程 有两个特征a和b PolynomialFeatures不仅会吧 a(平方) a(3次方)  b(平方) b(3次方)  
还会添加a和b的 各种组合
'''



