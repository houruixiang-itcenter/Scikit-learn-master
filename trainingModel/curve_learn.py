#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/2 下午11:09
# @Author  : Aries
# @Site    : 
# @File    : curve_learn.py
# @Software: PyCharm
'''
学习曲线
高阶多项式的训练数据拟合,很可能比简单的线性回归要好


3个例子
1.300多阶的多项式
2.线性模型
3.二次多项式模型


第二章中 我们评估性能
如果在训练集上变现良好 但是交叉验证时候表现糟糕 则就是过度拟合
如果在训练集上和交叉验证时候都表现糟糕 则就是拟合不足


还有一种方式就是观察曲线

'''
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.rand(m, 1)

'''
这个曲线在训练集和验证集上,关于'训练集大小'的性能函数
要生成这个曲线 只需要在不同大小的训练子集上多次训练模型即可
'''


def plot_learning_curves(model, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_errors, val_errors = [], []
    for i in range(1, len(x_train)):
        model.fit(x_train[:i], y_train[:i])
        y_train_predict = model.predict(x_train[:i])
        y_val_predict = model.predict(x_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:i]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val')
    plt.xlabel('train set size')
    plt.ylabel('RMSE')
    plt.ylim(0, 3)


'''
由于模型函数 是点集 泛化出来的函数 
所以随着训练集的增加  成本函数会发生变化
'''

# lin_reg = LinearRegression()
# plt.subplot(211)
# plot_learning_curves(lin_reg, x, y)
'''
这个模型 明显就是拟合不足的结果
无论是训练集预测  还是验证集 都表现的不好   最终训练集和验证集的RMSE都停留在某一个确定值十分接近
两条曲线均到达高低,十分接近,而且相当的高

解决方法:添加实例是于事无补,我们应该选择更加复杂的模型 或者找到更好的特征
'''
print('------------------------------------------10阶多项式--------------------------------------------------------')

polynomial_regssion = Pipeline([
    ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
    ('lin_reg', LinearRegression())

])
# polynomial_data = polynomial_regssion.fit_transform(x)
# lin_reg = LinearRegression()
# plt.subplot(212)

plot_learning_curves(polynomial_regssion, x, y)
plt.show()
'''
改进模型 过度拟合的方法之一是提供更多训练,知道验证误差接近训练误差
'''
'''
在机器学习领域,一个重要的理论结果就是.模型的泛化误差可以被表示为三个截然不同的误差之和:
1.偏差:
这部分误差主要来源于模型的选型,比如假设数据是线性的,实际上是二次的
2.方差:
用来描述数据的离散程度,这部分误差是对数据微小变化的过度敏感导致的,比如一次线性模型,使用高阶多项式模型会提高方差
3.不可避免的误差
这部分误差是由于数据的噪声所致,减少这部分误差的唯一方法就是清理数据
(例如修复数据源,如损坏的传感器,或者是检测并移除异常值)


经过这么久的学习,对过度拟合和拟合不足有了一个比较深刻的认识

# 前提是基于最优模型
过度拟合: 训练集上表现良好,验证集上表现较差  1.使用更大的训练集 2.函数模型降阶
拟合不足: 训练集和验证集上表现都较差 1.利用组合等方式增加特征值 3.使用高阶的函数模型



---
绘制曲线 实时的一点一点的绘制线性模型 
其中最重要的是PolynomialFeatures 他可以根据需求组合不同次幂的特征
这个学习曲线  是 训练集大小 vs 成本函数  的曲线 
可以理解 是不同模型的性能曲线
'''
