#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/5 下午3:53
# @Author  : Aries
# @Site    :
# @File    : decision_boundary.py
# @Software: PyCharm
'''
决策边界

----
案例:鸢尾回归<这是一个非常著名的数据集>

共有150朵鸢尾花,分别来自三个不同的品种:
1.Setosa鸢尾花
2.Versicolor鸢尾花
3.Virginica鸢尾花
数据里包含花的萼片以及花瓣的长度和宽度
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
print(list(iris))
'''
花瓣宽度
numpy [x,y]  
x:行数  :n/n:
y:列数  :n/n:
'''
x = iris['data'][:, 3:]
'''
if Virginica鸢尾花  : 1
else               : 0
'''
y = (iris['target'] == 2).astype(np.int)  #
print(x)
print(y)

print('----------------------------------------------训练逻辑回归模型-------------------------------------------------')
'''
训练逻辑回归模型
'''
log_reg = LogisticRegression()
log_reg.fit(x, y)

'''
我们来看看 花瓣宽度是0 --- 3cm 之间的i鸢尾花,模型估算出来的概率
'''
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
'''
逻辑回归预测概率
'''
y_proba = log_reg.predict_proba(x_new)
plt.plot(x_new, y_proba[:, 1], 'g-', label='Iris-Virginica')
plt.plot(x_new, y_proba[:, 0], 'b--', label='not Iris-Virginica')
# plt.show()

'''
下面来看预测值
会输出一个认为可能性最大的值
'''
print(log_reg.predict([[1.5],[1.7]]))

'''
与其他线性模型一样,逻辑回归模型可以用L1或L2惩罚函数来正则化,Scikit-Learn默认添加的是L2函数
'''
