#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/11 下午10:21
# @Author  : Aries
# @Site    :
# @File    : svm_reg.py
# @Software: PyCharm

'''
SVM回归
SVM不仅支持线性和非线性分类,而且还支持线性和非线性回归

---
其实线性和非线性回归与线性和非线性分类是目标反转的
线性和非线性分类是拟合两个类别之间可能得最宽的街道的同时限制间隔违例

而线性回归和非线性回归是让尽可能多的实例位于街道上,同时限制间隔违例(也就是不在街道上的实例)
注意街道的宽度由超参数ε决定

在间隔内添加更多的实例不会影响模型的预测,所以这个模型被称为'ε不敏感'
'''
from command.DataUtils import serialize_data, get_serialize_data
from sklearn.svm import LinearSVR
from sklearn import datasets
from sklearn.svm import SVR

moons = get_serialize_data('moons', 1)
x = moons[0]
y = moons[1]
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(x, y)

'''
同理 要解决非线性回归问题可以使用核化的SVM模型
params:
degree: 阶数
c:与正则化程度成反比
ε:决定回归模型的街道宽度
'''
'''
以下代码使用支持核技巧的SVR类,与SVC一样在有效解决数据爆炸的同时也相比于LinearSVR有较高的时间复杂度
'''
svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(x, y)
