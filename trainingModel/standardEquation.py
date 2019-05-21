# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/29 下午5:16
# @Author  : Aries
# @Site    :
# @File    : standardEquation.py
# @Software: PyCharm
'''
基于标准方程 来预测高斯噪音
'''

'''


基于标准方程: a = (X<T> * X)<-1> * X<T> * y
是在有些符号打不出来  理解就好了


a是使成本函数最小的值
y是所有的目标向量 就是应变量
'''
import numpy as np
import matplotlib.pyplot as plt
from trainingModel.utils.DataUtils import serialize_data, get_serialize_data
import matplotlib.pyplot as plt

'''
numpy.random.rand(d0,d1,…,dn)

rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
dn表格每个维度
返回值为指定维度的array
'''
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)

'''
python验证上面的标准方程
np.ones---按照参数
'''
x_b = np.c_[np.ones((100, 1)), x]  # add x0
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta_best)
serialize_data(theta_best, 'theta_best')
'''
[[4.49965693]
 [3.02319479]]

首先 我们期待的是 4-3  显然非常接近 ,噪声的存在使得其不可能完全还原为原本的函数
'''
'''
现在可以用  seita 做预测
'''
print('------------------------------------------------高斯噪音--------------------------------------------------------')
x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]
# todo  进行预测
y_predict = x_new_b.dot(theta_best)
print(y_predict)

'''
绘制预测结果
'''
plt.plot(x_new, y_predict, 'r-')
plt.plot(x, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()
print('-------------------------------------scikit-Learn等效代码 ---------------------------------------------------')
'''
下面是等效的scikit-Learn代码 
scikit-Learn将偏置项(intercept_)和特征权重(coef_)分离开了
'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
'''
下面这里就是 scikit-Learn内部计算完标准方程之后的结果
'''
print(lin_reg.intercept_,lin_reg.coef_)
'''
预测函数  
内部需要添加x0
'''
print(lin_reg.predict(x_new))

