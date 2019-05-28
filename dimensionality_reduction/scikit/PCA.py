#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/28 下午4:00
# @Author  : Aries
# @Site    :
# @File    : PCA.py
# @Software: PyCharm

'''
PCA是当下最流行的降维算法
他先是识别出最接近数据的超平面,然后将数据投影在上面

注意 数据集绝对不能有缺省 不然会有Nan值出现  影响code的运行
'''
from command.DataUtils import get_serialize_data
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dimensionality_reduction.scikit.utils.matplotUtils import explained_variance_ratio_vs_num

'''
保留差异性:将更多的数据投影,尽量最大程度的避免数据丢失

主成分:在训练中识别哪条轴对差异性的贡献度高
一般会有许多主成分的轴,最后根据这些转化成多维空间
'''

X = get_serialize_data('X', 5)
Y = get_serialize_data('Y', 5)

# 数据集中
'''
axis=1 求每一行的平均值
axis=0 求每一列的平均值
V.T 就是我们所有想要的成分
'''
X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X)
# # 主成分
c1 = V.T[:, 0]
c2 = V.T[:, 1]

# 将训练集投影到由前两个主成分定义的平面上

w2 = V.T[:, :2]
X2D = X_centered.dot(w2)
print(X2D)

print('------------------------------------------------PCA-----------------------------------------------------')
'''
使用Scikit-Learn
Scikit-Learn PCA类也是使用SVD分解来实现主成分分析,以下代码就是PCA将数据集降到二维
'''
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(X2D)

'''
可以直接访问主成分
'''
print(pca.components_)
print('------------------------------------------------方差解释器-----------------------------------------------------')
'''
还有一个非常有用的指标--方差解释率
表示每一个主成分对整个数据集的方差贡献度
'''
print(pca.explained_variance_ratio_)
print('---------------------------------------------选择正确数量的维度-------------------------------------------------')
'''
如何选择正确的维度:
1.选择主成分集合中,靠前的几个元素,使用靠前的主成分的方差解释率一次相加,直到得到足够大比例的方差,此时便是最好的维度

2.如果要可视化观察数据集,可以直接降到二维或者三维
'''
pca0 = PCA()
pca0.fit(X)
cumsum = np.cumsum(pca0.explained_variance_ratio_)
# argmax 取出最大索引
d = np.argmax(cumsum >= 0.95) + 1
print(d)
'''
从上面来看 7维的超空间就是很好的降维选择

当然还可以在PCA直接指定 希望方差概率累加为0.95 
'''
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(X_reduced)

'''
绘制解释方差和维度数量的函数
'''
explained_variance_ratio_vs_num(range(0, len(cumsum)), cumsum)
plt.show()