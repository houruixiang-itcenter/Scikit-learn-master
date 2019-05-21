#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 下午11:25
# @Author  : Aries
# @Site    : 
# @File    : svm_classifier.py
# @Software: PyCharm
'''
SVM是一个功能强大并且全面的机器学习模型
可以执行很多分类任务
适用于小中型的复杂数据集的分类


----
多特征的分类 会涉及到空间的知识 所以要有足够的空间感
'''
import warnings

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from command.DataUtils import get_serialize_data, serialize_data
from sklearn.datasets import make_moons
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

print('-----------------------------------------------大间隔分类--------------------------------------------------------')
'''
except:
决策边界最好是平行的直线 这样远离实例  拟合程度比较好 分类的边界也比较明确

但是SVM的分类  对特征缩放十分敏感 如果特征之间相差的量级太大  就会使得最宽的街道接近于水平 or 垂直  不利于分类
'''

print('-----------------------------------------------软间隔分类--------------------------------------------------------')
'''
硬间隔:  <不允许有异常值的出现
>
就是数据必须位于街道两侧 但是如果有异常数据 如:A类的数据,异常值在B类数据集合中 
这样就给硬间隔造成了困扰 导致分类器不能很好的泛化
'''
'''
软间隔
尽可能的在保持街道宽阔和限制 间隔违例<就是位于街道之上,甚至在错误一边的实例> 之间找到良好的平衡,这就是  软间隔分类
'''
# TODO 其实SVM分类器可以看做是不同类别之间拟合可能得最宽的街道
# TODO 所以C值变小,街道变宽,虽然间隔违例变多但是 分类的正确率也会随之提高
print('-----------------------------------------------线性SVM分类------------------------------------------------------')
'''
在scikit-learn中通过控制C的大小来控制模型的拟合程度 
C值越小则间隔违例也会越多  但是拟合程度会好一点 就是说分类的正确率提高 
而做为一个分类器我们更加关心的应当是分类的正确率


----
下面是加载鸢尾花数据集,特征缩放,使用C值进行正则化处理,并且使用之后的hinge损失函数
'''
iris = datasets.load_iris()
serialize_data(iris, 'iris', 1)
x = iris['data'][:, (2, 3)]  # 宽度和长度
y = (iris['target'] == 2).astype(np.float64)  # Iris -- Virginica

'''
loss:惩罚函数 
dual : bool, (default=True)

选择算法以解决双优化或原始优化问题。 当n_samples> n_features时，首选dual = False。
实例数 大于 特征数的时候  首选dual = False 
'''
svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge')),
])

svm_clf.fit(x, y)
print(svm_clf.predict([[5.5, 1.7]]))
print(svm_clf.classes_)
'''
与Logistic不同的是 SVM不会输出每个类别的概率
'''
print('-----------------------------------------------非线性SVM分类----------------------------------------------------')
'''
虽然许多情况下,线性SVM分类器是有效的,并且通常出人意料的好
但是有很多数据集远不是线性可分离的,处理非线性数据的方法之一
是利用trainingModel中PolynomialFeatures进行参数组合添加新特证

譬如  有许多数据模型数据不可分离  但是添加多项式之后就变得线性可分离

-----
下面来实现一个线性不可分离的实现
我们基于卫星数据来试一下
'''
moons = make_moons()

x_moons = moons[0]
y_moons = moons[1]
serialize_data(moons, 'moons', 1)

polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge'))
])
polynomial_svm_clf.fit(x_moons, y_moons)
print(polynomial_svm_clf.predict([['1.5183', '-0.3551']]))

print('-----------------------------------------------多项式核----------------------------------------------------')
'''
添加多项式特征实现起来非常简单,并且对所有机器学习有效
但是有两个痛点
1.阶数太低 不足以处理太过复杂的数据
2.阶数太高模型运行起来又太慢,还有过度拟合的风险


幸运的是 SVM有一个魔术般的数学技巧可以使用,叫做核技巧
它的效果和添加许多多项式特征,甚至非常高阶的多项式一样
但是并不是真正的添加,所以不会存在数量爆炸的风险 


?????? 那为什么呢 
先看代码                       
'''
poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(x_moons, y_moons)
print(poly_kernel_svm_clf.predict([['1.5183', '-0.3551']]))

print('----------------------------------添加相似特征 and 高斯RBF-------------------------------------')

rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(x_moons, y_moons)
print(rbf_kernel_svm_clf.predict([['1.5183', '-0.3551']]))

'''
注意:
LinearSVC/SVC  
C:与SVM分类器街道的宽窄负相关  C越小街道越大 间隔违例越多,准确率高  

SVC:
coef0:对多项式的敏感度,coef0越大就越敏感
gamma:影响相似特征的钟形曲线,gamma越小钟形曲线越宽 影响范围就越大

当然我们要通过网格搜索来找到最佳的超参数
'''
'''
时间复杂度
LinearSVC: 时间复杂度低,但是数据量大的时候,数据会爆炸
SVC:时间复杂度高,但是有核技巧,不会数据爆炸
'''

