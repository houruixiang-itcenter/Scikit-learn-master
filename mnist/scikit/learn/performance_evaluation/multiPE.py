#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/28 下午3:12
# @Author  : Aries
# @Site    :
# @File    : multiPE.py
# @Software: PyCharm
'''
多类别分类器的性能评估
'''
import warnings

import numpy
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

'''
评估准确率
'''
from mnist.scikit.learn.utils.DataUtils import get_serialize_data, plot_digits
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sgd_clf = get_serialize_data('sgd-multi')
x_train = get_serialize_data('X_train')  # type: numpy.ndarray
y_train = get_serialize_data('Y_train')
cross_scores = cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring='accuracy')
print(cross_scores)
'''
这个准确度是80%+  如果随机蒙的话 大概是10%  所以我觉得其实还可以 
但是还可以进一步优化  比如特征缩放 将所有的属性特征缩放到统一量级
'''
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train.astype(np.float64))
cross_scores_result = cross_val_score(sgd_clf, x_train_scaler, y_train, cv=3, scoring='accuracy')
print(cross_scores_result)

print('--------------------------------------------混淆矩阵------------------------------------------------------')
'''
对于分类器来说 
我认为准确率做为考量标准还是不够的 
我们还是应该 基于混淆举证 来下下面几个指标把
1.精确度  2.召回率  3,ROC
'''
y_train_pred = cross_val_predict(sgd_clf, x_train_scaler, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)

'''
从混淆矩阵来看  图片基本都集中在对角线上 由此可见 这个分类器还是不错的

接着我们把焦点聚集到错误图片的分布上来看
我们使用错误率 而不是错误的绝对值 后者会图片较多的类别不公平
'''
rows_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / rows_sums
print('rows_sums:  ', rows_sums)
print('norm_conf_mx:  ', norm_conf_mx)
'''
为了分析错误图片  
有必要 把对角线上所有错误的图片置为0
'''
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
'''
从图片中来看 
暗的表示错误的图片较少  亮的代表错误的图片较多
所以 就本例来看 1,2的图片较暗  所以被分到1,2错误图片较少  8,9比较多 
结合行和列来看 第5行  第8列就比较亮  说明 5被错误分为8的较多
结合行和列来看 第3行  第5列就比较亮  说明 3被错误分为5的较多
'''
print('--------------------------------------------具体错误的分析------------------------------------------------------')
'''
来重点看一下 3和5的混淆错误

这里需要涉及到一个布尔索引

In [24]: arr = np.arange(7)
In [25]: booling1 = np.array([True,False,False,True,True,False,False])
In [26]: arr[booling1]
Out[26]: array([0, 3, 4])

一目了然  这里不再做缀述
'''
cl_a, cl_b = 3, 5
x_aa = x_train[(y_train == cl_a) & (y_train_pred == cl_a)]
x_ab = x_train[(y_train == cl_a) & (y_train_pred == cl_b)]
x_ba = x_train[(y_train == cl_b) & (y_train_pred == cl_a)]
x_bb = x_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# print(x_aa)
# print(x_ab)
# print(x_ba)
# print(x_bb)
plt.figure(figsize=(8, 8))
plt.subplot(221)
plot_digits(np.r_[x_aa[:25]], images_per_row=5)
plt.subplot(222)
plot_digits(np.r_[x_ab[:25]], images_per_row=5)
plt.subplot(223)
plot_digits(np.r_[x_ba[:25]], images_per_row=5)
plt.subplot(224)
plot_digits(np.r_[x_bb[:25]], images_per_row=5)

plt.show()

'''
到此为止  我们就获取到了    
3的真正类  3的假正类 
5的真正类  5的假正类 


---- 到此为止 我们有分析到 
3和5 的区别就是 顶线和下端连接线的弧度 
3的顶线左偏则容易被认为是5
5的顶线右偏则容易被认为是3

所以减少错误率的一个方式就是在处理数据的时候  中心旋转让数字变正
'''

