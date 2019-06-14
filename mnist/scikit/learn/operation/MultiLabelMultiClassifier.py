#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/27 下午11:25
# @Author  : Aries
# @Site    :
# @File    : SGDMultiClassifier.py
# @Software: PyCharm
'''
多标签分类
'''
'''
多标签分类系统

目前我们的模型是这样的 一个实例只会有一个二元的分类器
但是 问题来了
如果一个图片里面有多个实例怎么办
eg  一个图片中有 1,3,5  当只有15的时候  我们应该是这样的输出 [1,0,1]
最常用的场景就是人脸识别 如 google相册

注意一个点: np_c是按行左右相加两个矩阵

a
Out[4]:
array([[1, 2, 3],
       [7, 8, 9]])

b
Out[5]:
array([[4, 5, 6],
       [1, 2, 3]])

c=np.c_[a,b]

c
Out[7]:
array([[1, 2, 3, 4, 5, 6],
       [7, 8, 9, 1, 2, 3]])

但是 np.array([7,8,9]) 被认为是3行

所以就有:
d= np.array([7,8,9])
 
e=np.array([1, 2, 3])
 
f=np.c_[d,e]
 
f
Out[12]: 
array([[7, 1],
       [8, 2],
       [9, 3]])
       
       
np_r是按列上下相加两个矩阵


-----------------------------------------------------------

'''
from sklearn.neighbors import KNeighborsClassifier

from mnist.scikit.learn.utils.DataUtils import get_serialize_data
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
import numpy as np

y_train = get_serialize_data('Y_train')
x_train = get_serialize_data('X_train')
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)

'''
这样 这个模型中 包括两个标签 第一个 是>7的数字 第二个是奇数
目的 让他同时识别>7 以及奇数
接下来测试一下 
'''
y_knn_predict = knn_clf.predict([x_train[36000]])
print('当前预测为 %s 实际数字 %s' % (y_knn_predict, y_train[36000]))

'''
接下来 需要评估下这个多标签分类器的性能 
这里我们使用F1分数
'''
y_train_knn_pred = cross_val_predict(knn_clf, x_train, y_multilabel, cv=3)
y_knn_score = f1_score(y_multilabel, y_train_knn_pred, average='macro')
print(y_knn_score)
