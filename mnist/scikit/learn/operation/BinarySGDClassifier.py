#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/17 下午10:57
# @Author  : Aries
# @Site    : 
# @File    : BinarySGDClassifier.py
# @Software: PyCharm
'''
二分分类器
5 and 非5
'''
from mnist.scikit.learn.utils.DataUtils import get_serialize_data, serialize_data
from sklearn.linear_model import SGDClassifier

x_train = get_serialize_data('X_train')
y_train = get_serialize_data('Y_train')
y_test = get_serialize_data('Y_test')

'''
为此分类任务创建目标向量
'''
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

'''
一个好的初始选择是随机低度下降(SGD)分类器
使用Scikit的SGDClassifier类即可
这个分类器的优势是能够有效处理非常大型的数据集,这部分是因为SDG独立处理训练实例,一次一个

预测器的话直接fit  x,y   即为训练
'''
'''
这里设置random_state是为了数据可复用
'''
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)
result = sgd_clf.predict(x_train[:30])
print(result)
print(y_train[:30])

'''
序列化分类器
'''
serialize_data(y_train_5, 'y_train_5')
serialize_data(y_test_5, 'y_test_5')
serialize_data(sgd_clf, 'sgd_clf')

