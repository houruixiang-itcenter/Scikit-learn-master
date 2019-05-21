#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 下午11:09
# @Author  : Aries
# @Site    : 
# @File    : outside_the_package_pe.py
# @Software: PyCharm
'''
bagging 性能评估
'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from command.DataUtils import get_serialize_data
from sklearn.metrics import accuracy_score

'''
由于bagging是放回式的抽样 所以注定会有部分训练实例不被用来训练
假定训练集的大小是m

随着m的升高 平均只对63%的训练实例进行采样
-----
剩余37%未被采样的训练实例称为包外实例
那么这37%的训练实例称为包外实例  --- 可用来进行性能评估,进而不再需要单独的测试集或者交叉验证集
'''
x_train = get_serialize_data('x_train', 3)
y_train = get_serialize_data('y_train', 3)
x_test = get_serialize_data('x_test', 3)
y_test = get_serialize_data('y_test', 3)
'''
在Bagging中加入参数 ood_score_就可以得到最终的评估分数
'''
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, max_samples=100,
    bootstrap=True, n_jobs=-1, oob_score=True, random_state=42
)
bag_clf.fit(x_train, y_train)
print(bag_clf.oob_score_)

'''
测试集评估
'''
y_pred = bag_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

'''
获取每个训练实例的包外决策函数,本例中基础预测器具备predict_proba方法,返回的是每个实例类别的概率
'''
print(bag_clf.oob_decision_function_)
