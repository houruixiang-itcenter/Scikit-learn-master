#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 下午10:31
# @Author  : Aries
# @Site    :
# @File    : bagging_pasting.py
# @Software: PyCharm

'''
bagging and pasting
'''
import warnings

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from command.DataUtils import get_serialize_data
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

'''
1.硬投票<特征> and 软投票<概率> 是集合不同的算法模型训练相同的数据

2.bagging<放回采样> and pasting<不放回的采样> 相同的算法模型  训练不同的数据
一旦预测器训练完成,集成就可以简单的聚合所有预测器的预测,类似于硬投票用于分类<多数预测> 或者 平均法进行回归

优势:可以通过不同的内核cpu内核甚至不同服务器,并行地训练预测器,预测也是并行的,所以易于拓展
'''
'''
下面来看 scikit代码
1.包含500个决策树分类器
2.随机从训练集中选取100个训练示例进行训练

对于bagging当然ok  但是对于pasting不放回的训练 数据集 = 模型数 * max_samples
所以我觉得 bagging更加好一些 而且可以进行包外评估
'''
'''
params:
n_estimators:算法模型数
max_samples:每次取样的最大实例
bootstrap: True--bagging False--pasting
n_jobs:指示scikit-learn用多少CPU进行训练和预测<-1表示使用所有可用的内核>
'''
x_train = get_serialize_data('x_train', 3)
y_train = get_serialize_data('y_train', 3)
x_test = get_serialize_data('x_test', 3)
y_test = get_serialize_data('y_test', 3)

bag_clf = BaggingClassifier(
	DecisionTreeClassifier(), n_estimators=500, max_samples=100,
	bootstrap=True, n_jobs=-1, random_state=42
)
bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

bag_clf1 = BaggingClassifier(
	DecisionTreeClassifier(), n_estimators=500, max_samples=100,
	bootstrap=False, n_jobs=-1, random_state=42
)

bag_clf1.fit(x_train, y_train)
y_pred1 = bag_clf.predict(x_test)
print(accuracy_score(y_test, y_pred1))

print('----------------------------------------------random Patches特征抽样--------------------------------------------')

'''
上面是对实例进行抽样 在bagging/pasting中 还可以对特征进行抽样 对应参数:bootstrap_features and max_features等 这叫做随机子空间法
'''
bag_clf2 = BaggingClassifier(
	DecisionTreeClassifier(), n_estimators=500, max_features=2,
	bootstrap=True, n_jobs=-1, random_state=42
)
bag_clf2.fit(x_train, y_train)
y_pred2 = bag_clf2.predict(x_test)
print(accuracy_score(y_test, y_pred2))
