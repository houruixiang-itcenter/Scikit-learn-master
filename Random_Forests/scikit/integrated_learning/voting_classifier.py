#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/19 下午5:26
# @Author  : Aries
# @Site    : 
# @File    : voting_classifier.py
# @Software: PyCharm
'''
在机器学习中  需要明白几个概念
最有效的模型:
1.随机向几千个人询问一个复杂的问题,在很多情况下要比专家回答的要好,这被称为群里智慧

----
2.同样类比,如果你聚合一组预测器(比如分类器或者回归)的预测,得到的预测结果也比最好的单个预测器要好,这样一组预测器,
我们称为集成,这种技术叫做集成学习

'''
import warnings

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from command.DataUtils import get_serialize_data, serialize_data
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


'''
下面来看一个投票分类器
假设你已经训练好了一堆分类器,每个分类器的准备率均为80%以上
大概包括:逻辑回归分类器  SVM分类器 随机森林分类器  K-近邻分类器  etc...

这样解释 
比如现在有一个包含1000个分类器的集成<且每个分类器只有51%是正确的 其实就是一个弱学习器 比随机多1%而已>
这时你的预测应该是建立在模型 达到51%的比率 
就是说 此时你应该以大多数模型的预测来做为预测值

比如有900个预测是类别A  100个预测是类别B  那么 你的预测比率就是90%

当然有一个前提就是: <控制变量法>
1.每个模型是完全独立的,错误互不相关
2.每个训练的数据不一样

下面是scikit训练的一个分类器
'''
moons = make_moons(n_samples=20000)
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)
'''
训练一个投票分类器  
硬投票法<hard> 把超过半数以上的投票结果作为要预测的分类，投票法处理回归问题，是将各个基分类器的回归结果简单求平均
'''
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)

x_train, x_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.2,random_state=42)


serialize_data(x_train, 'x_train', 3)
serialize_data(y_train, 'y_train', 3)
serialize_data(x_test, 'x_test', 3)
serialize_data(y_test, 'y_test', 3)

'''
接着 我们看看其准确率
'''
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

serialize_data(voting_clf,'voting_clf',3)

'''
软投票法<soft> 是加权概率然后求其平均值  
其实有时候他会对高度自信的投票更高的权重

'''
