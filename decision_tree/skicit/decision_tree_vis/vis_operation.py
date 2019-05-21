#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 下午8:34
# @Author  : Aries
# @Site    : 
# @File    : vis_operation.py
# @Software: PyCharm
'''
首先要了解决策树
我们需要先构建一个决策树 并将决策树可视化
'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from command.DataUtils import serialize_data

iris = load_iris()
x = iris.data[:, 2:]
y = iris.target
serialize_data(iris, 'iris', 2)

'''
max_depth:最大深度
'''
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(x, y)
serialize_data(tree_clf, 'tree_clf', 2)

print('----------------------------------------------决策树可视化--------------------------------------------------')
export_graphviz(
    tree_clf,
    out_file='/Users/houruixiang/python/Scikit-learn-master/decision_tree/assets/iris_tree.dot',
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

'''
从可视图可知参数:
length:花瓣长
width:花瓣宽
gini:基尼系数<不纯度>
value:各种类型数目<可以用于计算基尼系数>
'''

'''
skicit-learn中的CART算法仅仅生成二叉树,就是说其叶子节点只有  T or  F
比如ID3生成的决策树,其节点可以拥有2个以上的节点
'''
'''
决策树 和Logistic回归一样 可以输出每个类别的概率 然后选择最高的输出
仔细观察决策树可以看出来
有问题 会有不准确的情况出现
'''
print(tree_clf.predict_proba([[2, 1.5]]))  # length and width
print(tree_clf.predict([[2, 1.5]]))
print(tree_clf.classes_)
