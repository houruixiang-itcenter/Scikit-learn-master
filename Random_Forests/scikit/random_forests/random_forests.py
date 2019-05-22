#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22 下午10:11
# @Author  : Aries
# @Site    : 
# @File    : random_forests.py
# @Software: PyCharm

'''
随机森林
'''
import warnings

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

from command.DataUtils import get_serialize_data
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

'''
随机森林就是决策树的集成
通常使用bagging或者pasting方法训练,训练实例用max_sample来控制,然后传输给DecisionClassifier<前面的bagging分类器就是这样>

还有一种就是直接使用RandomForestClassifier
看下面的code
'''
'''
除了少数例外,RandomForestClassifier同时拥有DecisionClassifier的所有超参数,以及bagging的所有超参数
前者:控制树的生长
后者:控制集合本身

'''
x_train = get_serialize_data('x_train', 3)
y_train = get_serialize_data('y_train', 3)
x_test = get_serialize_data('x_test', 3)
y_test = get_serialize_data('y_test', 3)
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(x_train, y_train)
y_pred = rnd_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(rnd_clf.feature_importances_)

'''
下面的bagging分类器  等同于上面的随机森林
'''
bag_clf = BaggingClassifier(
	DecisionTreeClassifier(splitter='random', max_leaf_nodes=16),
	n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42
)
bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print('-------------------------------------------极端随机数---------------------------------------------------------')
'''
常规随机树:每次分裂会有一个最佳的阈值
极端随机树:每次分裂使用随机的阈值
这样对数据的敏感性会低于常规的随机树,然后以更高的偏差换取更低的方差
'''
ext_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)

ext_clf.fit(x_train, y_train)
y_pred = ext_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

print('-------------------------------------------特征重要性---------------------------------------------------------')
'''
如果查看单个决策树 会发现重要的特征会出现在靠近根结点的地方,而相对不重要的会出现在叶结点甚至没有

所以输出每个随机森中特征在每个决策树上的平均深度对我们的研究至关重要
通过feature_importances_可以看到每个特征的平均深度
上面已经输出 无奈特征太少 下面我们来看鸢尾花的示例:
'''
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris['data'], iris['target'])
for name, sorce in zip(iris['feature_names'], rnd_clf.feature_importances_):
	print(name, sorce)

'''
MNIST
'''

