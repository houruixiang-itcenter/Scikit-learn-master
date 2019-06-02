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

from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
	GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, tree
from sklearn.datasets import load_iris

from command.DataUtils import get_serialize_data
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(x_train, y_train)
y_pred = rnd_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(rnd_clf.feature_importances_)

'''
下面的bagging分类器  等同于上面的随机森林
'''
bag_clf = BaggingClassifier(
	DecisionTreeClassifier(splitter='random', max_leaf_nodes=16),
	n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
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
ext_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

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
MNIST数据集训练一个随机森林分类器
'''
# rnd_clf_iris = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# x_train_iris = get_serialize_data('X_train', 4)
# y_train_iris = get_serialize_data('Y_train', 4)
# rnd_clf_iris.fit(x_train_iris,y_train_iris)

print('-------------------------------------------提升法: AdaBoost----------------------------------------------------')
'''
新预测器对其前序进行纠正的方法之一,就是更多的关注前序你和不足的训练实例,从而使新的预测器不断的越来越专注难缠的问题,
这就是AdaBoost使用的技术

------------------------重点-----------------------------
1.实例的权重
其实这是一个串行循环的过程 :
首先使用第一个基础分类器进行预测,更新其错误的权重;
然后基于这个权重对第二个分类器进行预测,并更新权重,以此类推...
#######
每次提高错误实例的权重,下个算法模型就会高概率的关注错误的实例,进而根据这些错误实例调整算法模型

注意:AdaBoost调整的权重是训练实例中错误实例的权重,那么 这个模型的拟合区域会主要集中在错误实例之上

AdaBoost这种依序循环的学习技术跟梯度下降 有一些异曲同工之处,差别在于---不再是调整单个预测器的参数试成本函数最小化,而是不断在集成中加入
预测器,使模型越来越好

2.预测器的权重
预测器的准确率越高---权重就越高
预测器只是随机猜测---权重接近于0
预测器的准确率低于随机猜测的准确率---权重为负
----------------------------------------------------




由于这是串行的运行 所以这种方式相对于bagging和pasting而言很慢


预测器的准确率越高---权重就越高
预测器只是随机猜测---权重接近于0
预测器的准确率低于随机猜测的准确率---权重为负
'''
'''
提升法AdaBoost:
仅仅是计算每一个分类器的权重而已
'''
'''
最后 会获得每一个分类器的权重,然后利用投票法进行预测
'''
'''
下面来看scikit-learn代码
基于200个单层的决策树
'''
ada_clf = AdaBoostClassifier(
	DecisionTreeClassifier(max_depth=1), n_estimators=200,
	algorithm='SAMME.R', learning_rate=0.5
)
ada_clf.fit(x_train, y_train)
y_pred = ada_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print('-------------------------------------------提升法: 梯度提升------------------------------------------------------')
'''
AdaBoost vs 梯度提升
AdaBoost:每次迭代调整实例中的错误实例权重(还有预测器的权重)
梯度提升:新的预测器对前一个预测器的残差进行拟合

另一个受欢迎的提升法就是梯度提升

他也是在集成中逐步添加预测器,每一个都对前序做出改正,但是与AdaBoost不同的是:
他不是调整每一个实例的权重,而是让新的预测器针对前一个预测器的残差进行拟合
'''
x = 2 * np.random.rand(2000, 100)
y = 4 + 3 * x + np.random.rand(2000, 1)
'''
提供一个基础的预测器(决策树)
'''
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(x, y)
lin_reg = LinearRegression()
lin_reg.fit(x, y)

'''
针对第一个预测器的残差训练第二个预测器
'''
s = lin_reg.predict(x)
s1 = tree_reg1.predict(x)
y2 = y - tree_reg1.predict(x)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(x, y2)
'''
依次类推...
'''
y3 = y2 - tree_reg2.predict(x)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(x, y3)

x_new = 2 * np.random.rand(200, 100)
y_new = 4 + 3 * x_new + np.random.rand(200, 1)
print(mean_squared_error(y_new, tree_reg1.predict(x_new)))
print(mean_squared_error(y_new, tree_reg2.predict(x_new)))
print(mean_squared_error(y_new, tree_reg3.predict(x_new)))
y_pred = sum(tree.predict(x_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(y_pred)
print('-------------------------------------GradientBoostingRegressor-----------------------------------------')

x_train = 2 * np.random.rand(2000, 1)
y_train = 4 + 3 * x_train + np.random.rand(2000, 1)

'''
learning_rate:之前是学习率,现在可以理解为对每棵树的贡献进行缩放
-------------------------------------------------------
若将其设置为低值:
1.随机森林中树的数量小:拟合不足
2.随机森林中树的数量大:过度拟合
'''
# gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1)
# gbrt.fit(x_train, y_train)
#
# x_train_new = 2 * np.random.rand(200, 100)
# y_train_new = 4 + 3 * x + np.random.rand(200, 1)
#
# print(mean_squared_error(y_train_new, gbrt.predict(x_train_new)))

'''
针对低learning_rate下的模型,我们可以使用早期停止法进行寻找树的最佳数量
最简单的方式就是使用staged_predict方法(每个阶段返回一个迭代器)
Scikit-Learn
1.下面的代码训练一个120棵树的GBRT
2.使用早期停止法找到最优的树的数量
3.重新训练最优模型
'''
x_train = get_serialize_data('X_train', 6)
x_val = get_serialize_data('X_test', 6)
y_train = get_serialize_data('Y_train', 6)
y_val = get_serialize_data('Y_test', 6)

'''
降维 数据太大
注意使用这个降维 必须保证features > simple
'''
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for x_batch in np.array_split(x_train, n_batches):
	# 部分适配
	inc_pca.partial_fit(x_batch)

x_train = inc_pca.transform(x_train)

n_batches = 50
inc_pca = IncrementalPCA(n_components=154)
for x_batch in np.array_split(x_val, n_batches):
	# 部分适配
	inc_pca.partial_fit(x_batch)

x_val = inc_pca.transform(x_val)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(x_train, y_train)
print('原始的均方根误差: ', mean_squared_error(y_val, gbrt.predict(x_val)))

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(x_val)]
best_n_estimators = np.argmin(errors) + 1
print('最优的决策树数目: ', best_n_estimators)
'''
到此为止,便拿到了最优的决策树的数量
那么便可以基于此进行最优模型的训练
'''
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
gbrt_best.fit(x_train, y_train)
print('最优解的均方根误差: ', mean_squared_error(y_val, gbrt_best.predict(x_val)))

'''
要使用早起停止法 不一定非要先训练大量的数,然后回头进行早期停止的操作
在GradientBoostingRegressor中
1.加一个参数warm_start=True
2.当fit被调用时候,Scikit-learn会保留现有的树,从而允许增量训练,下面代码在误差连续5次没有改善时候自己停止训练

ok~ 下面看code的实现:
'''
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
min_val_error = float('inf')
error_going_up = 0
for n_estimators in range(1, 120):
	gbrt.n_estimators = n_estimators
	gbrt.fit(x_train, y_train)
	y_pred = gbrt.predict(x_val)
	val_error = mean_squared_error(y_val, y_pred)
	if val_error < min_val_error:
		min_val_error = val_error
		error_going_up = 0
	else:
		error_going_up += 1
		if error_going_up == 5:
			break  # 早期停止
