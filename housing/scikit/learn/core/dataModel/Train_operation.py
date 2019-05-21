#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/12 下午4:53
# @Author  : Aries
# @Site    :
# @File    : Train_operation.py
# @Software: PyCharm

'''
数据进过之前的处理  已经洗涤 优化  完善   自认为已经是一个很棒的数据集合
那么 接下来进行培训合评估模型的工作

Data columns (total 10 columns):
longitude             20640 non-null float64
latitude              20640 non-null float64
housing_median_age    20640 non-null float64
total_rooms           20640 non-null float64
total_bedrooms        20433 non-null float64
population            20640 non-null float64
households            20640 non-null float64
median_income         20640 non-null float64
median_house_value    20640 non-null float64
ocean_proximity       20640 non-null object

'''
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

from housing.scikit.learn.core.data.PipelineTransformation import get_pipeline_data
from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_unlabel_data, get_label_data
from housing.scikit.learn.utils.DataTestingUtils import calRMSE, display_scores, serialize_data, get_serialize_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor

print('-----------------------------------------------训练一个线性回归模型-----------------------------------------------')
'''
线性回归模型  --- LinearRegression
'''
housing = get_unlabel_data()  # type: DataFrame
housing_label = get_label_data()  # type: DataFrame
housing_prepared = get_pipeline_data(housing)
lin_reg = LinearRegression()
'''
@params:
1.  x  自变量
2.  y  函数 
'''
print(housing_prepared.shape)
# todo 线性回归的估值器
lin_reg.fit(housing_prepared, housing_label)
'''
至此一个线性回归的模型已经ok
测试几个数据集

这里测试时候需要注意 涉及到独热编码 有一个弊端  文本类型物种都要有  
所以我们使用上面做过转化的数据
'''
some_data = housing_prepared[:5]
some_labels = housing_label.iloc[:5]

print('------------------------------------------------测试回归模型---------------------------------------------------')
print(lin_reg.predict(some_data))
print(list(some_labels))

print('--------------------------------------------模型训练好之后来测量模型的均方根误差-------------------------------')
''''
就算当前模型的均方根误差
'''
housiong_predict = lin_reg.predict(housing_prepared)
RMSE = calRMSE(housing_label, housiong_predict)
print(RMSE)
print('-------------------------------DecisionTreeRegressor模型的均方根误差-------------------------------------------')

'''
oh  myGod  误差达到了  68628 美元 
显然差强人意  所以我们需要进一步的探索  纠错

显然这是拟合不足导致的误差,下面有几种解决方案
1.选择更加强大的模型
2.为算法训练提供更好的特征
3.减少对模型的限制

首先 可以确定这不是一个 正则化的模型 所以最后一种排除 
下面我们来尝试一个更加复杂的模型 
'''
'''
决策树模型 ---- DecisionTreeRegressor
'''
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_label)

# TODO  结合来评估这个模型的均方根误差
housiong_predict = tree_reg.predict(housing_prepared)
RMSE = calRMSE(housing_label, housiong_predict)
print(RMSE)
print('-------------------------------DecisionTreeRegressor决策树数据测试-------------------------------------------')
some_data = housing_prepared[:5]
some_labels = housing_label.iloc[:5]
print(tree_reg.predict(some_data))
print(list(some_labels))
'''
这个模型预测的完全正确  且其RMSE为0 真的这么准确么 
猜想:有可能这个模型过度拟合
'''
print('-------------------------------使用交叉验证的方式进行训练合评估-------------------------------------------')
'''
利用scikit-learn中的K-折交叉代码 将训练集分割为 10个子集<折叠>   其中1个进行评估 剩下9个进行训练
'''
print('-------------------------------评估决策树-------------------------------------------')
# todo 评估决策树
'''
交叉验证 cross_val_score 
@params-1: 预测模型实例
@params-2: 处理之后的数据 
@params-3: 标签
...
@params-5: 训练次数
'''
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_label, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)
print('------------------------------------------评估线性回归模型-----------------------------------------------')
# todo 评估线性回归模型
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_label, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
'''
end:  决策树看起来是很准确  其实是过度拟合  
根据上面的评估 我们知道 
决策树 其实比线性回归模型 还要糟糕
'''
print('---------------------------------------使用RandomForestRegressor模型-------------------------------------------')
'''
随机森林模型
'''
forest_reg = RandomForestRegressor()  # type: RandomForestRegressor
forest_reg.fit(housing_prepared, housing_label)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_label, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

'''
这个模型要比上面两种模型优秀许多  
但是但是有误差 训练集上的分数远远低于验证集

所以 意味着该模型对训练仍然过度拟合 
1.简化模型  --- 去除噪音
2.约束模型(就是正则化)
3.获取更多数据
'''
print('---------------------------------------序列化:当前注释 选择执行-------------------------------------------')
# serialize_data(lin_reg, 'lin_reg')
# serialize_data(tree_reg, 'tree_reg')
# serialize_data(forest_reg, 'forest_reg')
#
# serialize_data(lin_scores, 'lin_scores')
# serialize_data(tree_scores, 'tree_scores')
# serialize_data(forest_scores, 'forest_scores')
#
# result = get_serialize_data('forest_scores')
# print(np.sqrt(-result))

print('---------------------------------------评估-------------------------------------------')

