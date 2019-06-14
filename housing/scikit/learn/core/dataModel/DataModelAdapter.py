#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/14 下午10:36
# @Author  : Aries
# @Site    : 
# @File    : DataModelAdapter.py
# @Software: PyCharm
from pandas import DataFrame

from housing.scikit.learn.utils.DataTestingUtils import serialize_data

print('--------------------------------------------调参方式一:网格搜索---------------------------------------------------')
'''
对已经构建的模型进行微调
之前我们在Train_operation中已经使用了
线性回归模型
决策树模型
随机森林模型
但是预测结果不符合我们的预期
下面我们进行 微调模型

下面有几种方法
调整超参数  ---  比较枯燥
所以需要借助Scikit-Learn的GridSearchCV来替代你进行搜索

'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from housing.scikit.learn.core.data.PipelineTransformation import get_pipeline_data
from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_unlabel_data, get_label_data
import numpy as np

housing = get_unlabel_data()  # type: DataFrame
housing_label = get_label_data()  # type: DataFrame
housing_prepared = get_pipeline_data(housing)
'''
param_grid  这个就是告诉Scikit-Learn  
首先评估第一个dict的n_estimators max_features  总共3x4=12种组合  
然后第二个dict的n_estimators max_features 2x3=6种组合 
所以RandomForestRegressor模型的超参数组合为  12 + 6 = 18种超参数组合    cv = 5 所以就是  5x18 = 90 次训练
'''
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_label)

# todo 这样就获得了最佳的参数组合
best_combination = grid_search.best_estimator_
print('------------------------------------------最佳的参数组合-------------------------------------------------')
print(best_combination)
print('------------------------------------------评估网格搜索产物的分数-------------------------------------------------')
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
'''
最佳组合
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=6, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=None, oob_score=False,
           random_state=None, verbose=0, warm_start=False)
评估不同超参数下的分数
63950.986962931325 {'max_features': 2, 'n_estimators': 3}
56320.07504231676 {'max_features': 2, 'n_estimators': 10}
53022.78409292703 {'max_features': 2, 'n_estimators': 30}
60276.61761350148 {'max_features': 4, 'n_estimators': 3}
52705.85669162451 {'max_features': 4, 'n_estimators': 10}
50267.47942431185 {'max_features': 4, 'n_estimators': 30}
59554.699514400694 {'max_features': 6, 'n_estimators': 3}
52040.28861259152 {'max_features': 6, 'n_estimators': 10}
49888.19166658163 {'max_features': 6, 'n_estimators': 30}
58588.9199830536 {'max_features': 8, 'n_estimators': 3}
52172.63895825733 {'max_features': 8, 'n_estimators': 10}
50185.24539022929 {'max_features': 8, 'n_estimators': 30}
62255.186422943385 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54844.115506885726 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
59818.346827389745 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
52386.119945610655 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
58317.24330365975 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
52066.1078972357 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}

可以看到'max_features': 6, 'n_estimators': 30 时候分数为49888  这是目前为止最好的超参数
目前为止  便找到了超参数的最优解

除了1. 网格搜索 
还有2. 随机搜索 --- RandomizedSearchCV  用于数据量大的时候  每次迭代一次就生成一个随机数  
最后3. 集成方法 --- 多种模型互补
'''

serialize_data(grid_search, 'grid_search')



