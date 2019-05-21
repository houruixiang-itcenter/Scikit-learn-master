#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 下午9:41
# @Author  : Aries
# @Site    : 
# @File    : dataGroupOperation.py
# @Software: PyCharm
from pandas import DataFrame

from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_train_test_set

train_set, test_set = get_train_test_set()
# todo 为了不影响训练数据 这里基于训练数据进行处理
assert isinstance(train_set, DataFrame)
housing = train_set.copy(deep=True)

print('-----------------------------------数据组合-----------------------------------------')
# todo
# 在数据可视化的过程中 我们会注意到有些数据和房价中位数不是那么高 那么我们需要在数据上做文章   不同属性之间进行组合来创建洗的属性
# 通过检测其相关性 来检测其是否有意义
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedroom_per_household'] = housing['total_bedrooms']/housing['households']
housing['population_per_household'] = housing['population']/housing['households']
corr_matrix = housing.corr()
corr_matrix = corr_matrix['median_house_value'].sort_values(ascending=False)
print(corr_matrix)



