#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 下午10:49
# @Author  : Aries
# @Site    : 
# @File    : CombinedAttrTransformation.py
# @Software: PyCharm
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_train_test_set

train_set, test_set = get_train_test_set()
# todo 为了不影响训练数据 这里基于训练数据进行处理
assert isinstance(train_set, DataFrame)
housing = train_set.copy(deep=True)  # type: DataFrame
print(housing.values)
print('----------------------------------------自定义转换器--组合参数-----------------------------------------------')
'''
Scikit提供了足够多的转换器  但是大多是duck模型
自定义的转换器 ---  为了与Scikit无缝连接
我们使用 Scikit模块提供的BaseEstimator / TransformerMixin作为基类进行自定义转换器的开发 
'''

# todo 属性组合的转换器  --- 组合属性加法器
'''
对应是的3 4 5 6列 
'''
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # noting else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            '''
            np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。--- 添加行属性       
            np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。--- 添加列属性 
            这里我们就在原来基础上  添加了 rooms_per_household and population_per_household and bedrooms_per_room
            '''
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributeAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)
print(housing['longitude'])
print(type(housing.values))
print(housing_extra_attribs)
