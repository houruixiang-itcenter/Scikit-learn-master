#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/15 上午12:17
# @Author  : Aries
# @Site    : 
# @File    : DataParamsAdapter.py
# @Software: PyCharm
from sklearn.model_selection import GridSearchCV

from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_one_hot_attribs, get_unlabel_data
from housing.scikit.learn.utils.DataTestingUtils import get_serialize_data

grid_search = get_serialize_data('grid_search')  # type: GridSearchCV
# 独热属性  还原
cat_one_hot_attribs = get_one_hot_attribs()
unlabel_data = get_unlabel_data()
housing = unlabel_data.drop('ocean_proximity', axis=1)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

# todo 将重要性分数显示在对应的属性名称旁边
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_one_hot_attribs = list(cat_one_hot_attribs)
attributes = list(housing) + extra_attribs + cat_one_hot_attribs
sorted = sorted(zip(feature_importances, attributes), reverse=True)
print('-----------------------------------------查看数据集中属性对应的重要性分数-------------------------------------------')
print(sorted)
'''
[(0.331337407421411, 'median_income'), 
 (0.1523443854472078, 'INLAND'), 
 (0.11316200879181408, 'pop_per_hhold'), 
 (0.07093167499289693, 'rooms_per_hhold'), 
 (0.07002528090482524, 'bedrooms_per_room'), 
 (0.06977133932292791, 'longitude'), 
 (0.06302690563936922, 'latitude'), 
 (0.04312033116145733, 'housing_median_age'), 
 (0.017820406022924562, 'households'), 
 (0.01722442884164994, 'population'), 
 (0.017089864360138304, 'total_rooms'),
 (0.016618788047934183, 'total_bedrooms'), 
 (0.010089307589396159, '<1H OCEAN'), 
 (0.004202361254958904, 'NEAR OCEAN'), 
 (0.003164541391389724, 'NEAR BAY'),
 (7.096880969875023e-05, 'ISLAND')]
'''
