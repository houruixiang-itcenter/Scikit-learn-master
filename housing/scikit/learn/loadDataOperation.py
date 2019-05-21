#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/5 下午2:18
# @Author  : Aries
# @Site    : 
# @File    : loadDataOperation.py
# @Software: PyCharm
import inline as inline
import matplotlib

from housing.scikit.learn.housingUtils import load_housing_data

housing = load_housing_data()
# todo 使用DataFrames输出前五行
#    longitude  latitude  ...  median_house_value  ocean_proximity
# 0    -122.23     37.88  ...            452600.0         NEAR BAY
# 1    -122.22     37.86  ...            358500.0         NEAR BAY
# 2    -122.24     37.85  ...            352100.0         NEAR BAY
# 3    -122.25     37.85  ...            341300.0         NEAR BAY
# 4    -122.25     37.85  ...            342200.0         NEAR BAY

print(housing.head())
print('---------------------------数据集的属性---------------------------------')
# todo 快速获取数据集的简单描述
print(housing.info())
print('---------------------------类别---------------------------------')

# todo 获取数据集的类型种类
print(housing['ocean_proximity'].value_counts())
print('---------------------------数值属性的摘要---------------------------------')
print(housing.describe())
print('---------------------------各个属性的直方图---------------------------------')
import matplotlib.pyplot as plt


housing.hist(bins=50, figsize=(20, 15))
plt.show()

