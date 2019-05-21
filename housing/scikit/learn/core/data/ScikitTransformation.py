#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 下午10:04
# @Author  : Aries
# @Site    : 
# @File    : ScikitTransformation.py
# @Software: PyCharm
import scipy

import numpy
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer
import pandas as pd

from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_train_test_set

train_set, test_set = get_train_test_set()
# todo 为了不影响训练数据 这里基于训练数据进行处理
assert isinstance(train_set, DataFrame)
housing = train_set.copy(deep=True)

# todo step1:首先将预测器和标签分开  使用不同的转换
housing = train_set.drop('median_house_value', axis=1)
housing_labels = train_set['median_house_value'].copy(deep=True)

print('-------------------------------------------数据清理--中位数站位缺省位------------------------------------------------')
# todo 大部分的机器学习算法无法在确实的特征上工作,所以我们要创建一些函数来辅助他
# todo 我们在前面有看到total_bedrooms有缺失数据,需要我们解决一下
# 1.放弃这些相应的地区
'''
housing.dropna(subset=['total_bedrooms'])
'''
# 2.放弃这个属性
'''
housing.drop('total_bedrooms',axis=1)
'''
# 3.将确实的值设置为某个值(0,平均数或者中位数都可以)
'''
housing['total_bedrooms'].fillna(median)
'''
# todo 推荐  Scikit-Learn提供了一个非常容易上手的教程来处理缺失值: imputer
# 创建一个imputer实例,指定你要用属性的中位数值替换该属性的缺失值
# todo 估算器
imputer = SimpleImputer(strategy='median')
# todo issue 中位数只能在数值上做计算  所以要把不是数值的属性去掉
housing_new = housing.drop('ocean_proximity', axis=1)
median = housing_new.median().values
print(median)
# todo 将imputer适配到新的训练集:
imputer.fit(housing_new)
# todo 虽然当前训练集中只有'total_bedrooms'有缺省 但是考虑到日后新数据也会缺失 所以需要适配所有的属性
median = housing_new.median().values
print(median)
# todo 将中位数存储再statistics_中
imputer.statistics_
print('-----------------------------------------------转换前----------------------------------------------------------')
print(housing_new.info())
# todo 使用这个训练有素的imputer将缺失值转化为中位数完成转换  这个x是一个转换后的array  size为16512   与dataframe相同
x = imputer.transform(housing_new)
print('-----------------------------------------------转换后----------------------------------------------------------')
print(len(x))
print(type(x))
# 将其放回到dataframe中
housing_tr = pd.DataFrame(x, columns=housing_new.columns)
print(housing_tr.info())

print('---------------------------------------------对非文本的转换----------------------------------------------------------')
# 之前在计算中位数的时候我们摒弃了非数值的column  -- ocean_proximity  现在我们将其转化为数字然后进行转换
# todo Scikit-Learn 为这类任务提供了一个转换器--LabelEncoder:
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
print('---------------------------------------------原始文本------------------------------------------------')
print(housing_cat)
# 估值器适配并转换 推荐使用fit_transform  ==>   fit -> transform
housing_cat_encoded = encoder.fit_transform(housing_cat)  # type: numpy.ndarray
print('---------------------------------------------文本转数字  success------------------------------------------------')
print(type(housing_cat_encoded))
print(housing_cat_encoded)
# 查看编码的映射
print(encoder.classes_)
# todo 对比映射  其对应关系应该是这样的
'''
<1H OCEAN  ---- 0
NEAR OCEAN ---- 4
INLAND --- 1
NEAR BAY --- 3
ISLAND --- 2
机器学习会认为 0和1会相似一点   实则0和4要比  0和1相似许多

**** todo 所以我们引入独热编码
eg:
<1H OCEAN  ---- 10000
NEAR OCEAN ---- 01000
INLAND --- 00100
NEAR BAY --- 00010
ISLAND --- 00001
'''
# todo 利用scikit进行独热编码  --- OneHotEncoder  将整数分类值转化为独热编码  其中fit_transform需要一个二维数组
# todo 但是housing_cat_encoded是一维数组,所以需要重塑
encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))  # type: scipy.sparse.csr.csr_matrix
print(type(housing_cat_1hot))
print('-----------------------------------------scipy转化为numpy.ndarray----------------------------------------------')
print(housing_cat_1hot.toarray())
# 注意这里转化完后的独热编码是scipy稀疏矩阵  当独热编码转化完毕之后  可能是几千行矩阵 然后每一行只有一个1
# 使用稀疏矩阵可以直存储1 而不存储0

print('-------------------------------使用LabelBinarizer完成文本直接到独热向量的转化---------------------------------------')
'''
上面 首先把文本 转化为数值   然后再把数值转化为独热向量 
下面我们使用LabelBinarizer直接把文本转换为独热向量
'''
encoder = LabelBinarizer()  # 这里默认返回的numpy数组  通过在构造中传入参数sparse_output = True 来控制返回SciPy稀疏矩阵
housing_cat_2hot = encoder.fit_transform(housing_cat)
print(housing_cat_2hot)



