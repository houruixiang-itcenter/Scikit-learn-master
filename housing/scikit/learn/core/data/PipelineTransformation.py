#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 下午8:10
# @Author  : Aries
# @Site    :
# @File    : PipelineTransformation.py
# @Software: PyCharm


from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer

from housing.scikit.learn.core.data.CombinedAttrTransformation import CombinedAttributeAdder
from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_train_test_set
from housing.scikit.learn.utils.DataTestingUtils import serialize_data

train_set, test_set = get_train_test_set()
# todo 为了不影响训练数据 这里基于训练数据进行处理
assert isinstance(train_set, DataFrame)
housing = train_set.copy(deep=True)  # type: DataFrame
# todo step1:首先将预测器和标签分开  使用不同的转换
housing = train_set.drop('median_house_value', axis=1)  # 不改变原有数据集合
housing_labels = train_set['median_house_value'].copy(deep=True)

# print(housing.values)

'''
去掉文本属性 
'''
housing_num = housing.drop('ocean_proximity', axis=1)

print('-------------------------------------------转换流水线---------------------------------------------------')
'''
params .... 最后一个参数是估值器   
其他均为转换器
这个转换流水线类似于责任链是一个一个执行的<顺序>
然后 step1  ---> step2  类似流水线  下一个转换器会基于上一个处理完的数据进行处理
'''
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributeAdder(True)),
    ('std_scaler', StandardScaler())
])
housing_num_tr = pipeline.fit_transform(housing_num)
print(housing_num_tr)
print('----------------------------------------多流水线高效处理最终数据---------------------------------------------------')
'''
这里需要自定义一个数据选择器
'''


class DataFrameSeletor(BaseEstimator, TransformerMixin):

    def __init__(self, attrs):
        self.attrs = attrs

    def fit(self, X, y=None):
        return self  # noting to do

    def transform(self, X, y=None):
        return X[self.attrs].values


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=None):
        return self.encoder.transform(x)


'''
分开处理 数值和文本

'ocean_proxymity' 数值文本  ----  最终转化为独热编码

'''

'''
适用于训练集 fit_transform先拟合数据  再进行转换  
注意fot 仅仅是求的训练集的均值 方差 最大值 最小值这些固有属性
'''


def get_pipeline_data(data):
    '''
    get_pipeline_data -- 获取最终数据
    :param data:
    :return:
    '''
    global num_pipeline, cat_pipeline, full_pipeline
    num_attrs = list(housing_num)
    print(num_attrs)
    cat_attribs = ['ocean_proximity']
    num_pipeline = Pipeline([
        ('selector', DataFrameSeletor(num_attrs)),
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributeAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSeletor(cat_attribs)),
        ('label_binarizer', MyLabelBinarizer()),
    ])
    '''
    FeatureUnion  Scikit提供 用于整合两个numpy
    '''
    print('MyLabelBinarizer :::  ', cat_pipeline.fit_transform(data))
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])
    pipeline_data = full_pipeline.fit_transform(data)
    serialize_data(full_pipeline,'full_pipeline')
    return full_pipeline.fit_transform(data)


get_pipeline_data(housing)
print('-----------------------------------------------测试1------------------------------------------------')
housing_test1 = num_pipeline.fit_transform(housing)
print(housing_test1.shape)
print(housing_test1[0])
print('-----------------------------------------------测试2------------------------------------------------')
housing_test2 = cat_pipeline.fit_transform(housing)
print(housing_test2.shape)
print(housing_test2[0])
print('-----------------------------------------------最终数据------------------------------------------------')
housing_prepared = full_pipeline.fit_transform(housing)  # type: np.ndarray
print(type(housing_prepared))
print(housing_prepared)
print(housing_prepared.shape)
#  housing_tr = pd.DataFrame(housing_prepared, columns=housing.columns)
print('-----------------------------------------------对最终数据进行查看------------------------------------------------')
# print(housing_tr.info())
# print(housing.head(10))
