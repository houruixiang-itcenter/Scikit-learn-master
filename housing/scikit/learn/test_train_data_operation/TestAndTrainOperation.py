#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/5 下午4:08
# @Author  : Aries
# @Site    : 
# @File    : TestAndTrainOperation.py
# @Software: PyCharm
'''
获取不同需求数据的py-file
'''
from pandas import DataFrame
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer

from housing.scikit.learn.housingUtils import load_housing_data, spilt_train_test, unique_spilt_train_test_by
import numpy as np

housing = load_housing_data('/Users/houruixiang/python/Scikit-learn-master/housing/scikit/learn/datasets/housing')


# todo 产出训练数据和测试数据
# train_set, test_set = spilt_train_test(housing, 0.2)
# print(len(train_set), 'train + ', len(test_set), 'test ')
# 上面的方式 的确可以获取训练数据和测试数据 但是具有不唯一性 也就是说每次运行不一样
# todo 为了保证id的唯一性  使用每一行的经纬度生成  因为经纬度独一无二
# housing['id'] = housing['longitude'] * 1000 + housing['latitude']
# train_set, test_set = unique_spilt_train_test_by(housing, 0.2, 'id')
# print(len(train_set), 'train + ', len(test_set), 'test ')

# todo 当然可以使用scikit中的分割 来实现 但是这个分割的算法与第一种相似
# todo 同样对于数据更新不具备唯一性 所以这里我还是使用的自己写的那一套
# train_test_split(housing, test_size=0.2, random_state=42)
def get_train_test_set():
    '''
    获取训练和测试数据集 原始数据
    :return:
    '''
    global start_train_set, start_test_set
    # todo  新的调研 收入中位数hin重要   所以测试集和训练集应当是基于收入中位数的分层抽样
    # todo  对收入中位数进行处理 <收入中位数/1.5  且  大于5的value全部并入value=5的区域>
    print('----------------------------------------housing one column type--------------------------------------------')
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    print(type(housing['income_cat']),type(housing['median_income']))
    # todo params 参数1:指定<5的'income_cat'无需进行check  参数2:目标值5.0  参数3:是否替换,即当inplace为true的时候全部替换为目标值5.0
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        start_train_set = housing.loc[train_index]
        start_test_set = housing.loc[test_index]
    print(len(start_train_set), 'train + ', len(start_test_set), 'test ')
    print(housing['income_cat'].value_counts() / len(housing))

    # todo 到此为止我们训练集/测试集已经采集完成 接着我们需要把之前用于进行分层抽样的'income_cat'进行删除,回归数据的本来面目  <当然我觉得留下也不是不可以 >
    for set in (start_train_set, start_test_set):
        assert isinstance(set, DataFrame)
        set.drop(['income_cat'], axis=1, inplace=True)

    return start_train_set, start_test_set

def get_unlabel_data():
    '''
    获取非标签数据
    :return:
    '''
    train_set, test_set = get_train_test_set()
    housing = train_set.copy(deep=True)  # type: DataFrame
    # todo step1:首先将预测器和标签分开  使用不同的转换
    housing = housing.drop('median_house_value', axis=1)  # 不改变原有数据集合
    return housing


def get_label_data():
    '''
    获取标签数据 --- median_house_value
    :return:
    '''
    train_set, test_set = get_train_test_set()
    housing_labels = train_set['median_house_value'].copy(deep=True)
    return housing_labels


def get_test_unlabel_data():
    '''
    获取非标签数据
    :return:
    '''
    train_set, test_set = get_train_test_set()
    housing = test_set.copy(deep=True)  # type: DataFrame
    # todo step1:首先将预测器和标签分开  使用不同的转换
    housing = housing.drop('median_house_value', axis=1)  # 不改变原有数据集合
    return housing


def get_test_label_data():
    '''
    获取标签数据 --- median_house_value
    :return:
    '''
    train_set, test_set = get_train_test_set()
    housing_labels = test_set['median_house_value'].copy(deep=True)
    return housing_labels

def get_one_hot_attribs():
    housing = get_unlabel_data()
    housing_cat = housing['ocean_proximity']
    encoder = LabelBinarizer()  # 这里默认返回的numpy数组  通过在构造中传入参数sparse_output = True 来控制返回SciPy稀疏矩阵
    housing_cat_2hot = encoder.fit_transform(housing_cat)
    return encoder.classes_
