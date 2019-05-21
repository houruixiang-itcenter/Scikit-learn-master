#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 下午8:57
# @Author  : Aries
# @Site    : 
# @File    : DataUtils.py
# @Software: PyCharm
from sklearn.metrics import mean_squared_error
import numpy as np


def calRMSE(exceptVal, predictVal):
    '''
    :param exceptVal: --- 实际的期望值
    :param predictVal: ---- 当前模型的预测值
    :return:
    返回当前模型的均方根误差
    '''
    lin_mse = mean_squared_error(exceptVal, predictVal)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse


def display_scores(scores):
    '''
    使用交叉验证的方式评估模型的性能
    :param scores:
    :return:
    '''
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())


from sklearn.externals import joblib


def serialize_data(tag_model, tag):
    '''
    序列化data
    :param tag_model:
    :param tag:
    :return:
    '''
    joblib.dump(tag_model, tag)


def get_serialize_data(tag):
    '''
    根据tag反序列化
    :param tag:
    :return:
    返回反序列化结果
    '''
    return joblib.load(tag)
