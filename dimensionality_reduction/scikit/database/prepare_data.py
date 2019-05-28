#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/28 下午4:56
# @Author  : Aries
# @Site    : 
# @File    : prepare_data.py
# @Software: PyCharm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from command.DataUtils import serialize_data
from housing.scikit.learn.core.data.CombinedAttrTransformation import CombinedAttributeAdder
from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_unlabel_data, get_label_data
import numpy as np

X = get_unlabel_data()
X = X.drop('ocean_proximity',axis=1)
Y = get_label_data()
X = np.array(X)
Y = np.array(Y)
'''
处理X的缺省
'''
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributeAdder(True)),
    ('std_scaler', StandardScaler())
])
X = pipeline.fit_transform(X)


serialize_data(X, 'X', 5)
serialize_data(Y, 'Y', 5)
