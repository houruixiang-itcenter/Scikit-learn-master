#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/15 下午4:21
# @Author  : Aries
# @Site    :
# @File    : TestSetAssessModel.py
# @Software: PyCharm
'''
到此为止 我们的模型训练完成 紧接着
我们需要用测试集来评估当前模型
'''
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
import numpy as np

from housing.scikit.learn.core.data.PipelineTransformation import get_pipeline_data
from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_test_unlabel_data, get_test_label_data
from housing.scikit.learn.utils.DataTestingUtils import get_serialize_data, calRMSE

'''
首先获取我们当下最优的模型

'''
grid_search = get_serialize_data('grid_search')  # type: GridSearchCV
final_model = grid_search.best_estimator_
print(type(final_model))
x_test = get_test_unlabel_data()
y_test = get_test_label_data()
# x_test_prepared = get_pipeline_data(x_test)
full_pipeline = get_serialize_data('full_pipeline')
'''
注意测试集  进行运作时候需要直接进行transform  而不是fit_transform
'''
x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
final_rmse = calRMSE(y_test, final_predictions)
print('----------------------------------------------测试集评估--------------------------------------------------------')
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
print(final_model)
print(final_rmse)
