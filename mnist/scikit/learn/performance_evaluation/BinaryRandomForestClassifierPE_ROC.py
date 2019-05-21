#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18 下午8:10
# @Author  : Aries
# @Site    :
# @File    : BinarySGDClassifierPE.py
# @Software: PyCharm
'''
ROC曲线是与二元分类一起使用的工具
它描述的是灵敏度(召回率) 和 (1-特异度)的函数


-----
True Positive Rate（真正率 , TPR）或灵敏度（sensitivity）
TPR = TP /（TP + FN）
正样本预测结果数 / 正样本实际数

True Negative Rate（真负率 , TNR）或特指度（specificity）
TNR = TN /（TN + FP）
负样本预测结果数 / 负样本实际数

False Positive Rate （假正率, FPR）
FPR = FP /（FP + TN）
被预测为正的负样本结果数 /负样本实际数

False Negative Rate（假负率 , FNR）
FNR = FN /（TP + FN）
被预测为负的正样本结果数 / 正样本实际数
'''
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from mnist.scikit.learn.utils.DataUtils import get_serialize_data, plot_roc_curve, serialize_data
from mnist.scikit.learn.performance_evaluation.BinarySGDClassifierPE_ROC import plot_sgd_roc, get_sgd_fpr_and_tpr
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 忽略 警告warnings
warnings.filterwarnings('ignore')

y_train_5 = get_serialize_data('y_train_5')
X_train = get_serialize_data('X_train')

'''
创建 RandomForestClassifier
'''
forest_clf = RandomForestClassifier(random_state=42)
'''
获取每一个实例的决策分数
在获取决策分数的时候  与SGD稍有不同  就是method decision_function --> predict_proba
但是dict_proba() 返回的是一个数组  每行代表一个实例 每列代表一个类别<value是概率>
这个类别是根据概率划分的  即这个实例 70%是5  30%是非5
'''
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')

'''
而绘制roc要的是分数 而不是概率 一个简单的解决方案就是直接使用正类的概率做为分数值
u765r4 '''
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholdsr_forest = roc_curve(y_train_5, y_scores_forest)
fpr, tpr = get_sgd_fpr_and_tpr()
plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'RF')
plt.legend(loc='bottom right')
plt.show()

'''
从途中可以看到   RandomForest 的Roc比SGD更加偏上  然后AUC分数也跟多 精度和召回率也都不错  
因此他的性能优于SGD
'''
serialize_data(forest_clf, 'forest_clf')
