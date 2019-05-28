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

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

from mnist.scikit.learn.utils.DataUtils import get_serialize_data, plot_roc_curve
import matplotlib.pyplot as plt

# 忽略 警告warnings
warnings.filterwarnings('ignore')

y_train_5 = get_serialize_data('y_train_5')
X_train = get_serialize_data('X_train')
sgd_clf = get_serialize_data('sgd_clf')  # type: SGDClassifier
'''
返回每个实例决策的分数
'''
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)
'''
下面来看matplotlib绘制FPR对TPR的曲线
'''

'''

随着真正率的提高 假正率会随之提高  当然反之亦然  
然后图中 有一个连接对角线的虚线  这个叫做ROC曲线  --- 这是完全随机时候 FPR vs TPR的函数曲线
所以 FPR vs TPR的曲线在ROC曲线上方 离他越远便越完美 


当然还有一种评估分类器的方法是计算曲线下方的面积  叫做AUC <随机情况下 是0.5   最完美情况下是 1>
'''
print(roc_auc_score(y_train_5, y_scores))


def plot_sgd_roc():
    plot_roc_curve(fpr, tpr)
    plt.show()


plot_sgd_roc()


def get_sgd_fpr_and_tpr():
    return fpr, tpr
