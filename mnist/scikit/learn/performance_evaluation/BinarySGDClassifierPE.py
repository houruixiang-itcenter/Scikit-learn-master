#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18 下午8:10
# @Author  : Aries
# @Site    : 
# @File    : BinarySGDClassifierPE.py
# @Software: PyCharm
'''
对二元分类器进行性能考核与评估
对于评估分类器其实要比评估回归器要困难许多
'''
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from mnist.scikit.learn.utils.DataFetchUtils import get_mnist_data_and_target
import matplotlib.pyplot as plt

'''
继续使用交叉验证的方式
'''

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from mnist.scikit.learn.utils.DataUtils import get_serialize_data
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from mnist.scikit.learn.utils.DataUtils import plot_precision_recall_vs_threshold, plot_recall_vs_precision

warnings.filterwarnings('ignore')

'''
n_splits:折叠次数 + 迭代次数
StratifiedKFold 会将true and false 分开
'''
skfolds = StratifiedKFold(n_splits=3, random_state=42)
y_train_5 = get_serialize_data('y_train_5')
X_train = get_serialize_data('X_train')
sgd_clf = get_serialize_data('sgd_clf')  # type: SGDClassifier

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = X_train[train_index]
    y_train_flods = (y_train_5[train_index])
    x_test_flods = X_train[test_index]
    y_test_flods = (y_train_5[test_index])

    clone_clf.fit(x_train_folds, y_train_flods)
    y_yied = clone_clf.predict(x_test_flods)
    n_correct = sum(y_yied == y_test_flods)
    print(n_correct / len(y_yied))  # 其实说白了就是预测序列和真是序列的对比

print('---------------------------------------使用cross_val_scroe来评估模型---------------------------------------------')
'''
accuracy 准确率  会有warning
'''
cross_score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(cross_score)

'''
所有交叉折叠的数据集都超过了 95%
其实做怎么说呢 这一堆数据集合中 只有大约10%是5  就是说你猜出非5的概率会是  90%
所以这样来说 准确性并不是衡量分类器 最重要的指标
'''
print('--------------------------------------------------混淆矩阵------------------------------------------------------')
'''
从上面看来 准确性的预测不具有代表性 
所以这里引入  混淆矩阵 
'''
'''
获取一个 干净的预测值 
'''
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
'''
转换混淆矩阵
'''
y_train_prefect_predictions = confusion_matrix(y_train_5, y_train_pred)
print(y_train_prefect_predictions)

'''
一个完美的分类器只有真正类和真负类,所以他的混淆矩阵只会在其对角线上有非零值
但是一定会有误差 所以做不到这一点 

计算精度的公式  TP / TP + FP  TP 真正类  FP 假正类
  召回率的公式  TP / TP + FN
计算精度和召回率 
'''
print('--------------------------------------------------精度和召回率---------------------------------------------------')
precision = precision_score(y_train_5, y_train_pred)
print(precision)
recall = recall_score(y_train_5, y_train_pred)
print(recall)

'''
0.7355274261603375
0.8039107175797823

-- 这样来看 
就是说 这个分类器判定为5 的准确率只有 百分之  73
      然后只有  百分之80的5被分类器分出来
'''

print('-----------------------------------------精度和召回率组合为 F1分数--------------------------------------------')
'''
精度和召回率是极具代表性的指标  
所以我们可以把他组合起来 做为评判分类器的指标


F1分数是精度和召回率的谐波平均值    
谐波平均值 的特点是 给予较低值更高的权重  这样只有两者都很高时候  才会获得较高的F1分数
'''
print(f1_score(y_train_5, y_train_pred))

print('-----------------------------------------精度和召回率组权衡--------------------------------------------')

'''
SGDClassifier
                                                                                                        
就是基于决策函数计算一个分值  
然后该值大于阈值 则是预测正值  小于阈值  就是预测负值 
那么 在特定的情况下 我们需要权衡 精度和召回率 

---- 
1.比如商场监控小偷的系统:  召回率  > 精度
2.少儿教育视频平台: 精度 > 召回率
关注的点不同 
我们继续看  SGDClassifier --- 根据每个实例计算决策分数 然后与阈值判断
我们获取每个实例的分数 然后决策分数和阈值判断 
'''
x, y = get_mnist_data_and_target()
y_score = sgd_clf.decision_function([x[36000]])
'''
然后SGDClassifier的阈值为0  下面代码 与sgd的predict一致
'''
print(y_score > 0)
print(y[36000])

'''
提升阈值
'''

print(y_score > 200000)
'''
这样就降低了召回率 提高了精度
'''
print('-----------------------------------------精度和召回率组权衡--确定阈值--------------------------------------------')
'''
返回每个实例决策的分数
'''
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
'''
利用这个决策的分数 计算所有的阈值
return : 精度  召回率  阈值
'''
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

'''
到此为止 两张图就画了出来 
根据场景来选择合适的阈值

假定一个目标 90 精度
从途中来看 阈值达到 70000的时候 精度会达到  90
'''
y_train_pred_90 = (y_scores > 120000)
'''
查看一下他的精度 和召回率  --- 需要自己根据所绘制的图像来看
'''
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

'''
利用matplotlib  绘制 召回率/精度  阈值的函数图
'''
if not plt.isinteractive():
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    plot_recall_vs_precision(precisions, recalls)
    plt.show()
