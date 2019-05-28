#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 下午8:57
# @Author  : Aries
# @Site    : 
# @File    : DataUtils.py
# @Software: PyCharm
import os

from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# todo 全局存储序列化文件的位置
path = '/Users/houruixiang/python/Scikit-learn-master/command/assets'


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


def select_path(mode):
    if mode == 0:
        path = '/Users/houruixiang/python/Scikit-learn-master/command/assets'
    elif mode == 1:
        path = '/Users/houruixiang/python/Scikit-learn-master/SVM/assets'
    elif mode == 2:
        path = '/Users/houruixiang/python/Scikit-learn-master/decision_tree/assets'
    elif mode == 3:
        path = '/Users/houruixiang/python/Scikit-learn-master/Random_Forests/scikit/assets'
    elif mode == 5:
        path = '/Users/houruixiang/python/Scikit-learn-master/dimensionality_reduction/assets'
    return path


def serialize_data(tag_model, tag, mode=0):
    '''
    序列化data
    :param tag_model:序列化的对象
    :param tag:序列化文件存放位置
    :return:
    '''
    path = select_path(mode)
    tag = os.path.join(path, tag)
    joblib.dump(tag_model, tag)


def get_serialize_data(tag, mode):
    '''
    根据tag反序列化
    :param tag:
    :return:
    返回反序列化结果
    '''
    path = select_path(mode)
    tag = os.path.join(path, tag)
    return joblib.load(tag)


# todo 利用matplotlib  绘制 精度/召回率  vs 阈值的函数图
def plot_precision_recall_vs_threshold(precisions, recalls, threshiolds):
    plt.subplot(211)
    plt.plot(threshiolds, precisions[:-1], 'b--', label='precisions')
    plt.plot(threshiolds, recalls[:-1], 'g-', label='recalls')
    plt.xlabel('threshiolds')
    plt.legend(loc='upper left')
    plt.xlim(-600000, 600000)
    plt.ylim(0, 1)


# todo 利用matplotlib  绘制 召回率  vs 进度的函数图
def plot_recall_vs_precision(precisions, recalls):
    plt.subplot(212)
    plt.plot(recalls, precisions, label='precisions')
    plt.xlabel('recalls')
    plt.ylabel('precisions')
    plt.legend(loc='upper left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)


# todo 利用matplotlib  绘制 真正率  vs 假正率
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')  # 假正率
    plt.ylabel('TPR')  # 真正率


# todo plt 绘制数字  --- 多图绘制
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=plt.cm.binary, **options)
    plt.axis("off")


# todo plt 绘制数字  --- 单图绘制
def plot_digits_angle(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # images = [instance.reshape(size, size) for instance in instances]
    image = instances.reshape(size, size)
    # n_rows = (len(instances) - 1) // images_per_row + 1
    # row_images = []
    # n_empty = n_rows * images_per_row - len(instances)
    # images.append(np.zeros((size, size * n_empty)))
    # for row in range(n_rows):
    #     rimages = images[row * images_per_row: (row + 1) * images_per_row]
    #     row_images.append(np.concatenate(rimages, axis=1))
    # image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=plt.cm.binary, **options)
    plt.axis("off")
