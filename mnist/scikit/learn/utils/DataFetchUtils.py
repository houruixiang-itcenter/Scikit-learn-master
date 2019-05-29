#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/15 下午11:39
# @Author  : Aries
# @Site    : 
# @File    : DataFetchUtils.py
# @Software: PyCharm

'''
第一步获取数据
'''
import os

from sklearn.datasets import fetch_mldata, get_data_home

from mnist.scikit.learn.utils.DataUtils import serialize_data, get_serialize_data


def get_mnist():
    '''
    存储路径 使用绝对路径 这样不会受到文件位置的干扰
    :return:
    '''
    try:
        if get_serialize_data('mnist') is None:
            print('mnist is none')
            mnist = fetch_mldata('MNIST original')
            serialize_data(mnist, 'mnist')
        else:
            print('mnist is not none')
            mnist = get_serialize_data('mnist')
    except FileNotFoundError as e:
        print('mnist is FileNotFoundError')
        mnist = fetch_mldata('MNIST original')
        serialize_data(mnist, 'mnist')
    '''
      {'DESCR': 'mldata.org dataset: mnist-original', 
      'COL_NAMES': ['label', 'data'], 
      'target': array([0., 0., 0., ..., 9., 9., 9.]), 
      'data': array([[0, 0, 0, ..., 0, 0, 0],
             [0, 0, 0, ..., 0, 0, 0],
             [0, 0, 0, ..., 0, 0, 0],
             ...,
             [0, 0, 0, ..., 0, 0, 0],
             [0, 0, 0, ..., 0, 0, 0],
             [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)}
      '''
    return mnist


mnist = get_mnist()
'''
test
来看数据几种的这些数组
'''


def get_mnist_data_and_target():
    mnist = get_mnist()
    x, y = mnist['data'], mnist['target']
    return x, y


x, y = get_mnist_data_and_target()
print(x.shape)
'''
x:
(70000, 784)
总共有70000张图片 每张有784个特征  
每个特征代表一个像素点的强度  从0(白色)到255(黑色)
由于每个图片是28 * 28 像素 
所以你只需要随手抓取一个实例的特征向量.将其重新形成一个28 * 28 的数组即可

end---- 使用Matplotlib的imshow()函数将其显示出来
'''
print(y.shape)


