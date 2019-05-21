#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/28 下午7:15
# @Author  : Aries
# @Site    :
# @File    : test.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from mnist.scikit.learn.utils.DataUtils import get_serialize_data
from PIL import Image

# a = np.array([[1, 2, 3], [7, 8, 9]])
# b = np.array([[4, 5, 6], [1, 2, 3]])
#
# d = np.array([7, 8, 9])
# e = np.array([1, 2, 3])
#
# g = np.r_[a, b]
#
# h = np.r_[d, e]
# print(a)
# print(b)
# print(d)
# print(e)
#
# print('----------------------------------------np.r_-----------------------------------------------')
#
#
# print(g)
# print(h)
#
# print('----------------------------------------np.c_-----------------------------------------------')
#
# g1 = np.c_[a, b]
#
# h1 = np.c_[d, e]
#
# print(g1)
# print(h1)
# x_test = get_serialize_data('X_test')
# data = plt.imread('../assets/num1.jpeg',)
# im = Image.open('../assets/num1.jpeg')
# data1 = np.matrix(im.getdata(), dtype='float') / 255.0
# new_data = np.reshape(data1, (784, 1))
# # data1 = Image.open('../assets/num1.jpeg').getdata()
# print(data)
