#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 下午8:56
# @Author  : Aries
# @Site    :
# @File    : mnist_operation.py
# @Software: PyCharm
import numpy

from mnist.scikit.learn.utils.DataFetchUtils import get_mnist_data_and_target
import matplotlib
import matplotlib.pyplot as plt

'''
x:
(70000, 784)
总共有70000张图片 每张有784个特征  
每个特征代表一个像素点的强度  从0(白色)到255(黑色)
由于每个图片是28 * 28 像素 
所以你只需要随手抓取一个实例的特征向量.将其重新形成一个28 * 28 的数组即可

end---- 使用Matplotlib的imshow()函数将其显示出来
'''
x, y = get_mnist_data_and_target()

print('------------------------------------展示标签值------------------------------------------------')
print(y[36000])

if __name__ == '__main__':
    some_digit = x[36000]  # type: numpy.ndarray
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()
