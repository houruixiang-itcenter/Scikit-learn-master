#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/28 下午9:13
# @Author  : Aries
# @Site    : 
# @File    : matplotUtils.py
# @Software: PyCharm
import matplotlib.pyplot as plt


def explained_variance_ratio_vs_num(num, explained_variance_ratio):
    plt.xlabel('num')
    plt.ylabel('explained_variance_ratio')
    plt.plot(num, explained_variance_ratio)
    plt.xlim(0, 400)
    plt.ylim(0, 1)
