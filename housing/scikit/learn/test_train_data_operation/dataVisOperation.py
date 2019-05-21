#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 下午5:26
# @Author  : Aries
# @Site    :
# @File    : dataVisOperation.py
# @Software: PyCharm
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from housing.scikit.learn.test_train_data_operation.TestAndTrainOperation import get_train_test_set

train_set, test_set = get_train_test_set()
print('-----------------------------训练数据进行可视化处理-------------------------------------')
# todo 为了不影响训练数据 这里基于训练数据进行处理
assert isinstance(train_set, DataFrame)
housing = train_set.copy(deep=True)
print('housing:   ', len(housing))
# todo 将地理数据可视化
assert isinstance(housing, DataFrame)
# todo 这个alpha这个参数很重要 可以看出高密度据点  kind = scatter 这里代表是分散
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# plt.show()  # 但是有一个问题这除了看起来和加利福尼亚的地图板块一样  其他什么都看不出来
# todo 上面的表格 与我们的研究而言其实意义不大  接着看下面的可视化
# todo 其中圆的半径代表当地人口的数量<s> 颜色代表价格<c 颜色范围从蓝<低>到红<高>>
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population'] / 100,
             label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
# plt.show()

print('-----------------------------------寻找相关性-----------------------------------------')
print('-----------------------------------方式1-----------------------------------------')
# todo 由于数据量不大 可以使用corr()方法轻松算数没对属性之间的标准相关系数
corr_matrix = housing.corr()
# todo 看看每个属性和房屋中位数的相关性分别是多少:  ascending-True:升序排列  ascending-False:降序排列
corr_value = corr_matrix['median_house_value'].sort_values(ascending=False)
print(type(corr_value))
print(corr_value)
print('-----------------------------------方式2-----------------------------------------')
# todo 上面的操作的确可以获取各个属性与房价中位数的相关性
# todo  可以看到有几个和房价中位数有正相关  所以我们需要查看数据集中各个属性之间的相关性
# todo pandas中scatter_matrix会绘制每个属性和各个属性之间的相关性图  <为了方便观看 仅仅展示4个潜力属性>
attr = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attr], figsize=(12, 5))
plt.show()
# 在整理中发现 房价中位数为50k/35k时候   与收入中位数无关 这部分怪异数据不具有代表性  所以可以酌情移除掉










