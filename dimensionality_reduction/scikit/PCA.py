#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/28 下午4:00
# @Author  : Aries
# @Site    :
# @File    : PCA.py
# @Software: PyCharm

'''
PCA是当下最流行的降维算法
他先是识别出最接近数据的超平面,然后将数据投影在上面

注意 数据集绝对不能有缺省 不然会有Nan值出现  影响code的运行
'''
from os import path
from tempfile import mkdtemp

from command.DataUtils import get_serialize_data
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import matplotlib.pyplot as plt

from dimensionality_reduction.scikit.utils.matplotUtils import explained_variance_ratio_vs_num

'''
保留差异性:将更多的数据投影,尽量最大程度的避免数据丢失

主成分:在训练中识别哪条轴对差异性的贡献度高
一般会有许多主成分的轴,最后根据这些转化成多维空间
'''

X = get_serialize_data('X', 5)
Y = get_serialize_data('Y', 5)

# 数据集中
'''
axis=1 求每一行的平均值
axis=0 求每一列的平均值
V.T 就是我们所有想要的成分
'''
X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X)
# # 主成分
c1 = V.T[:, 0]
c2 = V.T[:, 1]

# 将训练集投影到由前两个主成分定义的平面上

w2 = V.T[:, :2]
X2D = X_centered.dot(w2)
print(X2D)

print('------------------------------------------------PCA-----------------------------------------------------')
'''
使用Scikit-Learn
Scikit-Learn PCA类也是使用SVD分解来实现主成分分析,以下代码就是PCA将数据集降到二维
'''
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(X2D)

'''
可以直接访问主成分
'''
print(pca.components_)
print('------------------------------------------------方差解释器-----------------------------------------------------')
'''
还有一个非常有用的指标--方差解释率
表示每一个主成分对整个数据集的方差贡献度
'''
print(pca.explained_variance_ratio_)
print('---------------------------------------------选择正确数量的维度-------------------------------------------------')
'''
如何选择正确的维度:
1.选择主成分集合中,靠前的几个元素,使用靠前的主成分的方差解释率一次相加,直到得到足够大比例的方差,此时便是最好的维度

2.如果要可视化观察数据集,可以直接降到二维或者三维
'''
pca0 = PCA()
pca0.fit(X)
cumsum = np.cumsum(pca0.explained_variance_ratio_)
# argmax 取出最大索引
d = np.argmax(cumsum >= 0.95) + 1
print(d)
'''
从上面来看 7维的超空间就是很好的降维选择

当然还可以在PCA直接指定 希望方差概率累加为0.95 
'''
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(X_reduced)

'''
绘制解释方差和维度数量的函数
'''
# explained_variance_ratio_vs_num(range(0, len(cumsum)), cumsum)
# plt.show()
print('---------------------------------------------MNIST-------------------------------------------------')
'''
housing 数据特征太少 我们来使用MNIST
'''
X_train = get_serialize_data('X_train', 5)
pca = PCA()
X_reduced = pca.fit_transform(X_train)
print(X_reduced)
cumsum = np.cumsum(pca.explained_variance_ratio_)
explained_variance_ratio_vs_num(range(0, 400), cumsum[:400])
# plt.show()

print('---------------------------------------------压缩PCA-------------------------------------------------')
'''
从上面MNIST的降维来看,MNIST原有特征维784
降维之后<我们选择95%的方差解释率的叠加>  这样特征会变为150多个 
这保留了绝大多数差异性的同时,数据集的大小压缩为原来的20%
这样便极大的提升了训练的速度
'''
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(X_reduced.shape[1])

'''
同理原始数据压缩之后 也可以解压缩恢复
但是由于我们是选取95%的方差解释率叠加进行压缩的
所以解压缩之后的数据会与原始数据存在差值,但是他很大可能的接近于原始数据

原始数据和重建数据(解压缩之后的数据)之间的均方距离,称为重建误差
'''
x_mnist_recovered = pca.inverse_transform(X_reduced)
print(x_mnist_recovered.shape[1])
print('---------------------------------------------增量PCA-------------------------------------------------')
'''
之前的PCA必须让整个数据集进入内存,才可以进行降维
---------------------------
幸运的是 我们可以使用增量PCA算法(IPCA):
这样将训练集分成一个个的小批量,一次给IPCA算法喂一个;
对于大型的数据集来说,这样做比较节约内存,而且可以支持在线应用PCA
????我觉得应该支持局部更新这样就很棒了



code :Scikit-Learn 方式一
1.将MNIST分成100个小批量(使用Numpy的array_split()函数)
2.然后一次次的喂给Scikit-Learn的IncrementalPCA
3.但是每次必须为每个小批量使用partial_fit()方法
'''
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for x_batch in np.array_split(X_train, n_batches):
	# 部分适配
	inc_pca.partial_fit(x_batch)

X_mnist_reduced = inc_pca.transform(X_train)

'''
方式二:
使用Numpy的memmap类,他允许巧妙的操控一个存储在磁盘中的二进制文件里的大型数组
而且memmap这个类仅在需要加载到内存中的时候才会加载
IncrementalPCA虽然是分割训练集进行批量的训练
但是在运行期间还是会占用内存

明天好好看看
'''
X_mm = np.memmap('../assets/X_train', mode='readonly', shape=(154, 6000))
batch_size = 100
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

print('---------------------------------------------随机PCA-------------------------------------------------')
'''
随机PCA的优势就是时间复杂度可控

当明确要求取前d个主成分(此时如果d 远远小于n),则使用这个会好一点
'''
rnd_pca = PCA(n_components=154, svd_solver='randomized')
X_rnd_reduced = rnd_pca.fit_transform(X_train)
print(X_rnd_reduced.shape)

print('---------------------------------------------核主成分分析-------------------------------------------------')
'''
降维时候同样可以使用核技巧
作用:使复杂的非线性投影降维成为可能

这里我们使用RBF(高斯相似度)核函数
'''
rbf_pca = KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
x_rbf_reduced = rbf_pca.fit_transform(X_train)
print(x_rbf_reduced.shape)