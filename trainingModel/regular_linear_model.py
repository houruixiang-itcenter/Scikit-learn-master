# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/3 下午2:30
# @Author  : Aries
# @Site    :
# @File    : regular_linear_model.py
# @Software: PyCharm
'''
正则线性模型
之前有说过介绍过度拟合的一个好方法 就是对模型正则化(即约束它):它拥有的自由度越低,就越不容易过度拟合数据
比如:将多项式模型正则化的简单方法就是降低多项式的阶数

对于线性模型来说,正则化通常是通过约束模型的权重来实现,接下来我们将会讨论以下三种不同的实现方法对权重进行约束:
1.岭回归
2.套索回归
3.弹性网络
'''
import warnings

from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

print('----------------------------------------------------岭回归------------------------------------------------------')
'''
岭回归<吉哄诺夫正则化> 
在成本函数中添加一个正则项 同时是的数据的成本函数最小 还要让权重最小
注意: 训练时候正则项添加到陈本函数中  
训练完之后需要使用原始的成本函数对模型进行评估

------
训练期间的成本函数和测试阶段成本函数不一样是非常常见的   训练时候的成本函数  有时候会使用优化过的衍生函数
α = 0: 岭回归就是一个线性模型
α = ∞  岭回归就是一个穿过数据平均值的水平线
'''
'''
下面来看闭式的岭回归
'''
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = x

'''
这个曲线在训练集和验证集上,关于'训练集大小'的性能函数
要生成这个曲线 只需要在不同大小的训练子集上多次训练模型即可
'''

ridge_reg = Ridge(alpha=0.1, solver='cholesky')

ridge_reg.fit(x, y)
y_predict = ridge_reg.predict([[1.5]])
plt.subplot(211)
plt.plot(x, ridge_reg.predict(x))

'''
随用随机梯度下降进行优化
'''
sgd_reg = SGDRegressor(penalty='l2')
sgd_reg.fit(x, y.ravel())
y_sgd_predict = sgd_reg.predict([[1.5]])
plt.subplot(212)
plt.plot(x, sgd_reg.predict(x))
print(y_predict, y_sgd_predict)
# plt.show()


print('----------------------------------------------------套索回归----------------------------------------------------')
'''
线性回归的另一种正则化,叫做最小绝对收缩和选择算子回归  简称lasso
它于岭回归一样,向成本函数增加一个正则项  但是它是L1范数 而不是 L2范数
Lasso 更加倾向于完全消除掉最不重要的特征的权重,也就是将其设置为0
换句话说Lasso回归会自动执行特征选择并输出一个稀疏矩阵


alpha：正则化强度。 较大的值指定较强的正则化。 Alpha对应于其他线性模型（如Logistic回归或LinearSVC）中的。
fit_intercept：是否计算该模型的截距。如果设置为False，将不会在计算中使用截距（比如，预处理数据已经中心化）
normalize：当fit_intercept设置为False时，该参数将会被忽略。如果为True，则回归前，回归变量X将会进行归一化，减去均值，然后除以L2范数。如果想要标准化，请在评估器（normalize参数为False）调用fit方法前调用StandardScaler，
copy_X：如果是True，x将被复制，否则，有可能被覆盖。
max_iter：共轭梯度求解器的最大迭代次数。对于‘sparse_cg’ 和 ‘lsqr’ 求解器，默认值由 scipy.sparse.linalg确定。对于‘sag’求解器，默认值是1000。
tol：求解方法精度
solver：类型： {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}，计算例程中使用的求解程序。

'''
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x, y)
print(lasso_reg.predict([[1.5]]))

print('---------------------------------------------------弹性回归----------------------------------------------------')
'''
弹性网络是岭回归与Lasso回归之间的中间地带
岭回归是一个默认的选择
如果用到的特征值只有少数几个 那么lasso和弹性网络,因为会把无关的特征降为0



-----
再者 弹性网络性能由于lasso
因为当特征数量超过实例数量或者几个特征强相关时,lasso非常不稳定
l1_ratio  -- r  决定L1和L2范数的权重

lasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
'''
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(x, y)
el_predict = elastic_net.predict([[1.5]])
print(el_predict)

print('---------------------------------------------------早期停止法----------------------------------------------------')
'''
还有一个与众不同的正则化方法 就是验证误差<RMSE>达到最小值时候停止训练,该方法叫做早期停止法


对于随机梯度下降和小批量下降来说曲线并没有那么平滑  可以离开最小值一段距离后停止训练<此时认为之后的训练不会比最小值更小>
这时我们可以将模型回滚倒最小值的状态
'''
'''
下面是早期停止法的基本实现
warm_start = True 调用fit()方法会从停下的地方继续开始训练.而不会重新开始
选择前一次的训练结果继续训练 而不需要从头训练
'''
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate='constant', eta0=0.0005)
minimum_val_error = float('inf')  # + ∞
best_epoch = None
best_model = None
for epoch in range(1, len(x_train)):
    sgd_reg.fit(x_train[:epoch], y_train[:epoch])
    y_val_predict = sgd_reg.predict(x_val)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = sgd_reg
print(best_epoch)
print(best_model.predict([[1.5]]))

print('---------------------------------------------------逻辑回归----------------------------------------------------')
'''
正如  第一章所说,一些回归算法也会用于分类(反之亦然)
-----
逻辑回归(Logistic回归.也称为罗吉思回归),被广泛用于估算一个实例属于某个特定类别的就概率
比如一个垃圾邮件的分类
正类:记为1---预估概率超过50%
负类:记为0---预估概率小于50%
'''
'''
同线性回归一样,基于输入的特征加权和偏置项
但是 不同的是  线性回归输出的是结果  但是逻辑回归输出的是数理逻辑
这里的梳理逻辑 p


----
y = 0  (p < 0.5)  正类
y = 1  (p >= 0.5)  负类

训练模型就是调整参数的过程 也就是 权重的过程 
我们来看 逻辑回归的参数


正类:c = -log(p)  正类中 当p=0时 参数特别大(这是对数函数的特性)  所以用来预测正类很合适
负类:c = -log(1-p) 负类中 当p=1时  参数特别大  所以用来预测负类很合
'''
# todo 成本函数
'''
逻辑回归的成本函数就是所有训练实例的平均成本,它可以记为一个单独的表达式,这个函数被程为损失函数
逻辑回归成本函数 就是log下的  p(y|t)
'''
