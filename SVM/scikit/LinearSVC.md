Sklearn.svm.LinearSVC(penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’,fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

penalty : string, ‘l1’ or ‘l2’ (default=’l2’)

指定惩罚中使用的规范。 'l2'惩罚是SVC中使用的标准。 'l1'导致稀疏的coef_向量。

loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)

指定损失函数。 “hinge”是标准的SVM损失（例如由SVC类使用），而“squared_hinge”是hinge损失的平方。

dual : bool, (default=True)

选择算法以解决双优化或原始优化问题。 当n_samples> n_features时，首选dual = False。

tol : float, optional (default=1e-4)

公差停止标准

C : float, optional (default=1.0)

错误项的惩罚参数

multi_class : string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’)

如果y包含两个以上的类，则确定多类策略。 “ovr”训练n_classes one-vs-rest分类器，而“crammer_singer”优化所有类的联合目标。 虽然crammer_singer在理论上是有趣的，因为它是一致的，但它在实践中很少使用，因为它很少能够提高准确性并且计算成本更高。 如果选择“crammer_singer”，则将忽略选项loss，penalty和dual。

fit_intercept : boolean, optional (default=True)

是否计算此模型的截距。 如果设置为false，则不会在计算中使用截距（即，预期数据已经居中）。

intercept_scaling : float, optional (default=1)

当self.fit_intercept为True时，实例向量x变为[x，self.intercept_scaling]，即具有等于intercept_scaling的常量值的“合成”特征被附加到实例向量。 截距变为intercept_scaling *合成特征权重注意！ 合成特征权重与所有其他特征一样经受l1 / l2正则化。 为了减小正则化对合成特征权重（并因此对截距）的影响，必须增加intercept_scaling。

class_weight : {dict, ‘balanced’}, optional

将类i的参数C设置为SVC的class_weight [i] * C. 如果没有给出，所有课程都应该有一个重量。 “平衡”模式使用y的值自动调整与输入数据中的类频率成反比的权重，如n_samples /（n_classes * np.bincount（y））

verbose : int, (default=0)

启用详细输出。 请注意，此设置利用liblinear中的每进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。

random_state : int, RandomState instance or None, optional (default=None)

在随机数据混洗时使用的伪随机数生成器的种子。 如果是int，则random_state是随机数生成器使用的种子; 如果是RandomState实例，则random_state是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。

max_iter : int, (default=1000)
