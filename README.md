**场景一:  预测加州的房价然后进行投资的预测**

提供:人口 人均GDP  经纬度 

选择性能指标 :
`_RMSE --- 均方根误差_  `
`_MAE --- 平均绝对误差_`


创建工作区:
首先 下载python模块:
- Jupyter---jupyter notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn


.... done

建议建立一个隔离区来使用

就是说隔离一部分python环境到这个隔离区中
隔离的环境路径  $HOME/bin
virtualenv env


激活环境上面的隔离环境
cd $HOME/ml
source env/bin/activate

然后这样 pip/pip3下载的包就会  集成导这里 
当然 你还需要在setting里面 切换一下python路径 来进行开发



- housing --- 加州房价预测
- mnist  ---- 数字 1~9 分类模型
- SVM --- SVM向量机 (分类 + 回归)
- trainingModel --- 训练模型
- decision_tree --- 决策树(分类 + 回归)
- Random_Forests --- 随机森林



