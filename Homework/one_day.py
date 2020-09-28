import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn as skl  # 导入sklearn库
from sklearn import datasets # 导入数据集
from sklearn.neighbors import KNeighborsClassifier # 从neighbors模块导入KN算法

'''
1.	导入sklearn库。
2.	查看当前sklearn的版本。
3.	导入sklearn库的内置数据集breast_cancer，表示医疗机构的乳腺癌数据，主要做分类或聚类。
4.	读取当前数据，并查看其数据的类型。
5.	获取当前数据的所有键的值。
6.	分别使用变量获取当前数据集的实际数据、数据集标签和数据特征名称内部信息数据，并查看数据以及数据的形状。完成后请认真观察其各项数据的关联性。
7.	导入sklearn库的数据集划分模块
8.	将当前数据集的80%数据划分为训练集，剩余的为测试集。并分别查看相关数据的形状。
            train_test_split
'''

if __name__ == "__main__":
    print('Sklearn版本:{}'.format(skl.__version__))
    bc = datasets.load_breast_cancer() # 导入医疗机构的乳腺癌数据
    pd_bc = pd.DataFrame(bc.data, columns=bc.feature_names) # 数据集转换DataFrame
    bc_X = bc.data  # 获取实际数据集
    bc_y = bc.target  # 获取数据集标签
    print(bc_y)
    bc_descr = pd_bc.info()  # 获取医疗机构乳腺癌数据内部信息
    print(bc_descr)
    train_X, test_X, train_y, test_y = train_test_split(bc_X, bc_y,train_size=0.8)
    print('相关数据的形状:train_X:{} train_y:{} test_X:{} test_y:{}'.format(train_X.shape,train_y.shape,test_X.shape,test_y.shape))
    print('查看数据类型:{}'.format(bc_X.dtype))
    print('获取当前数据的所有Keys:{}'.format(bc.keys()))
    