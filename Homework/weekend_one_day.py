import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

irs = datasets.load_iris()
irs_data = irs['data']
irs_target = irs['target']
irs_target_names = irs['target_names']

# 划分训练集和测试集
x_data, x_test, y_data, y_test = train_test_split(data=irs_data, columns=irs_target)
print(x_data)