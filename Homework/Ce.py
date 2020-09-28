import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets

abalone = pd.read_csv(r'Homework\data\abalone.csv')
# 查看前五行数据
print(abalone.head(5))
# 对当前数据进行探查
abalone.info()
# 获取前8列作为实际数据
x_data = abalone.iloc[ :, :-1]  # 实际数据
x_target = abalone.iloc[ :, -1]  # 数据标签--年龄